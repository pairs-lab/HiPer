"""
seliSlam.object_encoder.bye_encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-scene contrastive point cloud encoder.

Implements BYE (Huang et al. 2025, IEEE RA-L) with UAV extensions:
  - DGCNN or PointNet backbone
  - NTXent contrastive loss (SimCLR / Sohn 2016)
  - UAV-specific augmentations (altitude jitter, top-down angle)
  - Farthest-point sampling + voxel downsampling

Training (end of session):
  train_session(observations, augmentor)

Inference (per keyframe):
  encode(partial_point_cloud) -> embedding (embed_dim,)

The encoder is "per-scene": trained fresh on each deployment's
exploration data. A checkpoint is saved to disk between sessions.
"""

import os
import numpy as np
import rospy

_CKPT_PATH = os.path.expanduser("~/.seliSlam/checkpoints/bye_encoder.pt")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# PointNet backbone (simplified)
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:

    class _PointNetEncoder(nn.Module):
        """
        PointNet global feature extractor.
        Input : (B, 3, N)  — zero-centred point cloud
        Output: (B, embed_dim)
        """
        def __init__(self, embed_dim: int = 256):
            super().__init__()
            self.mlp1 = nn.Sequential(
                nn.Conv1d(3, 64,  1), nn.BatchNorm1d(64),  nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, embed_dim),
            )

        def forward(self, x):
            # x: (B, 3, N)
            feat = self.mlp1(x)                # (B, 256, N)
            feat = feat.max(dim=2).values      # (B, 256) global max-pool
            return self.mlp2(feat)             # (B, embed_dim)

    class _DGCNNEncoder(nn.Module):
        """
        Simplified DGCNN encoder (Wang et al. 2019).
        Uses k-NN graph edge convolutions for local geometry.
        Input : (B, 3, N)
        Output: (B, embed_dim)
        """
        def __init__(self, embed_dim: int = 256, k: int = 10):
            super().__init__()
            self.k = k
            self.conv1 = nn.Sequential(
                nn.Conv2d(6, 64, 1, bias=False),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
            self.conv2 = nn.Sequential(
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
            self.mlp = nn.Sequential(
                nn.Linear(192, 256), nn.ReLU(),
                nn.Linear(256, embed_dim),
            )

        def forward(self, x):
            B, C, N = x.shape
            # Build edge features for layer 1
            e1 = self._edge_conv(x, self.conv1)   # (B, 64, N)
            e2 = self._edge_conv(e1, self.conv2)  # (B, 128, N)
            # Concatenate + global pool
            cat = torch.cat([e1.max(2).values,
                              e2.max(2).values,
                              x.max(2).values], dim=1)  # (B, 64+128+C)
            return self.mlp(cat)                         # (B, embed_dim)

        def _edge_conv(self, x, conv):
            B, C, N = x.shape
            k = min(self.k, N)
            # Pairwise distances for k-NN
            xx = (x ** 2).sum(1, keepdim=True)          # (B,1,N)
            dist = xx + xx.transpose(2, 1) - 2 * torch.bmm(
                x.transpose(2, 1), x)                   # (B,N,N)
            idx  = dist.topk(k, dim=2, largest=False).indices  # (B,N,k)
            # Gather neighbours
            idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
            idx_flat  = (idx + idx_base).view(-1)
            x_flat    = x.transpose(2, 1).contiguous().view(B * N, C)
            nbrs      = x_flat[idx_flat].view(B, N, k, C)
            x_rep     = x.transpose(2, 1).unsqueeze(2).expand_as(nbrs)
            edge_feat = torch.cat([x_rep, nbrs - x_rep], dim=3)  # (B,N,k,2C)
            edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()
            out = conv(edge_feat)                        # (B, F, N, k)
            return out.max(dim=3).values                 # (B, F, N)

    class _ProjectionHead(nn.Module):
        """SimCLR projection head for contrastive loss computation."""
        def __init__(self, embed_dim: int = 256, proj_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim),
            )

        def forward(self, x):
            return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# NTXent loss (SimCLR / Sohn 2016 multi-class N-pair)
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:
    def ntxent_loss(embeddings: "torch.Tensor",
                    labels: "torch.Tensor",
                    temperature: float = 0.1) -> "torch.Tensor":
        """
        NTXent loss as used in BYE (SimCLR, Chen et al. 2020).
        embeddings : (B, proj_dim) L2-normalised
        labels     : (B,) integer object IDs
        """
        B = embeddings.shape[0]
        # Cosine similarity matrix
        sim = torch.mm(embeddings, embeddings.T) / temperature  # (B, B)

        # Mask out self-similarity
        eye_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        sim.masked_fill_(eye_mask, float("-inf"))

        # For each anchor i, the positive is any j with same label
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        label_eq.masked_fill_(eye_mask, False)

        # Log-softmax over all negatives
        log_prob = F.log_softmax(sim, dim=1)

        # Loss: mean over all (anchor, positive) pairs
        loss = -(log_prob * label_eq.float()).sum(1)
        denom = label_eq.float().sum(1).clamp(min=1)
        return (loss / denom).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main encoder class
# ─────────────────────────────────────────────────────────────────────────────

class BYEEncoder:
    """
    BYE per-scene encoder wrapper.

    Usage
    -----
    # Inference (per keyframe):
    emb = encoder.encode(partial_pts)   # (embed_dim,) numpy

    # End-of-session training:
    encoder.train_session(observations, augmentor)
    encoder.save_checkpoint()
    """

    def __init__(
        self,
        embed_dim: int = 256,
        backbone: str = "dgcnn",    # "dgcnn" | "pointnet"
        num_points: int = 1024,
        knn_k: int = 10,
        device: str = "cuda",
    ):
        self._embed_dim  = embed_dim
        self._num_points = num_points
        self._device     = None
        self._stub       = not _TORCH_OK

        if not self._stub:
            import torch
            self._device = torch.device(
                device if torch.cuda.is_available() else "cpu")
            if backbone == "dgcnn":
                self._encoder = _DGCNNEncoder(embed_dim, k=knn_k).to(self._device)
            else:
                self._encoder = _PointNetEncoder(embed_dim).to(self._device)
            self._proj_head = _ProjectionHead(embed_dim).to(self._device)
            self._optimiser = torch.optim.Adam(
                list(self._encoder.parameters()) +
                list(self._proj_head.parameters()),
                lr=3e-3,
            )

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, pts: np.ndarray) -> np.ndarray:
        """
        Encode a zero-centred partial point cloud.
        pts : (N, 3) float32
        Returns : (embed_dim,) float32 L2-normalised embedding.
        """
        if self._stub:
            return np.random.randn(self._embed_dim).astype(np.float32)

        pts = self._preprocess(pts)   # subsample to num_points
        import torch
        t = torch.from_numpy(pts.T[np.newaxis]).to(self._device)  # (1,3,N)
        self._encoder.eval()
        with torch.no_grad():
            emb = self._encoder(t).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(emb)
        return (emb / (norm + 1e-8)).astype(np.float32)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_session(
        self,
        observations: list,    # list of (pts: np.ndarray, object_id: int)
        augmentor=None,
        epochs: int = 300,
        batch_size: int = 64,
    ):
        """
        Train the encoder on this session's observations.
        observations: list of (pts, object_id) tuples.
        augmentor   : UAVPointCloudAugmentor instance (optional).
        """
        if self._stub or len(observations) < 2:
            return

        import torch
        self._encoder.train()
        self._proj_head.train()

        # Build dataset
        all_pts = []
        all_ids = []
        for pts, oid in observations:
            pp = self._preprocess(pts)
            all_pts.append(pp)
            all_ids.append(int(oid))
        all_pts = np.stack(all_pts, axis=0)   # (L, num_points, 3)
        all_ids = np.array(all_ids)

        n_samples = all_pts.shape[0]

        for epoch in range(epochs):
            # Sample batch (64 anchors + 1 positive per anchor = 128)
            idx = np.random.choice(n_samples,
                                   size=min(batch_size, n_samples),
                                   replace=False)
            batch_pts = all_pts[idx]
            batch_ids = all_ids[idx]

            # Augment
            if augmentor is not None:
                batch_pts = np.stack(
                    [augmentor.augment(p) for p in batch_pts])

            # Positive pairs: for each sample, find another with same id
            pos_pts = []
            for i, oid in enumerate(batch_ids):
                same = np.where(all_ids == oid)[0]
                if len(same) > 1:
                    j = np.random.choice(same[same != idx[i]])
                    pos = self._preprocess(all_pts[j])
                else:
                    pos = augmentor.augment(batch_pts[i]) \
                          if augmentor else batch_pts[i].copy()
                pos_pts.append(pos)
            pos_pts = np.stack(pos_pts)

            # Concatenate anchor + positive
            combined_pts = np.concatenate([batch_pts, pos_pts], axis=0)
            combined_ids = np.concatenate([batch_ids, batch_ids])

            t_pts = torch.from_numpy(
                combined_pts.transpose(0, 2, 1)).float().to(self._device)
            t_ids = torch.from_numpy(combined_ids).long().to(self._device)

            self._optimiser.zero_grad()
            h = self._encoder(t_pts)          # (2B, embed_dim)
            g = self._proj_head(h)            # (2B, proj_dim)
            g = F.normalize(g, dim=1)
            loss = ntxent_loss(g, t_ids)
            loss.backward()
            self._optimiser.step()

            if epoch % 50 == 0:
                rospy.loginfo(
                    f"[bye_encoder] Epoch {epoch:3d}/{epochs}  "
                    f"loss={loss.item():.4f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_checkpoint(self):
        if self._stub:
            return
        import torch
        os.makedirs(os.path.dirname(_CKPT_PATH), exist_ok=True)
        torch.save({
            "encoder":   self._encoder.state_dict(),
            "proj_head": self._proj_head.state_dict(),
        }, _CKPT_PATH)
        rospy.loginfo(f"[bye_encoder] Checkpoint saved to {_CKPT_PATH}")

    def try_load_checkpoint(self):
        if self._stub or not os.path.exists(_CKPT_PATH):
            return
        import torch
        ckpt = torch.load(_CKPT_PATH, map_location=self._device)
        self._encoder.load_state_dict(ckpt["encoder"])
        self._proj_head.load_state_dict(ckpt["proj_head"])
        rospy.loginfo(f"[bye_encoder] Loaded checkpoint from {_CKPT_PATH}")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, pts: np.ndarray) -> np.ndarray:
        """
        Voxel downsample then farthest-point sample to self._num_points.
        Returns (num_points, 3) float32.
        """
        if pts.shape[0] == 0:
            return np.zeros((self._num_points, 3), dtype=np.float32)

        # Voxel downsample (0.01 m grid)
        pts = self._voxel_downsample(pts, voxel_size=0.01)

        N = pts.shape[0]
        if N >= self._num_points:
            pts = self._fps(pts, self._num_points)
        else:
            # Repeat points to reach num_points (padding)
            reps = int(np.ceil(self._num_points / N))
            pts  = np.tile(pts, (reps, 1))[:self._num_points]

        return pts.astype(np.float32)

    @staticmethod
    def _voxel_downsample(pts: np.ndarray,
                           voxel_size: float = 0.01) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        voxel_indices = np.floor(pts / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return pts[unique_idx]

    @staticmethod
    def _fps(pts: np.ndarray, n: int) -> np.ndarray:
        """Farthest point sampling — O(N·n) numpy implementation."""
        N = pts.shape[0]
        selected = np.zeros(n, dtype=np.int32)
        distances = np.full(N, np.inf)
        farthest  = np.random.randint(0, N)
        for i in range(n):
            selected[i] = farthest
            centroid    = pts[farthest]
            d = np.sum((pts - centroid) ** 2, axis=1)
            distances   = np.minimum(distances, d)
            farthest    = np.argmax(distances)
        return pts[selected]
