"""
seliSlam.vio.depth_pose_net
~~~~~~~~~~~~~~~~~~~~~~~~~~~
DepthNet + PoseNet with a complete, mathematically correct
photometric reprojection loss.

Loss (Monodepth2 / CoVIO):
  L = L_photometric + gamma * L_smoothness + lambda * L_velocity

Photometric loss:
  1. View synthesis via inverse warping (back-project → transform → project)
  2. SSIM + L1 mixture (alpha = 0.85)
  3. Auto-masking: exclude pixels where the unwarped image is already
     closer to the target than the warped one
  4. Min-over-sources: take the minimum loss over all source frames

Rotation representation: axis-angle → SO(3) via Rodrigues formula.
"""

import os
import numpy as np
import rospy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    rospy.logwarn("[depth_pose_net] PyTorch not found — stub mode.")

from sensor_msgs.msg import Image as ROSImage

_CKPT_DIR = os.path.expanduser("~/.seliSlam/checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# Network sub-modules
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:

    class _DepthDecoder(nn.Module):
        """Multi-scale disparity decoder on top of ResNet-18 features."""

        def __init__(self, enc_ch: int = 512):
            super().__init__()
            self.up1 = nn.Sequential(
                nn.Conv2d(enc_ch, 256, 3, padding=1), nn.ELU(inplace=True))
            self.up2 = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1), nn.ELU(inplace=True))
            self.up3 = nn.Sequential(
                nn.Conv2d(128, 64,  3, padding=1), nn.ELU(inplace=True))
            self.up4 = nn.Sequential(
                nn.Conv2d(64,  32,  3, padding=1), nn.ELU(inplace=True))
            self.up5 = nn.Sequential(
                nn.Conv2d(32,  16,  3, padding=1), nn.ELU(inplace=True))
            self.head = nn.Conv2d(16, 1, 3, padding=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            for layer in (self.up1, self.up2, self.up3, self.up4, self.up5):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)
            return torch.sigmoid(self.head(x))      # disparity in (0, 1)

    class _PoseDecoder(nn.Module):
        """
        Takes two ResNet-18 feature maps → 6-DOF relative pose.
        Output: axis-angle (B,3), translation (B,3)
        """

        def __init__(self, enc_ch: int = 512):
            super().__init__()
            self.squeeze = nn.Conv2d(enc_ch * 2, 256, 1)
            self.conv1   = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
            self.conv2   = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
            self.head    = nn.Conv2d(256, 6, 1)
            self.head.weight.data *= 0.01
            self.head.bias.data.zero_()

        def forward(self, f_a: "torch.Tensor",
                    f_b: "torch.Tensor") -> tuple:
            x   = torch.cat([f_a, f_b], dim=1)
            x   = self.squeeze(x)
            x   = self.conv1(x)
            x   = self.conv2(x)
            out = self.head(x).mean(dim=[2, 3])     # global avg → (B, 6)
            return out[:, :3], out[:, 3:]            # axis-angle, translation


# ─────────────────────────────────────────────────────────────────────────────
# Geometric utilities
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_OK:

    def axisangle_to_matrix(aa: "torch.Tensor") -> "torch.Tensor":
        """
        Rodrigues rotation formula.
        aa : (B, 3)  axis * angle
        Returns (B, 4, 4) SE(3) rotation matrices (translation = 0).
        """
        B     = aa.shape[0]
        angle = aa.norm(dim=1, keepdim=True).clamp(min=1e-8)
        axis  = aa / angle                          # (B, 3)
        a     = angle.squeeze(1)                    # (B,)

        ca = torch.cos(a)
        sa = torch.sin(a)

        # Skew-symmetric K for axis
        zero = torch.zeros(B, device=aa.device, dtype=aa.dtype)
        K = torch.stack([
            torch.stack([ zero,      -axis[:,2],  axis[:,1]], dim=1),
            torch.stack([ axis[:,2],  zero,       -axis[:,0]], dim=1),
            torch.stack([-axis[:,1],  axis[:,0],  zero],      dim=1),
        ], dim=1)                                   # (B, 3, 3)

        eye3 = torch.eye(3, device=aa.device, dtype=aa.dtype)[None].expand(B,-1,-1)
        R = eye3 + sa[:,None,None] * K + (1 - ca)[:,None,None] * (K @ K)

        T = torch.eye(4, device=aa.device, dtype=aa.dtype)[None].expand(B,-1,-1).clone()
        T[:, :3, :3] = R
        return T                                    # (B, 4, 4)

    def build_T(aa: "torch.Tensor",
                t:  "torch.Tensor") -> "torch.Tensor":
        """Combine axis-angle + translation into (B,4,4) SE(3)."""
        T = axisangle_to_matrix(aa)
        T[:, :3, 3] = t
        return T

    def warp(src: "torch.Tensor",
             depth: "torch.Tensor",
             T:     "torch.Tensor",
             K:     "torch.Tensor") -> "torch.Tensor":
        """
        Inverse-warp src into the target frame.

        src   : (B, C, H, W)   source image
        depth : (B, 1, H, W)   predicted depth of the *target* frame
        T     : (B, 4, 4)      T_src_from_tgt  (target → source transform)
        K     : (B, 3, 3)      camera intrinsics

        Returns warped src : (B, C, H, W)

        Steps
        -----
        1. Pixel grid → normalised rays via K_inv
        2. Scale rays by depth → 3D points in target frame
        3. Transform to source frame via T
        4. Project to source pixels via K
        5. Bilinear sample src at projected coordinates
        """
        B, _, H, W = depth.shape
        dev = depth.device

        # Pixel grid
        u = torch.arange(W, dtype=torch.float32, device=dev)
        v = torch.arange(H, dtype=torch.float32, device=dev)
        vv, uu = torch.meshgrid(v, u, indexing="ij")          # (H, W)
        ones   = torch.ones_like(uu)
        pix    = torch.stack([uu, vv, ones], 0).view(1, 3, -1).expand(B,-1,-1)

        # Back-project
        K_inv  = torch.linalg.inv(K)                          # (B, 3, 3)
        rays   = K_inv.bmm(pix)                               # (B, 3, HW)
        d_flat = depth.view(B, 1, -1)
        pts    = rays * d_flat                                 # (B, 3, HW)

        # Homogeneous → transform
        ones3  = torch.ones(B, 1, H * W, device=dev)
        pts_h  = torch.cat([pts, ones3], dim=1)               # (B, 4, HW)
        pts_s  = T.bmm(pts_h)                                 # (B, 4, HW)

        # Project into source frame
        proj   = K.bmm(pts_s[:, :3])                          # (B, 3, HW)
        pix_s  = proj[:, :2] / proj[:, 2:3].clamp(min=1e-7)

        # Normalise to [-1, 1] for grid_sample
        pix_s[:, 0] = (pix_s[:, 0] / (W - 1)) * 2 - 1
        pix_s[:, 1] = (pix_s[:, 1] / (H - 1)) * 2 - 1
        grid = pix_s.permute(0, 2, 1).view(B, H, W, 2)       # (B, H, W, 2)

        return F.grid_sample(src, grid, mode="bilinear",
                             padding_mode="border",
                             align_corners=True)               # (B, C, H, W)

    def ssim_map(x: "torch.Tensor",
                 y: "torch.Tensor") -> "torch.Tensor":
        """Per-pixel SSIM. Returns (B, 1, H, W)."""
        C1, C2 = 0.01**2, 0.03**2

        def p(t):
            return F.avg_pool2d(t, 3, 1, 1)

        mx, my   = p(x), p(y)
        sx       = p(x*x)  - mx**2
        sy       = p(y*y)  - my**2
        sxy      = p(x*y)  - mx*my

        num = (2*mx*my + C1) * (2*sxy + C2)
        den = (mx**2 + my**2 + C1) * (sx + sy + C2)
        return (num / (den + 1e-8)).mean(1, keepdim=True)

    def photo_loss(tgt: "torch.Tensor",
                   warped: "torch.Tensor",
                   alpha: float = 0.85) -> "torch.Tensor":
        """α*(1-SSIM)/2 + (1-α)*|tgt-warped|  →  (B,1,H,W)"""
        l1   = (tgt - warped).abs().mean(1, keepdim=True)
        ssim = (1 - ssim_map(tgt, warped)) * 0.5
        return alpha * ssim + (1 - alpha) * l1

    def edge_smooth(disp: "torch.Tensor",
                    img:  "torch.Tensor") -> "torch.Tensor":
        """Edge-aware smoothness loss."""
        mean_d = disp.mean([2,3], keepdim=True).clamp(1e-7)
        dn = disp / mean_d
        gx_d = (dn[:,:,:,:-1] - dn[:,:,:,1:]).abs()
        gy_d = (dn[:,:,:-1,:] - dn[:,:,1:,:]).abs()
        gx_i = (img[:,:,:,:-1] - img[:,:,:,1:]).abs().mean(1,keepdim=True)
        gy_i = (img[:,:,:-1,:] - img[:,:,1:,:]).abs().mean(1,keepdim=True)
        return (gx_d * (-gx_i).exp()).mean() + (gy_d * (-gy_i).exp()).mean()

    def disp_to_depth(disp: "torch.Tensor",
                      min_d: float = 0.1,
                      max_d: float = 100.0) -> "torch.Tensor":
        scaled = 1.0/max_d + (1.0/min_d - 1.0/max_d) * disp
        return 1.0 / scaled.clamp(min=1e-6)


def default_K(H: int, W: int,
              device=None) -> "torch.Tensor":
    fx = fy = 0.8 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx],[0, fy, cy],[0, 0, 1]],
                      dtype=torch.float32)
    if device:
        K = K.to(device)
    return K.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DepthPoseNet:

    def __init__(self, freeze_encoder: bool = True,
                 device: str = "cuda"):
        self._freeze = freeze_encoder
        self._stub   = not _TORCH_OK
        self._K      = None

        if not self._stub:
            import torch
            self._dev = torch.device(
                device if torch.cuda.is_available() else "cpu")
            self._build()

        os.makedirs(_CKPT_DIR, exist_ok=True)
        self._ckpt = os.path.join(_CKPT_DIR, "generalizer.pt")

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        def _resnet18_enc():
            r = tvm.resnet18(weights=None)
            return nn.Sequential(
                r.conv1, r.bn1, r.relu, r.maxpool,
                r.layer1, r.layer2, r.layer3, r.layer4
            ).to(self._dev)

        self._enc      = _resnet18_enc()
        self._pose_enc = _resnet18_enc()
        self._ddec     = _DepthDecoder().to(self._dev)
        self._pdec     = _PoseDecoder().to(self._dev)

        if self._freeze:
            for net in (self._enc, self._pose_enc):
                for p in net.parameters():
                    p.requires_grad_(False)

        self._optim = torch.optim.Adam(
            list(self._ddec.parameters()) + list(self._pdec.parameters()),
            lr=1e-4)

    # ── Camera intrinsics API ─────────────────────────────────────────────────

    def update_K(self, fx, fy, cx, cy):
        if self._stub:
            return
        K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],
                          dtype=torch.float32, device=self._dev)
        self._K = K.unsqueeze(0)

    def _K_for(self, B: int, H: int, W: int) -> "torch.Tensor":
        K = self._K if self._K is not None else default_K(H, W, self._dev)
        return K.expand(B, -1, -1)

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(self, triplet: list) -> tuple:
        if self._stub:
            return (np.ones((192,640),np.float32),
                    {"translation": np.zeros(3,np.float32),
                     "rotation":    np.array([0,0,0,1],np.float32)})

        imgs = [self._img(f["img_msg"]) for f in triplet]

        for net in (self._enc, self._pose_enc, self._ddec, self._pdec):
            net.eval()

        with torch.no_grad():
            disp  = self._ddec(self._enc(imgs[1]))
            depth = disp_to_depth(disp)
            fa    = self._pose_enc(imgs[1])
            fb    = self._pose_enc(imgs[2])
            aa, t = self._pdec(fa, fb)
            T     = build_T(aa, t)

        d_np = depth.squeeze().cpu().numpy()
        t_np = T[0,:3,3].cpu().numpy()
        R_np = T[0,:3,:3].cpu().numpy()
        from scipy.spatial.transform import Rotation
        q_np = Rotation.from_matrix(R_np).as_quat().astype(np.float32)
        return d_np, {"translation": t_np, "rotation": q_np}

    # ── Online update ─────────────────────────────────────────────────────────

    def update_step(self, batch: list, n_cycles: int = 5) -> float:
        if self._stub:
            return 0.0

        self._enc.eval()
        self._pose_enc.eval()
        self._ddec.train()
        self._pdec.train()

        val = 0.0
        for _ in range(n_cycles):
            self._optim.zero_grad()
            L = self._loss(batch)
            L.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._ddec.parameters()) + list(self._pdec.parameters()),
                max_norm=1.0)
            self._optim.step()
            val = float(L.item())
        return val

    def _loss(self, batch: list) -> "torch.Tensor":
        total = None
        for triplet in batch:
            imgs = [self._img(f["img_msg"]) for f in triplet]
            vel  = float(triplet[-1].get("velocity", 0.0))
            B,C,H,W = imgs[0].shape
            K = self._K_for(B, H, W)

            # Depth
            disp  = self._ddec(self._enc(imgs[1]))
            depth = disp_to_depth(disp)

            # Poses (prev→curr, next→curr)
            enc = self._pose_enc
            aa_pc, t_pc = self._pdec(enc(imgs[0]), enc(imgs[1]))
            aa_nc, t_nc = self._pdec(enc(imgs[2]), enc(imgs[1]))
            T_pc = build_T(aa_pc, t_pc)
            T_nc = build_T(aa_nc, t_nc)

            # Warp
            w_prev = warp(imgs[0], depth, T_pc, K)
            w_next = warp(imgs[2], depth, T_nc, K)

            # Photometric loss maps
            ph_p = photo_loss(imgs[1], w_prev)
            ph_n = photo_loss(imgs[1], w_next)

            # Auto-mask (exclude pixels easier to predict without warping)
            id_p = photo_loss(imgs[1], imgs[0])
            id_n = photo_loss(imgs[1], imgs[2])
            m_p  = (ph_p < id_p).float().detach()
            m_n  = (ph_n < id_n).float().detach()

            # Min over sources + auto-mask
            l_photo = torch.min(ph_p * m_p, ph_n * m_n).mean()

            # Smoothness
            l_sm = edge_smooth(disp, imgs[1])

            # Velocity supervision
            pred_v = t_pc.norm(dim=1).mean()
            l_vel  = F.l1_loss(
                pred_v,
                torch.tensor(vel, device=self._dev, dtype=torch.float32))

            L = l_photo + 1e-3 * l_sm + 0.05 * l_vel
            total = L if total is None else total + L

        return total / max(len(batch), 1)

    # ── Replay buffer interface ───────────────────────────────────────────────

    def get_encoder_features(self, frame: dict) -> np.ndarray:
        if self._stub:
            return np.random.randn(512).astype(np.float32)
        img = self._img(frame["img_msg"])
        with torch.no_grad():
            self._enc.eval()
            f = self._enc(img)
        return f.mean([2,3]).squeeze().cpu().numpy()

    # ── ROS helpers ───────────────────────────────────────────────────────────

    def depth_to_ros(self, d: np.ndarray, header) -> ROSImage:
        msg = ROSImage()
        msg.header   = header
        msg.height   = d.shape[0]
        msg.width    = d.shape[1]
        msg.encoding = "32FC1"
        msg.step     = d.shape[1] * 4
        msg.data     = d.astype(np.float32).tobytes()
        return msg

    def backproject_to_cloud(self, d: np.ndarray,
                              img_msg: ROSImage) -> np.ndarray:
        H, W = d.shape
        if self._K is not None:
            Kn = self._K.squeeze().cpu().numpy()
            fx,fy,cx,cy = Kn[0,0],Kn[1,1],Kn[0,2],Kn[1,2]
        else:
            fx = fy = 0.8*W
            cx,cy = W/2.0, H/2.0
        uu,vv = np.meshgrid(np.arange(W,dtype=np.float32),
                             np.arange(H,dtype=np.float32))
        z = d.astype(np.float32)
        pts = np.stack([(uu-cx)*z/fx, (vv-cy)*z/fy, z], -1).reshape(-1,3)
        valid = (z.flatten()>0.05) & (z.flatten()<30.0)
        return pts[valid]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_generalizer(self):
        if self._stub:
            return
        torch.save({
            "enc":      self._enc.state_dict(),
            "pose_enc": self._pose_enc.state_dict(),
            "ddec":     self._ddec.state_dict(),
            "pdec":     self._pdec.state_dict(),
        }, self._ckpt)
        rospy.loginfo(f"[depth_pose_net] Saved → {self._ckpt}")

    def try_load_checkpoint(self):
        if self._stub or not os.path.exists(self._ckpt):
            return
        sd = torch.load(self._ckpt, map_location=self._dev)
        self._enc.load_state_dict(sd["enc"])
        self._pose_enc.load_state_dict(sd["pose_enc"])
        self._ddec.load_state_dict(sd["ddec"])
        self._pdec.load_state_dict(sd["pdec"])
        rospy.loginfo("[depth_pose_net] Checkpoint loaded.")

    # ── Image conversion ──────────────────────────────────────────────────────

    def _img(self, msg: ROSImage) -> "torch.Tensor":
        from cv_bridge import CvBridge
        import cv2
        cv_img = CvBridge().imgmsg_to_cv2(msg, "rgb8")
        cv_img = cv2.resize(cv_img, (640, 192))
        return torch.from_numpy(
            cv_img.astype(np.float32)/255.0
        ).permute(2,0,1).unsqueeze(0).to(self._dev)
