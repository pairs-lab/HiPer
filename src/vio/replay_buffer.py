"""
seliSlam.vio.replay_buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fixed-size experience replay buffer for the VIO front-end.
Implements the diversity-based update strategy from CoVIO
(Vödisch et al. 2023, §3.2.1):

  - Add a frame only if its cosine similarity to the nearest
    buffer frame is below `similarity_threshold`.
  - If the buffer is full, remove the frame most similar to
    all other frames (minimises diversity loss).

Features are extracted with the DepthNet encoder (frozen),
so no separate network is needed.
"""

import numpy as np
from typing import Callable, Optional, List


class DiversityReplayBuffer:
    """
    Fixed-size replay buffer that maximises stored image diversity.

    Parameters
    ----------
    max_size : int
        Hard cap on number of stored triplets.
    similarity_threshold : float
        A new frame is added only if max cosine sim to buffer < this.
    feature_extractor : Callable
        f(frame_dict) -> np.ndarray of shape (feature_dim,).
        Typically the DepthNet encoder run in eval mode.
    """

    def __init__(
        self,
        max_size: int = 100,
        similarity_threshold: float = 0.95,
        feature_extractor: Optional[Callable] = None,
    ):
        self._max_size   = max_size
        self._thresh     = similarity_threshold
        self._extract_fn = feature_extractor

        # Each entry: {"frame": frame_dict, "feature": np.ndarray}
        self._buffer: List[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def maybe_add(self, frame: dict) -> bool:
        """
        Attempt to add a frame to the buffer.
        Returns True if the frame was added.

        frame must be a dict with at least {"img_msg", "depth_msg",
        "velocity", "stamp"} keys (as produced by vio_frontend_node).
        """
        feat = self._get_feature(frame)
        if feat is None:
            # No extractor — always add (degenerate mode for testing)
            self._unconditional_add(frame, np.zeros(1))
            return True

        # Check similarity against existing buffer
        if len(self._buffer) > 0:
            sims = self._cosine_sims(feat)
            if sims.max() >= self._thresh:
                return False   # too similar to existing content

        self._unconditional_add(frame, feat)
        return True

    def sample(self, n: int) -> list:
        """
        Return up to n randomly sampled triplets (wrapped in a list).
        Each element is a list [triplet] matching the format expected
        by DepthPoseNet.update_step().
        """
        if len(self._buffer) == 0:
            return []
        n = min(n, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        return [[self._buffer[i]["frame"]] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_feature(self, frame: dict) -> Optional[np.ndarray]:
        if self._extract_fn is None:
            return None
        try:
            feat = self._extract_fn(frame)
            if feat is None:
                return None
            feat = np.array(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-8)
        except Exception:
            return None

    def _cosine_sims(self, feat: np.ndarray) -> np.ndarray:
        """Cosine similarity of `feat` against all buffer features."""
        stored = np.stack(
            [e["feature"] for e in self._buffer], axis=0)  # (B, D)
        dots   = stored @ feat                               # (B,)
        # Features are already L2-normalised
        return dots

    def _unconditional_add(self, frame: dict, feat: np.ndarray):
        entry = {"frame": frame, "feature": feat}
        if len(self._buffer) < self._max_size:
            self._buffer.append(entry)
        else:
            # Remove the most redundant entry (highest sum-similarity)
            remove_idx = self._most_redundant_index()
            self._buffer[remove_idx] = entry

    def _most_redundant_index(self) -> int:
        """
        Return the index of the entry whose feature is most similar
        to all other entries — i.e. the one adding least diversity.
        """
        feats = np.stack([e["feature"] for e in self._buffer], axis=0)
        # Gram matrix of cosine similarities
        gram  = feats @ feats.T                              # (B, B)
        # Sum of similarities for each entry (exclude self-similarity)
        np.fill_diagonal(gram, 0.0)
        scores = gram.sum(axis=1)                            # (B,)
        return int(np.argmax(scores))
