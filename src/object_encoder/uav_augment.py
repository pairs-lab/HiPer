"""
seliSlam.object_encoder.uav_augment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UAV-specific point cloud augmentation for BYE encoder training.

Ground-robot BYE (Huang et al. 2025) augments with:
  - random point jitter (0–0.03 m)
  - random rotation around each axis (0–30°)

UAV extension adds:
  - altitude jitter: random vertical translation simulating
    different flight altitudes (up to ±altitude_jitter_range m)
  - grazing angle simulation: tilt the point cloud by a random
    pitch angle to simulate non-nadir viewpoints
  - top-down masking: randomly remove points with z < threshold
    to simulate partial top-down views

These augmentations make the contrastive embeddings altitude-invariant,
which is critical for UAV re-identification across sessions where
the flight altitude may differ significantly.
"""

import numpy as np


class UAVPointCloudAugmentor:
    """
    Applies a sequence of UAV-specific augmentations to a
    zero-centred partial point cloud.

    Parameters
    ----------
    altitude_jitter_range : float
        Maximum vertical (z) shift in metres. Simulates re-visiting
        at a different altitude.
    angle_jitter_deg : float
        Maximum rotation around each axis in degrees (BYE default: 30).
    position_jitter_m : float
        Gaussian noise sigma for point positions (BYE default: 0.03 m).
    grazing_pitch_deg : float
        Maximum pitch angle for grazing-angle simulation.
    topdown_mask_prob : float
        Probability of applying top-down masking per call.
    """

    def __init__(
        self,
        altitude_jitter_range: float = 5.0,
        angle_jitter_deg: float = 30.0,
        position_jitter_m: float = 0.03,
        grazing_pitch_deg: float = 15.0,
        topdown_mask_prob: float = 0.3,
    ):
        self._alt_range    = altitude_jitter_range
        self._angle_range  = np.deg2rad(angle_jitter_deg)
        self._pos_jitter   = position_jitter_m
        self._graze_range  = np.deg2rad(grazing_pitch_deg)
        self._mask_prob    = topdown_mask_prob

    def augment(self, pts: np.ndarray) -> np.ndarray:
        """
        Apply all augmentations to a (N, 3) zero-centred point cloud.
        Returns an augmented copy of the same shape.
        """
        pts = pts.copy().astype(np.float32)

        # 1. Random rotation around X, Y, Z (BYE base augmentation)
        pts = self._random_rotate(pts)

        # 2. Random point position jitter (BYE base augmentation)
        pts += np.random.uniform(
            -self._pos_jitter, self._pos_jitter,
            size=pts.shape).astype(np.float32)

        # 3. UAV: altitude-perspective jitter via z-axis scaling.
        #    A global z-translation is cancelled by re-centering (BYE §IV-B),
        #    so we instead randomly SCALE the z-axis, which changes the
        #    aspect ratio of the object as seen from different altitudes.
        #    scale ∈ [1 - α, 1 + α] where α = alt_range / (alt_range + 10)
        if self._alt_range > 0:
            alpha     = self._alt_range / (self._alt_range + 10.0)
            z_scale   = float(np.random.uniform(1.0 - alpha, 1.0 + alpha))
            pts[:, 2] = pts[:, 2] * z_scale

        # 4. UAV: grazing angle simulation (random pitch tilt)
        pitch = np.random.uniform(-self._graze_range, self._graze_range)
        pts   = self._rotate_axis(pts, pitch, axis="x")

        # 5. UAV: top-down masking (randomly remove lower points)
        if np.random.rand() < self._mask_prob:
            pts = self._topdown_mask(pts)

        # Re-centre after augmentations
        pts -= pts.mean(axis=0)

        return pts.astype(np.float32)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _random_rotate(self, pts: np.ndarray) -> np.ndarray:
        """Random rotation around each of X, Y, Z by up to ±angle_range."""
        ax = np.random.uniform(-self._angle_range, self._angle_range)
        ay = np.random.uniform(-self._angle_range, self._angle_range)
        az = np.random.uniform(-self._angle_range, self._angle_range)
        pts = self._rotate_axis(pts, ax, "x")
        pts = self._rotate_axis(pts, ay, "y")
        pts = self._rotate_axis(pts, az, "z")
        return pts

    @staticmethod
    def _rotate_axis(pts: np.ndarray,
                      angle: float,
                      axis: str) -> np.ndarray:
        """Rotate point cloud by `angle` radians around `axis`."""
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            R = np.array([[1, 0,  0],
                          [0, c, -s],
                          [0, s,  c]], dtype=np.float32)
        elif axis == "y":
            R = np.array([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]], dtype=np.float32)
        else:  # z
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float32)
        return (R @ pts.T).T

    @staticmethod
    def _topdown_mask(pts: np.ndarray,
                       keep_fraction: float = 0.7) -> np.ndarray:
        """
        Remove the lowest (1 - keep_fraction) of points by z coordinate.
        Simulates the top-down UAV view missing lower parts of objects.
        """
        if pts.shape[0] < 10:
            return pts
        z_threshold = np.percentile(pts[:, 2],
                                    (1 - keep_fraction) * 100)
        mask = pts[:, 2] >= z_threshold
        kept = pts[mask]
        return kept if kept.shape[0] > 5 else pts
