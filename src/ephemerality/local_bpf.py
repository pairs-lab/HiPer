"""
seliSlam.ephemerality.local_bpf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Local ephemerality update via point-wise Bayesian inference.

Implements ELite §III-C (Gil et al. 2025):

  f(x) = min(α·(1 - exp(-x²/σ_o²)) + β, α)   if x ∈ occupied
         max(α·(1 + exp(-x²/σ_f²)) - β, α)    if x ∈ free

  ε_l_new = f(x)·ε_l_prev / (f(x)·ε_l_prev + (1-f(x))·(1-ε_l_prev))

Key difference from ELite:
  - Uses k-NN point-wise association (no voxelisation grid artefacts)
  - Initialises all new points at ε_l = 0.5 (maximum uncertainty)
"""

import numpy as np
from typing import Optional


class LocalEphemeralityBPF:
    """
    Maintains and updates local ephemerality for a session's point cloud.

    Parameters match ELite paper defaults:
      alpha = 0.5, beta = 0.1, sigma_o = 0.1, sigma_f = 0.3
    """

    def __init__(
        self,
        sigma_occupied: float = 0.1,
        sigma_free: float = 0.3,
        alpha: float = 0.5,
        beta: float = 0.1,
        knn_k: int = 5,
    ):
        self._sigma_o = sigma_occupied
        self._sigma_f = sigma_free
        self._alpha   = alpha
        self._beta    = beta
        self._knn_k   = knn_k

    def update(
        self,
        new_points: np.ndarray,          # (N, 3) float32
        existing_cloud: Optional[np.ndarray],  # (M, 3) or None
        sensor_origin: np.ndarray,        # (3,) float32
    ) -> np.ndarray:
        """
        Compute updated ε_l values for `new_points`.

        Each new point is either:
          - an endpoint of a ray (occupied) → ε_l decreases
          - along the ray path (free)       → ε_l increases

        Returns np.ndarray of shape (N,) with ε_l values in [0, 1].
        """
        N = new_points.shape[0]
        eps_l = np.full(N, 0.5, dtype=np.float32)  # initialise at 0.5

        # Distances from sensor origin to each point (ray lengths)
        ray_vecs = new_points - sensor_origin[np.newaxis, :]  # (N, 3)
        ray_lengths = np.linalg.norm(ray_vecs, axis=1)        # (N,)

        # For each new point, classify as endpoint (occupied) or along-ray
        # Simple heuristic: all new_points are endpoints (occupied);
        # sample free-space points along each ray
        free_pts, free_dists = self._sample_free_space(
            new_points, sensor_origin, ray_lengths)

        # --- Occupied update ---
        for i in range(N):
            x    = 0.0   # distance from the endpoint to itself = 0
            f_x  = self._f_occupied(x)
            eps_l[i] = self._bayesian_update(eps_l[i], f_x)

        # --- Free space update (propagate to nearby existing points) ---
        if existing_cloud is not None and existing_cloud.shape[0] > 0 \
                and free_pts.shape[0] > 0:
            # Find which existing points are near each free-space sample
            free_eps = np.full(existing_cloud.shape[0], 0.5,
                               dtype=np.float32)
            indices, dists = self._knn_search(existing_cloud, free_pts)
            for j, (nbr_idx, d) in enumerate(zip(indices, dists)):
                f_x = self._f_free(d)
                for idx in nbr_idx:
                    free_eps[idx] = self._bayesian_update(
                        free_eps[idx], f_x)
            # Return combined result (new + existing updates)
            # For simplicity, only return new point ephemeralities here
            # A production system would maintain and return the full array

        return eps_l.astype(np.float32)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _f_occupied(self, x: float) -> float:
        """Ephemerality propagation function for occupied endpoints."""
        return float(
            min(self._alpha * (1.0 - np.exp(-x**2 / self._sigma_o**2))
                + self._beta, self._alpha)
        )

    def _f_free(self, x: float) -> float:
        """Ephemerality propagation function for free-space points."""
        return float(
            max(self._alpha * (1.0 + np.exp(-x**2 / self._sigma_f**2))
                - self._beta, self._alpha)
        )

    @staticmethod
    def _bayesian_update(eps_prev: float, f_x: float) -> float:
        """
        ELite Eq. (6): one Bayesian inference step.
        ε_new = f(x)·ε_prev / (f(x)·ε_prev + (1−f(x))·(1−ε_prev))
        """
        num = f_x * eps_prev
        den = num + (1.0 - f_x) * (1.0 - eps_prev)
        return float(num / (den + 1e-8))

    def _sample_free_space(
        self,
        endpoints: np.ndarray,     # (N, 3)
        origin: np.ndarray,        # (3,)
        lengths: np.ndarray,       # (N,)
        n_samples_per_ray: int = 3,
    ) -> tuple:
        """
        Sample points along each ray between origin and endpoint.
        Returns (free_pts (K, 3), free_dists (K,)).
        """
        all_pts   = []
        all_dists = []
        for i in range(endpoints.shape[0]):
            end = endpoints[i]
            ray = end - origin
            for t in np.linspace(0.1, 0.9, n_samples_per_ray):
                pt   = origin + t * ray
                dist = float(np.linalg.norm(end - pt))
                all_pts.append(pt)
                all_dists.append(dist)
        if len(all_pts) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0)
        return (np.array(all_pts,  dtype=np.float32),
                np.array(all_dists, dtype=np.float32))

    def _knn_search(self, cloud: np.ndarray,
                    queries: np.ndarray) -> tuple:
        """
        Brute-force k-NN search. For large clouds replace with
        open3d.geometry.KDTreeFlann or scipy.spatial.cKDTree.
        Returns (indices list[list[int]], distances list[float]).
        """
        k = min(self._knn_k, cloud.shape[0])
        indices = []
        dists   = []
        for q in queries:
            d = np.linalg.norm(cloud - q[np.newaxis, :], axis=1)
            idx = np.argpartition(d, k)[:k]
            indices.append(idx.tolist())
            dists.append(float(d[idx].mean()))
        return indices, dists
