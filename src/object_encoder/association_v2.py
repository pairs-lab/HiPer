"""
hiper.object_encoder.association_v2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Association v2: match one session's object tracks to the mapped objects
(design.md, 'Association v2').

  score_ij = gamma_ij * app_ij + w_geo * exp(-d2_ij / 2)
  app_ij   = (1 - w) cos(BYE_i, BYE_j) + w cos(CLIP_i, CLIP_j); a modality
             missing on either side drops out and the remaining weight is
             renormalised (as in association.py)
  gamma_ij = exp(-(u_i + u_j) / (2 sigma_u^2)), u = tr(C) of the unit
             embeddings, in [0, 1] (no 1/d factor)
  d2_ij    = (z_i - x_j)^T (Sigma_j [+ P] + R_i)^-1 (z_i - x_j), with P the
             session pose covariance (pose inflation)

A one-to-one Hungarian assignment maximises the total score over the pairs
with score >= tau_match (pairs below the threshold never compete). A match
is tentative if u_i + u_j > tau_u, and a relocation if d2 > chi2_3(0.99).
Every term can be switched off for ablations (AssocConfig). Pure numpy /
scipy; the v1 module association.py is left unchanged.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_3_99 = 11.34   # chi-square quantile, 3 dof, p = 0.99


@dataclass
class AssocConfig:
    name: str = "full"
    w: float = 0.5               # CLIP weight: 0 = BYE only, 1 = CLIP only
    w_geo: float = 0.5           # weight of the geometric term
    sigma_u: float = 1.0         # gamma scale
    tau_u: float = 0.5           # tentative gate on u_i + u_j
    tau_match: float = 0.75      # minimum score of a match
    chi2_reloc: float = CHI2_3_99
    centroid_std: float = 0.05   # m, isotropic noise added to every track
                                 # centroid covariance (partial views)
    pose_inflation: bool = True  # add the session pose covariance
    use_gamma: bool = True
    use_tau_u: bool = True


@dataclass
class AssocSet:
    """Positions, covariances and appearance of tracks or of map objects.
    A zero row in ``clip`` / ``bye`` marks a missing embedding."""
    pos: np.ndarray                     # (n, 3)
    cov: np.ndarray                     # (n, 3, 3)
    u: np.ndarray                       # (n,) CLIP dispersion tr(C)
    clip: Optional[np.ndarray] = None   # (n, d_clip)
    bye: Optional[np.ndarray] = None    # (n, d_bye)

    def __len__(self) -> int:
        return len(self.pos)


@dataclass
class Match:
    track: int                   # track index
    obj: int = -1                # map-object index, -1 if unmatched
    score: float = float("nan")
    d2: float = float("nan")
    tentative: bool = False      # must not update beliefs or embeddings
    relocated: bool = False      # confident re-ID far from the mapped position


# ── Score terms ──────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine matrix (n, M) and the mask of pairs where both rows exist."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-8)[:, None] & (nb > 1e-8)[None, :]
    cos = (a / np.maximum(na, 1e-8)[:, None]) @ \
        (b / np.maximum(nb, 1e-8)[:, None]).T
    return np.where(ok, cos, 0.0), ok


def appearance(tracks: AssocSet, objects: AssocSet, w: float) -> np.ndarray:
    """app_ij with per-pair modality renormalisation; 0 if no modality."""
    num = np.zeros((len(tracks), len(objects)))
    den = np.zeros_like(num)
    for wm, a, b in ((1.0 - w, tracks.bye, objects.bye),
                     (w, tracks.clip, objects.clip)):
        if wm <= 0.0 or a is None or b is None:
            continue
        cos, ok = _cosine(a, b)
        num += wm * cos
        den += wm * ok
    return np.where(den > 0.0, num / np.maximum(den, 1e-12), 0.0)


def mahalanobis(tracks: AssocSet, objects: AssocSet,
                pose_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """Squared Mahalanobis distance (n, M) under Sigma_j [+ P] + R_i."""
    diff = tracks.pos[:, None, :] - objects.pos[None, :, :]
    S = tracks.cov[:, None] + objects.cov[None, :] + 1e-9 * np.eye(3)
    if pose_cov is not None:
        S = S + np.asarray(pose_cov, dtype=np.float64)
    sol = np.linalg.solve(S, diff[..., None])[..., 0]
    return np.einsum("nmk,nmk->nm", diff, sol)


def score_matrix(tracks: AssocSet, objects: AssocSet, cfg: AssocConfig,
                 pose_cov: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (score, d2, u_i + u_j), each (n, M)."""
    d2 = mahalanobis(tracks, objects,
                     pose_cov if cfg.pose_inflation else None)
    u_sum = np.asarray(tracks.u, dtype=np.float64)[:, None] + \
        np.asarray(objects.u, dtype=np.float64)[None, :]
    gamma = (np.exp(-u_sum / (2.0 * cfg.sigma_u ** 2)) if cfg.use_gamma
             else np.ones_like(u_sum))
    score = gamma * appearance(tracks, objects, cfg.w) + \
        cfg.w_geo * np.exp(-0.5 * d2)
    return score, d2, u_sum


# ── Assignment ───────────────────────────────────────────────────────────────

def assign(score: np.ndarray, d2: np.ndarray, u_sum: np.ndarray,
           cfg: AssocConfig) -> List[Match]:
    """Gated Hungarian assignment; one Match per track (row)."""
    n, M = score.shape
    out = [Match(i) for i in range(n)]
    if n == 0 or M == 0:
        return out
    valid = score >= cfg.tau_match
    rows, cols = linear_sum_assignment(-np.where(valid, score, 0.0))
    for i, j in zip(rows, cols):
        if not valid[i, j]:
            continue
        out[i] = Match(
            track=int(i), obj=int(j), score=float(score[i, j]),
            d2=float(d2[i, j]),
            tentative=bool(cfg.use_tau_u and u_sum[i, j] > cfg.tau_u),
            relocated=bool(d2[i, j] > cfg.chi2_reloc))
    return out


def associate(tracks: AssocSet, objects: AssocSet, cfg: AssocConfig,
              pose_cov: Optional[np.ndarray] = None) -> List[Match]:
    """Score all pairs and solve the assignment (one Match per track)."""
    if len(tracks) == 0 or len(objects) == 0:
        return [Match(i) for i in range(len(tracks))]
    return assign(*score_matrix(tracks, objects, cfg, pose_cov), cfg)
