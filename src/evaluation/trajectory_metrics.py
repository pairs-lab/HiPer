"""
seliSlam.evaluation.trajectory_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trajectory evaluation metrics for lifelong SLAM.

Implements the two metrics used on the OpenLORIS-Scene lifelong SLAM
benchmark (Shi et al., ICRA 2020) plus standard ATE:

  - ATE RMSE  : root-mean-square absolute trajectory error after
                timestamp association and (optional) Umeyama alignment.
  - C-rate    : correct rate — the proportion of ground-truth-stamped
                poses for which the estimator outputs a pose whose
                absolute position error is below a threshold.
                Periods where the estimator produced NO pose count as
                incorrect, so tracking failure is penalised (unlike ATE,
                which silently skips untracked frames).

No ROS dependency — pure numpy.
"""

from typing import Optional, Tuple

import numpy as np


# ── Timestamp association ─────────────────────────────────────────────────────

def associate_timestamps(
    t_a: np.ndarray,
    t_b: np.ndarray,
    max_dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-neighbour association of two sorted timestamp arrays.
    Returns (idx_a, idx_b) index arrays of matched pairs with |dt| <= max_dt.
    Each timestamp is used at most once.
    """
    t_a = np.asarray(t_a, dtype=np.float64)
    t_b = np.asarray(t_b, dtype=np.float64)
    idx_a, idx_b = [], []
    j = 0
    for i, ta in enumerate(t_a):
        # advance j to the closest t_b[j]
        while j + 1 < len(t_b) and abs(t_b[j + 1] - ta) <= abs(t_b[j] - ta):
            j += 1
        if j < len(t_b) and abs(t_b[j] - ta) <= max_dt:
            idx_a.append(i)
            idx_b.append(j)
            j += 1
            if j >= len(t_b):
                break
    return np.array(idx_a, dtype=np.int64), np.array(idx_b, dtype=np.int64)


# ── Umeyama alignment ─────────────────────────────────────────────────────────

def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Least-squares rigid (optionally similarity) transform aligning
    src (N,3) onto dst (N,3):  dst ≈ s * R @ src + t.
    Returns (R, t, s).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("umeyama_alignment expects matching (N,3) arrays")
    n = src.shape[0]
    if n < 3:
        # Degenerate: translation-only alignment
        t = dst.mean(axis=0) - src.mean(axis=0)
        return np.eye(3), t, 1.0

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d

    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_s = (xs ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return R, t, s


# ── ATE ───────────────────────────────────────────────────────────────────────

def ate_rmse(
    est_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    align: bool = True,
    with_scale: bool = False,
) -> float:
    """
    ATE RMSE (m) between associated position arrays (N,3).
    If align, an Umeyama rigid (or similarity) alignment is applied first.
    """
    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    if est_xyz.shape[0] == 0:
        return float("inf")
    if align:
        R, t, s = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)
        est_xyz = (s * (R @ est_xyz.T)).T + t
    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return float(np.sqrt((err ** 2).mean()))


# ── OpenLORIS C-rate ──────────────────────────────────────────────────────────

def correct_rate(
    est_t: np.ndarray,
    est_xyz: np.ndarray,
    gt_t: np.ndarray,
    gt_xyz: np.ndarray,
    tau_pos: float = 0.3,
    max_dt: float = 0.1,
    align: bool = True,
) -> float:
    """
    OpenLORIS-Scene correct rate (C-rate):
    the fraction of ground-truth timestamps for which the estimator
    produced a pose (within max_dt) whose absolute position error after
    alignment is below tau_pos. Ground-truth stamps with no estimate
    count as incorrect, so the metric penalises tracking loss.
    """
    est_t = np.asarray(est_t, dtype=np.float64)
    gt_t = np.asarray(gt_t, dtype=np.float64)
    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    if len(gt_t) == 0:
        return 0.0
    if len(est_t) == 0:
        return 0.0

    idx_gt, idx_est = associate_timestamps(gt_t, est_t, max_dt=max_dt)
    if len(idx_gt) == 0:
        return 0.0

    e = est_xyz[idx_est]
    g = gt_xyz[idx_gt]
    if align and len(idx_gt) >= 3:
        R, t, s = umeyama_alignment(e, g)
        e = (s * (R @ e.T)).T + t
    err = np.linalg.norm(e - g, axis=1)
    n_correct = int((err < tau_pos).sum())
    return float(n_correct) / float(len(gt_t))


# ── Trajectory integration helper ─────────────────────────────────────────────

def integrate_relative_poses(rel_T: list, T0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Chain a list of 4x4 relative transforms T_{k-1 -> k} into absolute
    poses. Returns an (N+1, 4, 4) array starting at T0 (default identity).
    """
    T = np.eye(4) if T0 is None else np.asarray(T0, dtype=np.float64)
    out = [T.copy()]
    for dT in rel_T:
        T = T @ np.asarray(dT, dtype=np.float64)
        out.append(T.copy())
    return np.stack(out, axis=0)
