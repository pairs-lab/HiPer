"""
seliSlam.object_encoder.association
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Score-level VLM-ensemble association (paper Eq. 13) with uncertainty
down-weighting and the tau_u tentative-match gate, solved jointly for
all instances of a keyframe via the Hungarian algorithm.

Pure numpy/scipy — no ROS — so the full association policy is unit-
testable and shared between the live node and the offline evaluator.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AssociationResult:
    instance_index: int
    object_id: str            # "" if unmatched (new object)
    score: float              # ensemble score of the assignment
    is_confident: bool        # score above threshold AND not tentative
    is_tentative: bool        # uncertainty gate tripped: do not let this
                              # match update the persistence belief


def _cosine_matrix(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Row-normalised cosine similarity matrix (n_q, n_bank)."""
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    return q @ b.T


def ensemble_associate(
    bye_queries: Optional[np.ndarray],       # (n, d_bye) or None
    clip_queries: Optional[np.ndarray],      # (n, d_clip) or None
    bank_ids: List[str],
    bye_bank: Optional[np.ndarray],          # (M, d_bye) or None
    clip_bank: Optional[np.ndarray],         # (M, d_clip) or None
    bank_uncertainties: Optional[np.ndarray] = None,   # (M,) u_j
    query_uncertainties: Optional[np.ndarray] = None,  # (n,) u_i
    w: float = 0.5,
    sigma_u: float = 0.25,
    tau_u: float = 0.5,
    score_threshold: float = 0.75,
) -> List[AssociationResult]:
    """
    Associate n query instances with M mapped objects.

    Ensemble score (Eq. 13): s = (1-w) * s_BYE + w * s_CLIP, computed at
    the SCORE level so the two embedding spaces never mix. When one
    modality is unavailable (no BYE bank in session 0, masked CLIP, ...)
    its weight is re-normalised onto the available one.

    Each score is down-weighted by gamma = exp(-(u_i + u_j) / 2 sigma_u^2)
    before a Hungarian assignment maximises total score one-to-one.
    Matches whose combined uncertainty exceeds tau_u are returned as
    tentative: they must not update the ephemerality posterior.
    """
    n = 0
    for q in (bye_queries, clip_queries):
        if q is not None and len(q) > 0:
            n = len(q)
            break
    M = len(bank_ids)
    if n == 0:
        return []
    if M == 0:
        return [AssociationResult(i, "", 0.0, False, False)
                for i in range(n)]

    # ── Score-level ensemble ──────────────────────────────────────────────────
    scores = np.zeros((n, M), dtype=np.float64)
    weight_sum = np.zeros((n, M), dtype=np.float64)
    if bye_queries is not None and bye_bank is not None and len(bye_bank) > 0:
        scores += (1.0 - w) * _cosine_matrix(bye_queries, bye_bank)
        weight_sum += (1.0 - w)
    if clip_queries is not None and clip_bank is not None \
            and len(clip_bank) > 0:
        scores += w * _cosine_matrix(clip_queries, clip_bank)
        weight_sum += w
    valid = weight_sum > 0
    scores[valid] /= weight_sum[valid]

    # ── Uncertainty down-weight gamma ─────────────────────────────────────────
    u_j = (np.asarray(bank_uncertainties, dtype=np.float64)
           if bank_uncertainties is not None else np.zeros(M))
    u_i = (np.asarray(query_uncertainties, dtype=np.float64)
           if query_uncertainties is not None else np.zeros(n))
    combined_u = u_i[:, None] + u_j[None, :]
    gamma = np.exp(-combined_u / (2.0 * sigma_u ** 2 + 1e-12))
    gated = scores * gamma

    # ── Hungarian assignment (maximise total gated score) ─────────────────────
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-gated)

    results: List[AssociationResult] = []
    assigned = {}
    for r, c in zip(rows, cols):
        assigned[int(r)] = int(c)
    for i in range(n):
        if i not in assigned:
            results.append(AssociationResult(i, "", 0.0, False, False))
            continue
        j = assigned[i]
        score = float(gated[i, j])
        if score < score_threshold:
            # Below threshold: treat as unmatched (new object candidate)
            results.append(AssociationResult(i, "", score, False, False))
            continue
        tentative = bool(combined_u[i, j] > tau_u)
        results.append(AssociationResult(
            instance_index=i,
            object_id=bank_ids[j],
            score=score,
            is_confident=not tentative,
            is_tentative=tentative,
        ))
    return results
