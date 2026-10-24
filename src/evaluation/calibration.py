"""
hiper.evaluation.calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Post-hoc calibration of predicted presence, applied identically to every
persistence model (report 2026-10-22-key-and-calibration).

  q = sigmoid(a z + b + sum_e (c_e + d_e z) 1[last evidence = e]),
  z = logit(p),  e in {create, miss, relocate, none}  (hit is the base)

fitted by L2-regularised logistic regression on the prediction rows of the
TUNING data (THOR validation; in leave-one-room-out, the four training
rooms). It changes probabilities only; prune decisions are unchanged.
"""

import math
from typing import Dict, List, Sequence

import numpy as np

EVIDENCE = ("create", "miss", "relocate", "none")
EPS = 1e-6


def _features(p: np.ndarray, ev: Sequence[str]) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    z = np.log(p) - np.log1p(-p)
    cols = [z]
    for e in EVIDENCE:
        ind = np.array([x == e for x in ev], float)
        cols += [ind, ind * z]
    return np.stack(cols, axis=1)


def fit(rows: Sequence[dict], C: float = 1.0) -> Dict:
    """Calibration map from labelled prediction rows of one model."""
    from sklearn.linear_model import LogisticRegression
    lab = [r for r in rows if r.get("pred_presence") is not None
           and r.get("label_next") is not None]
    y = np.array([int(r["label_next"]) for r in lab])
    if len(lab) == 0 or y.min() == y.max():
        return {"identity": True}
    X = _features([r["pred_presence"] for r in lab],
                  [r["last_evidence"] for r in lab])
    m = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    return {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def apply(rows: Sequence[dict], cal: Dict, suffix: str = "_cal") -> List[dict]:
    """Copies of the rows with calibrated pred_presence, model + suffix."""
    out = [dict(r, model=r["model"] + suffix) for r in rows]
    idx = [i for i, r in enumerate(out) if r.get("pred_presence") is not None]
    if not idx or cal.get("identity"):
        return out
    X = _features([out[i]["pred_presence"] for i in idx],
                  [out[i]["last_evidence"] for i in idx])
    s = X @ np.asarray(cal["coef"]) + cal["intercept"]
    q = 1.0 / (1.0 + np.exp(-s))
    for i, v in zip(idx, q):
        out[i]["pred_presence"] = float(v)
    return out
