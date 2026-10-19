"""
hiper.ephemerality.priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class-conditioned vanishing rates lambda_c (s^-1) for the persistence
models in persistence_v2 and persistence_baselines.

Two sources, both returning (table, default_lambda):
  load_lambda_table  the hand-written indoor table
                     (config/class_priors_indoor.yaml: lifetimes in days,
                     lambda_c = 1 / (days * 86400));
  learn_class_rates  rates estimated from the ground-truth change histories
                     of the TRAIN split (the "class-rate MLE" baseline).

Keys are AI2-THOR object types ("AlarmClock"). Every table also holds the
CLIP vocabulary spelling of each key ("alarm clock", see vocab_label), so
a lookup by either spelling returns the same rate.

Estimator of learn_class_rates
------------------------------
An object is at risk of vanishing while it is present at a location. The
sessions sample this process: for consecutive sessions k, k+1 (gap
d = t_{k+1} - t_k) an object present at k either
  survives - present at k+1 and neither moved nor re-appeared since k:
             exposure d, no event;
  vanishes - absent at k+1, or moved / re-appeared since k: one event at
             an unknown time inside the gap (interval censored).
Absent objects are not at risk. A moved or re-appeared object starts a new
exposure at its new location (the exponential model is memoryless).

Per class c, with S_c the summed exposure of survived gaps and d_1..d_n the
gaps that ended in a vanish, the exponential likelihood is
  L_c(lambda) = exp(-lambda S_c) prod_j (1 - exp(-lambda d_j)).
The pooled rate lambda_0 maximises the product of L_c over all classes.
Each class rate is the MAP estimate under a Gamma(1 + k, k / lambda_0)
prior, k = prior_strength (k pseudo-vanishes in k / lambda_0 pseudo-
exposure, mode lambda_0), i.e. the unique root of the score
  k / lambda - k / lambda_0 - S_c + sum_j d_j / (exp(lambda d_j) - 1) = 0
(the log-posterior is concave). Equivalently
  lambda_c = (n + k) / (E_c + k / lambda_0),
  E_c = S_c + sum_j E[T | T < d_j],  E[T | T < d] = 1/lambda - d/(e^{lambda d} - 1),
i.e. occurrence / exposure with each censored vanish time replaced by its
conditional mean (the EM fixed point). A class without vanishes gets
k / (S_c + k / lambda_0); a class absent from the data gets lambda_0
through default_lambda. k -> 0 gives the per-class MLE.
"""

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

DAY = 86400.0
INDOOR_PRIORS_YAML = str(
    Path(__file__).resolve().parents[2] / "config" / "class_priors_indoor.yaml")

# (time [s], gt_objects) of one session; gt_objects as in SessionData
SessionGT = Tuple[float, List[dict]]


def vocab_label(name: str) -> str:
    """CLIP vocabulary spelling of an AI2-THOR type ("TVStand" -> "tv stand")."""
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return " ".join(words).lower() if words else name


def _with_vocab_aliases(table: Dict[str, float]) -> Dict[str, float]:
    out = dict(table)
    for key, lam in table.items():
        out.setdefault(vocab_label(key), lam)
    return out


def load_lambda_table(
        yaml_path: str = INDOOR_PRIORS_YAML) -> Tuple[Dict[str, float], float]:
    """Return ({class: lambda_c [s^-1]}, default_lambda) from a lifetime table."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    table = {str(k): 1.0 / (float(days) * DAY)
             for k, days in cfg["lifetimes_days"].items()}
    return _with_vocab_aliases(table), 1.0 / (float(cfg["default_days"]) * DAY)


# ── Rates learned from ground-truth histories ───────────────────────────────

def exposure_counts(gt_histories: Iterable[Sequence[SessionGT]]
                    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Per class: summed exposure of survived gaps S_c [s] and the list of gaps
    [s] that ended in a vanish. ``gt_histories`` holds one entry per scene,
    each a sequence of (time, gt_objects) per session, e.g.
    ``[(s.time, s.gt_objects) for s in sessions]``.
    """
    survived: Dict[str, float] = defaultdict(float)
    vanished: Dict[str, List[float]] = defaultdict(list)
    for scene in gt_histories:
        sessions = sorted(scene, key=lambda s: float(s[0]))
        for (t0, objs0), (t1, objs1) in zip(sessions, sessions[1:]):
            d = float(t1) - float(t0)
            if d <= 0.0:
                raise ValueError("sessions must have distinct times")
            nxt = {o["gt_id"]: o for o in objs1}
            for o in objs0:
                if not o["present"]:
                    continue
                cls = str(o["type"])
                n = nxt.get(o["gt_id"])
                if (n is None or not n["present"] or n.get("moved_since_prev")
                        or n.get("appeared_since_prev")):
                    vanished[cls].append(d)
                else:
                    survived[cls] += d
    classes = set(survived) | set(vanished)
    return ({c: survived.get(c, 0.0) for c in classes},
            {c: vanished.get(c, []) for c in classes})


def _map_rate(survived: float, gaps: np.ndarray, k: float,
              beta: float) -> float:
    """
    Root of k/lam - beta - S + sum_j d_j / (exp(lam d_j) - 1) (decreasing in
    lam). Because 1/lam - d < d / (e^{lam d} - 1) < 1/lam, the root lies in
    [(n + k) / (beta + S + sum d), (n + k) / (beta + S)].
    """
    n = len(gaps)
    if n + k <= 0.0:
        return 0.0
    if beta + survived <= 0.0:
        raise ValueError("no survived exposure and no prior: rate unbounded")
    if n == 0:
        return k / (beta + survived)

    def score(lam: float) -> float:
        x = lam * gaps
        return (k / lam - beta - survived
                + float(np.sum(gaps * np.exp(-x) / -np.expm1(-x))))

    lo = (n + k) / (beta + survived + float(gaps.sum()))
    hi = (n + k) / (beta + survived)
    for _ in range(100):                     # bisection in log(lambda)
        mid = math.sqrt(lo * hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def learn_class_rates(gt_histories: Iterable[Sequence[SessionGT]],
                      prior_strength: float = 1.0
                      ) -> Tuple[Dict[str, float], float]:
    """
    Class vanishing rates [s^-1] from ground-truth histories (TRAIN split
    only). Returns ({class: lambda_c}, lambda_0); see the module docstring
    for the estimator. ``prior_strength`` k is the prior's weight in
    pseudo-vanishes at the pooled rate lambda_0.
    """
    survived, vanished = exposure_counts(gt_histories)
    all_gaps = np.array([d for g in vanished.values() for d in g], float)
    lam0 = _map_rate(sum(survived.values()), all_gaps, 0.0, 0.0)
    k = float(prior_strength)
    table = {}
    for cls in sorted(survived):
        if lam0 <= 0.0:                      # no vanish anywhere in the data
            table[cls] = 0.0
            continue
        beta = k / lam0 if k > 0.0 else 0.0
        table[cls] = _map_rate(survived[cls], np.array(vanished[cls], float),
                               k, beta)
    return _with_vocab_aliases(table), lam0
