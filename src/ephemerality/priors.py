"""
hiper.ephemerality.priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class-conditioned vanishing rates lambda_c (s^-1) for the persistence
models in persistence_v2 and persistence_baselines.

Three sources. The first two return (table, default_lambda):
  load_lambda_table  the hand-written indoor table
                     (config/class_priors_indoor.yaml: lifetimes in days,
                     lambda_c = 1 / (days * 86400));
  learn_class_rates  rates estimated from the ground-truth change histories
                     of the TRAIN split (the "class-rate MLE" baseline).

  NeighbourSurvivalPrior  a per-object rate retrieved from the training
                     objects nearest in DINOv2 appearance (no class label).

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
                if n is not None and n["present"] is None:
                    continue           # unknown next state (3RScan): skip
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


# ── Survival prior retrieved by appearance (embedding neighbours) ──────────

def geo_features(extent, height, pixel_area) -> np.ndarray:
    """Geometry key of an object: [log of its sorted extents (3, floored at
    1 cm), height of its centroid, log of its mean mask area]. Missing parts
    (no extent, no height, pixel_area <= 0 as for rendered real data) are
    NaN and drop out of the distance."""
    out = np.full(5, np.nan)
    if extent is not None:
        e = np.sort(np.maximum(np.asarray(extent, np.float64), 0.01))[::-1]
        out[:3] = np.log(e[:3])
    if height is not None and np.isfinite(height):
        out[3] = float(height)
    if pixel_area is not None and float(pixel_area) > 0.0:
        out[4] = math.log(max(float(pixel_area), 1e-5))
    return out


def object_exposures(sessions_by_scene) -> List[dict]:
    """
    One record per tracked ground-truth object of TRAINING scenes:
    ``{"emb": unit mean DINOv2 embedding of its tracks, "clip": unit mean
    CLIP embedding (or None), "geo": geo_features of its mean extent,
    height and mask area, "S": survived exposure [s], "gaps": gaps [s] that
    ended in a vanish}`` (same at-risk rules as exposure_counts).
    ``sessions_by_scene``: iterable of session lists (object_session.
    SessionData with tracks and gt_objects).
    """
    recs = []
    for sessions in sessions_by_scene:
        ss = sorted(sessions, key=lambda s: s.session_index)
        acc: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        for s in ss:
            for t in s.tracks:
                if t.gt_id is None or t.dino_mean is None:
                    continue
                a = acc[t.gt_id]
                a["dino"].append(np.asarray(t.dino_mean, float))
                if t.clip_mean is not None:
                    c = np.asarray(t.clip_mean, float)
                    a["clip"].append(c / max(np.linalg.norm(c), 1e-12))
                if t.extent is not None:
                    a["extent"].append(np.asarray(t.extent, float))
                a["height"].append(float(np.asarray(t.centroid)[2]))
                a["pix"].append(float(t.pixel_area))
        surv: Dict[str, float] = defaultdict(float)
        gaps: Dict[str, list] = defaultdict(list)
        for s0, s1 in zip(ss, ss[1:]):
            d = float(s1.time) - float(s0.time)
            nxt = {o["gt_id"]: o for o in s1.gt_objects}
            for o in s0.gt_objects:
                if not o["present"]:
                    continue
                n = nxt.get(o["gt_id"])
                if n is not None and n["present"] is None:
                    continue           # unknown next state (3RScan): skip
                if (n is None or not n["present"] or n.get("moved_since_prev")
                        or n.get("appeared_since_prev")):
                    gaps[o["gt_id"]].append(d)
                else:
                    surv[o["gt_id"]] += d
        for g, a in acc.items():
            m = np.mean(a["dino"], axis=0)
            nm = np.linalg.norm(m)
            if nm <= 1e-12:
                continue
            clip = None
            if a["clip"]:
                c = np.mean(a["clip"], axis=0)
                clip = c / max(np.linalg.norm(c), 1e-12)
            recs.append({
                "emb": m / nm, "clip": clip,
                "geo": geo_features(
                    np.mean(a["extent"], 0) if a["extent"] else None,
                    float(np.mean(a["height"])), float(np.mean(a["pix"]))),
                "S": surv.get(g, 0.0), "gaps": list(gaps.get(g, []))})
    return recs


def weighted_map_rate(w: np.ndarray, S: np.ndarray, gaps: np.ndarray,
                      gap_w: np.ndarray, k: float, lam0: float) -> float:
    """MAP rate of objects with weights w (exposures S, vanish gaps with
    weights gap_w) under the Gamma(1 + k, k / lam0) prior: the root of
    k/l - k/lam0 - sum w S + sum_j gw_j d_j / (e^{l d_j} - 1)."""
    Sw = float(np.dot(w, S))
    n = float(np.sum(gap_w))
    beta = k / lam0
    if n + k <= 0.0:
        return lam0
    if n == 0.0:
        return k / (beta + Sw)

    def score(lam: float) -> float:
        x = lam * gaps
        return k / lam - beta - Sw + float(
            np.sum(gap_w * gaps * np.exp(-x) / -np.expm1(-x)))

    lo = (n + k) / (beta + Sw + float(np.dot(gap_w, gaps)))
    hi = (n + k) / (beta + Sw)
    for _ in range(80):                      # bisection in log(lambda)
        mid = math.sqrt(lo * hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


class NeighbourSurvivalPrior:
    """
    Survival rate of a new object retrieved from the training objects most
    similar to it. Distance to training record i (all parts on the scale of
    the DINOv2 cosine distance; r_c, r_g = ratios of median pairwise
    distances over the records):
      d_i = [a (1 - s_dino) + (1 - a) r_c (1 - s_clip) + w_g r_g d_geo]
            / (1 + w_g),     a = 1 - clip_weight, w_g = geo_weight,
    d_geo = mean over the query's available geometry features of the squared
    standardised difference, / 2. The K nearest records get kernel weights
    w_i = exp(-(d_i - d_min) / tau), and the rate is their weighted MAP
    estimate shrunk to the pooled rate lambda_0 with strength k
    (pseudo-vanishes). With clip_weight = geo_weight = 0 this is pure
    DINOv2 retrieval, w_i = exp((s_i - s_max) / tau). Hard class keys
    (learn_class_rates) are the special case of 0/1 weights on the objects
    sharing a label; here no label is needed.
    """

    def __init__(self, records: Sequence[dict], lam0: Optional[float] = None,
                 clip_weight: float = 0.0, geo_weight: float = 0.0):
        if not records:
            raise ValueError("no training records")
        self.E = np.stack([r["emb"] for r in records]).astype(np.float64)
        self.S = np.array([r["S"] for r in records], dtype=np.float64)
        self.gaps = [np.asarray(r["gaps"], dtype=np.float64) for r in records]
        if lam0 is None:
            lam0 = _map_rate(float(self.S.sum()),
                             np.concatenate(self.gaps), 0.0, 0.0)
        self.lam0 = float(lam0)
        self.clip_weight = float(clip_weight)
        self.geo_weight = float(geo_weight)
        iu = np.triu_indices(len(self.E), 1)
        # scale ratios need >= 2 records; with one record they do not matter
        med = (lambda x: float(np.median(x)) if x.size else 1.0)  # noqa: E731
        med_d = med((1.0 - self.E @ self.E.T)[iu])
        self.C = None
        if self.clip_weight > 0.0:
            if any(r.get("clip") is None for r in records):
                raise ValueError("clip_weight > 0 needs a CLIP key per record")
            self.C = np.stack([r["clip"] for r in records]).astype(np.float64)
            self.r_c = med_d / max(med((1.0 - self.C @ self.C.T)[iu]), 1e-12)
        self.G = None
        if self.geo_weight > 0.0:
            G = np.stack([np.asarray(r["geo"], np.float64) for r in records])
            self.g_mu = np.nanmean(G, axis=0)
            sd = np.nanstd(G, axis=0)
            self.g_sd = np.where(sd > 1e-12, sd, 1.0)
            self.G = np.nan_to_num((G - self.g_mu) / self.g_sd)
            sq = ((self.G ** 2).sum(1)[:, None] + (self.G ** 2).sum(1)[None]
                  - 2.0 * self.G @ self.G.T)
            dg = np.maximum(sq, 0.0) / (2.0 * self.G.shape[1])
            self.r_g = med_d / max(med(dg[iu]), 1e-12)
        self._cache: Dict = {}

    @staticmethod
    def _unit(v):
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if np.isfinite(n) and n > 1e-12 else None

    def distance(self, dino, clip=None, geo=None) -> np.ndarray:
        """Distance of one query to every record (see the class doc)."""
        d = 1.0 - self.E @ dino
        if self.C is not None and clip is not None:
            a = 1.0 - self.clip_weight
            d = a * d + (1.0 - a) * self.r_c * (1.0 - self.C @ clip)
        if self.G is not None and geo is not None:
            ok = np.isfinite(geo)
            if ok.any():
                gq = (geo[ok] - self.g_mu[ok]) / self.g_sd[ok]
                dg = ((self.G[:, ok] - gq) ** 2).sum(1) / (2.0 * ok.sum())
                d = (d + self.geo_weight * self.r_g * dg) / (
                    1.0 + self.geo_weight)
        return d

    def rate(self, query, K: int = 100, tau: Optional[float] = 0.05,
             k: float = 1.0) -> float:
        """query: the DINOv2 embedding of the new object, or a dict with
        "dino" and optionally "clip" and "geo" (geo_features)."""
        if isinstance(query, dict):
            dino = self._unit(query.get("dino"))
            clip = self._unit(query.get("clip"))
            geo = query.get("geo")
            geo = None if geo is None else np.asarray(geo, np.float64)
        else:
            dino, clip, geo = self._unit(query), None, None
        if dino is None:
            return self.lam0
        K = int(min(max(K, 1), len(self.E)))
        key = (dino.tobytes(), None if clip is None else clip.tobytes(),
               None if geo is None else geo.tobytes(), K, tau, k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        nkey = key[:4]
        nb = self._cache.get(nkey)
        if nb is None:
            d = self.distance(dino, clip, geo)
            idx = np.argpartition(d, K - 1)[:K]
            nb = (idx, d[idx])
            self._cache[nkey] = nb
        idx, dk = nb
        w = np.exp(-(dk - dk.min()) / tau) if tau else np.ones(K)
        gaps = np.concatenate([self.gaps[i] for i in idx])
        gap_w = np.concatenate([np.full(len(self.gaps[i]), wi)
                                for i, wi in zip(idx, w)])
        lam = weighted_map_rate(w, self.S[idx], gaps, gap_w, k, self.lam0)
        if len(self._cache) > 400000:
            self._cache.clear()
        self._cache[key] = lam
        return lam
