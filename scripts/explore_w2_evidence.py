#!/usr/bin/env python3
"""
scripts/explore_w2_evidence.py
==============================
VALIDATION-ONLY exploration, idea w2 (slug w2_evidence): a learned evidence
model for the persistence filter. Never reads test / test2 / test3 /
obchange data or results.

The filter update depends on an event only through its likelihood ratio
L1 / L0 (all two-slice weights are normalised), so the evidence model is a
log-likelihood ratio per event (constant now:
  HIT  LLR = log((1 - P_M) / P_F),  MISS LLR = log(P_M / (1 - P_F))):
  HIT : LLR = a_h + b_s (score - c) + b_d (log1p d2 - c) + b_n (log1p n_obs - c)
              [+ b_v (not_visible - c)]                       (ext_hit_vis)
  MISS: LLR = a_m + g_n (log1p last_nobs - c) + g_f (log1p n_fused - c)
              [+ g_a (log1p age - c)]                         (ext_miss_age)
7 parameters (at most 5 free in an ext_* variant); c = means over the TRAIN
events. Features are observable event fields only: match score,
Mahalanobis^2 d2 and frame count n_obs of the hit track; for a miss, n_obs
of the last track that set the object (create, hit or relocate) and the
number of fused sessions so far. ext_*: visible flag of a hit (mapped
position in view) and age = k - created (sessions since creation, from the
event's k). eps_l is 0 and tentative is False on every hit / miss event of
the cached streams, and visible is True on every miss, so they carry no
information there. L1 = sigmoid(LLR), L0 = sigmoid(-LLR) (only the ratio
matters).

Fit: theta minimises the filter's own prediction NLL (mean over scenes) on
the TRAIN streams (train + train2, their labels). The appearance prior is
cross-fitted so that no train object sees its own record: train streams use
a prior built from 20 train2 scenes (FloorPlan 6-10, 206-210, 306-310,
406-410), train2 streams the prior from records_train (the val prior).
Prior parameters stay at the frozen run-v2 values during the fit (HiPer
kappa0 = 32, PF lifetime 32 d, retrieved prior K = 300, tau = 0.05, k = 1).
The same fit is done separately for HiPer, PF(lambda) and PF+retrieved.

Robust floor/ceiling: p' = (1 - eps) p + eps / 2, eps in EPS_GRID.

Val selection (declared grids; NBR parameters frozen):
  ref            DET_GRID x prior grid                 (the reference)
  ref_eps        DET_GRID x prior grid x EPS_GRID
  learned        train theta,          prior grid x EPS_GRID
  learned_slopes train slopes, DET_GRID intercepts, prior grid x EPS_GRID
  abl_*          learned with only the hit part / only the miss part /
                 only intercepts / one miss feature (each refitted on
                 train), prior grid x EPS_GRID
  ref_wide       stronger reference: DET_GRID with p_false also in
                 P_FALSE_EXTRA, x prior grid x EPS_GRID
  miss_slopes_det  DET_GRID constants + the train miss slopes of abl_miss_only
                 (hit LLR constant), DET_GRID x prior grid x EPS_GRID
                 (learned_slopes uses all train slopes with DET intercepts;
                 kept as a recorded failure: its hit slopes were fitted with
                 a saturated hit intercept)
  miss_slopes_wide as miss_slopes_det on the ref_wide constants grid (val-tuned
                 constants instead of train intercepts; simplification check)
  ext_hit_vis    abl_miss_only + hit not-visible slope (refitted on train,
                 warm start at abl_miss_only), prior grid x EPS_GRID
  ext_miss_age   abl_miss_only + miss log1p(age) slope (same)
prior grid: HiPer KAPPA_FINAL, PF LIFETIME_GRID_DAYS, PF+retrieved none.
2-fold check over the sorted val scenes (A = even index, B = odd index):
per variant (its grid selected on one fold, scored on the other) and nested
(the variant is ALSO chosen on the selection fold, pooled over the variant
sets in POOLS; the pools train_fitted_evidence and constant_or_train_fitted
were added post hoc, after the 'all' pool result was seen).

A fast vectorised copy of the filter does the search; it is checked against
the harness (explore_common.run_model with a filter subclass, assert
|diff| < 1e-6) on the frozen references and on the selected configuration
of every variant in HARNESS_VARIANTS.

  python scripts/explore_w2_evidence.py [--rebuild]
Outputs: RESULTS/explore_cache/w2_evidence/{results.json, run.log,
built.pkl (cache of the dense arrays; --rebuild recomputes it)}
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HIT, MISS, NONE, HierarchicalPersistenceFilter)
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import gt_labels  # noqa: E402

OUT = ec.CACHE / "w2_evidence"
DT = 86400.0
M_CLIP = 1e-4
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]
P_FALSE_EXTRA = [1e-6, 1e-4, 1e-3]          # ref_wide only
NBR = dict(K=300, tau=0.05, k=1.0)          # frozen run-v2 values
CODE = {"create": 1, "hit": 2, "miss": 3, "none": 4, "relocate": 5}
PNAMES = ["a_hit", "b_score", "b_log1p_d2", "b_log1p_nobs",
          "a_miss", "g_log1p_last_nobs", "g_log1p_nfused",
          "b_not_visible", "g_log1p_age"]          # last two: ext_* only
HIT_IDX, MISS_IDX = [1, 2, 3], [5, 6]
REF_VARIANTS = ["ref", "ref_eps", "ref_wide"]       # constant evidence
POOLS = {   # nested 2-fold: the variant is also chosen on the selection fold
    "ref_only": ["ref"],
    "constant_evidence": REF_VARIANTS,
    "learned_evidence": ["learned", "learned_slopes", "miss_slopes_det",
                         "miss_slopes_wide", "abl_hit_only", "abl_miss_only",
                         "abl_intercepts_only", "abl_miss_nobs",
                         "abl_miss_nf", "ext_hit_vis", "ext_miss_age"],
}
POOLS["all"] = REF_VARIANTS + POOLS["learned_evidence"]
# Added POST HOC (after the run of 23:17 showed that the 'all' pool picks
# miss_slopes_wide on both folds and does not transfer): pools without the
# variants whose evidence constants are val-tuned (learned_slopes,
# miss_slopes_*); all evidence parameters of these come from train, val
# tunes only the prior and eps.
POOLS["train_fitted_evidence"] = ["learned", "abl_hit_only", "abl_miss_only",
                                  "abl_intercepts_only", "abl_miss_nobs",
                                  "abl_miss_nf", "ext_hit_vis",
                                  "ext_miss_age"]
POOLS["constant_or_train_fitted"] = (REF_VARIANTS
                                     + POOLS["train_fitted_evidence"])
TRAIN_EPS_VARIANTS = ("learned", "abl_miss_only", "ext_miss_age")
LOG = []


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ── Data ─────────────────────────────────────────────────────────────────────

def nbr_prior(records):
    return te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]} for r in records])


def build(streams, prior, tag):
    """Dense [object, session] arrays of one set of streams. y (label_next)
    is used for scoring only; every feature is an observable event field."""
    scenes = sorted(streams)
    K = max(len(streams[s]["times"]) for s in scenes)
    objs = [(si, sc, oid, ob) for si, sc in enumerate(scenes)
            for oid, ob in streams[sc]["objects"].items()]
    N = len(objs)
    code = np.zeros((N, K), np.int8)
    t = np.zeros((N, K))
    tn = np.full((N, K), np.nan)
    y = np.full((N, K), -1, np.int8)
    raw = {f: np.full((N, K), np.nan)
           for f in ("score", "d2", "n_obs", "last_nobs", "nf", "notvis",
                     "age")}
    m_nbr = np.full(N, np.nan)
    sidx = np.zeros(N, np.int64)
    keys = []
    for i, (si, sc, oid, ob) in enumerate(objs):
        st = streams[sc]
        T, gt = st["times"], st["gt"]
        sidx[i] = si
        keys.append((sc, oid))
        last_nobs, nf = None, 0
        for ev in ob["events"]:
            k, e = ev["k"], ev["ev"]
            assert ev["t"] == T[k] and float(ev.get("eps_l") or 0.0) == 0.0
            code[i, k] = CODE[e]
            t[i, k] = ev["t"]
            if k + 1 < len(T):
                tn[i, k] = T[k + 1]
            _, nxt = gt_labels(gt, ev["gt_id"], ev["pos_session"], k)  # SCORE
            if nxt is not None:
                y[i, k] = nxt
            for f in ("score", "d2", "n_obs"):
                if ev[f] is not None:
                    raw[f][i, k] = float(ev[f])
            if last_nobs is not None:
                raw["last_nobs"][i, k] = float(last_nobs)
            raw["nf"][i, k] = nf
            raw["age"][i, k] = float(k - ob["created"])
            if e == "hit":
                raw["notvis"][i, k] = 0.0 if ev["visible"] else 1.0
            if e in ("create", "hit", "relocate"):
                nf += 1
                last_nobs = ev["n_obs"]
        if prior is not None:
            emb = ob["events"][0].get("emb")
            lam = (prior.rate(emb, K=NBR["K"], tau=NBR["tau"], k=NBR["k"])
                   if emb is not None else prior.lam0)
            m_nbr[i] = min(max(math.exp(-lam * DT), M_CLIP), 1.0 - M_CLIP)
    msk = (code > 0) & (y >= 0)
    D = dict(tag=tag, scenes=scenes, K=K, code=code, t=t, tn=tn, y=y,
             raw=raw, m_nbr=m_nbr, sidx=sidx, keys=keys, msk=msk,
             yy=y[msk].astype(np.float64),
             ss=np.broadcast_to(sidx[:, None], code.shape)[msk],
             cnt=np.bincount(np.broadcast_to(sidx[:, None], code.shape)[msk],
                             minlength=len(scenes)).astype(np.float64))
    return D


def concat(D1, D2, tag):
    """Stack two built sets (scene indices of D2 shifted)."""
    off = len(D1["scenes"])
    assert D1["K"] == D2["K"]
    D = dict(tag=tag, scenes=D1["scenes"] + D2["scenes"], K=D1["K"])
    for f in ("code", "t", "tn", "y", "m_nbr"):
        D[f] = np.concatenate([D1[f], D2[f]])
    D["raw"] = {f: np.concatenate([D1["raw"][f], D2["raw"][f]])
                for f in D1["raw"]}
    D["sidx"] = np.concatenate([D1["sidx"], D2["sidx"] + off])
    D["keys"] = D1["keys"] + D2["keys"]
    D["msk"] = (D["code"] > 0) & (D["y"] >= 0)
    D["yy"] = D["y"][D["msk"]].astype(np.float64)
    D["ss"] = np.broadcast_to(D["sidx"][:, None], D["code"].shape)[D["msk"]]
    D["cnt"] = np.bincount(D["ss"], minlength=len(D["scenes"])).astype(float)
    return D


def centers(D):
    """Feature means over the TRAIN hit / miss events (fixed constants)."""
    r, c = D["raw"], D["code"]
    h, m = c == 2, c == 3
    return {"score": float(np.mean(r["score"][h])),
            "ld2": float(np.mean(np.log1p(r["d2"][h]))),
            "lnobs": float(np.mean(np.log1p(r["n_obs"][h]))),
            "llast": float(np.mean(np.log1p(r["last_nobs"][m]))),
            "lnf": float(np.mean(np.log1p(r["nf"][m]))),
            "notvis": float(np.mean(r["notvis"][h])),
            "lage": float(np.mean(np.log1p(r["age"][m])))}


def attach_features(D, c):
    r = D["raw"]
    z = lambda a: np.nan_to_num(a, nan=0.0)  # noqa: E731
    D["F"] = np.stack([
        z(r["score"] - c["score"]), z(np.log1p(r["d2"]) - c["ld2"]),
        z(np.log1p(r["n_obs"]) - c["lnobs"]),
        z(np.log1p(r["last_nobs"]) - c["llast"]),
        z(np.log1p(r["nf"]) - c["lnf"]),
        z(r["notvis"] - c["notvis"]),
        z(np.log1p(r["age"]) - c["lage"])])


def llr(theta, D):
    F = D["F"]
    Lh = (theta[0] + theta[1] * F[0] + theta[2] * F[1] + theta[3] * F[2]
          + theta[7] * F[5])
    Lm = theta[4] + theta[5] * F[3] + theta[6] * F[4] + theta[8] * F[6]
    return Lh, Lm


def det_intercepts(pm, pf):
    return math.log((1.0 - pm) / pf), math.log(pm / (1.0 - pf))


def theta_ref(pm, pf):
    ah, am = det_intercepts(pm, pf)
    return np.array([ah, 0, 0, 0, am, 0, 0, 0, 0], float)


# ── Vectorised filter (persistence_v2 observe/relocate/presence, rho = 0,
#    pi_new = 1, no environment level) ────────────────────────────────────────

def _surv(a, b, r, m, learn):
    if not learn:
        return m ** r
    return np.exp(gammaln(a + r) + gammaln(a + b) - gammaln(a)
                  - gammaln(a + b + r))


def simulate(D, Lh, Lm, m, kappa0, learn):
    code, t, tn = D["code"], D["t"], D["tn"]
    N, K = code.shape
    pi, a, b, tl = np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N)
    P = np.full((N, K), np.nan)
    for k in range(K):
        c = code[:, k]
        cr = c == 1
        pi[cr], tl[cr] = 1.0, t[cr, k]
        a[cr], b[cr] = kappa0 * m[cr], kappa0 * (1.0 - m[cr])
        i = np.flatnonzero((c == 2) | (c == 3))
        if i.size:
            r = np.maximum(0.0, (t[i, k] - tl[i]) / DT)
            S = _surv(a[i], b[i], r, m[i], learn)
            L = np.where(c[i] == 2, Lh[i, k], Lm[i, k])
            L1, L0 = expit(L), expit(-L)
            p = pi[i]
            w11, w10, w00 = p * S * L1, p * (1.0 - S) * L0, (1.0 - p) * L0
            z = w11 + w10 + w00
            w11, w10 = w11 / z, w10 / z
            if learn:
                a[i] += r * w11
                b[i] += w10
            pi[i], tl[i] = w11, t[i, k]
        rl = c == 5
        if learn:
            b[rl] += pi[rl]
        pi[rl], tl[rl] = 1.0, t[rl, k]
        if k < K - 1:
            i = np.flatnonzero(c > 0)
            r = np.maximum(0.0, (tn[i, k] - tl[i]) / DT)
            P[i, k] = pi[i] * _surv(a[i], b[i], r, m[i], learn)
    return P


def scene_nll(D, P, eps=0.0):
    p = P[D["msk"]]
    if eps:
        p = (1.0 - eps) * p + 0.5 * eps
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    yy = D["yy"]
    loss = -(yy * np.log(q) + (1.0 - yy) * np.log1p(-q))
    return np.bincount(D["ss"], weights=loss,
                       minlength=len(D["scenes"])) / D["cnt"]


# ── Families ────────────────────────────────────────────────────────────────

def fam_spec(meta):
    op, pp, np_ = meta["ours_params"], meta["pf_params"], meta["pf_nbr_params"]
    return {
        "hiper": dict(learn=True, prior="nbr", pkey="kappa0",
                      pgrid=list(te.KAPPA_FINAL), pfrozen=op["kappa0"],
                      det=(op["p_miss"], op["p_false"]), tau_pi=op["tau_pi"]),
        "pf": dict(learn=False, prior="life", pkey="lifetime_days",
                   pgrid=list(te.LIFETIME_GRID_DAYS),
                   pfrozen=pp["lifetime_days"],
                   det=(pp["p_miss"], pp["p_false"]), tau_pi=pp["tau_pi"]),
        "pf_nbr": dict(learn=False, prior="nbr", pkey=None, pgrid=[None],
                       pfrozen=None, det=(np_["p_miss"], np_["p_false"]),
                       tau_pi=np_["tau_pi"]),
    }


def prior_args(D, spec, pval):
    """(m per object, kappa0) of one prior setting."""
    if spec["prior"] == "nbr":
        m = D["m_nbr"]
    else:
        m0 = min(max(math.exp(-1.0 / pval), M_CLIP), 1.0 - M_CLIP)
        m = np.full(len(D["sidx"]), m0)
    kappa = pval if spec["pkey"] == "kappa0" else 1.0
    return m, kappa


def run(D, spec, theta, pval):
    m, kappa = prior_args(D, spec, pval)
    Lh, Lm = llr(theta, D)
    return simulate(D, Lh, Lm, m, kappa, spec["learn"])


# ── Train fit ───────────────────────────────────────────────────────────────

def fit(Dtr, spec, free, theta0):
    """Minimise the mean-over-scenes prediction NLL on the train streams
    over the free entries of theta (others fixed at theta0)."""
    free = np.asarray(free, bool)
    m, kappa = prior_args(Dtr, spec, spec["pfrozen"])

    def f(x):
        th = theta0.copy()
        th[free] = x
        Lh, Lm = llr(th, Dtr)
        P = simulate(Dtr, Lh, Lm, m, kappa, spec["learn"])
        return float(np.mean(scene_nll(Dtr, P)))

    x0 = theta0[free].copy()
    f0 = f(x0)
    best, nfev = None, 0
    for method in ("L-BFGS-B", "Powell"):
        start = x0 if best is None else best.x
        res = minimize(f, start, method=method,
                       options={"maxfev": 1500} if method == "Powell"
                       else {"maxiter": 300})
        nfev += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    th = theta0.copy()
    th[free] = best.x
    return th, f0, float(best.fun), nfev


# ── Val grids ───────────────────────────────────────────────────────────────

def grid_rows(Dv, spec, variant, thetas):
    """Per-scene val NLL of every configuration of one variant.
    thetas: {'learned': theta} for theta-based variants."""
    rows = []
    det = te.grid(te.DET_GRID)
    if variant in ("ref_wide", "miss_slopes_wide"):
        det = te.grid({"p_miss": te.DET_GRID["p_miss"],
                       "p_false": P_FALSE_EXTRA + te.DET_GRID["p_false"]})
    eps_grid = [0.0] if variant == "ref" else EPS_GRID
    for pval in spec["pgrid"]:
        if variant in ("ref", "ref_eps", "ref_wide", "learned_slopes",
                       "miss_slopes_det", "miss_slopes_wide"):
            for d in det:
                th = theta_ref(d["p_miss"], d["p_false"])
                if variant == "learned_slopes":
                    th[HIT_IDX] = thetas["learned"][HIT_IDX]
                    th[MISS_IDX] = thetas["learned"][MISS_IDX]
                if variant in ("miss_slopes_det", "miss_slopes_wide"):
                    th[MISS_IDX] = thetas["abl_miss_only"][MISS_IDX]
                P = run(Dv, spec, th, pval)
                for eps in eps_grid:
                    rows.append(({**d, spec["pkey"] or "prior": pval,
                                  "eps": eps}, th, scene_nll(Dv, P, eps)))
        else:
            th = thetas[variant]
            P = run(Dv, spec, th, pval)
            for eps in eps_grid:
                rows.append(({spec["pkey"] or "prior": pval, "eps": eps},
                             th, scene_nll(Dv, P, eps)))
    return rows


def select(rows, idx=None):
    def score(r):
        v = r[2] if idx is None else r[2][idx]
        return float(np.mean(v))
    return min(rows, key=score)


def two_fold(rows, n_scenes):
    A = np.arange(0, n_scenes, 2)
    B = np.arange(1, n_scenes, 2)
    rA, rB = select(rows, A), select(rows, B)
    on_b = float(np.mean(rA[2][B]))
    on_a = float(np.mean(rB[2][A]))
    return {"params_selected_on_A": rA[0], "nll_on_B": on_b,
            "params_selected_on_B": rB[0], "nll_on_A": on_a,
            "mean_heldout_nll": 0.5 * (on_a + on_b)}


def nested_two_fold(rows_by_variant, pool, n_scenes):
    """2-fold check in which the VARIANT is also chosen on the selection
    fold: the best (variant, configuration) over all rows of the pool on
    A is scored on B, and vice versa."""
    A = np.arange(0, n_scenes, 2)
    B = np.arange(1, n_scenes, 2)
    cand = [(v, r) for v in pool for r in rows_by_variant[v]]
    vA, rA = min(cand, key=lambda x: float(np.mean(x[1][2][A])))
    vB, rB = min(cand, key=lambda x: float(np.mean(x[1][2][B])))
    on_b, on_a = float(np.mean(rA[2][B])), float(np.mean(rB[2][A]))
    return {"pool": pool, "n_configs": len(cand),
            "variant_selected_on_A": vA, "params_selected_on_A": rA[0],
            "nll_on_B": on_b, "variant_selected_on_B": vB,
            "params_selected_on_B": rB[0], "nll_on_A": on_a,
            "mean_heldout_nll": 0.5 * (on_a + on_b)}


# ── Harness check (explore_common.run_model with a filter subclass) ─────────

class EvidenceHPF(HierarchicalPersistenceFilter):
    """persistence_v2 filter whose HIT / MISS likelihood ratio is looked up
    per event (table {(oid, t): LLR}, built from observable event fields),
    and whose prediction is mixed with an outlier probability eps."""

    def __init__(self, table, eps=0.0, **kw):
        super().__init__(**kw)
        self.table, self.eps = table, float(eps)

    def observe(self, oid, t, evidence, eps_l=0.0, **info):
        st = self._s[oid]
        if evidence == NONE:
            return
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        S = self._survival(st, r)
        L = self.table.get((oid, float(t)), 0.0)  # creation hit: no-op
        L1, L0 = float(expit(L)), float(expit(-L))
        pi = st.pi
        w11 = pi * S * L1
        w10 = pi * (1.0 - S) * L0
        w01 = (1.0 - pi) * self.rho * L1
        w00 = (1.0 - pi) * (1.0 - self.rho) * L0
        z = w11 + w10 + w01 + w00
        w11, w10, w01 = w11 / z, w10 / z, w01 / z
        if self.learn_rate:
            st.a += r * w11
            st.b += w10
        st.pi = w11 + w01
        st.t = float(t)
        st.last_evidence = evidence

    def presence(self, oid, t):
        p = super().presence(oid, t)
        return (1.0 - self.eps) * p + 0.5 * self.eps


def harness(Dv, streams, spec, theta, pval, eps, nbr, name):
    Lh, Lm = llr(theta, Dv)
    tables = {sc: {} for sc in Dv["scenes"]}
    code = Dv["code"]
    for i, (sc, oid) in enumerate(Dv["keys"]):
        for k in np.flatnonzero((code[i] == 2) | (code[i] == 3)):
            tables[sc][(oid, float(Dv["t"][i, k]))] = float(
                Lh[i, k] if code[i, k] == 2 else Lm[i, k])
    if spec["prior"] == "nbr":
        kw = dict(default_lambda=nbr.lam0, use_class_prior=False,
                  rate_fn=functools.partial(nbr.rate, K=NBR["K"],
                                            tau=NBR["tau"], k=NBR["k"]))
    else:
        kw = dict(default_lambda=1.0 / (pval * DT), use_class_prior=False)
    if spec["learn"]:
        kw.update(kappa0=pval, variance_gate=False)
    kw.update(learn_rate=spec["learn"], tau_pi=spec["tau_pi"])
    return ec.run_model(streams, lambda sc: EvidenceHPF(tables[sc], eps,
                                                        **kw), name)


def ref_harness(streams, meta):
    """The frozen references through te.make_model (unchanged code)."""
    out = {}
    for name, fam, p in (("hiper", "hpf_nbr", meta["ours_params"]),
                         ("pf", "pf", meta["pf_params"]),
                         ("pf_nbr", "pf_nbr", meta["pf_nbr_params"])):
        out[name] = ec.run_model(streams, lambda sc: te.make_model(fam, p),
                                 name)
    return out


# ── Diagnostics ─────────────────────────────────────────────────────────────

def breakdown(D, P, eps=0.0):
    """Val NLL share by (event at k, label_next) (scoring only)."""
    p = P[D["msk"]]
    p = (1.0 - eps) * p + 0.5 * eps
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    yy = D["yy"]
    loss = -(yy * np.log(q) + (1.0 - yy) * np.log1p(-q))
    ev = D["code"][D["msk"]]
    inv = {v: k for k, v in CODE.items()}
    out = {}
    for c in np.unique(ev):
        for lab in (0, 1):
            s = (ev == c) & (yy == lab)
            if s.any():
                out[f"{inv[int(c)]}|y={lab}"] = {
                    "n": int(s.sum()), "sum_nll": float(loss[s].sum()),
                    "mean_nll": float(loss[s].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-harness", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the cached arrays (built.pkl)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    meta = ec.meta()
    spec = fam_spec(meta)

    # Priors: val as the reference (records_train); train cross-fitted
    rec_tr, rec_tr2 = ec.load("records", "train"), ec.load("records", "train2")
    sub2 = [r for r in rec_tr2
            if 6 <= int(r["scene"].replace("FloorPlan", "")) % 100 <= 10]
    log("records train", len(rec_tr), "train2", len(rec_tr2),
        "train2 subset (20 scenes)", len(sub2),
        len({r["scene"] for r in sub2}))
    pr_val = nbr_prior(rec_tr)
    pr_for_train = nbr_prior(sub2)
    streams_val = ec.load("streams", "val")
    cache = OUT / "built.pkl"
    Dv = None
    if cache.exists() and not args.rebuild:
        Dv, Dtr = pickle.load(open(cache, "rb"))
        if "age" in Dv["raw"] and "notvis" in Dv["raw"]:
            log("loaded", cache)
        else:
            log("cache lacks the ext_* raw features: rebuilding")
            Dv = None
    if Dv is None:
        Dv = build(streams_val, pr_val, "val")
        Dtr = concat(build(ec.load("streams", "train"), pr_for_train,
                           "train"),
                     build(ec.load("streams", "train2"), pr_val, "train2"),
                     "train+train2")
        pickle.dump((Dv, Dtr), open(cache, "wb"))
    c = centers(Dtr)
    attach_features(Dv, c)
    attach_features(Dtr, c)
    log("built val", Dv["code"].shape, "train", Dtr["code"].shape,
        "centers", {k: round(v, 4) for k, v in c.items()})
    res = {"centers": c, "eps_grid": EPS_GRID, "nbr_frozen": NBR,
           "train_prior": "train streams: 20 train2 scenes; train2 streams: "
                          "records_train", "families": {}}

    # 1. Engine reproduces the frozen references
    eng_ref = {}
    for f, sp in spec.items():
        th = theta_ref(*sp["det"])
        eng_ref[f] = float(np.mean(scene_nll(Dv, run(Dv, sp, th,
                                                     sp["pfrozen"]))))
    log("engine reference val NLL", eng_ref)
    res["engine_reference_val_nll"] = eng_ref

    # 2. Train fits (HiPer, PF, PF+retrieved), full + ablations
    # free masks over PNAMES (9 entries; the last two are the ext_* slopes)
    T_, F_ = True, False
    parts = {"learned": [T_] * 7 + [F_, F_],
             "abl_hit_only": [T_, T_, T_, T_, T_, F_, F_, F_, F_],
             "abl_miss_only": [T_, F_, F_, F_, T_, T_, T_, F_, F_],
             "abl_intercepts_only": [T_, F_, F_, F_, T_, F_, F_, F_, F_],
             "abl_miss_nobs": [T_, F_, F_, F_, T_, T_, F_, F_, F_],
             "abl_miss_nf": [T_, F_, F_, F_, T_, F_, T_, F_, F_],
             # ext_*: abl_miss_only + one more field (warm start there)
             "ext_hit_vis": [T_, F_, F_, F_, T_, T_, T_, T_, F_],
             "ext_miss_age": [T_, F_, F_, F_, T_, T_, T_, F_, T_]}
    thetas = {}
    for f, sp in spec.items():
        thetas[f] = {}
        fits = {}
        for vname, free in parts.items():
            th0 = (thetas[f]["abl_miss_only"] if vname.startswith("ext_")
                   else theta_ref(*sp["det"]))
            th, f0, f1, nfev = fit(Dtr, sp, free, th0)
            thetas[f][vname] = th
            ph = expit(th[0])
            fits[vname] = {"theta": dict(zip(PNAMES, th.tolist())),
                           "train_nll_start": f0, "train_nll_fit": f1,
                           "n_evals": nfev,
                           "llr_hit_at_mean": th[0], "llr_miss_at_mean": th[4]}
            log(f, vname, "train NLL %.5f -> %.5f" % (f0, f1),
                {k: round(v, 3) for k, v in zip(PNAMES, th)},
                "P(hit-ratio)=%.4f" % ph)
        res["families"][f] = {"train_fit": fits}

    # 3. Val grids, selection, 2-fold check
    n_sc = len(Dv["scenes"])
    variants = POOLS["all"]
    sel, all_rows = {}, {}
    for f, sp in spec.items():
        fam = res["families"][f]
        fam["val"] = {}
        sel[f], all_rows[f] = {}, {}
        for v in variants:
            rows = grid_rows(Dv, sp, v, thetas[f])
            all_rows[f][v] = rows
            best = select(rows)
            noeps = select([r for r in rows if r[0]["eps"] == 0.0])
            cv = two_fold(rows, n_sc)
            fam["val"][v] = {
                "n_configs": len(rows), "best_params": best[0],
                "best_val_nll": float(np.mean(best[2])),
                "best_val_nll_eps0": float(np.mean(noeps[2])),
                "best_params_eps0": noeps[0],
                "per_scene_nll": dict(zip(Dv["scenes"],
                                          best[2].tolist())),
                "cv2": cv}
            sel[f][v] = (best[0], best[1])
            log(f, v, "configs", len(rows), "val NLL %.5f" %
                np.mean(best[2]), best[0], "| eps0 %.5f" % np.mean(noeps[2]),
                "| cv2 held-out %.5f" % cv["mean_heldout_nll"])
        # nested 2-fold: the variant is also chosen on the selection fold
        fam["nested_cv2"] = {name: nested_two_fold(all_rows[f], pool, n_sc)
                             for name, pool in POOLS.items()}
        for name, r in fam["nested_cv2"].items():
            log(f, "nested cv2", name, "held-out %.5f" %
                r["mean_heldout_nll"], "| on A:", r["variant_selected_on_A"],
                r["params_selected_on_A"], "| on B:",
                r["variant_selected_on_B"], r["params_selected_on_B"])

    # 4. Train-optimal eps (information) and the fully train-fitted model:
    #    train theta, frozen prior, eps = argmin of the TRAIN NLL (no val
    #    tuning at all)
    for f, sp in spec.items():
        for v in TRAIN_EPS_VARIANTS:
            P = run(Dtr, sp, thetas[f][v], sp["pfrozen"])
            tr = {str(e): float(np.mean(scene_nll(Dtr, P, e)))
                  for e in EPS_GRID}
            res["families"][f][f"train_nll_by_eps_{v}"] = tr
            e_tr = float(min(tr, key=tr.get))
            Pv = run(Dv, sp, thetas[f][v], sp["pfrozen"])
            vv = float(np.mean(scene_nll(Dv, Pv, e_tr)))
            res["families"][f][f"train_fitted_{v}"] = {
                "prior": sp["pfrozen"], "eps_train_opt": e_tr,
                "val_nll": vv}
            log(f, f"train NLL by eps ({v})",
                {k: round(x, 5) for k, x in tr.items()},
                "| fully train-fitted: val NLL %.5f (eps %s, prior %s)"
                % (vv, e_tr, sp["pfrozen"]))

    # 5. Loss breakdown (val, scoring only): reference vs learned (HiPer)
    for f, sp in spec.items():
        bd = {}
        for v in ("ref", "ref_eps", "ref_wide", "learned", "abl_miss_only",
                  "miss_slopes_det", "miss_slopes_wide", "ext_hit_vis",
                  "ext_miss_age"):
            p, th = sel[f][v]
            P = run(Dv, sp, th, p.get(sp["pkey"] or "prior"))
            bd[v] = breakdown(Dv, P, p["eps"])
        res["families"][f]["val_breakdown"] = bd

    res["summary"] = summarize(res, variants)
    log("summary", json.dumps(res["summary"], default=float))
    save(res, t_start)

    # 6. Harness check of the selected configurations
    if not args.skip_harness:
        te.NBR["prior"] = pr_val
        te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
        res["harness_reference"] = ref_harness(streams_val, meta)
        log("harness reference", res["harness_reference"])
        best_v = res["summary"]["best_hiper_variant"]
        for f, sp in spec.items():
            hv = {}
            for v in dict.fromkeys(HARNESS_VARIANTS[f] + [best_v]):
                p, th = sel[f][v]
                pval = p.get(sp["pkey"] or "prior")
                h = harness(Dv, streams_val, sp, th, pval, p["eps"], pr_val,
                            f"{f}_{v}")
                eng = res["families"][f]["val"][v]["best_val_nll"]
                hv[v] = {**h, "engine_nll": eng,
                         "abs_diff": abs(h["nll"] - eng)}
                log("harness", f, v, {k: round(x, 5) for k, x in h.items()},
                    "engine %.6f" % eng)
                assert abs(h["nll"] - eng) < 1e-6, (f, v, h["nll"], eng)
            res["families"][f]["harness"] = hv
    save(res, t_start)
    log("done", res["seconds"], "s ->", OUT / "results.json")


HARNESS_VARIANTS = {   # + the best HiPer variant in every family
    "hiper": ["ref", "ref_eps", "ref_wide", "learned", "abl_miss_only",
              "miss_slopes_det", "miss_slopes_wide", "ext_hit_vis",
              "ext_miss_age"],
    "pf": ["ref", "ref_eps", "ref_wide", "learned", "abl_miss_only"],
    "pf_nbr": ["ref", "ref_eps", "ref_wide", "learned", "abl_miss_only"]}
DECOMP = ["ref", "ref_eps", "ref_wide", "abl_intercepts_only",
          "abl_miss_nobs", "abl_miss_nf", "abl_miss_only", "learned",
          "miss_slopes_det", "miss_slopes_wide", "ext_hit_vis",
          "ext_miss_age"]


def save(res, t_start):
    res["seconds"] = round(time.time() - t_start, 1)
    with open(OUT / "results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(OUT / "run.log", "w") as fh:
        fh.write("\n".join(LOG) + "\n")


def summarize(res, variants):
    """Best HiPer variant by val NLL (the declared criterion), its 2-fold
    check, and the same variant applied to the two PF baselines."""
    fam = res["families"]
    rank = {f: sorted(((v, fam[f]["val"][v]["best_val_nll"],
                        fam[f]["val"][v]["cv2"]["mean_heldout_nll"])
                       for v in variants), key=lambda x: x[1])
            for f in fam}
    best_v = rank["hiper"][0][0]
    out = {"ranking": rank, "best_hiper_variant": best_v,
           "best": {f: {"variant": best_v,
                        "val_nll": fam[f]["val"][best_v]["best_val_nll"],
                        "params": fam[f]["val"][best_v]["best_params"],
                        "cv2_heldout": fam[f]["val"][best_v]["cv2"][
                            "mean_heldout_nll"]} for f in fam},
           "reference": {f: {"val_nll": fam[f]["val"]["ref"]["best_val_nll"],
                             "params": fam[f]["val"]["ref"]["best_params"],
                             "cv2_heldout": fam[f]["val"]["ref"]["cv2"][
                                 "mean_heldout_nll"]} for f in fam}}
    # per-scene consistency of the best variant (val, 20 scenes)
    out["paired"] = {}
    for f in fam:
        V = fam[f]["val"]
        for b in ("ref", "ref_wide", "ref_eps", "learned", "abl_miss_only"):
            if b == best_v:
                continue
            pa, pb = V[best_v]["per_scene_nll"], V[b]["per_scene_nll"]
            d = np.array([pa[s_] - pb[s_] for s_ in sorted(pa)])
            bt = pm.paired_bootstrap(pa, pb)
            out["paired"][f"{f}:{best_v}-{b}"] = {
                "mean_diff": float(d.mean()), "median_diff": float(
                    np.median(d)), "scenes_better": int((d < 0).sum()),
                "n_scenes": int(d.size), "ci_lo": bt["ci_lo"],
                "ci_hi": bt["ci_hi"], "p_value": bt["p_value"]}
    out["fair_baselines"] = [
        {"name": f, "val_nll_before": out["reference"][f]["val_nll"],
         "val_nll_after": out["best"][f]["val_nll"],
         "cv2_before": out["reference"][f]["cv2_heldout"],
         "cv2_after": out["best"][f]["cv2_heldout"]} for f in ("pf", "pf_nbr")]
    # nested 2-fold (variant chosen on the selection fold too)
    out["nested_cv2"] = {f: {name: {
        "mean_heldout_nll": r["mean_heldout_nll"],
        "selected": [r["variant_selected_on_A"], r["variant_selected_on_B"]]}
        for name, r in fam[f]["nested_cv2"].items()} for f in fam}
    # effect decomposition: val NLL (eps selected / eps = 0) and cv2
    out["decomposition"] = {f: [
        {"variant": v, "val_nll": fam[f]["val"][v]["best_val_nll"],
         "val_nll_eps0": fam[f]["val"][v]["best_val_nll_eps0"],
         "cv2_heldout": fam[f]["val"][v]["cv2"]["mean_heldout_nll"],
         "params": fam[f]["val"][v]["best_params"]} for v in DECOMP]
        for f in fam}
    out["train_fitted"] = {f: {v: fam[f][f"train_fitted_{v}"]
                               for v in TRAIN_EPS_VARIANTS} for f in fam}
    return out


if __name__ == "__main__":
    main()
