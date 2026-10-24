#!/usr/bin/env python3
"""
scripts/explore_w3_mapprior.py
==============================
VALIDATION-ONLY exploration, idea w3 (slug w3_mapprior): learn HiPer's
appearance-retrieved survival prior from TRAIN *map objects* instead of
TRAIN ground-truth objects.

The current prior (priors.NeighbourSurvivalPrior on records_train) keys each
GT object by its mean DINOv2 embedding over all its tracks and uses its GT
stay/vanish history. The filter, however, predicts for MAP objects, keyed by
the creating track's embedding, whose label process (lifelong_objects.
gt_labels) also contains identity switches and relocations.

Record sources (declared before any result was seen):
  gt    GT-object records (records_<split>): mean-DINOv2 key, GT history.
        = the current prior (reference with split train).
  ckey  ABLATION, key only: map-object creation embedding as key, history
        of the GT object that created the map object (records_<split>).
  map   THE IDEA: map-object creation embedding as key, map history read
        from the TRAIN labels: at session k the map object is at risk iff
        gt_present_now(k) = 1; it survives the gap k -> k+1 (exposure
        t_{k+1} - t_k) iff label_next(k) = 1, else one vanish in that gap
        (same at-risk rules as priors.object_exposures, applied to the map
        object's label process).
  mix   union of gt and map records; each map record's kernel weight is
        multiplied by omega in OMEGA_GRID, weights renormalised to max 1
        (omega = 1: plain union, the unmodified NeighbourSurvivalPrior).
Training size: SIZES = train, train+train2 (for every source).

Every source is re-tuned on THOR val with te.NBR_GRID x te.DET_GRID x
te.KAPPA_FINAL (HiPer, 2688 configs; the reference was tuned on the same
grid) and te.NBR_GRID x te.DET_GRID (PF+retrieved, 384 configs). PF(lambda)
has no retrieved prior; it is reported with the pooled rate lambda_0 of
each record set (DET_GRID), next to its val-tuned lifetime.

Grid scores use an exact vectorised re-implementation of the replay
(HierarchicalPersistenceFilter, rho = 0, pi_new = 1, no environment level)
and of NeighbourSurvivalPrior.rate; `check` proves equality with the
official harness (ec.run_model). Each selected config is then re-scored
with the official harness (the NLL / Brier / ECE / AUROC reported).

Robustness: 2-fold check over val scenes (sorted names; fold A = even
index, fold B = odd index): select on A, score on B and vice versa; mean
held-out NLL. Labels of val are used only for scoring and for this
declared selection. Never reads test / test2 / test3 / obchange.

  python scripts/explore_w3_mapprior.py check
  python scripts/explore_w3_mapprior.py run --workers 3
  python scripts/explore_w3_mapprior.py diag    # after run (diagnostic)
  python scripts/explore_w3_mapprior.py ablate  # POST-HOC: gt + ckey union
  python scripts/explore_w3_mapprior.py ablate_gkmap  # POST-HOC: GT-mean
                                                # key + map history
  python scripts/explore_w3_mapprior.py summary # nested 2-fold, after run
Outputs: RESULTS/explore_cache/w3_mapprior/
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.ephemerality.priors import (  # noqa: E402
    NeighbourSurvivalPrior, _map_rate, weighted_map_rate)
from hiper.evaluation.lifelong_objects import gt_labels  # noqa: E402

DAY = 86400.0
OUT = ec.CACHE / "w3_mapprior"
CODE = {"create": 1, "hit": 2, "miss": 3, "none": 4, "relocate": 5}
M_CLIP = 1e-4                     # HierarchicalPersistenceFilter default
REF_NBR = dict(K=300, tau=0.05, k_nbr=1.0)   # frozen run v2 (meta.json)

# ── Declared grids (fixed before any result was seen) ────────────────────────
SOURCES = ("gt", "ckey", "map", "mix")
SIZES = {"train": ("train",), "train12": ("train", "train2")}
OMEGA_GRID = [0.5, 1.0, 2.0]                 # mix only
NBR = te.grid(te.NBR_GRID)                   # 24: K x tau x k_nbr
DET = te.grid(te.DET_GRID)                   # 16: p_miss x p_false
KAPPA = list(te.KAPPA_FINAL)                 # 7


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if (np.isfinite(n) and n > 1e-12) else None


# ── Training records ─────────────────────────────────────────────────────────

def gt_records(splits):
    return [{"emb": r["emb"], "S": float(r["S"]),
             "gaps": [float(d) for d in r["gaps"]]}
            for sp in splits for r in ec.load("records", sp)]


def map_history(stream, ob):
    """(S [s], vanish gaps [s]) of one TRAIN map object from the TRAIN
    labels: at risk at k iff gt_present_now(k) = 1, survives k -> k+1 iff
    label_next(k) = 1."""
    gt, T = stream["gt"], stream["times"]
    S, gaps = 0.0, []
    for ev in ob["events"]:
        k = ev["k"]
        now, nxt = gt_labels(gt, ev["gt_id"], ev["pos_session"], k)
        if nxt is None or now != 1:
            continue
        d = float(T[k + 1]) - float(T[k])
        if nxt == 1:
            S += d
        else:
            gaps.append(d)
    return S, gaps


def map_records(splits, history="map"):
    """One record per TRAIN map object keyed by its creation embedding;
    history 'map' (own label process) or 'ckey' (GT history of the GT object
    that created it)."""
    out = []
    for sp in splits:
        streams = ec.load("streams", sp)
        gtrec = {(r["scene"], r["gt_id"]): r for r in ec.load("records", sp)}
        for sc in sorted(streams):
            st = streams[sc]
            for oid, ob in st["objects"].items():
                e = _unit(ob["events"][0].get("emb"))
                if e is None:
                    continue
                if history == "map":
                    S, gaps = map_history(st, ob)
                else:
                    r = gtrec.get((sc, ob["events"][0]["gt_id"]))
                    if r is None:
                        continue
                    S, gaps = float(r["S"]), [float(d) for d in r["gaps"]]
                out.append({"emb": e, "S": S, "gaps": gaps})
    return out


def gkmap_records(splits):
    """POST-HOC ablation (declared after `run`, 2026-10-20 23:15): key = mean
    DINOv2 of the GT object that created the map object (the key of `gt`),
    history = the map object's own label history (the history of `map`).
    Completes the 2 x 2 table key {GT mean, creation track} x history
    {GT object, map object}; same objects as ckey."""
    out = []
    for sp in splits:
        streams = ec.load("streams", sp)
        gtrec = {(r["scene"], r["gt_id"]): r for r in ec.load("records", sp)}
        for sc in sorted(streams):
            st = streams[sc]
            for oid, ob in st["objects"].items():
                if _unit(ob["events"][0].get("emb")) is None:
                    continue
                r = gtrec.get((sc, ob["events"][0]["gt_id"]))
                if r is None:
                    continue
                S, gaps = map_history(st, ob)
                out.append({"emb": np.asarray(r["emb"], np.float64),
                            "S": S, "gaps": gaps})
    return out


def records_for(source, size, omega=1.0):
    """(records, per-record weights) of one declared variant."""
    splits = SIZES[size]
    if source == "gt":
        recs = gt_records(splits)
        return recs, np.ones(len(recs))
    if source in ("map", "ckey"):
        recs = map_records(splits, source)
        return recs, np.ones(len(recs))
    if source == "gkmap":     # POST-HOC: GT-mean key, map history
        recs = gkmap_records(splits)
        return recs, np.ones(len(recs))
    if source == "gtckey":    # POST-HOC ablation of mix: no map history
        g, c = gt_records(splits), map_records(splits, "ckey")
        return g + c, np.ones(len(g) + len(c))
    g, m = gt_records(splits), map_records(splits, "map")
    return g + m, np.concatenate([np.ones(len(g)), np.full(len(m), omega)])


def pooled_summary(recs):
    S = sum(r["S"] for r in recs)
    gaps = np.array([d for r in recs for d in r["gaps"]], float)
    lam0 = _map_rate(S, gaps, 0.0, 0.0)
    return {"n_records": len(recs), "exposure_days": S / DAY,
            "n_vanish": int(gaps.size), "lam0_per_day": lam0 * DAY,
            "lam0_days": 1.0 / (lam0 * DAY),
            "n_no_exposure": int(sum(1 for r in recs
                                     if r["S"] == 0 and not r["gaps"]))}


class WeightedNbrPrior(NeighbourSurvivalPrior):
    """NeighbourSurvivalPrior with per-record weights rw (mix variants):
    w_i = rw_i exp((s_i - s_max) / tau), renormalised to max 1. With all
    rw = 1 it returns exactly NeighbourSurvivalPrior.rate."""

    def __init__(self, records, rw):
        super().__init__(records)
        self.rw = np.asarray(rw, np.float64)

    def rate(self, emb, K=100, tau=0.05, k=1.0):
        e = np.asarray(emb, dtype=np.float64)
        n = np.linalg.norm(e)
        if not np.isfinite(n) or n < 1e-12:
            return self.lam0
        K = int(min(max(K, 1), len(self.E)))
        e = e / n
        key = ("w", e.tobytes(), K, tau, k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        idx, s = self._neighbours(e, K)
        w = np.exp((s - s.max()) / tau) if tau else np.ones(K)
        w = self.rw[idx] * w
        w = w / w.max()
        gaps = np.concatenate([self.gaps[i] for i in idx])
        gap_w = np.concatenate([np.full(len(self.gaps[i]), wi)
                                for i, wi in zip(idx, w)])
        lam = weighted_map_rate(w, self.S[idx], gaps, gap_w, k, self.lam0)
        self._cache[key] = lam
        return lam


def official_prior(recs, rw):
    if np.all(rw == 1.0):
        return NeighbourSurvivalPrior(recs)
    return WeightedNbrPrior(recs, rw)


# ── Fast, exact neighbour rates (vectorised over val objects) ───────────────

class FastNbr:
    """NeighbourSurvivalPrior.rate for many queries at once. Gaps take few
    distinct values, so each record's gaps are stored as counts per value;
    the bisection (80 steps in log lambda) is the one of
    priors.weighted_map_rate."""

    def __init__(self, recs, rw):
        self.E = np.stack([r["emb"] for r in recs]).astype(np.float64)
        self.S = np.array([r["S"] for r in recs], np.float64)
        allg = [np.asarray(r["gaps"], np.float64) for r in recs]
        self.gv = np.unique(np.concatenate(allg))
        self.G = np.zeros((len(recs), len(self.gv)))
        for i, g in enumerate(allg):
            for d in g:
                self.G[i, np.searchsorted(self.gv, d)] += 1.0
        self.lam0 = _map_rate(float(self.S.sum()), np.concatenate(allg),
                              0.0, 0.0)
        self.rw = np.asarray(rw, np.float64)
        self._sims = None

    def prepare(self, Q):
        self._sims = Q @ self.E.T
        self._part = {}

    def rates(self, K, tau, k):
        sims = self._sims
        K = int(min(max(K, 1), self.E.shape[0]))
        if K not in self._part:
            idx = np.argpartition(-sims, K - 1, axis=1)[:, :K]
            self._part[K] = (idx, np.take_along_axis(sims, idx, 1))
        idx, s = self._part[K]
        w = np.exp((s - s.max(1, keepdims=True)) / tau) if tau \
            else np.ones_like(s)
        w = self.rw[idx] * w
        w = w / w.max(1, keepdims=True)
        Sw = (w * self.S[idx]).sum(1)
        C = np.einsum("nk,nkj->nj", w, self.G[idx])
        n = C.sum(1)
        beta = k / self.lam0
        gv = self.gv[None, :]
        lo = (n + k) / (beta + Sw + C @ self.gv)
        hi = (n + k) / (beta + Sw)
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            x = mid[:, None] * gv
            sc = k / mid - beta - Sw + (C * gv * np.exp(-x)
                                        / -np.expm1(-x)).sum(1)
            pos = sc > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        lam = np.sqrt(lo * hi)
        lam = np.where(n == 0.0, k / (beta + Sw), lam)
        if n.size and (n + k <= 0.0).any():
            lam = np.where(n + k <= 0.0, self.lam0, lam)
        return lam


# ── Fast, exact replay of the persistence filter on the val streams ─────────

class ValArrays:
    """Dense (object x session) event arrays of the val streams. Y holds
    label_next (EVAL ONLY: scoring and the declared selection)."""

    def __init__(self, streams):
        self.scenes = sorted(streams)
        objs = [(si, sc, oid, ob) for si, sc in enumerate(self.scenes)
                for oid, ob in streams[sc]["objects"].items()]
        KS = max(len(streams[sc]["times"]) for sc in self.scenes)
        N = len(objs)
        self.EV = np.zeros((N, KS), np.int8)
        self.EPS = np.zeros((N, KS))
        self.T = np.full((N, KS), np.nan)
        self.Y = np.full((N, KS), np.nan)
        self.obj_scene = np.array([o[0] for o in objs])
        self.keys = [(sc, oid) for _, sc, oid, _ in objs]
        embs = []
        for n, (si, sc, oid, ob) in enumerate(objs):
            st = streams[sc]
            times = st["times"]
            self.T[n, :len(times)] = times
            for ev in ob["events"]:
                k = ev["k"]
                self.EV[n, k] = CODE[ev["ev"]]
                self.EPS[n, k] = float(ev["eps_l"])
                _, nxt = gt_labels(st["gt"], ev["gt_id"], ev["pos_session"],
                                   k)
                if nxt is not None:
                    self.Y[n, k] = float(nxt)
            embs.append(_unit(ob["events"][0].get("emb")))
        d = next(e.size for e in embs if e is not None)
        self.qok = np.array([e is not None for e in embs])
        self.Q = np.stack([e if e is not None else np.zeros(d)
                           for e in embs])
        self.mask = (self.EV != 0) & ~np.isnan(self.Y)
        self.row_scene = np.broadcast_to(self.obj_scene[:, None],
                                         self.EV.shape)[self.mask]
        self.n_scene = np.bincount(self.row_scene,
                                   minlength=len(self.scenes))
        self.y = self.Y[self.mask]

    def m_from_rates(self, lam, lam0):
        lam = np.where(self.qok, lam, lam0)
        return np.clip(np.exp(-lam * DAY), M_CLIP, 1.0 - M_CLIP)

    def scene_nll(self, P):
        p = np.clip(np.clip(P[self.mask], 0.0, 1.0), 1e-6, 1.0 - 1e-6)
        y = self.y
        nll = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        return (np.bincount(self.row_scene, weights=nll,
                            minlength=len(self.scenes)) / self.n_scene)


def _surv(a, b, r, m, learn):
    """E[p^r] (Beta) or m^r (fixed rate); 1 where r <= 0."""
    out = np.ones_like(r)
    pos = r > 0.0
    if pos.any():
        if learn:
            aa, bb, rr = a[pos], b[pos], r[pos]
            out[pos] = np.exp(gammaln(aa + rr) + gammaln(aa + bb)
                              - gammaln(aa) - gammaln(aa + bb + rr))
        else:
            out[pos] = m[pos] ** r[pos]
    return out


def run_filter(A, m, p_miss, p_false, kappa=None, learn=True, rho=0.0,
               pi_new=1.0, dt_ref=DAY):
    """Predictions P[n, k] = presence at t_{k+1} after the events of k, as
    replay(stream, HierarchicalPersistenceFilter(...)) produces them."""
    N, KS = A.EV.shape
    pi, a, b, tl = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    P = np.full((N, KS), np.nan)
    kap = None if kappa is None else np.broadcast_to(
        np.asarray(kappa, np.float64), (N,))
    for k in range(KS):
        ev, t = A.EV[:, k], A.T[:, k]
        c = ev == CODE["create"]
        if c.any():
            pi[c] = pi_new
            if learn:
                a[c] = kap[c] * m[c]
                b[c] = kap[c] * (1.0 - m[c])
            tl[c] = t[c]
        # create is followed by observe(HIT) at r = 0 (apply_event)
        u = c | (ev == CODE["hit"]) | (ev == CODE["miss"])
        if u.any():
            r = np.maximum(0.0, (t[u] - tl[u]) / dt_ref)
            S = _surv(a[u], b[u], r, m[u], learn)
            hit = ev[u] != CODE["miss"]
            L1 = np.where(hit, 1.0 - p_miss, p_miss)
            L0 = np.where(hit, p_false, 1.0 - p_false)
            w = 1.0 - np.clip(A.EPS[u, k], 0.0, 1.0)
            L1 = np.where(hit, w * L1 + (1.0 - w), L1)
            L0 = np.where(hit, w * L0 + (1.0 - w), L0)
            p = pi[u]
            w11 = p * S * L1
            w10 = p * (1.0 - S) * L0
            w01 = (1.0 - p) * rho * L1
            w00 = (1.0 - p) * (1.0 - rho) * L0
            z = w11 + w10 + w01 + w00
            ok = z > 0.0
            zs = np.where(ok, z, 1.0)
            w11, w10, w01 = w11 / zs, w10 / zs, w01 / zs
            ui = np.flatnonzero(u)[ok]
            if learn:
                a[ui] += (r * w11)[ok]
                b[ui] += w10[ok]
            pi[ui] = (w11 + w01)[ok]
            tl[ui] = t[ui]
        rl = ev == CODE["relocate"]
        if rl.any():
            if learn:
                b[rl] += pi[rl]
            pi[rl] = pi_new
            tl[rl] = t[rl]
        if k + 1 < KS:
            q = ev != 0
            r = np.maximum(0.0, (A.T[q, k + 1] - tl[q]) / dt_ref)
            S = _surv(a[q], b[q], r, m[q], learn)
            P[q, k] = pi[q] * S + (1.0 - pi[q]) * rho
    return P


class Tiled:
    """The val arrays repeated n times (one copy per kappa0 value), so one
    run_filter call scores every kappa0 of a grid row."""

    def __init__(self, A, n):
        self.n, self.N = n, A.EV.shape[0]
        self.EV = np.tile(A.EV, (n, 1))
        self.EPS = np.tile(A.EPS, (n, 1))
        self.T = np.tile(A.T, (n, 1))


def hpf_all_kappa(A, At, m, p_miss, p_false):
    """[kappa, scene] NLL of HiPer for every kappa0 in KAPPA."""
    kap = np.repeat(np.asarray(KAPPA, np.float64), At.N)
    P = run_filter(At, np.tile(m, At.n), p_miss, p_false, kappa=kap)
    return np.stack([A.scene_nll(P[l * At.N:(l + 1) * At.N])
                     for l in range(At.n)])


# ── One variant: full grids ──────────────────────────────────────────────────

_W = {}


def variant_job(spec):
    """Grid of one record variant. Returns per-scene val NLL arrays:
    hpf [nbr, det, kappa, scene], pfn [nbr, det, scene]."""
    t0 = time.time()
    A = _W["A"]
    source, size, omega = spec
    recs, rw = records_for(source, size, omega)
    fast = FastNbr(recs, rw)
    fast.prepare(A.Q)
    At = Tiled(A, len(KAPPA))
    nS = len(A.scenes)
    hpf = np.full((len(NBR), len(DET), len(KAPPA), nS), np.nan)
    pfn = np.full((len(NBR), len(DET), nS), np.nan)
    for i, g in enumerate(NBR):
        m = A.m_from_rates(fast.rates(g["K"], g["tau"], g["k_nbr"]),
                           fast.lam0)
        for j, d in enumerate(DET):
            pfn[i, j] = A.scene_nll(run_filter(A, m, d["p_miss"],
                                               d["p_false"], learn=False))
            hpf[i, j] = hpf_all_kappa(A, At, m, d["p_miss"], d["p_false"])
    return {"spec": spec, "hpf": hpf, "pfn": pfn,
            "pooled": pooled_summary(recs), "seconds": time.time() - t0}


def official_job(args):
    """Re-score one selected config with the official harness."""
    spec, family, params = args
    t0 = time.time()
    recs, rw = records_for(*spec)
    te.NBR["prior"] = official_prior(recs, rw)
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    s = ec.run_model(_W["streams"], lambda sc: te.make_model(family, params),
                     family)
    return {"spec": spec, "family": family, "params": params, **s,
            "seconds": time.time() - t0}


# ── Selection helpers ────────────────────────────────────────────────────────

def hpf_params(i, j, l):
    return {**DET[j], "kappa0": KAPPA[l], "variance_gate": False, **NBR[i],
            "tau_pi": 0.3}


def pfn_params(i, j):
    return {**DET[j], **NBR[i], "tau_pi": 0.5}


def best_index(arr, scenes=None):
    """argmin over the leading axes of the mean over `scenes` (last axis)."""
    v = arr if scenes is None else arr[..., scenes]
    mean = v.mean(-1)
    flat = int(np.nanargmin(mean))
    return np.unravel_index(flat, mean.shape), float(mean.ravel()[flat])


def two_fold(arr, nS):
    """Select on fold A (even scene index), score on B, and vice versa.
    arr: [..., scene]; returns the mean held-out NLL and the picks."""
    A_ = np.arange(0, nS, 2)
    B_ = np.arange(1, nS, 2)
    out = {}
    held = []
    for name, sel, ev in (("A->B", A_, B_), ("B->A", B_, A_)):
        idx, nll_sel = best_index(arr, sel)
        nll_held = float(arr[idx][ev].mean())
        held.append(nll_held)
        out[name] = {"index": [int(x) for x in idx],
                     "nll_select_fold": nll_sel, "nll_heldout": nll_held}
    out["mean_heldout_nll"] = float(np.mean(held))
    return out


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_check():
    """Fast evaluator == official harness on the three references."""
    streams = ec.load("streams")
    A = ValArrays(streams)
    meta = ec.meta()
    recs, rw = records_for("gt", "train")
    fast = FastNbr(recs, rw)
    fast.prepare(A.Q)
    nbr = NeighbourSurvivalPrior(recs)
    out = {"lam0_fast": fast.lam0, "lam0_official": nbr.lam0}
    worst = 0.0
    for g in (REF_NBR, dict(K=10, tau=0.02, k_nbr=4.0),
              dict(K=100, tau=0.1, k_nbr=1.0)):
        lf = fast.rates(g["K"], g["tau"], g["k_nbr"])
        lo = np.array([nbr.rate(A.Q[n], K=g["K"], tau=g["tau"],
                                k=g["k_nbr"]) for n in range(len(A.Q))])
        rel = float(np.max(np.abs(lf - lo) / lo))
        worst = max(worst, rel)
        print("rates", g, "max rel diff", rel, flush=True)
    out["rate_max_rel_diff"] = worst
    te.NBR["prior"] = nbr
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    res = {}
    for name, fam, p in (("ours", "hpf_nbr", meta["ours_params"]),
                         ("pf_nbr", "pf_nbr", meta["pf_nbr_params"]),
                         ("pf", "pf", meta["pf_params"])):
        off = ec.run_model(streams, lambda sc: te.make_model(fam, p), name)
        if fam == "pf":
            m = np.full(len(A.Q), min(max(math.exp(-1.0 / p["lifetime_days"]),
                                          M_CLIP), 1 - M_CLIP))
        else:
            m = A.m_from_rates(fast.rates(p["K"], p["tau"], p["k_nbr"]),
                               fast.lam0)
        P = run_filter(A, m, p["p_miss"], p["p_false"],
                       kappa=p.get("kappa0"), learn=(fam == "hpf_nbr"))
        fnll = float(A.scene_nll(P).mean())
        res[name] = {"official_nll": off["nll"], "fast_nll": fnll,
                     "abs_diff": abs(off["nll"] - fnll)}
        print(name, res[name], flush=True)
    # one row-level comparison (HiPer): predictions of every labelled row
    p = meta["ours_params"]
    rows = []
    for sc in sorted(streams):
        rows += te.replay(streams[sc], te.make_model("hpf_nbr", p), "ours")
    m = A.m_from_rates(fast.rates(p["K"], p["tau"], p["k_nbr"]), fast.lam0)
    P = run_filter(A, m, p["p_miss"], p["p_false"], kappa=p["kappa0"])
    pos = {key: n for n, key in enumerate(A.keys)}
    lab = [r for r in rows if r["label_next"] is not None]
    diff = [abs(r["pred_presence"] - P[pos[(r["scene"], r["oid"])], r["k"]])
            for r in lab]
    ydiff = sum(int(r["label_next"] != A.Y[pos[(r["scene"], r["oid"])],
                                           r["k"]]) for r in lab)
    out["n_rows_official"] = len(lab)
    out["n_rows_fast"] = int(A.mask.sum())
    out["row_pred_max_abs_diff"] = float(np.max(diff))
    out["row_label_mismatches"] = int(ydiff)
    # the kappa-tiled grid path == one run_filter call per kappa0
    allk = hpf_all_kappa(A, Tiled(A, len(KAPPA)), m, p["p_miss"],
                         p["p_false"])
    single = np.stack([A.scene_nll(run_filter(A, m, p["p_miss"],
                                              p["p_false"], kappa=kap))
                       for kap in KAPPA])
    out["tiled_vs_single_max_abs_diff"] = float(np.abs(allk - single).max())
    out["models"] = res
    print(json.dumps(out, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "check.json", "w"), indent=1)


def cmd_run(workers):
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    streams = ec.load("streams")
    A = ValArrays(streams)
    _W.update(A=A, streams=streams)
    nS = len(A.scenes)
    specs = [(s, z, 1.0) for s in ("gt", "ckey", "map") for z in SIZES]
    specs += [("mix", z, o) for z in SIZES for o in OMEGA_GRID]
    with Pool(workers) as pool:
        res = list(pool.imap_unordered(variant_job, specs))
    res.sort(key=lambda r: specs.index(r["spec"]))
    print("grids done", round(time.time() - t_start, 1), "s", flush=True)
    np.savez_compressed(
        OUT / "scene_nll_grids.npz", scenes=np.array(A.scenes),
        **{f"hpf__{s}__{z}__{o}": r["hpf"] for r in res
           for (s, z, o) in [r["spec"]]},
        **{f"pfn__{s}__{z}__{o}": r["pfn"] for r in res
           for (s, z, o) in [r["spec"]]})

    variants = {}
    jobs = []
    for r in res:
        s, z, o = r["spec"]
        name = f"{s}@{z}" + (f"@w{o:g}" if s == "mix" else "")
        (i, j, l), nll_h = best_index(r["hpf"])
        (pi_, pj), nll_p = best_index(r["pfn"])
        variants[name] = {
            "source": s, "size": z, "omega": o, "pooled": r["pooled"],
            "grid_seconds": round(r["seconds"], 1),
            "hiper": {"fast_val_nll": nll_h,
                      "params": hpf_params(i, j, l),
                      "two_fold": two_fold(r["hpf"], nS)},
            "pf_nbr": {"fast_val_nll": nll_p, "params": pfn_params(pi_, pj),
                       "two_fold": two_fold(r["pfn"], nS)}}
        jobs.append((r["spec"], "hpf_nbr", hpf_params(i, j, l)))
        jobs.append((r["spec"], "pf_nbr", pfn_params(pi_, pj)))
    # mix as ONE family: omega selected with the other params
    for z in SIZES:
        stack = np.stack([r["hpf"] for r in res
                          if r["spec"][0] == "mix" and r["spec"][1] == z])
        stack_p = np.stack([r["pfn"] for r in res
                            if r["spec"][0] == "mix" and r["spec"][1] == z])
        variants[f"mix@{z}@family"] = {
            "hiper_two_fold_omega_selected": two_fold(stack, nS),
            "pf_nbr_two_fold_omega_selected": two_fold(stack_p, nS)}
    # reference with the frozen neighbour params (K=300, tau=0.05, k=1)
    ref = next(r for r in res if r["spec"] == ("gt", "train", 1.0))
    i_ref = NBR.index({"K": REF_NBR["K"], "tau": REF_NBR["tau"],
                       "k_nbr": REF_NBR["k_nbr"]})
    ref_fixed = {"hiper": two_fold(ref["hpf"][i_ref], nS),
                 "pf_nbr": two_fold(ref["pfn"][i_ref], nS),
                 "hiper_fast_val_nll_full": best_index(ref["hpf"][i_ref])[1]}

    with Pool(workers) as pool:
        offs = list(pool.imap_unordered(official_job, jobs))
    for o in offs:
        s, z, w = o["spec"]
        name = f"{s}@{z}" + (f"@w{w:g}" if s == "mix" else "")
        key = "hiper" if o["family"] == "hpf_nbr" else "pf_nbr"
        variants[name][key]["official"] = {
            k: o[k] for k in ("nll", "brier", "ece", "auroc")}

    # PF(lambda): pooled rate lambda_0 of each record set, DET_GRID
    pf_pooled = {}
    for name, v in variants.items():
        if "pooled" not in v or v["source"] == "mix":
            continue
        days = v["pooled"]["lam0_days"]
        m = np.full(len(A.Q), min(max(math.exp(-1.0 / days), M_CLIP),
                                  1 - M_CLIP))
        arr = np.stack([A.scene_nll(run_filter(A, m, d["p_miss"],
                                               d["p_false"], learn=False))
                        for d in DET])
        (j,), nll = best_index(arr)
        pf_pooled[name] = {"lifetime_days": days, "fast_val_nll": nll,
                           "params": {**DET[j], "lifetime_days": days},
                           "two_fold": two_fold(arr, nS)}
    # PF(lambda) reference: val-tuned lifetime (te.LIFETIME_GRID_DAYS)
    arr = np.stack([[A.scene_nll(run_filter(
        A, np.full(len(A.Q), min(max(math.exp(-1.0 / L), M_CLIP),
                                 1 - M_CLIP)),
        d["p_miss"], d["p_false"], learn=False)) for d in DET]
        for L in te.LIFETIME_GRID_DAYS])
    (li, dj), nll = best_index(arr)
    pf_ref = {"fast_val_nll": nll,
              "params": {**DET[dj], "lifetime_days": te.LIFETIME_GRID_DAYS[li]},
              "two_fold": two_fold(arr, nS)}

    out = {
        "idea": "w3_mapprior: survival prior retrieved from TRAIN map "
                "objects (creation embedding + map label history)",
        "command": "PYTHONDONTWRITEBYTECODE=1 python "
                   "scripts/explore_w3_mapprior.py run --workers "
                   f"{workers}",
        "declared_grids": {"sources": SOURCES, "sizes": SIZES,
                           "omega": OMEGA_GRID, "nbr": te.NBR_GRID,
                           "det": te.DET_GRID, "kappa": KAPPA,
                           "n_hiper_configs": len(NBR) * len(DET)
                           * len(KAPPA),
                           "n_pf_nbr_configs": len(NBR) * len(DET)},
        "reference_frozen": {"hiper_val_nll": 0.29091,
                             "pf_val_nll": 0.35914,
                             "pf_nbr_val_nll": 0.29829},
        "reference_fixed_nbr_two_fold": ref_fixed,
        "variants": variants, "pf_pooled_rate": pf_pooled,
        "pf_val_tuned_lifetime": pf_ref,
        "val_scenes": A.scenes, "n_val_rows": int(A.mask.sum()),
        "seconds": round(time.time() - t_start, 1)}
    json.dump(out, open(OUT / "results.json", "w"), indent=1, default=float)
    # short table
    print(f"{'variant':24s} {'n_rec':>6s} {'lam0_d':>7s} {'HiPer':>8s} "
          f"{'HiPer_off':>9s} {'2fold':>8s} {'PF+r':>8s} {'2fold':>8s}")
    for name, v in variants.items():
        if "hiper" not in v:
            print(f"{name:24s} omega-selected 2fold HiPer "
                  f"{v['hiper_two_fold_omega_selected']['mean_heldout_nll']:.5f}"
                  f" PF+r {v['pf_nbr_two_fold_omega_selected']['mean_heldout_nll']:.5f}")
            continue
        h, p = v["hiper"], v["pf_nbr"]
        print(f"{name:24s} {v['pooled']['n_records']:6d} "
              f"{v['pooled']['lam0_days']:7.2f} {h['fast_val_nll']:8.5f} "
              f"{h['official']['nll']:9.5f} "
              f"{h['two_fold']['mean_heldout_nll']:8.5f} "
              f"{p['fast_val_nll']:8.5f} "
              f"{p['two_fold']['mean_heldout_nll']:8.5f}")
    print("reference fixed nbr 2fold HiPer",
          round(ref_fixed["hiper"]["mean_heldout_nll"], 5),
          "PF+r", round(ref_fixed["pf_nbr"]["mean_heldout_nll"], 5))
    for name, v in pf_pooled.items():
        print("PF pooled", name, round(v["lifetime_days"], 2),
              round(v["fast_val_nll"], 5),
              round(v["two_fold"]["mean_heldout_nll"], 5))
    print("PF val-tuned", pf_ref["params"], round(pf_ref["fast_val_nll"], 5),
          round(pf_ref["two_fold"]["mean_heldout_nll"], 5))
    print("total", out["seconds"], "s")


def cmd_diag():
    """DIAGNOSTIC of the configs selected by `run` (val labels and the
    event gt_id field are used for SCORING breakdowns only): NLL by identity
    kept / switched and by evidence x label, agreement of the retrieved
    rates with the reference prior, scenes won."""
    res = json.load(open(OUT / "results.json"))
    streams = ec.load("streams")
    A = ValArrays(streams)
    SW = np.zeros(A.EV.shape, bool)          # row: identity switched
    obj_sw = np.zeros(A.EV.shape[0], bool)   # object: ever switched
    for n, (sc, oid) in enumerate(A.keys):
        ob = streams[sc]["objects"][oid]
        g0 = ob["events"][0]["gt_id"]
        for ev in ob["events"]:
            SW[n, ev["k"]] = ev["gt_id"] != g0
        obj_sw[n] = SW[n].any()
    v = res["variants"]
    names = [n for n in v if "hiper" in v[n]]
    best = min(names, key=lambda n: v[n]["hiper"]["fast_val_nll"])
    pick = list(dict.fromkeys(["gt@train", "gt@train12", "ckey@train",
                               "map@train", "map@train12", best]))
    inv = {c: e for e, c in CODE.items()}
    sw_row, ev_row = SW[A.mask], A.EV[A.mask]
    out, base = {"best_variant": best}, None
    for name in pick:
        spec = (v[name]["source"], v[name]["size"], v[name]["omega"])
        recs, rw = records_for(*spec)
        fast = FastNbr(recs, rw)
        fast.prepare(A.Q)
        p = v[name]["hiper"]["params"]
        lam = np.where(A.qok, fast.rates(p["K"], p["tau"], p["k_nbr"]),
                       fast.lam0)
        lam_ref_nbr = np.where(A.qok, fast.rates(
            REF_NBR["K"], REF_NBR["tau"], REF_NBR["k_nbr"]), fast.lam0)
        m = A.m_from_rates(lam, fast.lam0)
        P = run_filter(A, m, p["p_miss"], p["p_false"], kappa=p["kappa0"])
        q = np.clip(np.clip(P[A.mask], 0, 1), 1e-6, 1 - 1e-6)
        rn = -(A.y * np.log(q) + (1 - A.y) * np.log1p(-q))
        cats = {}
        for lab, sel in (("identity kept", ~sw_row),
                         ("identity switched", sw_row)):
            cats[lab] = {"n": int(sel.sum()),
                         "sum_nll_per_row": float(rn[sel].sum() / rn.size),
                         "mean_nll": float(rn[sel].mean())}
        for c in sorted(set(ev_row.tolist())):
            for y in (0.0, 1.0):
                sel = (ev_row == c) & (A.y == y)
                if sel.any():
                    cats[f"{inv[c]}, y={int(y)}"] = {
                        "n": int(sel.sum()),
                        "sum_nll_per_row": float(rn[sel].sum() / rn.size),
                        "mean_nll": float(rn[sel].mean())}
        d = {"params": p, "val_nll": float(A.scene_nll(P).mean()),
             "pooled_row_nll": float(rn.mean()),
             "mean_log10_rate_per_day": float(np.log10(lam * DAY).mean()),
             "mean_m": float(m.mean()), "categories": cats,
             "scene_nll": A.scene_nll(P).tolist(),
             "_lam": lam, "_lam_ref_nbr": lam_ref_nbr}
        if base is None:
            base = d
        else:
            ln, lb = np.log(lam), np.log(base["_lam"])
            lr, lbr = np.log(lam_ref_nbr), np.log(base["_lam_ref_nbr"])
            d["vs_gt@train"] = {
                "pearson_log_rate_selected": float(np.corrcoef(ln, lb)[0, 1]),
                "pearson_log_rate_ref_nbr": float(np.corrcoef(lr, lbr)[0, 1]),
                "mean_log_rate_ratio_switched_obj": float(
                    (ln - lb)[obj_sw].mean()),
                "mean_log_rate_ratio_kept_obj": float(
                    (ln - lb)[~obj_sw].mean()),
                "scenes_better": int(np.sum(np.array(d["scene_nll"])
                                            < np.array(base["scene_nll"]))),
                # paired bootstrap over val scenes (variant - gt@train);
                # optimistic: both configs were selected on these scenes
                "paired_bootstrap_scene_nll": ec.pm.paired_bootstrap(
                    dict(zip(A.scenes, d["scene_nll"])),
                    dict(zip(A.scenes, base["scene_nll"]))),
                "category_delta_sum_nll_per_row": {
                    c: d["categories"][c]["sum_nll_per_row"]
                    - base["categories"][c]["sum_nll_per_row"]
                    for c in d["categories"] if c in base["categories"]}}
        out[name] = d
        print(name, "val NLL", round(d["val_nll"], 5),
              {c: (x["n"], round(x["sum_nll_per_row"], 5))
               for c, x in cats.items()}, flush=True)
        if "vs_gt@train" in d:
            print("   vs gt@train", json.dumps(d["vs_gt@train"]), flush=True)
    for d in out.values():
        if isinstance(d, dict):
            d.pop("_lam", None)
            d.pop("_lam_ref_nbr", None)
    out["n_objects_switched"] = int(obj_sw.sum())
    out["n_rows_switched"] = int(sw_row.sum())
    # Fair pairs (same training data, or same source): per-scene NLL of
    # each side at its own full-val selected config (scene_nll_grids.npz)
    Z = np.load(OUT / "scene_nll_grids.npz")
    scenes = [str(x) for x in Z["scenes"]]

    def sel(name, fam="hpf"):
        s_, z_, o_ = v[name]["source"], v[name]["size"], v[name]["omega"]
        arr = Z[f"{fam}__{s_}__{z_}__{o_}"]
        idx, _ = best_index(arr)
        return dict(zip(scenes, arr[idx].tolist()))

    pairs = [("map@train", "gt@train"), ("map@train12", "gt@train12"),
             ("ckey@train", "gt@train"), ("ckey@train12", "gt@train12"),
             ("map@train", "ckey@train"), ("map@train12", "ckey@train12"),
             ("mix@train@w1", "gt@train"), ("mix@train12@w1", "gt@train12"),
             ("gt@train12", "gt@train"), ("map@train12", "map@train")]
    out["pairs"] = {}
    for a_, b_ in pairs:
        row = {}
        for fam, key in (("hpf", "hiper"), ("pfn", "pf_nbr")):
            sa, sb = sel(a_, fam), sel(b_, fam)
            row[key] = {
                "val_nll_a": v[a_][key]["fast_val_nll"],
                "val_nll_b": v[b_][key]["fast_val_nll"],
                "twofold_a": v[a_][key]["two_fold"]["mean_heldout_nll"],
                "twofold_b": v[b_][key]["two_fold"]["mean_heldout_nll"],
                "scenes_a_better": int(sum(sa[x] < sb[x] for x in scenes)),
                "paired_bootstrap": ec.pm.paired_bootstrap(sa, sb)}
        out["pairs"][f"{a_} - {b_}"] = row
        h = row["hiper"]
        print(f"{a_:16s} - {b_:12s} HiPer val {h['val_nll_a'] - h['val_nll_b']:+.5f}"
              f" 2fold {h['twofold_a'] - h['twofold_b']:+.5f} wins "
              f"{h['scenes_a_better']}/20 CI [{h['paired_bootstrap']['ci_lo']:+.4f},"
              f" {h['paired_bootstrap']['ci_hi']:+.4f}] | PF+r val "
              f"{row['pf_nbr']['val_nll_a'] - row['pf_nbr']['val_nll_b']:+.5f}"
              f" 2fold {row['pf_nbr']['twofold_a'] - row['pf_nbr']['twofold_b']:+.5f}",
              flush=True)
    json.dump(out, open(OUT / "diag.json", "w"), indent=1)


def cmd_ablate(workers, sources=("gtckey",),
               out_name="ablation_gtckey.json"):
    """POST-HOC ablations (declared after `run`, not used to pick the best
    variant). gtckey: union of GT-object records and creation-keyed GT
    histories = mix@w1 without the map label history. gkmap: GT-mean key
    with the map label history (ablate_gkmap). Same grids and 2-fold
    protocol as `run`; each is also compared per scene with gt@<size> at
    its own full-val selected config (needs scene_nll_grids.npz of `run`)."""
    t_start = time.time()
    streams = ec.load("streams")
    A = ValArrays(streams)
    _W.update(A=A, streams=streams)
    nS = len(A.scenes)
    specs = [(s, z, 1.0) for s in sources for z in SIZES]
    with Pool(min(workers, len(specs))) as pool:
        res = list(pool.imap_unordered(variant_job, specs))
    res.sort(key=lambda r: specs.index(r["spec"]))
    Z = np.load(OUT / "scene_nll_grids.npz")
    if [str(x) for x in Z["scenes"]] != list(A.scenes):
        raise SystemExit("scene order of scene_nll_grids.npz differs")
    out, jobs = {}, []
    for r in res:
        s, z, o = r["spec"]
        (i, j, l), nll_h = best_index(r["hpf"])
        (pi_, pj), nll_p = best_index(r["pfn"])
        out[f"{s}@{z}"] = {
            "source": s, "size": z, "omega": o, "pooled": r["pooled"],
            "hiper": {"fast_val_nll": nll_h, "params": hpf_params(i, j, l),
                      "two_fold": two_fold(r["hpf"], nS)},
            "pf_nbr": {"fast_val_nll": nll_p, "params": pfn_params(pi_, pj),
                       "two_fold": two_fold(r["pfn"], nS)}}
        # per scene vs gt@<size>, each at its own full-val selected config
        # (optimistic for both: selected on these scenes)
        for fam, key in (("hpf", "hiper"), ("pfn", "pf_nbr")):
            mine = r[fam][best_index(r[fam])[0]]
            g = Z[f"{fam}__gt__{z}__1.0"]
            ref = g[best_index(g)[0]]
            out[f"{s}@{z}"][key]["vs_gt_same_size"] = {
                "val_nll_gt": float(ref.mean()),
                "scenes_better_than_gt": int(np.sum(mine < ref)),
                "paired_bootstrap": ec.pm.paired_bootstrap(
                    dict(zip(A.scenes, mine.tolist())),
                    dict(zip(A.scenes, ref.tolist()))),
                "scene_nll": mine.tolist(), "gt_scene_nll": ref.tolist()}
        jobs.append((r["spec"], "hpf_nbr", hpf_params(i, j, l)))
        jobs.append((r["spec"], "pf_nbr", pfn_params(pi_, pj)))
    with Pool(min(workers, len(jobs))) as pool:
        offs = list(pool.imap_unordered(official_job, jobs))
    for o in offs:
        s, z, _ = o["spec"]
        key = "hiper" if o["family"] == "hpf_nbr" else "pf_nbr"
        out[f"{s}@{z}"][key]["official"] = {
            k: o[k] for k in ("nll", "brier", "ece", "auroc")}
    out["seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(OUT / out_name, "w"), indent=1,
              default=float)
    for name, v in out.items():
        if name == "seconds":
            continue
        h, p = v["hiper"], v["pf_nbr"]
        print(f"{name:20s} n={v['pooled']['n_records']} HiPer "
              f"{h['fast_val_nll']:.5f} (official {h['official']['nll']:.5f})"
              f" 2fold {h['two_fold']['mean_heldout_nll']:.5f} | PF+r "
              f"{p['fast_val_nll']:.5f} 2fold "
              f"{p['two_fold']['mean_heldout_nll']:.5f}", flush=True)
        for key in ("hiper", "pf_nbr"):
            c = v[key]["vs_gt_same_size"]
            b = c["paired_bootstrap"]
            print(f"   {key:6s} - gt@{v['size']}: "
                  f"{v[key]['fast_val_nll'] - c['val_nll_gt']:+.5f} wins "
                  f"{c['scenes_better_than_gt']}/{nS} CI "
                  f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] p {b['p_value']:.3f}",
                  flush=True)
    print("total", out["seconds"], "s")


def cmd_summary():
    """Selection-aware summary (added 2026-10-20 23:25, after `run`; no new
    variant). From the saved grids of `run`: nested 2-fold in which the
    idea's variant (map, or mix with its omega) is chosen INSIDE each fold
    together with its other params, per training size; the same over every
    declared source incl. gt (the whole val-selection procedure).
    DIAGNOSTIC: PF(lambda) on a finer lifetime grid (the pooled train rates,
    ~22 d, fall between te.LIFETIME_GRID_DAYS 16 and 32). summary.json."""
    Z = np.load(OUT / "scene_nll_grids.npz")
    nS = len(Z["scenes"])
    idea = [("map", 1.0)] + [("mix", o) for o in OMEGA_GRID]
    every = [("gt", 1.0), ("ckey", 1.0)] + idea
    out = {}
    for z in SIZES:
        for tag, lst in (("idea", idea), ("all_declared", every)):
            row = {}
            for fam, key in (("hpf", "hiper"), ("pfn", "pf_nbr")):
                stack = np.stack([Z[f"{fam}__{s}__{z}__{o}"] for s, o in lst])
                tf = two_fold(stack, nS)
                for f in ("A->B", "B->A"):
                    s_, o_ = lst[tf[f]["index"][0]]
                    tf[f]["variant"] = f"{s_}@w{o_:g}"
                idx, nll = best_index(stack)
                s_, o_ = lst[idx[0]]
                row[key] = {"full_val_variant": f"{s_}@w{o_:g}",
                            "full_val_nll": nll, "two_fold": tf}
                # per scene vs gt@<size>, both at their full-val selected
                # config (optimistic for both: selected on these scenes)
                g = Z[f"{fam}__gt__{z}__1.0"]
                ref, mine = g[best_index(g)[0]], stack[idx]
                row[key]["vs_gt_same_size"] = {
                    "val_nll_gt": float(ref.mean()),
                    "scenes_better_than_gt": int(np.sum(mine < ref)),
                    "paired_bootstrap": ec.pm.paired_bootstrap(
                        dict(zip(map(str, Z["scenes"]), mine.tolist())),
                        dict(zip(map(str, Z["scenes"]), ref.tolist())))}
            out[f"{tag}@{z}"] = row
            print(f"{tag + '@' + z:22s}", {
                k: (v["full_val_variant"], round(v["full_val_nll"], 5),
                    round(v["two_fold"]["mean_heldout_nll"], 5),
                    v["two_fold"]["A->B"]["variant"],
                    v["two_fold"]["B->A"]["variant"])
                for k, v in row.items()}, flush=True)
            for k, v in row.items():
                c = v["vs_gt_same_size"]
                b = c["paired_bootstrap"]
                print(f"   {k:6s} - gt@{z}: {b['diff']:+.5f} wins "
                      f"{c['scenes_better_than_gt']}/{nS} CI "
                      f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] "
                      f"p {b['p_value']:.3f}", flush=True)
    A = ValArrays(ec.load("streams"))
    fine = [12, 16, 18, 20, 22, 24, 26, 28, 32, 40]
    arr = np.stack([[A.scene_nll(run_filter(
        A, np.full(len(A.Q), min(max(math.exp(-1.0 / L), M_CLIP),
                                 1 - M_CLIP)),
        d["p_miss"], d["p_false"], learn=False)) for d in DET]
        for L in fine])
    per = {}
    for li, L in enumerate(fine):
        (j,), nll = best_index(arr[li])
        per[str(L)] = {"val_nll": nll, **DET[j]}
    (li, dj), nll = best_index(arr)
    out["diag_pf_lifetime_fine"] = {
        "lifetimes_days": fine, "per_lifetime_det_best": per,
        "best": {"lifetime_days": fine[li], **DET[dj], "val_nll": nll},
        "two_fold": two_fold(arr, nS)}
    print("PF(lambda) fine lifetime grid:",
          {L: round(v["val_nll"], 5) for L, v in per.items()},
          "2fold", round(out["diag_pf_lifetime_fine"]["two_fold"]
                         ["mean_heldout_nll"], 5), flush=True)
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "diag", "ablate",
                                    "ablate_gkmap", "summary"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.workers > 3:
        raise SystemExit("at most 3 worker processes")
    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "diag":
        cmd_diag()
    elif args.cmd == "ablate":
        cmd_ablate(args.workers)
    elif args.cmd == "ablate_gkmap":
        cmd_ablate(args.workers, ("gkmap",), "ablation_gkmap.json")
    elif args.cmd == "summary":
        cmd_summary()
    else:
        cmd_run(args.workers)


if __name__ == "__main__":
    main()
