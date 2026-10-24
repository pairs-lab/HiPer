#!/usr/bin/env python3
"""
scripts/explore_w5_retrieval.py
===============================
VALIDATION-ONLY exploration w5 (workflow 'widen-margin'): a richer retrieval
for HiPer's appearance-retrieved survival prior. Never reads test / test2 /
test3 / obchange data or results.

Reference (run v2, priors.NeighbourSurvivalPrior): the K nearest TRAIN objects
(20 scenes) by DINOv2 cosine similarity s, kernel w = exp((s - s_max) / tau),
weighted censored-exponential MAP rate with Gamma shrinkage k to the pooled
rate. Val NLL 0.29091 (HiPer), 0.29829 (PF + retrieved), 0.35914 (PF(lambda)).

Declared variants (fixed before any result was seen)
  (a) kernel  rel   w = exp((s - s_max) / tau)                   [reference]
              abs   w = exp((s - 1) / tau)  (far neighbours carry little
                    evidence; the Gamma shrinkage dominates for OOD objects)
  (b) emb     dino [reference] | clip | cat75 | cat50 | ens
              catA : distance alpha (1 - s_dino) + (1 - alpha) r_c (1 - s_clip)
                     (the concatenation of the two re-weighted unit vectors)
              ens  : mean of the dino and the clip prior means m (same K/tau/k)
  (c) geo     w_g in {0 [reference], 0.1, 0.3, 1}: distance
              d = (d_app + w_g r_g d_geo) / (1 + w_g); d_geo = mean squared
              difference of standardised [log sorted extent (3, floor 1 cm),
              height, log pixel_area] / 2
  (d) keys    train (20 scenes) [reference] | train+train2 (60 scenes)
  r_c, r_g put the CLIP / geometry distances on the DINOv2 scale (ratio of the
  median pairwise distances among the KEY records, no val data); s = 1 - d.
  NBR grid of every variant: K {10,30,100,300,1000} x tau {.02,.05,.1,.2,.5}
  x k {0.25,1,4}; te.NBR_GRID (run v2) is its subset K<=300, tau<=0.1, k>=1.
  Filter grid as run v2: te.DET_GRID x te.KAPPA_FINAL (HiPer, variance gate
  off) and te.DET_GRID (PF + retrieved). Criterion: mean-over-scenes val NLL.
  PF(lambda) has no retrieved prior, so the idea does not apply to it.

Robustness: 2-fold over sorted val scenes (A = even index, B = odd index),
params (and, in the 'pipeline' row, the variant too) selected on one fold,
scored on the other. Independent check: the 40 (kernel, emb, geo) structures
with their val-selected params (keys = train) re-scored on the TRAIN streams
with keys = train2 records (scenes never used for selection; train labels
are fair to use).

Speed: a vectorised replay (all objects x all filter configs of one prior at
once) replaces explore_common.run_model inside the grid. It is checked
against the official replay (per-scene NLL, every prediction) and every
reported number is re-scored with explore_common.run_model.

  python scripts/explore_w5_retrieval.py            # full run, 3 workers
  python scripts/explore_w5_retrieval.py --smoke    # bug finding only
  python scripts/explore_w5_retrieval.py --edge     # post-hoc grid-edge
        extension (after the full run; see NBR_EDGE): k_nbr down to 1/16,
        w_g up to 10, geometry drop-one; applied to the reference as well
  python scripts/explore_w5_retrieval.py --diag     # scoring-only loss
        breakdown of the final pick vs the frozen reference
  python scripts/explore_w5_retrieval.py --check2   # independent check 2
        (TRAIN2 streams, keys = train; keys factor via train + other half
        of train2) + check 1b; added after the edge run, no selection
  python scripts/explore_w5_retrieval.py --geodiag  # geometry drop-one with
        the final params fixed, on val / check 1 / check 2 (scoring only)
  python scripts/explore_w5_retrieval.py --verify   # I1: current code
        reproduces grid.npz where the edge grid overlaps it
Outputs: RESULTS/explore_cache/w5_retrieval/{results.json, grid.npz,
  edge_results.json, edge_grid.npz, diag_breakdown.json,
  check2_results.json, geodiag_results.json, verify_results.json, *.log}
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

from hiper.evaluation.lifelong_objects import replay  # noqa: E402

OUT = ec.CACHE / "w5_retrieval"
DAY = 86400.0
M_CLIP = 1e-4
NLL_EPS = 1e-6
CODE = {"create": 1, "hit": 2, "miss": 3, "none": 4, "relocate": 5}

# ── Declared grids ───────────────────────────────────────────────────────────
KERNELS = ("rel", "abs")
EMBS = {"dino": 1.0, "cat75": 0.75, "cat50": 0.5, "clip": 0.0, "ens": None}
GEO_W = (0.0, 0.1, 0.3, 1.0)
KEYSETS = {"train": ("train",), "train+train2": ("train", "train2")}
NBR_EXT = dict(K=[10, 30, 100, 300, 1000], tau=[0.02, 0.05, 0.1, 0.2, 0.5],
               k_nbr=[0.25, 1.0, 4.0])
REF = ("rel", "dino", 0.0, "train")
REF_NBR = {"K": 300, "tau": 0.05, "k_nbr": 1.0}      # run v2 selection
NBR = te.grid(NBR_EXT)
NBR_V2 = [j for j, g in enumerate(NBR) if g["K"] in te.NBR_GRID["K"]
          and g["tau"] in te.NBR_GRID["tau"]
          and g["k_nbr"] in te.NBR_GRID["k_nbr"]]
HPF_CFG = [{**d, "kappa0": float(k)} for d in te.grid(te.DET_GRID)
           for k in te.KAPPA_FINAL]
PF_CFG = te.grid(te.DET_GRID)


def vname(spec) -> str:
    kernel, emb, wg, keys = spec
    return f"{kernel}|{emb}|g{wg:g}|{keys}"


def all_specs():
    return [(kn, e, g, ks) for kn in KERNELS for e in EMBS for g in GEO_W
            for ks in KEYSETS]


# ── Retrieval prior (vectorised NeighbourSurvivalPrior.rate) ─────────────────

def geo_raw(extent, height, pix) -> np.ndarray:
    e = np.sort(np.maximum(np.asarray(extent, np.float64), 0.01))[::-1]
    return np.concatenate([np.log(e),
                           [float(height), math.log(max(float(pix), 1e-5))]])


def _geo_d(A, B) -> np.ndarray:
    sq = ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :]
          - 2.0 * A @ B.T)
    return np.maximum(sq, 0.0) / (2.0 * A.shape[1])


class Keys:
    """Training objects (records_<split>.pkl) used as retrieval keys."""

    def __init__(self, recs):
        self.n = len(recs)
        self.dino = np.stack([r["emb"] for r in recs]).astype(np.float64)
        self.clip = np.stack([r["clip"] for r in recs]).astype(np.float64)
        self.S = np.array([r["S"] for r in recs], np.float64)
        allg = np.concatenate([np.asarray(r["gaps"], np.float64)
                               for r in recs])
        self.dvals = np.unique(allg)
        self.C = np.zeros((self.n, len(self.dvals)))
        for i, r in enumerate(recs):
            for g in r["gaps"]:
                self.C[i, np.searchsorted(self.dvals, g)] += 1.0
        G = np.stack([geo_raw(r["extent"], r["height"], r["pixel_area"])
                      for r in recs])
        self.g_mu, self.g_sd = G.mean(0), G.std(0)
        self.G = (G - self.g_mu) / self.g_sd
        # pooled rate exactly as the reference prior computes it
        self.lam0 = te.NeighbourSurvivalPrior(
            [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
             for r in recs]).lam0
        iu = np.triu_indices(self.n, 1)
        dd = np.median(1.0 - (self.dino @ self.dino.T)[iu])
        dc = np.median(1.0 - (self.clip @ self.clip.T)[iu])
        dg = np.median(_geo_d(self.G, self.G)[iu])
        self.r_c, self.r_g = float(dd / dc), float(dd / dg)
        self.med = {"dino": float(dd), "clip": float(dc), "geo": float(dg)}

    def geo_ratio(self, dims) -> float:
        """r_g of a subset of the geometry features (edge mode)."""
        key = tuple(dims)
        if not hasattr(self, "_rg"):
            self._rg = {}
        if key not in self._rg:
            G = self.G[:, list(dims)]
            iu = np.triu_indices(self.n, 1)
            self._rg[key] = float(self.med["dino"]
                                  / np.median(_geo_d(G, G)[iu]))
        return self._rg[key]

    def distance(self, Q, alpha: float, wg: float, dims=None) -> np.ndarray:
        """(n_query, n_keys) distance d = 1 - s of one embedding variant;
        dims = subset of the geometry features (None = all five)."""
        if alpha == 1.0:
            d = 1.0 - Q["dino"] @ self.dino.T
        elif alpha == 0.0:
            d = self.r_c * (1.0 - Q["clip"] @ self.clip.T)
        else:
            d = (alpha * (1.0 - Q["dino"] @ self.dino.T)
                 + (1.0 - alpha) * self.r_c * (1.0 - Q["clip"] @ self.clip.T))
        if wg > 0.0:
            gq = (Q["geo"] - self.g_mu) / self.g_sd
            if dims is None:
                d = (d + wg * self.r_g * _geo_d(gq, self.G)) / (1.0 + wg)
            else:
                dl = list(dims)
                d = (d + wg * self.geo_ratio(dl)
                     * _geo_d(gq[:, dl], self.G[:, dl])) / (1.0 + wg)
        return d


def queries(feats, order) -> dict:
    """Creation-track features of the map objects, in evaluator order."""
    return {"dino": np.stack([feats[k]["dino"] for k in order]),
            "clip": np.stack([feats[k]["clip"] for k in order]),
            "geo": np.stack([geo_raw(feats[k]["extent"], feats[k]["height"],
                                     feats[k]["pixel_area"])
                             for k in order])}


def map_rate_vec(Sw, nu, dvals, k, lam0) -> np.ndarray:
    """priors.weighted_map_rate for many objects at once; the vanish gaps are
    grouped by their (few) distinct values: nu[o, u] = summed weight of the
    gaps equal to dvals[u]."""
    n = nu.sum(1)
    beta = k / lam0
    lam = np.full(len(Sw), lam0)
    zero = n == 0.0
    lam[zero] = k / (beta + Sw[zero])
    nz = ~zero
    if nz.any():
        S_, nu_, n_ = Sw[nz], nu[nz], n[nz]
        lo = (n_ + k) / (beta + S_ + nu_ @ dvals)
        hi = (n_ + k) / (beta + S_)
        for _ in range(80):                  # bisection in log(lambda)
            mid = np.sqrt(lo * hi)
            x = mid[:, None] * dvals[None, :]
            sc = (k / mid - beta - S_
                  + (nu_ * dvals * np.exp(-x) / -np.expm1(-x)).sum(1))
            pos = sc > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        lam[nz] = np.sqrt(lo * hi)
    return lam


class Retrieval:
    """Prior survival means m(K, tau, k) of every query for one variant."""

    def __init__(self, keys: Keys, Q: dict, kernel: str, emb: str,
                 wg: float, dims=None):
        self.keys, self.kernel = keys, kernel
        alphas = [1.0, 0.0] if emb == "ens" else [EMBS[emb]]
        self.D = [keys.distance(Q, a, wg, dims) for a in alphas]
        self._nb = {}

    def _neigh(self, i, K):
        hit = self._nb.get((i, K))
        if hit is None:
            D = self.D[i]
            K = min(K, D.shape[1])
            idx = np.argpartition(D, K - 1, axis=1)[:, :K]
            hit = (np.take_along_axis(D, idx, 1), self.keys.S[idx],
                   self.keys.C[idx])
            self._nb = {kk: v for kk, v in self._nb.items() if kk[1] == K}
            self._nb[(i, K)] = hit
        return hit

    def means(self, K, tau, k) -> np.ndarray:
        ms = []
        for i in range(len(self.D)):
            Dk, Sk, Ck = self._neigh(i, K)
            if self.kernel == "rel":
                w = np.exp(-(Dk - Dk.min(1, keepdims=True)) / tau)
            else:
                w = np.exp(-Dk / tau)
            lam = map_rate_vec(np.einsum("ok,ok->o", w, Sk),
                               np.einsum("ok,oku->ou", w, Ck),
                               self.keys.dvals, k, self.keys.lam0)
            ms.append(np.clip(np.exp(-lam * DAY), M_CLIP, 1.0 - M_CLIP))
        m = ms[0] if len(ms) == 1 else np.mean(ms, 0)
        return np.clip(m, M_CLIP, 1.0 - M_CLIP)


# ── Vectorised replay ────────────────────────────────────────────────────────

def logpoch(x, r: float):
    """ln Gamma(x + r) - ln Gamma(x) for an array x and a scalar r > 0
    (exact product for the integer part of r)."""
    n = int(math.floor(r + 1e-9))
    frac = r - n
    if frac > 1e-9:
        out = gammaln(x + frac) - gammaln(x)
        y = x + frac
    else:
        out, y = 0.0, x
    if n > 0:
        p = y.copy()
        for i in range(1, n):
            p *= y + i
        out = out + np.log(p)
    return out


class FastEval:
    """Replay of explore_common.run_model for HierarchicalPersistenceFilter
    (rho = 0, pi_new = 1, no environment level, dt_ref = 1 day) for many
    filter configs at once: per-scene NLL on the rows with a label. Labels
    come from one official replay (scoring only)."""

    def __init__(self, streams, labels):
        self.scenes = sorted(streams)
        nk = len(streams[self.scenes[0]]["times"])
        order, ev, sidx = [], [], []
        for si, sc in enumerate(self.scenes):
            st = streams[sc]
            assert len(st["times"]) == nk
            for oid, ob in st["objects"].items():
                row = np.zeros(nk, np.int8)
                for e in ob["events"]:
                    assert row[e["k"]] == 0
                    row[e["k"]] = CODE[e["ev"]]
                    if e["ev"] in ("hit", "create"):
                        assert float(e["eps_l"]) == 0.0
                order.append((sc, oid))
                ev.append(row)
                sidx.append(si)
        self.order, self.ev = order, np.array(ev)
        self.sidx = np.array(sidx)
        self.O, self.K, self.S = len(order), nk, len(self.scenes)
        T = np.array([streams[sc]["times"] for sc in self.scenes],
                     np.float64)[self.sidx]
        y = np.full((self.O, nk), np.nan)
        for o, (sc, oid) in enumerate(order):
            for k in range(nk):
                v = labels.get((sc, oid, k))
                if v is not None:
                    y[o, k] = float(v)
        exist, lab = self.ev > 0, np.isfinite(y)
        assert np.array_equal(lab[:, :-1], exist[:, :-1])
        assert not lab[:, -1].any()
        self.y = y
        t_last = np.full(self.O, np.nan)
        r_ev = np.zeros((self.O, nk))
        r_pred = np.full((self.O, nk), np.nan)
        for k in range(nk):
            e = self.ev[:, k]
            hm = (e == 2) | (e == 3)
            r_ev[hm, k] = np.maximum(0.0, (T[hm, k] - t_last[hm]) / DAY)
            upd = (e == 1) | hm | (e == 5)
            t_last[upd] = T[upd, k]
            if k + 1 < nk:
                ex = e > 0
                r_pred[ex, k] = np.maximum(
                    0.0, (T[ex, k + 1] - t_last[ex]) / DAY)
        for k in range(1, nk):     # the update at k reuses S of prediction k-1
            hm = (self.ev[:, k] == 2) | (self.ev[:, k] == 3)
            assert np.array_equal(r_ev[hm, k], r_pred[hm, k - 1])
        self.r_pred = r_pred
        self.groups = []
        for k in range(nk - 1):
            ex = np.flatnonzero(self.ev[:, k] > 0)
            rs = r_pred[ex, k]
            self.groups.append([(float(r), ex[rs == r]) for r in np.unique(rs)])
        self.onehot = np.zeros((self.O, self.S))
        self.onehot[np.arange(self.O), self.sidx] = 1.0
        self.n_lab = lab.sum(1) @ self.onehot
        self.i_cr = [np.flatnonzero(self.ev[:, k] == 1) for k in range(nk)]
        self.i_hm = [np.flatnonzero((self.ev[:, k] == 2)
                                    | (self.ev[:, k] == 3))
                     for k in range(nk)]
        self.hit = [self.ev[self.i_hm[k], k] == 2 for k in range(nk)]
        self.i_rl = [np.flatnonzero(self.ev[:, k] == 5) for k in range(nk)]
        self.i_lab = [np.flatnonzero(lab[:, k]) for k in range(nk)]

    def run(self, m, p_miss, p_false, kappa=None, return_pred=False):
        """Per-scene NLL (C, S) of C filter configs; the rate is learned iff
        kappa is given (HiPer), else fixed at m (PF). m: (O,) prior means."""
        # layout (objects, configs): row gathers of the event subsets are
        # contiguous copies
        learn = kappa is not None
        C = len(p_miss)
        pmi = np.asarray(p_miss, np.float64)[None, :]
        pfa = np.asarray(p_false, np.float64)[None, :]
        pi = np.ones((self.O, C))
        Sv = np.ones((self.O, C))
        if learn:
            kap = np.asarray(kappa, np.float64)[None, :]
            a = np.zeros((self.O, C))
            b = np.zeros((self.O, C))
        nll = np.zeros((self.S, C))
        pred = np.full((C, self.O, self.K), np.nan) if return_pred else None
        for k in range(self.K):
            cr = self.i_cr[k]
            if cr.size:
                pi[cr] = 1.0
                if learn:
                    a[cr] = kap * m[cr][:, None]
                    b[cr] = kap * (1.0 - m[cr])[:, None]
            hm = self.i_hm[k]
            if hm.size:
                hit = self.hit[k][:, None]
                L1 = np.where(hit, 1.0 - pmi, pmi)
                L0 = np.where(hit, pfa, 1.0 - pfa)
                p, S = pi[hm], Sv[hm]
                w11 = p * S * L1
                w10 = p * (1.0 - S) * L0
                w00 = (1.0 - p) * L0
                z = w11 + w10 + w00
                w11, w10 = w11 / z, w10 / z
                if learn:
                    a[hm] += self.r_pred[hm, k - 1][:, None] * w11
                    b[hm] += w10
                pi[hm] = w11
            rl = self.i_rl[k]
            if rl.size:
                if learn:
                    b[rl] += pi[rl]
                pi[rl] = 1.0
            if k + 1 >= self.K:
                continue
            for r, g in self.groups[k]:
                if r <= 0.0:
                    Sv[g] = 1.0
                elif learn:
                    ag = a[g]
                    Sv[g] = np.exp(logpoch(ag, r) - logpoch(ag + b[g], r))
                else:
                    Sv[g] = np.power(m[g], r)[:, None]
            lab = self.i_lab[k]
            pk = pi[lab] * Sv[lab]
            if return_pred:
                pred[:, lab, k] = pk.T
            q = np.clip(pk, NLL_EPS, 1.0 - NLL_EPS)
            yk = self.y[lab, k][:, None]
            nll += self.onehot[lab].T @ -(yk * np.log(q)
                                          + (1.0 - yk) * np.log1p(-q))
        out = (nll / self.n_lab[:, None]).T
        return (out, pred) if return_pred else out


HPF_ARR = (np.array([c["p_miss"] for c in HPF_CFG]),
           np.array([c["p_false"] for c in HPF_CFG]),
           np.array([c["kappa0"] for c in HPF_CFG]))
PF_ARR = (np.array([c["p_miss"] for c in PF_CFG]),
          np.array([c["p_false"] for c in PF_CFG]))

# ── Grid job (one variant) ───────────────────────────────────────────────────

DATA = {}


def run_variant(spec):
    t0 = time.time()
    kernel, emb, wg, ks = spec
    ret = Retrieval(DATA["keys"][ks], DATA["Q"], kernel, emb, wg)
    fe = DATA["fe"]
    hpf = np.zeros((len(NBR), len(HPF_CFG), fe.S))
    pfn = np.zeros((len(NBR), len(PF_CFG), fe.S))
    mstat = np.zeros((len(NBR), 3))
    for j, g in enumerate(NBR):
        m = ret.means(g["K"], g["tau"], g["k_nbr"])
        mstat[j] = (m.mean(), m.std(), np.mean(m < 0.5))
        hpf[j] = fe.run(m, *HPF_ARR)
        pfn[j] = fe.run(m, *PF_ARR[:2])
    return spec, hpf, pfn, mstat, time.time() - t0


# ── Official replay (explore_common harness) ─────────────────────────────────

def official_rows(streams, factory, name):
    rows = []
    for sc in sorted(streams):
        rows += replay(streams[sc], factory(sc), name)
    return rows


def official_prior_model(streams_keyed, order, m, cfg, learn, name):
    """explore_common.run_model with a per-object Beta prior (mean m)."""
    kappa = cfg.get("kappa0", 4.0)
    prior = {f"{sc}/{oid}": (float(m[o]), kappa)
             for o, (sc, oid) in enumerate(order)}
    kw = dict(p_miss=cfg["p_miss"], p_false=cfg["p_false"], kappa0=kappa,
              variance_gate=False, use_class_prior=False, learn_rate=learn)
    return ec.run_model(streams_keyed,
                        lambda sc: ec.PerObjectPriorHPF(prior, **kw), name)


def labels_of(rows):
    return {(r["scene"], r["oid"], int(r["k"])): r["label_next"]
            for r in rows}


# ── Analysis helpers ─────────────────────────────────────────────────────────

def best_of(arr, nbr_idx=None, mask=None):
    """(nbr index, cfg index, value) of the min mean NLL over scenes in mask."""
    nbr_idx = list(range(arr.shape[0])) if nbr_idx is None else list(nbr_idx)
    sub = arr[nbr_idx]
    v = sub.mean(2) if mask is None else sub[:, :, mask].mean(2)
    j, c = np.unravel_index(int(np.argmin(v)), v.shape)
    return nbr_idx[j], int(c), float(v[j, c])


def params_of(fam, j, c):
    cfg = HPF_CFG[c] if fam == "hpf" else PF_CFG[c]
    return {**NBR[j], **cfg}


def cv2(arr, A, B, nbr_idx=None):
    ja, ca, _ = best_of(arr, nbr_idx, A)
    jb, cb, _ = best_of(arr, nbr_idx, B)
    on_b = float(arr[ja, ca, B].mean())
    on_a = float(arr[jb, cb, A].mean())
    return {"A_to_B": on_b, "B_to_A": on_a, "mean_heldout": 0.5 * (on_a + on_b),
            "params_A": (ja, ca), "params_B": (jb, cb)}


def cv2_pipeline(res, fam, A, B, specs):
    """Variant AND params selected on one fold, scored on the other."""
    out = {}
    for sel, ev_ in (("A", B), ("B", A)):
        mask = A if sel == "A" else B
        best = None
        for s in specs:
            arr = res[vname(s)][fam]
            j, c, v = best_of(arr, None, mask)
            if best is None or v < best[0]:
                best = (v, s, j, c, float(arr[j, c, ev_].mean()))
        out[sel] = {"variant": vname(best[1]), "sel_nll": best[0],
                    "heldout_nll": best[4], **params_of(fam, best[2], best[3])}
    out["mean_heldout"] = 0.5 * (out["A"]["heldout_nll"]
                                 + out["B"]["heldout_nll"])
    return out


def jdump(x):
    if isinstance(x, dict):
        return {str(k): jdump(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jdump(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return jdump(x.tolist())
    return x


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global NBR, HPF_CFG, HPF_ARR
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    meta = ec.meta()
    streams = ec.load("streams", "val")
    feats = ec.load("feats", "val")
    recs = {sp: ec.load("records", sp) for sp in ("train", "train2")}
    keyed = ec.keyed(streams)

    # 1. Official reference (I1) + labels (scoring only)
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in recs["train"]])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    off = {}
    rows_ref = official_rows(
        streams, lambda sc: te.make_model("hpf_nbr", meta["ours_params"]),
        "ours")
    off["ours_frozen"] = ec.scores(rows_ref)
    off["pf_frozen"] = ec.run_model(
        streams, lambda sc: te.make_model("pf", meta["pf_params"]), "pf")
    off["pf_nbr_frozen"] = ec.run_model(
        streams, lambda sc: te.make_model("pf_nbr", meta["pf_nbr_params"]),
        "pf_nbr")
    say("official frozen:", {k: round(v["nll"], 5) for k, v in off.items()})
    fe = FastEval(streams, labels_of(rows_ref))
    Q = queries(feats, fe.order)
    keys = {ks: Keys([r for sp in sps for r in recs[sp]])
            for ks, sps in KEYSETS.items()}
    for ks, kk in keys.items():
        say(f"keys {ks}: n={kk.n} lam0={1 / kk.lam0 / DAY:.3f} d "
            f"r_c={kk.r_c:.3f} r_g={kk.r_g:.3f} medians={kk.med} "
            f"gap values={list(kk.dvals / DAY)}")

    # 2. Fast evaluator == official replay (reference, PF, PF + retrieved)
    chk = {}
    op, pp, pn = (meta["ours_params"], meta["pf_params"],
                  meta["pf_nbr_params"])
    nbr_ref = te.NBR["prior"]
    m_ref_orig = np.array([
        min(max(math.exp(-nbr_ref.rate(streams[sc]["objects"][oid]["events"]
                                       [0]["emb"], K=op["K"], tau=op["tau"],
                                       k=op["k_nbr"]) * DAY), M_CLIP),
            1 - M_CLIP) for sc, oid in fe.order])
    ret_ref = Retrieval(keys["train"], Q, "rel", "dino", 0.0)
    m_ref = ret_ref.means(op["K"], op["tau"], op["k_nbr"])
    chk["prior_mean_max_abs_diff_vs_NeighbourSurvivalPrior"] = float(
        np.abs(m_ref - m_ref_orig).max())
    nll_f, pred_f = fe.run(m_ref, [op["p_miss"]], [op["p_false"]],
                           [op["kappa0"]], return_pred=True)
    pmax = 0.0
    idx = {key: o for o, key in enumerate(fe.order)}
    for r in rows_ref:
        if r["label_next"] is None:
            continue
        o = idx[(r["scene"], r["oid"])]
        pmax = max(pmax, abs(pred_f[0, o, int(r["k"])] - r["pred_presence"]))
    chk["hiper_pred_max_abs_diff"] = pmax
    chk["hiper_nll_fast"] = float(nll_f.mean())
    chk["hiper_nll_official"] = off["ours_frozen"]["nll"]
    m_pf = np.full(fe.O, min(max(math.exp(-1.0 / pp["lifetime_days"]),
                                 M_CLIP), 1 - M_CLIP))
    chk["pf_nll_fast"] = float(fe.run(m_pf, [pp["p_miss"]],
                                      [pp["p_false"]]).mean())
    chk["pf_nll_official"] = off["pf_frozen"]["nll"]
    m_pn = ret_ref.means(pn["K"], pn["tau"], pn["k_nbr"])
    chk["pf_nbr_nll_fast"] = float(fe.run(m_pn, [pn["p_miss"]],
                                          [pn["p_false"]]).mean())
    chk["pf_nbr_nll_official"] = off["pf_nbr_frozen"]["nll"]
    say("fast vs official:", chk)
    assert chk["prior_mean_max_abs_diff_vs_NeighbourSurvivalPrior"] < 1e-9
    assert chk["hiper_pred_max_abs_diff"] < 1e-9
    for f in ("hiper", "pf", "pf_nbr"):
        assert abs(chk[f + "_nll_fast"] - chk[f + "_nll_official"]) < 1e-9, f

    specs = all_specs()
    if args.smoke:
        specs = [REF, ("abs", "cat50", 0.3, "train+train2")]
        NBR[:] = [g for g in NBR if g["K"] in (30, 300)
                  and g["tau"] in (0.05, 0.2) and g["k_nbr"] == 1.0]
        HPF_CFG[:] = HPF_CFG[::8]
        HPF_ARR = (np.array([c["p_miss"] for c in HPF_CFG]),
                   np.array([c["p_false"] for c in HPF_CFG]),
                   np.array([c["kappa0"] for c in HPF_CFG]))
    t0 = time.time()
    m0 = ret_ref.means(300, 0.05, 1.0)
    fe.run(m0, *HPF_ARR)
    say(f"timing: one prior x {len(HPF_CFG)} HiPer configs "
        f"{time.time() - t0:.2f}s; {len(specs)} variants x {len(NBR)} priors")

    # 3. Grid over the declared variants (fork: workers share DATA)
    DATA.update(fe=fe, Q=Q, keys=keys)
    res = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for spec, hpf, pfn, mstat, sec in pool.imap_unordered(run_variant,
                                                              specs):
            res[vname(spec)] = {"hpf": hpf, "pf_nbr": pfn, "mstat": mstat}
            j, c, v = best_of(hpf)
            jp, cp, vp = best_of(pfn)
            say(f"{vname(spec):28s} HiPer {v:.5f} PF+nbr {vp:.5f} "
                f"({sec:.0f}s) {params_of('hpf', j, c)}")
    np.savez_compressed(
        OUT / ("grid_smoke.npz" if args.smoke else "grid.npz"),
        variants=np.array([vname(s) for s in specs]),
        hpf=np.stack([res[vname(s)]["hpf"] for s in specs]).astype(np.float32),
        pf_nbr=np.stack([res[vname(s)]["pf_nbr"] for s in specs]).astype(
            np.float32),
        scenes=np.array(fe.scenes))

    # 4. Analysis
    A = np.arange(fe.S) % 2 == 0
    B = ~A
    j_ref = next(j for j, g in enumerate(NBR) if g == REF_NBR) \
        if REF_NBR in NBR else None
    table = {}
    for s in specs:
        nm = vname(s)
        row = {}
        for fam in ("hpf", "pf_nbr"):
            arr = res[nm][fam]
            j, c, v = best_of(arr)
            row[fam] = {"val_nll": v, "params": params_of(fam, j, c)}
            if not args.smoke:
                j2, c2, v2 = best_of(arr, NBR_V2)
                row[fam]["val_nll_v2grid"] = v2
                row[fam]["params_v2grid"] = params_of(fam, j2, c2)
            row[fam]["cv2"] = cv2(arr, A, B)
            row[fam]["cv2"]["params_A"] = params_of(
                fam, *row[fam]["cv2"]["params_A"])
            row[fam]["cv2"]["params_B"] = params_of(
                fam, *row[fam]["cv2"]["params_B"])
        pb = row["hpf"]["params"]
        jb = NBR.index({k: pb[k] for k in ("K", "tau", "k_nbr")})
        row["prior_mean_stats_at_hpf_best"] = dict(zip(
            ("mean_m", "std_m", "frac_m_below_0.5"), res[nm]["mstat"][jb]))
        table[nm] = row
    ref_nm = vname(REF)
    out = {"declared": {"kernels": KERNELS, "embs": list(EMBS),
                        "geo_w": GEO_W, "keysets": KEYSETS, "nbr": NBR_EXT,
                        "nbr_v2": te.NBR_GRID, "det": te.DET_GRID,
                        "kappa": te.KAPPA_FINAL, "ref": vname(REF),
                        "ref_nbr": REF_NBR, "folds": {
                            "A": [s for s, a in zip(fe.scenes, A) if a],
                            "B": [s for s, b in zip(fe.scenes, B) if b]}},
           "keys": {ks: {"n": kk.n, "lam0_days": 1 / kk.lam0 / DAY,
                         "r_c": kk.r_c, "r_g": kk.r_g, "medians": kk.med}
                    for ks, kk in keys.items()},
           "official_frozen": off, "fast_vs_official": chk,
           "variants": table}
    # reference selected the same way (NBR fixed at the run v2 choice)
    if j_ref is not None:
        rh = res[ref_nm]["hpf"]
        cvr = cv2(rh, A, B, [j_ref])
        cvr["params_A"] = params_of("hpf", *cvr["params_A"])
        cvr["params_B"] = params_of("hpf", *cvr["params_B"])
        out["ref_cv2_nbr_fixed"] = cvr
        out["ref_val_nll_nbr_fixed"] = best_of(rh, [j_ref])[2]
        rp = res[ref_nm]["pf_nbr"]
        cvp = cv2(rp, A, B, [j_ref])
        cvp["params_A"] = params_of("pf_nbr", *cvp["params_A"])
        cvp["params_B"] = params_of("pf_nbr", *cvp["params_B"])
        out["pf_nbr_ref_cv2_nbr_fixed"] = cvp
    for fam in ("hpf", "pf_nbr"):
        out[f"pipeline_cv2_{fam}"] = cv2_pipeline(res, fam, A, B, specs)
    best_nm = min(table, key=lambda n: table[n]["hpf"]["val_nll"])
    best_spec = next(s for s in specs if vname(s) == best_nm)
    out["best_hpf_variant"] = best_nm
    pbest_nm = min(table, key=lambda n: table[n]["pf_nbr"]["val_nll"])
    out["best_pf_nbr_variant"] = pbest_nm

    # one-factor ablation (from the reference) and leave-one-out (from best)
    def swap(spec, i, v):
        s = list(spec)
        s[i] = v
        return tuple(s)

    levels = [KERNELS, tuple(EMBS), GEO_W, tuple(KEYSETS)]
    fnames = ["kernel", "emb", "geo", "keys"]
    abl = []
    for i, lv in enumerate(levels):
        for v in lv:
            s = swap(REF, i, v)
            if vname(s) in table:
                abl.append({"from": "reference", "factor": fnames[i],
                            "level": v, "variant": vname(s),
                            "hpf": table[vname(s)]["hpf"]["val_nll"],
                            "hpf_cv2": table[vname(s)]["hpf"]["cv2"]
                            ["mean_heldout"],
                            "pf_nbr": table[vname(s)]["pf_nbr"]["val_nll"]})
    for i in range(4):
        s = swap(best_spec, i, REF[i])
        if s != best_spec and vname(s) in table:
            abl.append({"from": "best", "factor": fnames[i],
                        "level": f"{best_spec[i]} -> {REF[i]}",
                        "variant": vname(s),
                        "hpf": table[vname(s)]["hpf"]["val_nll"],
                        "hpf_cv2": table[vname(s)]["hpf"]["cv2"]
                        ["mean_heldout"],
                        "pf_nbr": table[vname(s)]["pf_nbr"]["val_nll"]})
    # best level per factor, marginal over the others
    marg = {}
    for i, lv in enumerate(levels):
        marg[fnames[i]] = {str(v): min(table[vname(s)]["hpf"]["val_nll"]
                                       for s in specs if s[i] == v)
                           for v in lv if any(s[i] == v for s in specs)}
    out["ablation"] = abl
    out["marginal_best_per_level"] = marg

    # per-scene NLL and paired bootstrap (scenes) of the best vs reference
    from hiper.evaluation import persistence_metrics as pm

    def per_scene(nm, fam, params):
        cfgs = HPF_CFG if fam == "hpf" else PF_CFG
        j = NBR.index({k: params[k] for k in ("K", "tau", "k_nbr")})
        c = cfgs.index({k: params[k] for k in cfgs[0]})
        return res[nm][fam][j, c]

    ps = {}
    frozen = {"hpf": meta["ours_params"], "pf_nbr": meta["pf_nbr_params"]}
    for fam in ("hpf", "pf_nbr"):
        bn = best_nm if fam == "hpf" else pbest_nm
        cands = {"best": (bn, table[bn][fam]["params"]),
                 "ref_extgrid": (ref_nm, table[ref_nm][fam]["params"])}
        try:
            cands["ref_frozen"] = (ref_nm, frozen[fam])
            per_scene(*cands["ref_frozen"][:1], fam, frozen[fam])
        except ValueError:
            cands.pop("ref_frozen")
        for k_, (nm, p) in cands.items():
            ps[f"{fam}_{k_}"] = dict(zip(fe.scenes,
                                         per_scene(nm, fam, p).tolist()))
        for k_ in ("ref_extgrid", "ref_frozen"):
            if f"{fam}_{k_}" in ps:
                out[f"paired_{fam}_best_vs_{k_}"] = pm.paired_bootstrap(
                    ps[f"{fam}_best"], ps[f"{fam}_{k_}"])
    out["per_scene_nll"] = ps

    # 5. Official re-score of the selected configs (all metrics)
    def official_cfg(nm, fam, params, name):
        s = next(x for x in specs if vname(x) == nm)
        r_ = Retrieval(keys[s[3]], Q, s[0], s[1], s[2])
        m = r_.means(params["K"], params["tau"], params["k_nbr"])
        return official_prior_model(keyed, fe.order, m, params,
                                    fam == "hpf", name)

    sel = {"hiper_best": (best_nm, "hpf", table[best_nm]["hpf"]["params"]),
           "hiper_ref_extgrid": (ref_nm, "hpf",
                                 table[ref_nm]["hpf"]["params"]),
           "pf_nbr_ref_extgrid": (ref_nm, "pf_nbr",
                                  table[ref_nm]["pf_nbr"]["params"]),
           "pf_nbr_best": (pbest_nm, "pf_nbr",
                           table[pbest_nm]["pf_nbr"]["params"]),
           "pf_nbr_on_hiper_best_variant": (
               best_nm, "pf_nbr", table[best_nm]["pf_nbr"]["params"])}
    for name, (nm, fam, params) in sel.items():
        o = official_cfg(nm, fam, params, name)
        off[name] = {**o, "variant": nm, "params": params,
                     "fast_nll": table[nm][fam]["val_nll"]}
        say(f"official {name:30s} {nm:28s} nll {o['nll']:.5f} "
            f"(fast {table[nm][fam]['val_nll']:.5f}) brier {o['brier']:.5f} "
            f"ece {o['ece']:.5f} auroc {o['auroc']:.5f}")
        assert abs(o["nll"] - table[nm][fam]["val_nll"]) < 1e-8, name

    # 6. Independent check on the TRAIN streams (keys = train2 records)
    out["train_check"] = train_check(table, specs, meta, recs, say)

    out["seconds"] = round(time.time() - t_start, 1)
    out["log"] = log
    fn = OUT / ("results_smoke.json" if args.smoke else "results.json")
    with open(fn, "w") as f:
        json.dump(jdump(out), f, indent=1)
    say("wrote", fn, f"({out['seconds']}s)")


def train_check(table, specs, meta, recs, say):
    """Each (kernel, emb, geo) structure with its val-selected params (keys =
    train, 20 scenes) re-scored on the TRAIN streams with keys = the train2
    records (40 other scenes). No selection on these scenes."""
    st = ec.load("streams", "train")
    ft = ec.load("feats", "train")
    rows = official_rows(st, lambda sc: te.make_model("pf",
                                                      meta["pf_params"]),
                         "pf")
    fe = FastEval(st, labels_of(rows))
    Q = queries(ft, fe.order)
    k2 = Keys(recs["train2"])
    out = {"pf_lambda_frozen": float(fe.run(
        np.full(fe.O, math.exp(-1.0 / meta["pf_params"]["lifetime_days"])),
        [meta["pf_params"]["p_miss"]], [meta["pf_params"]["p_false"]]).mean())}
    op = meta["ours_params"]
    for s in specs:
        if s[3] != "train":
            continue
        nm = vname(s)
        r_ = Retrieval(k2, Q, s[0], s[1], s[2])
        row = {}
        for fam in ("hpf", "pf_nbr"):
            p = table[nm][fam]["params"]
            m = r_.means(p["K"], p["tau"], p["k_nbr"])
            kap = [p["kappa0"]] if fam == "hpf" else None
            row[fam] = float(fe.run(m, [p["p_miss"]], [p["p_false"]],
                                    kap).mean())
        if s == REF:
            m = r_.means(op["K"], op["tau"], op["k_nbr"])
            row["hpf_frozen_v2_params"] = float(fe.run(
                m, [op["p_miss"]], [op["p_false"]], [op["kappa0"]]).mean())
            pn = meta["pf_nbr_params"]
            m = r_.means(pn["K"], pn["tau"], pn["k_nbr"])
            row["pf_nbr_frozen_v2_params"] = float(fe.run(
                m, [pn["p_miss"]], [pn["p_false"]]).mean())
        out[nm.rsplit("|", 1)[0]] = row
        say(f"train-check {nm.rsplit('|', 1)[0]:20s} HiPer {row['hpf']:.5f}"
            f" PF+nbr {row['pf_nbr']:.5f}")
    return out


# ── Post-hoc grid-edge extension (--edge) ────────────────────────────────────
# Declared 2026-10-20 ~14:47 JST AFTER the first main-grid rows: most variants
# selected k_nbr = 0.25 (lowest declared value) and w_g = 1 (highest). The
# extension is applied to the reference structure too (fair baseline), every
# selection is over the union main + edge grid, and the 2-fold check repeats
# the whole selection (variant + params) on the union.
NBR_EDGE = dict(K=[30, 100, 300, 1000], tau=[0.02, 0.05, 0.1, 0.2],
                k_nbr=[0.0625, 0.125, 0.25])
GEO_W_EDGE = (3.0, 10.0)
GEO_DROP = {"all": None, "no_extent": (3, 4), "no_height": (0, 1, 2, 4),
            "no_pixel": (0, 1, 2, 3)}


def ename(s) -> str:
    return vname(s[:4]) + ("" if s[4] == "all" else f"|{s[4]}")


def parse_name(nm):
    p = nm.split("|")
    return (p[0], p[1], float(p[2][1:]), p[3], p[4] if len(p) > 4 else "all")


def run_edge(spec):
    t0 = time.time()
    kernel, emb, wg, ks, sub = spec
    ret = Retrieval(DATA["keys"][ks], DATA["Q"], kernel, emb, wg,
                    GEO_DROP[sub])
    fe, nb = DATA["fe"], DATA["nbr_e"]
    hpf = np.zeros((len(nb), len(HPF_CFG), fe.S))
    pfn = np.zeros((len(nb), len(PF_CFG), fe.S))
    for j, g in enumerate(nb):
        m = ret.means(g["K"], g["tau"], g["k_nbr"])
        hpf[j] = fe.run(m, *HPF_ARR)
        pfn[j] = fe.run(m, *PF_ARR[:2])
    return spec, hpf, pfn, time.time() - t0


def _best(arr, mask=None):
    v = arr.mean(2) if mask is None else arr[:, :, mask].mean(2)
    j, c = np.unravel_index(int(np.argmin(v)), v.shape)
    return int(j), int(c), float(v[j, c])


def setup_val(meta, recs):
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in recs["train"]])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    streams = ec.load("streams", "val")
    rows_ref = official_rows(
        streams, lambda sc: te.make_model("hpf_nbr", meta["ours_params"]),
        "ours")
    fe = FastEval(streams, labels_of(rows_ref))
    return streams, rows_ref, fe


def edge(workers):
    t_start = time.time()
    res_main = json.load(open(OUT / "results.json"))
    gz = np.load(OUT / "grid.npz")
    main_names = [str(x) for x in gz["variants"]]
    H, P = gz["hpf"], gz["pf_nbr"]
    meta = ec.meta()
    recs = {sp: ec.load("records", sp) for sp in ("train", "train2")}
    streams, _, fe = setup_val(meta, recs)
    assert fe.scenes == [str(x) for x in gz["scenes"]]
    Q = queries(ec.load("feats", "val"), fe.order)
    keys = {ks: Keys([r for sp in sps for r in recs[sp]])
            for ks, sps in KEYSETS.items()}
    bk, be, bg, bks, _ = parse_name(res_main["best_hpf_variant"])
    specs = [("rel", "dino", 0.0, "train", "all"),
             ("rel", "dino", 0.0, "train+train2", "all"),
             (bk, be, bg, bks, "all"), (bk, be, bg, "train", "all")]
    specs += [("rel", "dino", w, "train", "all") for w in (1.0,) + GEO_W_EDGE]
    if bg > 0:
        specs += [(bk, be, w, bks, "all") for w in GEO_W_EDGE]
        specs += [(bk, be, bg, bks, s) for s in GEO_DROP if s != "all"]
    specs = list(dict.fromkeys(specs))
    NBR_E = te.grid(NBR_EDGE)
    DATA.update(fe=fe, Q=Q, keys=keys, nbr_e=NBR_E)
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for spec, hpf, pfn, sec in pool.imap_unordered(run_edge, specs):
            res[spec] = (hpf, pfn)
            print(f"edge {ename(spec):36s} HiPer {_best(hpf)[2]:.5f} "
                  f"PF+nbr {_best(pfn)[2]:.5f} ({sec:.0f}s)", flush=True)
    np.savez_compressed(OUT / "edge_grid.npz",
                        variants=np.array([ename(s) for s in specs]),
                        hpf=np.stack([res[s][0] for s in specs]).astype(
                            np.float32),
                        pf_nbr=np.stack([res[s][1] for s in specs]).astype(
                            np.float32))
    A = np.arange(fe.S) % 2 == 0
    B = ~A
    cfgs = {"hpf": HPF_CFG, "pf_nbr": PF_CFG}
    allv = {nm: {"hpf": (H[i], NBR), "pf_nbr": (P[i], NBR)}
            for i, nm in enumerate(main_names)}
    out = {"declared_post_hoc": {"nbr_edge": NBR_EDGE,
                                 "geo_w_edge": GEO_W_EDGE,
                                 "geo_drop": GEO_DROP,
                                 "main_best": res_main["best_hpf_variant"]},
           "variants": {}}
    for s in specs:
        nm = ename(s)
        hpf, pfn = res[s]
        if nm in allv:      # union with the main grid
            u = {"hpf": (np.concatenate([H[main_names.index(nm)], hpf]),
                         NBR + NBR_E),
                 "pf_nbr": (np.concatenate([P[main_names.index(nm)], pfn]),
                            NBR + NBR_E)}
        else:
            u = {"hpf": (hpf, NBR_E), "pf_nbr": (pfn, NBR_E)}
        allv[nm] = u
        row = {}
        for fam, arr_e in (("hpf", hpf), ("pf_nbr", pfn)):
            j, c, v = _best(arr_e)
            row[fam] = {"edge_val_nll": v,
                        "edge_params": {**NBR_E[j], **cfgs[fam][c]}}
            arr, nl = u[fam]
            j, c, v = _best(arr)
            ja, ca, _ = _best(arr, A)
            jb, cb, _ = _best(arr, B)
            on_b, on_a = float(arr[ja, ca, B].mean()), float(
                arr[jb, cb, A].mean())
            row[fam].update({
                "union_val_nll": v,
                "union_params": {**nl[j], **cfgs[fam][c]},
                "union_cv2": {"A_to_B": on_b, "B_to_A": on_a,
                              "mean_heldout": 0.5 * (on_a + on_b),
                              "params_A": {**nl[ja], **cfgs[fam][ca]},
                              "params_B": {**nl[jb], **cfgs[fam][cb]}}})
        out["variants"][nm] = row
    # whole selection (variant + params) on one fold, scored on the other
    for fam in ("hpf", "pf_nbr"):
        pipe = {}
        for sel, ev_ in (("A", B), ("B", A)):
            mask = A if sel == "A" else B
            best = None
            for nm, d in allv.items():
                arr, nl = d[fam]
                j, c, v = _best(arr, mask)
                if best is None or v < best[0]:
                    best = (v, nm, nl[j], cfgs[fam][c],
                            float(arr[j, c, ev_].mean()))
            pipe[sel] = {"variant": best[1], "sel_nll": best[0],
                         "heldout_nll": best[4], **best[2], **best[3]}
        pipe["mean_heldout"] = 0.5 * (pipe["A"]["heldout_nll"]
                                      + pipe["B"]["heldout_nll"])
        out[f"pipeline_cv2_{fam}_union"] = pipe
    # final picks over everything (val NLL), fair reference over the union
    finals = {}
    for fam in ("hpf", "pf_nbr"):
        best = None
        for nm, d in allv.items():
            arr, nl = d[fam]
            j, c, v = _best(arr)
            if best is None or v < best[0]:
                best = (v, nm, {**nl[j], **cfgs[fam][c]})
        finals[f"{fam}_best"] = best
        arr, nl = allv[vname(REF)][fam]
        j, c, v = _best(arr)
        finals[f"{fam}_ref_union"] = (v, vname(REF), {**nl[j], **cfgs[fam][c]})
    keyed = ec.keyed(streams)
    off = {}
    for name, (v, nm, p) in finals.items():
        kn, em, wg, ks, sub = parse_name(nm)
        m = Retrieval(keys[ks], Q, kn, em, wg, GEO_DROP[sub]).means(
            p["K"], p["tau"], p["k_nbr"])
        o = official_prior_model(keyed, fe.order, m, p,
                                 name.startswith("hpf"), name)
        off[name] = {**o, "variant": nm, "params": p, "fast_nll": v}
        print(f"official {name:18s} {nm:36s} nll {o['nll']:.5f} (fast "
              f"{v:.5f}) brier {o['brier']:.5f} ece {o['ece']:.5f} auroc "
              f"{o['auroc']:.5f}", flush=True)
        assert abs(o["nll"] - v) < 1e-6, name
    out["official_final"] = off
    # per-scene paired bootstrap: final best vs fair reference / frozen
    from hiper.evaluation import persistence_metrics as pm

    def scene_vec(fam, nm, p):
        arr, nl = allv[nm][fam]
        j = nl.index({k: p[k] for k in ("K", "tau", "k_nbr")})
        c = cfgs[fam].index({k: p[k] for k in cfgs[fam][0]})
        return dict(zip(fe.scenes, arr[j, c].astype(float).tolist()))

    for fam in ("hpf", "pf_nbr"):
        b = scene_vec(fam, finals[f"{fam}_best"][1], finals[f"{fam}_best"][2])
        r = scene_vec(fam, vname(REF), finals[f"{fam}_ref_union"][2])
        out[f"paired_{fam}_best_vs_ref_union"] = pm.paired_bootstrap(b, r)
    # independent check: TRAIN streams, keys = train2 (no selection there)
    st = ec.load("streams", "train")
    rows = official_rows(st, lambda sc: te.make_model(
        "pf", meta["pf_params"]), "pf")
    fe_t = FastEval(st, labels_of(rows))
    Q_t = queries(ec.load("feats", "train"), fe_t.order)
    k2 = Keys(recs["train2"])
    tc = {}
    op, pn = meta["ours_params"], meta["pf_nbr_params"]
    jobs = [("hpf_ref_frozen", vname(REF), "hpf", op),
            ("pf_nbr_ref_frozen", vname(REF), "pf_nbr", pn)]
    for name, (v, nm, p) in finals.items():
        jobs.append((name, nm, "hpf" if name.startswith("hpf") else "pf_nbr",
                     p))
    for label, nm, fam, p in jobs:
        kn, em, wg, _, sub = parse_name(nm)
        m = Retrieval(k2, Q_t, kn, em, wg, GEO_DROP[sub]).means(
            p["K"], p["tau"], p["k_nbr"])
        kap = [p["kappa0"]] if fam == "hpf" else None
        tc[label] = {"variant_structure": nm.rsplit("|", 1)[0]
                     if sub == "all" else nm, "keys": "train2",
                     "nll": float(fe_t.run(m, [p["p_miss"]], [p["p_false"]],
                                           kap).mean())}
        print(f"train-check {label:18s} {nm:36s} {tc[label]['nll']:.5f}",
              flush=True)
    tc["pf_lambda_frozen"] = float(fe_t.run(
        np.full(fe_t.O, math.exp(-1.0 / meta["pf_params"]["lifetime_days"])),
        [meta["pf_params"]["p_miss"]], [meta["pf_params"]["p_false"]]).mean())
    out["train_check"] = tc
    out["seconds"] = round(time.time() - t_start, 1)
    with open(OUT / "edge_results.json", "w") as f:
        json.dump(jdump(out), f, indent=1)
    print("wrote", OUT / "edge_results.json", f"({out['seconds']}s)")


def diag():
    """SCORING-ONLY loss breakdown (labels used to group rows, nothing is
    fitted): frozen reference vs the selected best HiPer variant (final pick
    of edge_results.json if present, else results.json), official replay
    rows, by (event at k, label_next) and by GT group.
    Writes diag_breakdown.json."""
    res = json.load(open(OUT / "results.json"))
    if (OUT / "edge_results.json").exists():
        fin = json.load(open(OUT / "edge_results.json"))["official_final"]
        res = {"best_hpf_variant": fin["hpf_best"]["variant"],
               "variants": {fin["hpf_best"]["variant"]: {
                   "hpf": {"params": fin["hpf_best"]["params"]}}}}
    meta = ec.meta()
    streams = ec.load("streams", "val")
    feats = ec.load("feats", "val")
    recs = {sp: ec.load("records", sp) for sp in ("train", "train2")}
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in recs["train"]])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    rows_ref = official_rows(
        streams, lambda sc: te.make_model("hpf_nbr", meta["ours_params"]),
        "ours")
    fe = FastEval(streams, labels_of(rows_ref))
    Q = queries(feats, fe.order)
    nm = res["best_hpf_variant"]
    p = res["variants"][nm]["hpf"]["params"]
    kernel, emb, wg, ks, sub = parse_name(nm)
    keys = Keys([r for sp in KEYSETS[ks] for r in recs[sp]])
    m = Retrieval(keys, Q, kernel, emb, wg, GEO_DROP[sub]).means(
        p["K"], p["tau"], p["k_nbr"])
    prior = {f"{sc}/{oid}": (float(m[o]), p["kappa0"])
             for o, (sc, oid) in enumerate(fe.order)}
    kw = dict(p_miss=p["p_miss"], p_false=p["p_false"], kappa0=p["kappa0"],
              variance_gate=False, use_class_prior=False)
    rows_b = official_rows(ec.keyed(streams),
                           lambda sc: ec.PerObjectPriorHPF(prior, **kw),
                           "best")
    nb = {(r["scene"], r["oid"], r["k"]): r for r in rows_b}

    def nll(pr, y):
        q = min(max(pr, NLL_EPS), 1.0 - NLL_EPS)
        return -math.log(q) if y == 1 else -math.log1p(-q)

    from collections import defaultdict
    acc = {"evidence_label": defaultdict(lambda: [0, 0.0, 0.0]),
           "group_label": defaultdict(lambda: [0, 0.0, 0.0])}
    tot = [0.0, 0.0]
    for r in rows_ref:
        y = r["label_next"]
        if y is None:
            continue
        rb = nb[(r["scene"], r["oid"], r["k"])]
        assert rb["label_next"] == y
        a_, b_ = nll(r["pred_presence"], y), nll(rb["pred_presence"], y)
        tot[0] += a_
        tot[1] += b_
        for key, kk in (("evidence_label", (r["last_evidence"], y)),
                        ("group_label", (r["gt_group"], y))):
            c = acc[key][str(kk)]
            c[0] += 1
            c[1] += a_
            c[2] += b_
    out = {"variant": nm, "params": p, "rows": sum(
        c[0] for c in acc["evidence_label"].values()),
        "sum_nll_ref": tot[0], "sum_nll_best": tot[1]}
    for key, d in acc.items():
        out[key] = {k: {"n": c[0], "mean_nll_ref": c[1] / c[0],
                        "mean_nll_best": c[2] / c[0],
                        "sum_diff_best_minus_ref": c[2] - c[1]}
                    for k, c in sorted(d.items(),
                                       key=lambda kv: kv[1][2] - kv[1][1])}
    with open(OUT / "diag_breakdown.json", "w") as f:
        json.dump(jdump(out), f, indent=1)
    print(json.dumps(jdump(out), indent=1))


# ── Independent check 2 (--check2) ──────────────────────────────────────────
# Added 2026-10-20 ~23:20 JST, after the main and edge runs, because check 1
# (TRAIN streams, keys = train2) disagreed with val on geometry. Nothing is
# selected here: every structure keeps its val-selected params; train /
# train2 labels are used for scoring only.
#   check 2  TRAIN2 streams (40 scenes, never used for selection) with keys =
#            the TRAIN records, i.e. the exact key set of the val rows with
#            keys = train, so their val-selected params transfer unchanged.
#            Keys factor: keys = train + the OTHER half of train2 (sorted
#            train2 scenes, even / odd index; an object's own scene is never
#            a key) with the params of the val rows with keys = train+train2.
#   check 1b TRAIN streams, keys = train2 (as check 1), but with the params
#            of the val rows with keys = train+train2 (60 key scenes; train2
#            has 40, closer than the 20 of the train rows used by check 1).

def check2():
    t_start = time.time()
    res = json.load(open(OUT / "results.json"))
    edge_res = json.load(open(OUT / "edge_results.json"))
    table = res["variants"]
    meta = ec.meta()
    op, pn = meta["ours_params"], meta["pf_nbr_params"]
    recs = {sp: ec.load("records", sp) for sp in ("train", "train2")}
    from hiper.evaluation import persistence_metrics as pm

    def say(*a):
        print(" ".join(str(x) for x in a), flush=True)

    # check 2 data; labels from the official frozen HiPer replay (te.NBR =
    # keys train), which also checks the fast replay on these streams
    st = ec.load("streams", "train2")
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in recs["train"]])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    rows = official_rows(st, lambda sc: te.make_model("hpf_nbr", op), "ours")
    fe = FastEval(st, labels_of(rows))
    Q = queries(ec.load("feats", "train2"), fe.order)
    k_tr = Keys(recs["train"])
    half = {sc: i % 2 for i, sc in enumerate(fe.scenes)}
    k_half = {h: Keys(recs["train"] + [r for r in recs["train2"]
                                       if half[r["scene"]] != h])
              for h in (0, 1)}
    q_idx = {h: np.flatnonzero(np.array([half[sc] == h
                                         for sc, _ in fe.order]))
             for h in (0, 1)}
    Qh = {h: {k: v[q_idx[h]] for k, v in Q.items()} for h in (0, 1)}

    def means2(struct, keyset, p):
        kn, em, wg, sub = struct
        if keyset == "train":
            return Retrieval(k_tr, Q, kn, em, wg, GEO_DROP[sub]).means(
                p["K"], p["tau"], p["k_nbr"])
        m = np.zeros(fe.O)
        for h in (0, 1):
            m[q_idx[h]] = Retrieval(k_half[h], Qh[h], kn, em, wg,
                                    GEO_DROP[sub]).means(
                p["K"], p["tau"], p["k_nbr"])
        return m

    def score(fe_, m, fam, p):
        kap = [p["kappa0"]] if fam == "hpf" else None
        return fe_.run(m, [p["p_miss"]], [p["p_false"]], kap)[0]

    out = {"check2": {"streams": "train2", "n_scenes": fe.S,
                      "n_objects": fe.O, "keys_train_n": k_tr.n,
                      "keys_train_plus_half_n": [k_half[0].n, k_half[1].n]}}
    c2 = out["check2"]
    ref_s = ("rel", "dino", 0.0, "all")
    v = score(fe, means2(ref_s, "train", op), "hpf", op)
    c2["hpf_ref_frozen"] = float(v.mean())
    c2["hpf_ref_frozen_official"] = ec.scores(rows)["nll"]
    assert abs(c2["hpf_ref_frozen"] - c2["hpf_ref_frozen_official"]) < 1e-9
    c2["pf_nbr_ref_frozen"] = float(score(fe, means2(ref_s, "train", pn),
                                          "pf_nbr", pn).mean())
    c2["pf_nbr_ref_frozen_official"] = ec.run_model(
        st, lambda sc: te.make_model("pf_nbr", pn), "pf_nbr")["nll"]
    assert abs(c2["pf_nbr_ref_frozen"]
               - c2["pf_nbr_ref_frozen_official"]) < 1e-9
    c2["pf_lambda_frozen_official"] = ec.run_model(
        st, lambda sc: te.make_model("pf", meta["pf_params"]), "pf")["nll"]
    say("check2 frozen:", {k: round(x, 5) for k, x in c2.items()
                           if isinstance(x, float)})
    per2 = {"hpf_ref_frozen": v}
    # every main-grid structure, keys = train and keys = train + half train2
    c2["structures"] = {}
    for s in all_specs():
        if s[3] != "train":
            continue
        st_ = s[:3] + ("all",)
        key = vname(s).rsplit("|", 1)[0]
        row = {}
        for ks in ("train", "train+train2"):
            nm = vname(s[:3] + (ks,))
            for fam in ("hpf", "pf_nbr"):
                p = table[nm][fam]["params"]
                vv = score(fe, means2(st_, ks, p), fam, p)
                row[f"{fam}|{ks}"] = float(vv.mean())
                per2[f"{nm}|{fam}"] = vv
        c2["structures"][key] = row
        say(f"check2 {key:16s} " + " ".join(f"{k} {x:.5f}"
                                           for k, x in row.items()))
    # edge variants with their union-selected params
    c2["edge"] = {}
    for nm, rv in edge_res["variants"].items():
        kn, em, wg, ks, sub = parse_name(nm)
        row = {}
        for fam in ("hpf", "pf_nbr"):
            p = rv[fam]["union_params"]
            vv = score(fe, means2((kn, em, wg, sub), ks, p), fam, p)
            row[fam] = float(vv.mean())
            per2[f"edge:{nm}|{fam}"] = vv
        c2["edge"][nm] = row
        say(f"check2 edge {nm:40s} HiPer {row['hpf']:.5f} "
            f"PF+nbr {row['pf_nbr']:.5f}")
    sc2 = list(fe.scenes)

    def paired(a, b):
        return pm.paired_bootstrap(dict(zip(sc2, per2[a].tolist())),
                                   dict(zip(sc2, per2[b].tolist())))

    fb = edge_res["official_final"]["hpf_best"]["variant"]
    fr = edge_res["official_final"]["hpf_ref_union"]["variant"]
    c2["paired"] = {
        "final_best_vs_ref_union": paired(f"edge:{fb}|hpf",
                                          f"edge:{fr}|hpf"),
        "final_best_vs_ref_frozen": paired(f"edge:{fb}|hpf",
                                           "hpf_ref_frozen"),
        "ref_union_vs_ref_frozen": paired(f"edge:{fr}|hpf",
                                          "hpf_ref_frozen")}
    # check 1b: TRAIN streams, keys = train2, train+train2-row params
    st1 = ec.load("streams", "train")
    rows1 = official_rows(st1, lambda sc: te.make_model(
        "pf", meta["pf_params"]), "pf")
    fe1 = FastEval(st1, labels_of(rows1))
    Q1 = queries(ec.load("feats", "train"), fe1.order)
    k2 = Keys(recs["train2"])
    c1 = {}
    for s in all_specs():
        if s[3] != "train+train2":
            continue
        nm = vname(s)
        key = nm.rsplit("|", 1)[0]
        row = {}
        r_ = Retrieval(k2, Q1, s[0], s[1], s[2])
        for fam in ("hpf", "pf_nbr"):
            p = table[nm][fam]["params"]
            row[fam] = float(score(fe1, r_.means(p["K"], p["tau"],
                                                 p["k_nbr"]), fam, p).mean())
        c1[key] = row
        say(f"check1b {key:16s} HiPer {row['hpf']:.5f} "
            f"PF+nbr {row['pf_nbr']:.5f}")
    out["check1b"] = c1
    # factor table: delta NLL (variant - base) on val, check 1, 1b and 2
    c1a = res["train_check"]

    def val(nm):
        return table[nm]["hpf"]["val_nll"]

    def c2v(nm):
        key, ks = nm.rsplit("|", 1)
        return c2["structures"][key][f"hpf|{ks}"]

    def c1v(nm):
        key, ks = nm.rsplit("|", 1)
        if ks == "train":
            return c1a[key]["hpf"]
        return c1[key]["hpf"]

    def frow(factor, base, var):
        d = {"factor": factor, "base": base, "variant": var,
             "val": val(var) - val(base), "check2": c2v(var) - c2v(base)}
        same_keys = base.rsplit("|", 1)[1] == var.rsplit("|", 1)[1]
        d["check1"] = c1v(var) - c1v(base) if same_keys else None
        return d

    R = vname(REF)
    facs = [("kernel abs", R, "abs|dino|g0|train")]
    facs += [(f"emb {e}", R, f"rel|{e}|g0|train")
             for e in ("cat75", "cat50", "clip", "ens")]
    facs += [(f"geo {g:g}", R, f"rel|dino|g{g:g}|train")
             for g in (0.1, 0.3, 1.0)]
    facs += [("keys train+train2", R, "rel|dino|g0|train+train2")]
    Bn = res["best_hpf_variant"]
    bk, be, bg, bks, _ = parse_name(Bn)
    facs += [(f"best: part emb {be}", f"rel|dino|g{bg:g}|{bks}", Bn),
             (f"best: part geo {bg:g}", f"{bk}|{be}|g0|{bks}", Bn),
             (f"best: part keys {bks}", f"{bk}|{be}|g{bg:g}|train", Bn),
             ("best + kernel abs", Bn, f"abs|{be}|g{bg:g}|{bks}")]
    # every row: delta = NLL(with the part) - NLL(without), < 0 = it helps
    out["factor_deltas"] = [frow(*f) for f in facs]
    for d in out["factor_deltas"]:
        say(f"factor {d['factor']:22s} val {d['val']:+.5f} check1 "
            f"{'n/a' if d['check1'] is None else format(d['check1'], '+.5f')}"
            f" check2 {d['check2']:+.5f}")
    out["per_scene_check2"] = {k: dict(zip(sc2, v_.tolist()))
                               for k, v_ in per2.items()
                               if k.startswith("edge:") or k ==
                               "hpf_ref_frozen"}
    out["seconds"] = round(time.time() - t_start, 1)
    with open(OUT / "check2_results.json", "w") as f:
        json.dump(jdump(out), f, indent=1)
    say("wrote", OUT / "check2_results.json", f"({out['seconds']}s)")


# ── Geometry diagnostic (--geodiag) ─────────────────────────────────────────
# Added 2026-10-20 ~23:30 JST after check 2: geometry helps on val and
# check 2 but hurts on check 1. Scoring only, nothing is selected: the final
# best structure (edge_results.json) with its params FIXED, geometry parts
# removed one at a time, on val (keys = train+train2), check 1 (TRAIN
# streams, keys = train2) and check 2 (TRAIN2 streams, keys = train + the
# other half of train2). Labels via gt_labels, the function replay() uses.

GEO_SUBSETS = {"all": None, "no_extent": (3, 4), "no_height": (0, 1, 2, 4),
               "no_pixel": (0, 1, 2, 3), "extent_only": (0, 1, 2)}


def labels_direct(streams):
    from hiper.evaluation.lifelong_objects import gt_labels
    return {(sc, oid, int(ev["k"])): gt_labels(st["gt"], ev["gt_id"],
                                               ev["pos_session"], ev["k"])[1]
            for sc, st in streams.items()
            for oid, ob in st["objects"].items() for ev in ob["events"]}


def geodiag():
    t_start = time.time()
    edge_res = json.load(open(OUT / "edge_results.json"))
    fin = edge_res["official_final"]["hpf_best"]
    kn, em, wg, ks, _ = parse_name(fin["variant"])
    p = fin["params"]
    recs = {sp: ec.load("records", sp) for sp in ("train", "train2")}
    from hiper.evaluation import persistence_metrics as pm
    structs = {"best": (wg, "all"), "no_geo": (0.0, "all"),
               "no_extent": (wg, "no_extent"), "no_height": (wg, "no_height"),
               "no_pixel": (wg, "no_pixel"),
               "extent_only": (wg, "extent_only"),
               "g0.3": (0.3, "all"), "g0.1": (0.1, "all")}
    out = {"variant": fin["variant"], "params": p, "datasets": {}}
    for ds in ("val", "check1", "check2"):
        split = {"val": "val", "check1": "train", "check2": "train2"}[ds]
        st = ec.load("streams", split)
        fe = FastEval(st, labels_direct(st))
        Q = queries(ec.load("feats", split), fe.order)
        if ds == "val":
            parts = [(np.arange(fe.O), Keys([r for sp in KEYSETS[ks]
                                             for r in recs[sp]]))]
        elif ds == "check1":
            parts = [(np.arange(fe.O), Keys(recs["train2"]))]
        else:
            half = {sc: i % 2 for i, sc in enumerate(fe.scenes)}
            parts = [(np.flatnonzero(np.array([half[sc] == h
                                               for sc, _ in fe.order])),
                      Keys(recs["train"] + [r for r in recs["train2"]
                                            if half[r["scene"]] != h]))
                     for h in (0, 1)]
        per = {}
        for nm, (w_, sub) in structs.items():
            m = np.zeros(fe.O)
            for idx, kk in parts:
                q = {k: v[idx] for k, v in Q.items()}
                m[idx] = Retrieval(kk, q, kn, em, w_,
                                   GEO_SUBSETS[sub]).means(
                    p["K"], p["tau"], p["k_nbr"])
            per[nm] = fe.run(m, [p["p_miss"]], [p["p_false"]],
                             [p["kappa0"]])[0]
        d = {"n_scenes": fe.S,
             "nll": {nm: float(v.mean()) for nm, v in per.items()},
             "paired_best_vs_no_geo": pm.paired_bootstrap(
                 dict(zip(fe.scenes, per["best"].tolist())),
                 dict(zip(fe.scenes, per["no_geo"].tolist()))),
             "per_scene_best_minus_no_geo": dict(zip(
                 fe.scenes, (per["best"] - per["no_geo"]).tolist()))}
        if ds == "val":
            d["fast_best_vs_official"] = [d["nll"]["best"], fin["nll"]]
            assert abs(d["nll"]["best"] - fin["nll"]) < 1e-9
        out["datasets"][ds] = d
        print(ds, {k: round(v, 5) for k, v in d["nll"].items()},
              "paired best-no_geo", {k: round(v, 5) for k, v in
                                     d["paired_best_vs_no_geo"].items()},
              flush=True)
    out["seconds"] = round(time.time() - t_start, 1)
    with open(OUT / "geodiag_results.json", "w") as f:
        json.dump(jdump(out), f, indent=1)
    print("wrote", OUT / "geodiag_results.json", f"({out['seconds']}s)")


def verify():
    """I1: the main grid (grid.npz, written by main()) is reproduced by the
    current code: every edge_grid.npz value (recomputed by --edge) at a
    (variant, K, tau, k_nbr, filter config) also in the main grid must match."""
    gz, ez = np.load(OUT / "grid.npz"), np.load(OUT / "edge_grid.npz")
    mn = [str(x) for x in gz["variants"]]
    en = [str(x) for x in ez["variants"]]
    ne = te.grid(NBR_EDGE)
    pairs = [(je, NBR.index(g)) for je, g in enumerate(ne) if g in NBR]
    out = {"n_nbr_overlap": len(pairs), "variants": {}}
    for ie, nm in enumerate(en):
        if nm not in mn:
            continue
        im = mn.index(nm)
        d = {fam: float(max(np.abs(gz[fam][im, jm].astype(np.float64)
                                   - ez[fam][ie, je].astype(np.float64)).max()
                            for je, jm in pairs))
             for fam in ("hpf", "pf_nbr")}
        out["variants"][nm] = d
    out["max_abs_diff"] = max(v for d in out["variants"].values()
                              for v in d.values())
    print(json.dumps(out, indent=1))
    with open(OUT / "verify_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    elif "--diag" in sys.argv:
        diag()
    elif "--edge" in sys.argv:
        edge(3)
    elif "--check2" in sys.argv:
        check2()
    elif "--geodiag" in sys.argv:
        geodiag()
    else:
        main()
