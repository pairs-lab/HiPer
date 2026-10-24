#!/usr/bin/env python3
"""
scripts/explore_w4_ebconc.py
============================
VALIDATION-ONLY exploration, idea w4 (slug w4_ebconc): an empirical-Bayes
concentration per object (the hierarchy proper).

HiPer (run v2) gives every new map object the prior
  p ~ Beta(kappa0 m, kappa0 (1 - m))
over its per-day survival probability p, with the mean m retrieved from the
K = 300 nearest TRAIN objects in DINOv2 appearance (kernel weights
w_i = exp((s_i - s_max) / tau), weighted MAP rate) and ONE global kappa0 = 32.
Here the prior's shape is fitted per object, by maximising the
kernel-weighted marginal likelihood of the neighbours' ground-truth TRAIN
histories (survived exposure S_i and vanish gaps d_il, in days):
  J(theta) = sum_i w_i log Int p^S_i prod_l (1 - p^d_il) g(p; theta) dp
           + k_pop * mean_{all train records j} log Int (same, record j) dp
k_pop pseudo-records of the whole TRAIN population shrink the fit (the analog
of k_nbr in the MAP rate). The integral is a quadrature on a logit-spaced
p-grid (960 cells, logit -10..14); Beta cell masses are exact differences of
the regularised incomplete beta (lower tail for p <= 1/2, upper tail above,
so U-shaped Betas with a or b < 1 are integrated correctly); one likelihood
vector per record is cached. theta grids: logit m in [-3, 9.2] (step 0.2),
kappa = 2^(j/2) in [1/8, 8192].

Variants (each re-tuned on THOR val over te.DET_GRID x its own options):
  eb_mk   g = Beta(kappa m, kappa (1 - m)); m and kappa both fitted.
  eb_k    m = HiPer's retrieved MAP mean (unchanged); kappa fitted.
  mix     g = w0 delta(p_s) + (1 - w0) Beta(kappa m, kappa (1 - m)),
          p_s = 1 - 1e-4 per day ('static'); (w0, m, kappa) fitted
          (w0 in {0, .05, ..., .95, .99}; logit m step 0.4; kappa = 2^j).
          Filter MixPriorHPF: per component a presence pi_c, exact posterior
          component weights, HiPer's Beta update inside the Beta component.
  mix_map mix, but the Beta component's mean is the MAP rate (as in HiPer,
          shrinkage k = 1) of the neighbours weighted by w_i (1 - r_i),
          r_i = P(record i is static | its history, fitted mixture): the
          atom absorbs the static neighbours, the Beta the movers
          (repair after the quick look: the EB-fitted mean is worse than
          the MAP mean, see hpf_ebmean / pf_ebmean).
  mix_k0  mix with kappa fixed to a global kappa0 in te.KAPPA_FINAL.
Every fitted kappa is multiplied by ONE global scale c (option kscale): the
fit sees clean ground-truth histories, the filter noisy evidence, so the
val-optimal level differs; c replaces the reference's tuned global kappa0
and keeps the per-object relative concentrations. Option kcap then caps the
scaled kappa (kappa_i = min(c kappa_i, kcap)): ~1 % of the single-Beta fits
and >10 % of the mixture's Beta components sit at the grid edge 8192.
Ablations / fair baselines (the idea applied where it can be):
  hpf_ebmean  eb_mk's fitted mean with a global kappa0 in te.KAPPA_FINAL
              (separates the mean from the concentration).
  pf_ebmean   PF + retrieved with eb_mk's fitted mean (no rate learning).
  pf_mix      PF + retrieved with the mix prior: component weights updated,
              Beta component frozen at its mean (no rate learning).
Reference: HiPer v2 = te.make_model("hpf_nbr") with K=300, tau=0.05,
k_nbr=1, variance_gate False, tau_pi 0.3, over te.DET_GRID x te.KAPPA_FINAL;
PF + retrieved = te.make_model("pf_nbr"), K=300, tau=0.05, k_nbr=1 over
te.DET_GRID.

Declared option grids (per variant, x te.DET_GRID; tau_pi = 0.3 and
variance_gate = False everywhere, as in the reference; neither changes NLL;
K = 300 neighbours as in the reference):
  eb_mk, eb_k, mix, mix_map : tau_w in {0.02, 0.05, 0.1}
                              x k_pop in {0, 0.25, 1} x kscale in {1, 2, 4, 8}
                              x kcap in {none, 128}
  mix_k0           : kappa0 in te.KAPPA_FINAL (tau_w 0.05, k_pop 0.25)
  hpf_ebmean       : kappa0 in te.KAPPA_FINAL (eb_mk mean at its best option)
  pf_ebmean        : eb_mk mean at its best option (DET only)
  pf_mix, pf_mixmap: mix / mix_map prior at its best option (DET only)
History: a first look (--quick, reference detection params only) used
tau_w {0.05, 0.1} x k_pop {0.25, 1, 4} without kscale; every EB variant was
worse than the reference there (fitted kappa median ~8 vs tuned kappa0 32),
which led to kscale, k_pop = 0, tau_w = 0.02 and mix_map. A second quick
look (tau_w {0.05, 0.1} x k_pop {0, 0.25, 1} x kscale, reference detection
params; log_quick.txt) gave eb_k 0.29000 vs reference 0.29091. kcap was
added after that look and before the full run (from the fitted-kappa
quantiles only, no val score). The grids above were then fixed before the
full run; the 2-fold CV covers them.

Data: TRAIN records (records_train.pkl) only, the same records the
reference prior uses; val streams for replay. Val labels are used ONLY to
score and to select the declared options (NLL, mean over scenes).
Robustness: 2-fold CV over sorted val scenes (fold A = even index, fold B =
odd): select on one fold, score on the other, mean of the two held-out NLLs,
for the best variant and for the reference selected the same way.

  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w4_ebconc.py [--workers 3]
  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w4_ebconc.py --followup
  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w4_ebconc.py --pflam
  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w4_ebconc.py --checkquad
  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w4_ebconc.py --summary
(run in this order; OMP/OPENBLAS/MKL_NUM_THREADS=1 keeps it at 3 cores)
Outputs: RESULTS/explore_cache/w4_ebconc/{results.json, grid.jsonl,
fits_summary.json, log.txt, priors_val.pkl}; --followup (mechanism
controls: eb_mk's and eb_k's fitted kappa permuted across objects,
globally and within deciles of logit m, 3 seeds, same grid; per-variant
2-fold CV and per-scene pairing vs the reference; ablation by option)
writes {results_followup.json, grid_followup.jsonl, log_followup.txt};
--pflam (PF(lambda) + static atom, fair baseline) writes
{results_pflam.json, grid_pflam.jsonl}; --checkquad (grid quadrature vs
exact closed form) writes check_quadrature.json; --summary merges all into
summary.json.
"""

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.special import betainc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HIT, MISS, NONE, HierarchicalPersistenceFilter, _log_beta_moment)
from hiper.ephemerality.priors import weighted_map_rate  # noqa: E402
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import replay  # noqa: E402

DAY = 86400.0
OUT = ec.CACHE / "w4_ebconc"
P_STATIC = 1.0 - 1e-4
REF_NBR = dict(K=300, tau=0.05, k_nbr=1.0)
FIXED = dict(tau_pi=0.3, variance_gate=False)
TAU_W = [0.02, 0.05, 0.1]
K_POP = [0.0, 0.25, 1.0]
C_SCALE = [1.0, 2.0, 4.0, 8.0]
K_CAP = [None, 128.0]          # None = no cap
MIX_K0_OPT = (0.05, 0.25)
W0_GRID = np.array([0.05 * i for i in range(20)] + [0.99])
METRICS = ("nll", "brier", "ece", "auroc")


# ── Quadrature grid over p (per-day survival) ───────────────────────────────

class PGrid:
    """Cells in logit(p) between edges ze (0 is an edge). Mass outside the
    edges goes to the first / last cell."""

    def __init__(self, lo=-10.0, hi=14.0, step=0.025):
        n = int(round((hi - lo) / step))
        self.ze = lo + step * np.arange(n + 1)
        self.ze[np.abs(self.ze) < 1e-9] = 0.0
        if not np.any(self.ze == 0.0):
            raise ValueError("0 must be an edge")
        zm = 0.5 * (self.ze[:-1] + self.ze[1:])
        self.logp = -np.log1p(np.exp(-zm))           # log p at cell centres
        self.n = zm.size
        self.left = self.ze[1:] <= 0.0               # cells with p <= 1/2

    def masses(self, a, b):
        """(P, G) Beta(a, b) masses of every cell, rows sum to 1."""
        a = np.asarray(a, float)[:, None]
        b = np.asarray(b, float)[:, None]
        x = 1.0 / (1.0 + np.exp(-self.ze))[None, :]
        y = 1.0 / (1.0 + np.exp(self.ze))[None, :]   # 1 - x, exact
        cdf = betainc(a, b, x)                       # accurate for p <= 1/2
        sf = betainc(b, a, y)                        # accurate for p > 1/2
        q = np.where(self.left[None, :], cdf[:, 1:] - cdf[:, :-1],
                     sf[:, :-1] - sf[:, 1:])
        q[:, 0] += cdf[:, 0]
        q[:, -1] += sf[:, -1]
        return np.clip(q, 0.0, None)


def record_loglik(recs, logp):
    """(N, G): log p^S prod_l (1 - p^d_l) at every cell, S and d in days."""
    logp = np.asarray(logp, float)
    out = np.empty((len(recs), logp.size))
    for i, r in enumerate(recs):
        v = (float(r["S"]) / DAY) * logp
        for d in r["gaps"]:
            v = v + np.log(-np.expm1((float(d) / DAY) * logp))
        out[i] = v
    return out


def log_marginal(logL, Q):
    """(N, P): log Int L_i(p) g_theta(p) dp by quadrature."""
    c = logL.max(axis=1, keepdims=True)
    M = np.exp(logL - c) @ Q.T
    return np.log(np.maximum(M, 1e-300)) + c


def theta_grid(z_step=0.2, k_half=True):
    zm = np.round(np.arange(-3.0, 9.2 + 1e-9, z_step), 6)
    ks = (2.0 ** (np.arange(-6, 27) / 2.0) if k_half
          else 2.0 ** np.arange(-3, 14, dtype=float))
    Z, Kp = np.meshgrid(zm, ks, indexing="ij")
    m = 1.0 / (1.0 + np.exp(-Z.ravel()))
    return m, Kp.ravel()


# ── Priors per val object ────────────────────────────────────────────────────

def val_objects(streams):
    """[(key 'scene/oid', unit DINOv2 embedding or None)] in a fixed order."""
    out = []
    for sc in sorted(streams):
        for oid, ob in streams[sc]["objects"].items():
            e = ob["events"][0].get("emb")
            if e is not None:
                e = np.asarray(e, np.float64)
                n = np.linalg.norm(e)
                e = e / n if np.isfinite(n) and n > 1e-12 else None
            out.append((f"{sc}/{oid}", e))
    return out


F: Dict = {}       # fit globals (inherited through fork)


def _fit_setup(recs, nbr):
    g = PGrid()
    logL = record_loglik(recs, g.logp)
    cL = logL.max(axis=1, keepdims=True)
    ls = np.array([(r["S"] / DAY) * math.log(P_STATIC) + sum(
        math.log(-math.expm1((d / DAY) * math.log(P_STATIC)))
        for d in r["gaps"]) for r in recs])
    m1, k1 = theta_grid(0.2, True)                    # single Beta, fine
    LM1 = log_marginal(logL, g.masses(k1 * m1, k1 * (1.0 - m1)))
    m2, k2 = theta_grid(0.4, False)                   # mixture, coarse
    LM2 = log_marginal(logL, g.masses(k2 * m2, k2 * (1.0 - m2)))
    c2 = np.maximum(ls, LM2.max(axis=1))
    F.update(g=g, nbr=nbr, Lexp=np.exp(logL - cL), cL=cL, m1=m1, k1=k1,
             LM1=LM1, pop1=LM1.mean(axis=0), m2=m2, k2=k2, c2=c2,
             A2=np.exp(ls - c2), B2=np.exp(LM2 - c2[:, None]),
             kq=2.0 ** (np.arange(-6, 27) / 2.0))
    allr = np.arange(len(recs))
    popmix = np.zeros((W0_GRID.size, m2.size))
    for s in range(0, len(recs), 100):                # mean over records
        popmix += _fmix(allr[s:s + 100]).sum(axis=1)
    F["popmix"] = popmix / len(recs) + c2.mean()


def _fmix(rows):
    """(n_w0, len(rows), P2): log(w0 A + (1 - w0) B) - c per record."""
    W0 = W0_GRID[:, None, None]
    return np.log(np.maximum(W0 * F["A2"][rows][None, :, None]
                             + (1.0 - W0) * F["B2"][rows][None, :, :],
                             1e-300))


def _fit_one(item):
    """EB fits of one val object for every (tau_w, k_pop) option."""
    key, e, m_ret = item
    f = F
    g, kq = f["g"], f["kq"]
    Qk = g.masses(kq * m_ret, kq * (1.0 - m_ret))
    LMk = np.log(np.maximum(f["Lexp"] @ Qk.T, 1e-300)) + f["cL"]
    popk = LMk.mean(axis=0)
    out = {}
    if e is not None:
        idx, s = f["nbr"]._neighbours(e, REF_NBR["K"])
        Fm = _fmix(idx)                               # (n_w0, K, P2)
    for tw in TAU_W:
        if e is not None:
            w = np.exp((s - s.max()) / tw)
            J1 = w @ f["LM1"][idx]
            Jk = w @ LMk[idx]
            Jm = np.einsum("k,wkp->wp", w, Fm) + float(w @ f["c2"][idx])
        else:
            J1 = np.zeros_like(f["pop1"])
            Jk = np.zeros_like(popk)
            Jm = np.zeros_like(f["popmix"])
        for kp in K_POP:
            o = f"t{tw:g}_k{kp:g}"
            j = int(np.argmax(J1 + kp * f["pop1"]))
            out[f"eb_mk|{o}"] = (float(f["m1"][j]), float(f["k1"][j]))
            j = int(np.argmax(Jk + kp * popk))
            out[f"eb_k|{o}"] = (m_ret, float(kq[j]))
            Jt = Jm + kp * f["popmix"]
            wi, j = np.unravel_index(int(np.argmax(Jt)), Jt.shape)
            w0 = float(W0_GRID[wi])
            out[f"mix|{o}"] = (w0, float(f["m2"][j]), float(f["k2"][j]))
            if e is not None:   # Beta mean: MAP rate of the movers
                A, B = f["A2"][idx], f["B2"][idx, j]
                rs = w0 * A / np.maximum(w0 * A + (1.0 - w0) * B, 1e-300)
                wb = w * (1.0 - rs)
                nb = f["nbr"]
                lam = weighted_map_rate(
                    wb, nb.S[idx], np.concatenate([nb.gaps[i] for i in idx]),
                    np.concatenate([np.full(len(nb.gaps[i]), x)
                                    for i, x in zip(idx, wb)]),
                    REF_NBR["k_nbr"], nb.lam0)
                mb = min(max(math.exp(-lam * DAY), 1e-4), 1.0 - 1e-4)
            else:
                mb = m_ret
            out[f"mix_map|{o}"] = (w0, mb, float(f["k2"][j]))
            if (tw, kp) == MIX_K0_OPT:
                for k0 in te.KAPPA_FINAL:
                    cols = np.where(np.isclose(f["k2"], k0))[0]
                    Jc = Jt[:, cols]
                    wi, jj = np.unravel_index(int(np.argmax(Jc)), Jc.shape)
                    out[f"mix_k0|k{k0:g}"] = (float(W0_GRID[wi]),
                                              float(f["m2"][cols[jj]]),
                                              float(k0))
    return key, out


def fit_priors(recs, objs, nbr, workers, log):
    """Every declared prior: {option key: {object key: tuple}} (cached)."""
    t0 = time.time()
    tag = {"tau_w": TAU_W, "k_pop": K_POP, "w0": W0_GRID.tolist(),
           "mix_k0_opt": list(MIX_K0_OPT), "ref_nbr": REF_NBR,
           "p_static": P_STATIC, "n_recs": len(recs), "n_objs": len(objs)}
    cache = OUT / "priors_val.pkl"
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        if d["tag"] == tag:
            log(f"fits loaded from {cache}")
            for key, e in objs:          # warm the reference rate cache
                if e is not None:
                    nbr.rate(e, K=REF_NBR["K"], tau=REF_NBR["tau"],
                             k=REF_NBR["k_nbr"])
            return d["pri"], d["summ"]
    items = []
    for key, e in objs:                  # also warms nbr's rate cache
        lam = (nbr.lam0 if e is None else
               nbr.rate(e, K=REF_NBR["K"], tau=REF_NBR["tau"],
                        k=REF_NBR["k_nbr"]))
        items.append((key, e, min(max(math.exp(-lam * DAY), 1e-4),
                                  1.0 - 1e-4)))
    _fit_setup(recs, nbr)
    log(f"fit setup: cells {F['g'].n}, single-Beta thetas {F['m1'].size}, "
        f"mixture thetas {F['m2'].size} x w0 {W0_GRID.size}; "
        f"{time.time() - t0:.1f}s")
    with Pool(workers) as pool:
        res = pool.map(_fit_one, items, chunksize=8)
    pri: Dict[str, dict] = defaultdict(dict)
    for (key, _, m_ret), (k2, out) in zip(items, res):
        assert key == k2
        pri["ret"][key] = (m_ret, 32.0)
        for name, v in out.items():
            pri[name][key] = v
    pri = dict(pri)
    summ = {}
    for name, d in pri.items():
        v = np.array(list(d.values()), float)
        w0 = v[:, 0] if v.shape[1] == 3 else np.zeros(len(v))
        m, k = v[:, -2], v[:, -1]
        summ[name] = {
            "logit_m_q": np.quantile(np.log(m / (1 - m)),
                                     [0.1, 0.5, 0.9]).round(3).tolist(),
            "log2_kappa_q": np.quantile(np.log2(k), [0.1, 0.25, 0.5, 0.75,
                                                     0.9]).round(2).tolist(),
            "frac_b_lt_1": float(np.mean(k * (1 - m) < 1.0)),
            "frac_a_lt_1": float(np.mean(k * m < 1.0)),
            "w0_q": np.quantile(w0, [0.1, 0.25, 0.5, 0.75, 0.9]
                                ).round(3).tolist()}
    pickle.dump({"tag": tag, "pri": pri, "summ": summ}, open(cache, "wb"))
    log(f"fits done in {time.time() - t0:.1f}s")
    return pri, summ


# ── Two-component filter: static atom + Beta ─────────────────────────────────

@dataclass
class _MixState:
    class_label: str
    pi_s: float          # P(present at t | static component)
    pi_b: float          # P(present at t | Beta component)
    ws: float            # P(static component | evidence so far)
    a: float
    b: float
    t: float
    last_evidence: str = HIT
    m: Optional[float] = None


class MixPriorHPF(HierarchicalPersistenceFilter):
    """HiPer with the prior w0 delta(p_s) + (1 - w0) Beta(kappa m,
    kappa (1 - m)) per object (keyed 'scene/oid', use with ec.keyed()).
    Each component keeps its own presence; the component weights get the
    exact posterior update (marginal likelihood of each session's evidence
    under the component); inside the Beta component the update is HiPer's
    (a += r w11, b += w10, component-conditional). With learn_rate=False
    the Beta component is frozen at its mean m (PF + mixture)."""

    name = "hpf_mix"

    def __init__(self, prior: Dict[str, tuple], p_static: float = P_STATIC,
                 **kw):
        super().__init__(**kw)
        if self.env_alpha is not None:
            raise ValueError("env level not supported")
        self.prior = prior
        self.log_ps = math.log(p_static)
        self.p_static = p_static

    def create(self, oid, class_label, t, emb=None):
        w0, m, kappa = self.prior[class_label]
        m = min(max(float(m), self.m_clip), 1.0 - self.m_clip)
        kappa = float(kappa)
        self._s[oid] = _MixState(class_label=class_label, pi_s=self.pi_new,
                                 pi_b=self.pi_new, ws=float(w0),
                                 a=kappa * m, b=kappa * (1.0 - m),
                                 t=float(t), m=m)

    def _ss(self, r):
        return math.exp(self.log_ps * r) if r > 0.0 else 1.0

    def _sb(self, st, r):
        if r <= 0.0:
            return 1.0
        if not self.learn_rate:
            return st.m ** r
        return math.exp(_log_beta_moment(st.a, st.b, r))

    def observe(self, oid, t, evidence, eps_l=0.0, **info):
        st = self._s[oid]
        if evidence == NONE:
            return
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        if evidence == HIT:
            L1, L0 = 1.0 - self.p_miss, self.p_false
            w = 1.0 - min(max(float(eps_l), 0.0), 1.0)
            L1, L0 = w * L1 + (1.0 - w), w * L0 + (1.0 - w)
        elif evidence == MISS:
            L1, L0 = self.p_miss, 1.0 - self.p_false
        else:
            raise ValueError(f"unknown evidence {evidence!r}")
        rho = self.rho
        comp = []
        for pi, S in ((st.pi_s, self._ss(r)), (st.pi_b, self._sb(st, r))):
            w11 = pi * S * L1
            w10 = pi * (1.0 - S) * L0
            w01 = (1.0 - pi) * rho * L1
            w00 = (1.0 - pi) * (1.0 - rho) * L0
            comp.append((w11, w10, w01, w00, w11 + w10 + w01 + w00))
        (s11, _, s01, _, zs), (b11, b10, b01, _, zb) = comp
        den = st.ws * zs + (1.0 - st.ws) * zb
        if den <= 0.0:
            return
        st.ws = st.ws * zs / den
        if zs > 0.0:
            st.pi_s = (s11 + s01) / zs
        if zb > 0.0:
            st.pi_b = (b11 + b01) / zb
            if self.learn_rate:
                st.a += r * b11 / zb
                st.b += b10 / zb
        st.t = float(t)
        st.last_evidence = evidence

    def relocate(self, oid, t):
        """Left the mapped location (re-identified elsewhere): likelihood
        pi_c (1 - S_c) + (1 - pi_c) per component; one vanish for the Beta
        component (HiPer: b += pi); presence restarts."""
        st = self._s[oid]
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        ls = st.pi_s * (1.0 - self._ss(r)) + (1.0 - st.pi_s)
        lb = st.pi_b * (1.0 - self._sb(st, r)) + (1.0 - st.pi_b)
        den = st.ws * ls + (1.0 - st.ws) * lb
        if den > 0.0:
            st.ws = st.ws * ls / den
        if self.relocation_resets_rate:
            m = st.m
            kap = st.a + st.b
            st.a, st.b = kap * m, kap * (1.0 - m)
        elif self.learn_rate:
            st.b += st.pi_b
        st.pi_s = st.pi_b = self.pi_new
        st.t = float(t)
        st.last_evidence = HIT

    def presence(self, oid, t):
        st = self._s[oid]
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        ps = st.pi_s * self._ss(r) + (1.0 - st.pi_s) * self.rho
        pb = st.pi_b * self._sb(st, r) + (1.0 - st.pi_b) * self.rho
        return st.ws * ps + (1.0 - st.ws) * pb

    def rate_mean(self, oid):
        st = self._s[oid]
        mb = st.m if not self.learn_rate else st.a / (st.a + st.b)
        return st.ws * self.p_static + (1.0 - st.ws) * mb

    def rate_var(self, oid):
        st = self._s[oid]
        if not self.learn_rate:
            mb, e2b = st.m, st.m * st.m
        else:
            s = st.a + st.b
            mb = st.a / s
            e2b = st.a * (st.a + 1.0) / (s * (s + 1.0))
        mean = st.ws * self.p_static + (1.0 - st.ws) * mb
        e2 = st.ws * self.p_static ** 2 + (1.0 - st.ws) * e2b
        return max(e2 - mean * mean, 0.0)

    def export(self, oid):
        raise NotImplementedError

    def load(self, oid, state):
        raise NotImplementedError


# ── Replay + metrics ─────────────────────────────────────────────────────────

W: Dict = {}       # worker globals (inherited through fork)


def make(family, params, opt):
    p = {**FIXED, **params}
    if family == "ref":
        return te.make_model("hpf_nbr", {**p, **REF_NBR})
    if family == "pfnbr":
        p.pop("variance_gate")
        return te.make_model("pf_nbr", {**p, **REF_NBR})
    if family == "pflam":
        p.pop("variance_gate")
        return te.make_model("pf", p)
    k0 = p.pop("kappa0", None)
    c = p.pop("kscale", None)
    cap = p.pop("kcap", None)
    prior = W["priors"][opt]
    if k0 is not None:     # global concentration override
        prior = {k: (v[:-1] + (float(k0),)) for k, v in prior.items()}
    if c is not None:      # global scale on the fitted concentrations
        prior = {k: (v[:-1] + (float(c) * v[-1],)) for k, v in prior.items()}
    if cap is not None:    # cap on the (scaled) concentrations
        prior = {k: (v[:-1] + (min(float(v[-1]), float(cap)),))
                 for k, v in prior.items()}
    if family == "beta":
        return ec.PerObjectPriorHPF(prior=prior, **p)
    if family == "beta_pf":
        return ec.PerObjectPriorHPF(prior=prior, learn_rate=False, **p)
    if family == "mix":
        return MixPriorHPF(prior=prior, **p)
    if family == "mix_pf":
        return MixPriorHPF(prior=prior, learn_rate=False, **p)
    raise ValueError(family)


def scene_metrics(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    out = {}
    for sc in sorted(by):
        p, y = pm.labeled_arrays(by[sc])
        out[sc] = {"nll": pm.nll(p, y), "brier": pm.brier(p, y),
                   "ece": pm.ece(p, y), "auroc": pm.auroc(p, y)}
    return out


def mean_over(per, scenes=None):
    scenes = sorted(per) if scenes is None else scenes
    return {k: float(np.nanmean([per[s][k] for s in scenes]))
            for k in METRICS}


def job(args):
    variant, family, opt, params = args
    streams = (W["raw"] if family in ("ref", "pfnbr", "pflam")
               else W["keyed"])
    rows = []
    for sc in sorted(streams):
        rows += replay(streams[sc], make(family, params, opt), variant)
    return variant, family, opt, params, scene_metrics(rows)


def run_jobs(pool, jobs, results, log, sink=None):
    t0 = time.time()
    for variant, family, opt, params, per in pool.imap_unordered(job, jobs):
        results.append({"variant": variant, "family": family, "opt": opt,
                        "params": params, "per_scene": per,
                        **mean_over(per)})
        if sink is not None:          # incremental grid file
            sink.write(json.dumps(results[-1], default=float) + "\n")
            sink.flush()
    log(f"  {len(jobs)} configs in {time.time() - t0:.1f}s")


def paired(results, v1, v2):
    """Per-scene NLL/AUROC differences v1 - v2 at each variant's full-val
    selected config (diagnostic; the selection itself used all val)."""
    b1, b2 = best_of(results, v1), best_of(results, v2)
    sc = sorted(b1["per_scene"])
    out = {}
    for k in ("nll", "auroc"):
        d = np.array([b1["per_scene"][s][k] - b2["per_scene"][s][k]
                      for s in sc])
        better = d < 0 if k == "nll" else d > 0
        out[k] = {"mean_diff": float(np.nanmean(d)),
                  "n_scenes_better": int(np.sum(better)),
                  "n_scenes": len(sc),
                  "per_scene": dict(zip(sc, d.round(6).tolist()))}
    return out


def best_of(results, variant, scenes=None):
    cand = [r for r in results if r["variant"] == variant]
    return min(cand, key=lambda r: mean_over(r["per_scene"], scenes)["nll"])


def cv2(results, variants, scenes):
    """Select on one fold of scenes, score on the other; mean of the two
    held-out NLLs."""
    cand = [r for r in results if r["variant"] in variants]
    A, B = scenes[0::2], scenes[1::2]
    out = {}
    for name, sel, ev in (("A->B", A, B), ("B->A", B, A)):
        b = min(cand, key=lambda r: mean_over(r["per_scene"], sel)["nll"])
        held = mean_over(b["per_scene"], ev)
        out[name] = {"variant": b["variant"], "opt": b["opt"],
                     "params": b["params"],
                     "heldout_nll": held["nll"],
                     "heldout_auroc": held["auroc"]}
    out["mean_heldout_nll"] = 0.5 * (out["A->B"]["heldout_nll"]
                                     + out["B->A"]["heldout_nll"])
    return out


def _opt_dims():
    return {
        "tau_w": lambda r: r["opt"].split("|")[1].split("_")[0][1:],
        "k_pop": lambda r: r["opt"].split("|")[1].split("_")[1][1:],
        "kscale": lambda r: r["params"].get("kscale"),
        "kcap": lambda r: r["params"].get("kcap", "none"),
    }


def marginals(results, variants=("eb_mk", "eb_k", "mix", "mix_map")):
    """Ablation by option: best val NLL (and the AUROC of that config) of
    each variant with one option fixed at each of its values."""
    out = {}
    for v in variants:
        cand = [r for r in results if r["variant"] == v]
        if not cand:
            continue
        out[v] = {}
        for dname, f in _opt_dims().items():
            out[v][dname] = {}
            for x in sorted({str(f(r)) for r in cand}):
                b = min((r for r in cand if str(f(r)) == x),
                        key=lambda r: r["nll"])
                out[v][dname][x] = {"nll": b["nll"], "auroc": b["auroc"]}
    return out


def followup(args, log):
    """Mechanism control (declared before the main grid finished; widened
    from the NLL-top single-Beta variant to BOTH single-Beta variants after
    eb_mk and eb_k came out within 0.0004 of each other): each variant's
    fitted concentrations are permuted across val objects, (a) globally
    (keeps the kappa distribution, breaks the per-object fit) and (b) within
    deciles of logit m (keeps the m-kappa relation), 3 seeds each; every
    control is re-tuned over the same DET x kscale x kcap grid at the
    variant's selected (tau_w, k_pop). Means stay per object."""
    t_start = time.time()
    streams = ec.load("streams", "val")
    recs = [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
            for r in ec.load("records", "train")]
    nbr = te.NeighbourSurvivalPrior(recs)
    te.NBR["prior"] = nbr
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    priors, _ = fit_priors(recs, val_objects(streams), nbr, args.workers,
                           log)
    results = [json.loads(x) for x in open(OUT / "grid.jsonl")]
    ctl, sel = {}, {}
    for v in ("eb_mk", "eb_k"):
        b = best_of(results, v)
        sel[v] = b
        base = priors[b["opt"]]
        keys = sorted(base)
        m = np.array([base[k][0] for k in keys])
        kap = np.array([base[k][1] for k in keys])
        z = np.log(m / (1.0 - m))
        dec = np.digitize(z, np.quantile(z, np.linspace(0.1, 0.9, 9)))
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            pa = rng.permutation(len(keys))
            ctl[f"{v}/perm_all_s{seed}"] = {
                k: (float(m[i]), float(kap[pa[i]]))
                for i, k in enumerate(keys)}
            kb = kap.copy()
            for d in np.unique(dec):
                ii = np.where(dec == d)[0]
                kb[ii] = kap[rng.permutation(ii)]
            ctl[f"{v}/perm_mbin_s{seed}"] = {
                k: (float(m[i]), float(kb[i])) for i, k in enumerate(keys)}
        log(f"followup: controls of {v} at {b['opt']} (val NLL "
            f"{b['nll']:.5f}, AUROC {b['auroc']:.5f})")
    W.update(raw=streams, keyed=ec.keyed(streams), priors={**priors, **ctl})
    det = te.grid(te.DET_GRID)
    caps = [{} if c is None else {"kcap": c} for c in K_CAP]
    res = []
    sink = open(OUT / "grid_followup.jsonl", "w")
    with Pool(args.workers) as pool:
        for name in ctl:
            run_jobs(pool, [(name, "beta", name, {**d, "kscale": c, **cp})
                            for d in det for c in C_SCALE for cp in caps],
                     res, log, sink)
    sink.close()
    best = {}
    for name in ctl:
        bb = best_of(res, name)
        best[name] = {"params": bb["params"],
                      **{k: bb[k] for k in METRICS}}
        log(f"  {name:22s} nll {bb['nll']:.5f} auroc {bb['auroc']:.5f} "
            f"{bb['params']}")
    summ = {}
    for v in sel:
        for kind in ("perm_all", "perm_mbin"):
            xs = [best[f"{v}/{kind}_s{s}"] for s in (0, 1, 2)]
            summ[f"{v}/{kind}"] = {
                k: {"mean": float(np.mean([x[k] for x in xs])),
                    "min": float(np.min([x[k] for x in xs])),
                    "max": float(np.max([x[k] for x in xs]))}
                for k in METRICS}
            s_ = summ[f"{v}/{kind}"]
            log(f"  {v}/{kind}: nll mean {s_['nll']['mean']:.5f} "
                f"[{s_['nll']['min']:.5f}, {s_['nll']['max']:.5f}]  auroc "
                f"mean {s_['auroc']['mean']:.5f}  (fitted: nll "
                f"{sel[v]['nll']:.5f} auroc {sel[v]['auroc']:.5f})")
    ref = best_of(results, "ref")
    scenes = sorted(streams)
    per_variant = {}
    for v in ("eb_mk", "eb_k", "mix", "mix_map", "mix_k0", "hpf_ebmean"):
        per_variant[v] = {"cv2": cv2(results, [v], scenes),
                          "paired_vs_ref": paired(results, v, "ref")}
        log(f"  {v:10s} CV2 held-out NLL "
            f"{per_variant[v]['cv2']['mean_heldout_nll']:.5f}; scenes "
            f"better than ref: NLL "
            f"{per_variant[v]['paired_vs_ref']['nll']['n_scenes_better']}/20,"
            f" AUROC "
            f"{per_variant[v]['paired_vs_ref']['auroc']['n_scenes_better']}"
            f"/20")
    out = {"variants": {v: {"opt": b["opt"], "params": b["params"],
                            **{k: b[k] for k in METRICS}}
                        for v, b in sel.items()},
           "reference_best": {"params": ref["params"],
                              **{k: ref[k] for k in METRICS}},
           "reference_cv2": cv2(results, ["ref"], scenes),
           "per_variant": per_variant,
           "controls": best, "controls_summary": summ,
           "marginals": marginals(results),
           "seconds": round(time.time() - t_start, 1)}
    json.dump(out, open(OUT / "results_followup.json", "w"), indent=1,
              default=float)
    log(f"followup done in {time.time() - t_start:.1f}s")


PFLAM_W0 = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]   # 0.0 = PF(lambda) (sanity);
# 0.75, 0.9 added after w0 = 0.5 came out at the grid edge


def pflam(args, log):
    """Fair baseline for the mixture part (added after the full run, when
    the static atom hurt HiPer but helped PF + retrieved): PF(lambda) with
    the prior w0 delta(p_s) + (1 - w0) delta(exp(-1/L)) per day, ONE global
    w0 (PF(lambda) has no retrieval), component weights updated exactly
    (MixPriorHPF, learn_rate=False). Grid: te.DET_GRID x
    te.LIFETIME_GRID_DAYS x w0 in PFLAM_W0; PF(lambda) itself on DET x
    LIFETIME (w0 = 0 must reproduce it exactly)."""
    t_start = time.time()
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    streams = ec.load("streams", "val")
    keyed = ec.keyed(streams)
    keys = [ob["class_label"] for st in keyed.values()
            for ob in st["objects"].values()]
    pri = {}
    for L in te.LIFETIME_GRID_DAYS:
        mL = math.exp(-1.0 / L)
        for w0 in PFLAM_W0:
            pri[f"pflam|L{L:g}|w{w0:g}"] = {k: (w0, mL, 1.0) for k in keys}
    W.update(raw=streams, keyed=keyed, priors=pri)
    det = te.grid(te.DET_GRID)
    res = []
    sink = open(OUT / "grid_pflam.jsonl", "w")
    with Pool(args.workers) as pool:
        log("PF(lambda): DET x LIFETIME")
        run_jobs(pool, [("pf_lam", "pflam", None,
                         {**d, "lifetime_days": L})
                        for d in det for L in te.LIFETIME_GRID_DAYS],
                 res, log, sink)
        log(f"PF(lambda) + static atom: DET x LIFETIME x w0 {PFLAM_W0}")
        run_jobs(pool, [(f"pf_lam_atom_w{w0:g}", "mix_pf",
                         f"pflam|L{L:g}|w{w0:g}", d)
                        for d in det for L in te.LIFETIME_GRID_DAYS
                        for w0 in PFLAM_W0], res, log, sink)
    sink.close()
    # sanity: w0 = 0 reproduces PF(lambda) config by config
    ref = {(r["params"]["p_miss"], r["params"]["p_false"],
            r["params"]["lifetime_days"]): r["nll"]
           for r in res if r["variant"] == "pf_lam"}
    gap = max(abs(r["nll"] - ref[(r["params"]["p_miss"],
                                  r["params"]["p_false"],
                                  float(r["opt"].split("|")[1][1:]))])
              for r in res if r["variant"] == "pf_lam_atom_w0")
    log(f"sanity: max |NLL(w0 = 0) - NLL(PF(lambda))| = {gap:.2e}")
    for r in res:                 # one variant name for the atom family
        if r["variant"].startswith("pf_lam_atom") and \
                r["variant"] != "pf_lam_atom_w0":
            r["variant"] = "pf_lam_atom"
    scenes = sorted(streams)
    out = {"sanity_max_abs_diff_w0": gap}
    for v in ("pf_lam", "pf_lam_atom"):
        b = best_of(res, v)
        out[v] = {"opt": b["opt"], "params": b["params"],
                  **{k: b[k] for k in METRICS},
                  "cv2": cv2(res, [v], scenes)}
        log(f"{v:12s} nll {b['nll']:.5f} brier {b['brier']:.5f} ece "
            f"{b['ece']:.5f} auroc {b['auroc']:.5f} {b['opt']} "
            f"{b['params']}  CV2 {out[v]['cv2']['mean_heldout_nll']:.5f}")
    out["seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(OUT / "results_pflam.json", "w"), indent=1,
              default=float)
    log(f"pflam done in {time.time() - t_start:.1f}s")


def checkquad(args, log):
    """Independent check of the quadrature: grid log-marginals vs the exact
    closed form sum_T (-1)^|T| B(a + S + sum_T d, b) / B(a, b) over subsets
    T of the vanish gaps (mpmath, 60 digits), 8 TRAIN records x 8 Beta
    shapes including U-shaped ones (a or b < 1)."""
    import itertools
    import mpmath as mp
    mp.mp.dps = 60
    recs = [{"S": r["S"], "gaps": r["gaps"]}
            for r in ec.load("records", "train")]
    rng = np.random.default_rng(0)
    idx = [int(i) for i in rng.choice(len(recs), 6, replace=False)]
    idx.append(int(np.argmax([r["S"] for r in recs])))
    idx.append(int(np.argmax([len(r["gaps"]) for r in recs])))
    thetas = [(0.97, 32.0), (0.97, 2.0), (0.999, 0.25), (0.6, 0.5),
              (0.9999, 8192.0), (0.3, 4.0), (0.95, 0.125), (0.99, 512.0)]
    g = PGrid()
    a = np.array([k * m for m, k in thetas])
    b = np.array([k * (1.0 - m) for m, k in thetas])
    LM = log_marginal(record_loglik([recs[i] for i in idx], g.logp),
                      g.masses(a, b))
    rows = []
    for ii, i in enumerate(idx):
        S = recs[i]["S"] / DAY
        ds = [d / DAY for d in recs[i]["gaps"]]
        for jj, (m, k) in enumerate(thetas):
            aa, bb = mp.mpf(k * m), mp.mpf(k * (1.0 - m))
            lb0 = mp.log(mp.beta(aa, bb))
            tot = mp.mpf(0)
            for n in range(len(ds) + 1):
                for T in itertools.combinations(ds, n):
                    x = mp.mpf(S) + sum(mp.mpf(d) for d in T)
                    tot += (-1) ** n * mp.exp(mp.log(mp.beta(aa + x, bb))
                                              - lb0)
            ex = float(mp.log(tot))
            rows.append({"record": i, "S_days": S, "n_gaps": len(ds),
                         "m": m, "kappa": k, "a": k * m, "b": k * (1 - m),
                         "grid": float(LM[ii, jj]), "exact": ex,
                         "abs_diff": abs(float(LM[ii, jj]) - ex)})
    d = np.array([r["abs_diff"] for r in rows])
    out = {"max_abs_diff": float(d.max()),
           "median_abs_diff": float(np.median(d)),
           "q95_abs_diff": float(np.quantile(d, 0.95)), "rows": rows}
    json.dump(out, open(OUT / "check_quadrature.json", "w"), indent=1)
    log(f"quadrature vs exact (log units, {len(rows)} cases): max "
        f"{d.max():.2e}, median {np.median(d):.2e}, 95% "
        f"{np.quantile(d, 0.95):.2e}")


def summary(args, log):
    """One table from results.json, results_followup.json,
    results_pflam.json and check_quadrature.json."""
    r = json.load(open(OUT / "results.json"))
    f = json.load(open(OUT / "results_followup.json"))
    p = json.load(open(OUT / "results_pflam.json"))
    q = json.load(open(OUT / "check_quadrature.json"))
    ref_cv = f["reference_cv2"]["mean_heldout_nll"]
    tab = {}
    for v, b in r["best"].items():
        row = {"opt": b["opt"], "params": b["params"],
               **{k: b[k] for k in METRICS},
               "delta_nll_vs_ref": b["nll"] - r["best"]["ref"]["nll"]}
        if v in f["per_variant"]:
            pv = f["per_variant"][v]
            row.update(
                cv2_heldout_nll=pv["cv2"]["mean_heldout_nll"],
                cv2_delta_vs_ref=pv["cv2"]["mean_heldout_nll"] - ref_cv,
                scenes_better_nll=pv["paired_vs_ref"]["nll"][
                    "n_scenes_better"],
                scenes_better_auroc=pv["paired_vs_ref"]["auroc"][
                    "n_scenes_better"])
        tab[v] = row
    tab["ref"]["cv2_heldout_nll"] = ref_cv
    out = {
        "selection": "val NLL, mean over 20 THOR val scenes",
        "reference": "HiPer v2 (hpf_nbr) over te.DET_GRID x te.KAPPA_FINAL",
        "table": tab,
        "top_variant_by_val_nll": r["top_variant"],
        "cv2_top_variant": r["cv2"]["variant"]["mean_heldout_nll"],
        "cv2_reference": r["cv2"]["reference"]["mean_heldout_nll"],
        "cv2_select_among_all_w4_variants":
            r["cv2"]["all_w4_variants"]["mean_heldout_nll"],
        "kappa_permutation_controls": f["controls_summary"],
        "ablation_by_option": f["marginals"],
        "fair_baselines": {
            "pf_lambda": {"before": p["pf_lam"], "after_static_atom":
                          p["pf_lam_atom"]},
            "pf_retrieved": {"before": r["best"]["pf_nbr"],
                             "after_ebmean": r["best"]["pf_ebmean"],
                             "after_mix": r["best"]["pf_mix"],
                             "after_mixmap": r["best"]["pf_mixmap"]}},
        "sanity": {"wrappers_reproduce_reference_nll": r["sanity_nll"],
                   "pf_lambda_w0_0_max_abs_diff":
                       p["sanity_max_abs_diff_w0"],
                   "quadrature_vs_exact_max_abs_log_diff":
                       q["max_abs_diff"],
                   "quadrature_vs_exact_median_abs_log_diff":
                       q["median_abs_diff"]}}
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, default=float)
    for v, row in tab.items():
        log(f"{v:11s} nll {row['nll']:.5f} ({row['delta_nll_vs_ref']:+.5f})"
            f" brier {row['brier']:.5f} ece {row['ece']:.5f} auroc "
            f"{row['auroc']:.5f}"
            + (f"  CV2 {row['cv2_heldout_nll']:.5f}"
               if "cv2_heldout_nll" in row else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="reference detection params only (bug finding)")
    ap.add_argument("--followup", action="store_true",
                    help="after the full run: permutation controls + "
                         "ablation by option (reads grid.jsonl)")
    ap.add_argument("--pflam", action="store_true",
                    help="fair baseline: PF(lambda) + static atom")
    ap.add_argument("--checkquad", action="store_true",
                    help="quadrature vs exact closed form (TRAIN records)")
    ap.add_argument("--summary", action="store_true",
                    help="merge the result files into summary.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / ("log_quick.txt" if args.quick else
                       "log_followup.txt" if args.followup else
                       "log_pflam.txt" if args.pflam else
                       "log_checkquad.txt" if args.checkquad else
                       "log_summary.txt" if args.summary else "log.txt"),
                "w")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if args.followup:
        followup(args, log)
        return
    if args.pflam:
        pflam(args, log)
        return
    if args.checkquad:
        checkquad(args, log)
        return
    if args.summary:
        summary(args, log)
        return
    t_start = time.time()
    meta = ec.meta()
    streams = ec.load("streams", "val")
    recs = [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
            for r in ec.load("records", "train")]
    nbr = te.NeighbourSurvivalPrior(recs)
    te.NBR["prior"] = nbr
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    objs = val_objects(streams)
    log(f"val scenes {len(streams)}, objects {len(objs)}, "
        f"train records {len(recs)}")
    priors, fit_summ = fit_priors(recs, objs, nbr, args.workers, log)
    json.dump(fit_summ, open(OUT / "fits_summary.json", "w"), indent=1)
    W.update(raw=streams, keyed=ec.keyed(streams), priors=priors)

    det = (te.grid(te.DET_GRID) if not args.quick else
           [{"p_miss": meta["ours_params"]["p_miss"],
             "p_false": meta["ours_params"]["p_false"]}])
    scenes = sorted(streams)
    results = []
    opts = [f"t{tw:g}_k{kp:g}" for tw in TAU_W for kp in K_POP]
    # sanity: the per-object wrappers reproduce the reference exactly
    # (retrieved mean, kappa 32; mixture with w0 = 0)
    d0 = {"p_miss": meta["ours_params"]["p_miss"],
          "p_false": meta["ours_params"]["p_false"]}
    W["priors"]["ret_mix"] = {k: (0.0,) + v for k, v in priors["ret"].items()}
    sink = open(OUT / ("grid_quick.jsonl" if args.quick else "grid.jsonl"),
                "w")
    caps = [{} if c is None else {"kcap": c} for c in K_CAP]
    with Pool(args.workers) as pool:
        chk = []
        run_jobs(pool, [
            ("chk_ref", "ref", None, {**d0, "kappa0": 32}),
            ("chk_beta", "beta", "ret", {**d0, "kappa0": 32}),
            ("chk_mix", "mix", "ret_mix", {**d0, "kappa0": 32})], chk, log)
        for r in chk:
            log(f"  sanity {r['variant']:9s} nll {r['nll']:.6f}")
        nll = {r["variant"]: r["nll"] for r in chk}
        sanity = {k: v for k, v in nll.items()}
        if max(nll.values()) - min(nll.values()) > 1e-9:
            log("SANITY FAILED: wrappers do not reproduce the reference")

        log("reference HiPer: DET x KAPPA_FINAL")
        run_jobs(pool, [("ref", "ref", None, {**d, "kappa0": k})
                        for d in det for k in te.KAPPA_FINAL], results, log,
                 sink)
        log("reference PF + retrieved: DET")
        run_jobs(pool, [("pf_nbr", "pfnbr", None, d) for d in det],
                 results, log, sink)
        for v, fam in (("eb_mk", "beta"), ("eb_k", "beta"), ("mix", "mix"),
                       ("mix_map", "mix")):
            log(f"{v}: DET x {opts} x kscale {C_SCALE} x kcap {K_CAP}")
            run_jobs(pool, [(v, fam, f"{v}|{o}", {**d, "kscale": c, **cp})
                            for d in det for o in opts for c in C_SCALE
                            for cp in caps],
                     results, log, sink)
        log("mix_k0: DET x KAPPA_FINAL")
        run_jobs(pool, [("mix_k0", "mix", f"mix_k0|k{k:g}", d) for d in det
                        for k in te.KAPPA_FINAL], results, log, sink)
        b_mk = best_of(results, "eb_mk")
        b_mix = best_of(results, "mix")
        b_mm = best_of(results, "mix_map")
        log(f"hpf_ebmean / pf_ebmean at eb_mk option {b_mk['opt']}; "
            f"pf_mix at {b_mix['opt']}; pf_mixmap at {b_mm['opt']}")
        run_jobs(pool, [("hpf_ebmean", "beta", b_mk["opt"],
                         {**d, "kappa0": k})
                        for d in det for k in te.KAPPA_FINAL], results, log,
                 sink)
        run_jobs(pool, [("pf_ebmean", "beta_pf", b_mk["opt"], d)
                        for d in det], results, log, sink)
        run_jobs(pool, [("pf_mix", "mix_pf", b_mix["opt"], d)
                        for d in det], results, log, sink)
        run_jobs(pool, [("pf_mixmap", "mix_pf", b_mm["opt"], d)
                        for d in det], results, log, sink)
    sink.close()
    variants = ["ref", "pf_nbr", "eb_mk", "eb_k", "mix", "mix_map", "mix_k0",
                "hpf_ebmean", "pf_ebmean", "pf_mix", "pf_mixmap"]
    best = {}
    for v in variants:
        b = best_of(results, v)
        best[v] = {"opt": b["opt"], "params": b["params"],
                   **{k: b[k] for k in METRICS}}
        log(f"{v:11s} nll {b['nll']:.5f} brier {b['brier']:.5f} ece "
            f"{b['ece']:.5f} auroc {b['auroc']:.5f} opt {b['opt']} "
            f"{b['params']}")
    mine = ["eb_mk", "eb_k", "mix", "mix_map", "mix_k0"]
    top = min(mine, key=lambda v: best[v]["nll"])
    cv = {"best_variant": top,
          "variant": cv2(results, [top], scenes),
          "reference": cv2(results, ["ref"], scenes),
          "all_w4_variants": cv2(results, mine, scenes),
          "folds": {"A": scenes[0::2], "B": scenes[1::2]}}
    log(f"CV2 {top}: {cv['variant']['mean_heldout_nll']:.5f}  reference: "
        f"{cv['reference']['mean_heldout_nll']:.5f}  all w4 variants: "
        f"{cv['all_w4_variants']['mean_heldout_nll']:.5f}")
    # official harness scoring (pm.summarize) of the selected configs
    official = {}
    for v in ("ref", top, "pf_nbr", "pf_ebmean", "pf_mix", "pf_mixmap"):
        b = best_of(results, v)
        fam = b["family"]
        stv = W["raw"] if fam in ("ref", "pfnbr") else W["keyed"]
        official[v] = ec.run_model(stv, lambda sc, f=fam, b=b: make(
            f, b["params"], b["opt"]), v)
        log(f"official {v:10s} " + " ".join(
            f"{k} {official[v][k]:.5f}" for k in METRICS))
    res = {"idea": "w4_ebconc", "quick": args.quick,
           "sanity_nll": sanity, "best": best, "top_variant": top,
           "delta_nll_top_vs_ref": best[top]["nll"] - best["ref"]["nll"],
           "cv2": cv, "official": official,
           "paired_top_vs_ref": paired(results, top, "ref"),
           "paired_pf_ebmean_vs_pf_nbr": paired(results, "pf_ebmean",
                                                "pf_nbr"),
           "fits_summary": fit_summ,
           "fair_baselines": [
               {"name": "PF+retrieved, EB-fitted mean (pf_ebmean)",
                "val_nll_before": best["pf_nbr"]["nll"],
                "val_nll_after": best["pf_ebmean"]["nll"]},
               {"name": "PF+retrieved, static atom + frozen Beta (pf_mix)",
                "val_nll_before": best["pf_nbr"]["nll"],
                "val_nll_after": best["pf_mix"]["nll"]},
               {"name": "PF+retrieved, static atom + frozen MAP-mean Beta "
                        "(pf_mixmap)",
                "val_nll_before": best["pf_nbr"]["nll"],
                "val_nll_after": best["pf_mixmap"]["nll"]}],
           "seconds": round(time.time() - t_start, 1)}
    json.dump(res, open(OUT / ("results_quick.json" if args.quick
                               else "results.json"), "w"),
              indent=1, default=float)
    log(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
