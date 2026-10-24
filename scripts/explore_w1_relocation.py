#!/usr/bin/env python3
"""
scripts/explore_w1_relocation.py
================================
VALIDATION-ONLY exploration, idea w1: relocation handling in HiPer.
Never reads test / test2 / test3 / obchange. Writes only under
RESULTS/explore_cache/w1_relocation/.

Reference (HiPer v2): after a RELOCATE event the filter sets pi = pi_new = 1
and adds b += pi (the old place counts as one vanish); (a, b) stay with the
object. Declared variants (a subclass of the filter, RelocHPF):

 (a) pi_reloc     presence right after a relocation, in PI_RELOC
 (b) rate update  w_v          weight of the vanish count, b += w_v * pi
                  kappa_reloc  cap on the Beta concentration a + b at a
                               relocation (mean kept): a moved object belongs
                               to the mobile population, whose rate spread is
                               wider than the global kappa0 assumes
                  h_mult       hazard multiplier for objects that have
                               relocated at least once (survival exponent x
                               h_mult for every later gap)
 (c) identity     q = P(correct identity | score, ln d2), logistic regression
                  fitted on TRAIN + TRAIN2 association rows with
                  relocated_pred (label: matched_gt_id == gt_id). The rate
                  belief becomes the moment-matched Beta of
                    q * same-object update + (1 - q) * new-object update,
                  new object = the map entry's own prior mean m0 at
                  concentration min(kappa0, kappa_reloc), plus (mix1) or
                  without (mix0) the vanish count.

Every variant is re-tuned over te.DET_GRID x te.KAPPA_FINAL (HiPer) with the
retrieved-prior parameters fixed at the frozen run (K, tau, k_nbr). The
components (a) and the h_mult part of (b) also apply to fixed-rate filters,
so PF(lambda) (DET x LIFETIME grid) and PF+retrieved (DET grid) are re-tuned
with PI_RELOC x H_MULT as well (w_v, kappa_reloc and (c) act on the learned
rate only and are no-ops there).

Selection criterion: val NLL, mean over scenes (te._scores). The tuning loop
uses a lean replay (same event order, same presence query as
lifelong_objects.replay, NLL only); the selected configs are re-scored with
the full replay + explore_common.scores, and the lean NLL is checked against
it.

  PYTHONDONTWRITEBYTECODE=1 python scripts/explore_w1_relocation.py [--smoke]
      val grids, ablations, 2-fold check -> results.json, grid_*.npz
  ... --train-stats    relocation statistics on TRAIN/TRAIN2 labels
                       -> train_stats.json (motivates the grids)
  ... --curves         per-knob val NLL curves from grid_*.npz -> curves.json
  ... --train2-check   val-selected configs replayed on TRAIN2 (never tuned
                       on) -> results_train2_check.json
  ... --summary        sign tests, effect at the reference's inner params,
                       HiPer margins before/after -> summary.json
"""

import argparse
import functools
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HIT, MISS, NONE, HierarchicalPersistenceFilter, _HPFState)
from hiper.evaluation import lifelong_objects as lo  # noqa: E402

OUT = ec.CACHE / "w1_relocation"
DAY = 86400.0

# ── Declared grids (fixed before the val run) ───────────────────────────────
PI_RELOC = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]          # (a)
W_V = [1.0, 2.0, 4.0, 8.0]                          # (b) vanish weight
# (b) None = no cap. For scale: the generator's lognormal hazard (sigma 1)
# implies a Beta concentration ~6 for pickupable and ~24 for moveable
# survival over one day (design knowledge, not val labels).
KAPPA_RELOC = [None, 4.0, 8.0, 16.0]
# (b) moved-object hazard. 5.0 added 2026-10-20 (resume, before the full val
# run; only a 3-scene smoke had been seen): on TRAIN / TRAIN2 labels the
# hazard after a relocation is 4-5.6x that of never-relocated hits
# (--train-stats, train_stats.json), so 3.0 could be a grid edge.
H_MULT = [1.0, 1.5, 2.0, 3.0, 5.0]
C_MODE = ["off", "mix0", "mix1"]                    # (c)
REF_X = dict(pi_reloc=1.0, w_v=1.0, kappa_reloc=None, h_mult=1.0,
             c_mode="off")
COMP = {"a": ("pi_reloc",), "b": ("w_v", "kappa_reloc", "h_mult"),
        "c": ("c_mode",)}

Q = {"coef": None}          # logistic identity model (w0, w_score, w_lnd2)
G = {}                      # worker globals: prepared streams


# ── Filter variant ───────────────────────────────────────────────────────────

def beta_mix(q, a1, b1, a2, b2):
    """Moment-matched Beta of q Beta(a1, b1) + (1 - q) Beta(a2, b2)."""
    s1, s2 = a1 + b1, a2 + b2
    m1, m2 = a1 / s1, a2 / s2
    v1 = m1 * (1.0 - m1) / (s1 + 1.0)
    v2 = m2 * (1.0 - m2) / (s2 + 1.0)
    mu = q * m1 + (1.0 - q) * m2
    e2 = q * (v1 + m1 * m1) + (1.0 - q) * (v2 + m2 * m2)
    var = max(e2 - mu * mu, 1e-12)
    kap = max(mu * (1.0 - mu) / var - 1.0, 1e-6)
    return mu * kap, (1.0 - mu) * kap


class RelocHPF(HierarchicalPersistenceFilter):
    """HierarchicalPersistenceFilter with the declared relocation variants.
    With REF_X it reproduces the parent exactly (checked in main)."""

    wants_reloc_info = True

    def __init__(self, pi_reloc=1.0, w_v=1.0, kappa_reloc=None, h_mult=1.0,
                 c_mode="off", q_coef=None, prior_lam=None, **kw):
        super().__init__(**kw)
        # {oid: rate_fn(emb)} precomputed once per scene (speed only; the
        # NBR params are fixed across the grid, checked against rate_fn)
        self.prior_lam = prior_lam
        self.pi_reloc = float(pi_reloc)
        self.w_v = float(w_v)
        self.kappa_reloc = None if kappa_reloc is None else float(kappa_reloc)
        self.h_mult = float(h_mult)
        self.c_mode = c_mode
        self.q_coef = q_coef
        if c_mode != "off" and q_coef is None:
            raise ValueError("c_mode needs the identity model")
        self._moved = set()

    def create(self, oid, class_label, t, emb=None):
        if self.prior_lam is None or self.rate_fn is None:
            return super().create(oid, class_label, t, emb=emb)
        lam = self.prior_lam.get(oid) if emb is not None else None
        own = None
        if lam is not None:                 # == parent with rate_fn(emb)
            own = math.exp(-float(lam) * self.dt_ref)
            own = min(max(own, self.m_clip), 1.0 - self.m_clip)
        m = own if own is not None else self.prior_mean(class_label)
        self._s[oid] = _HPFState(class_label=class_label, pi=self.pi_new,
                                 a=self.kappa0 * m,
                                 b=self.kappa0 * (1.0 - m), t=float(t),
                                 m=own)

    def _gx(self, oid):
        g = self.env_gamma()
        if self.h_mult != 1.0 and oid in self._moved:
            return g * self.h_mult
        return g

    def q_identity(self, score, d2):
        w0, ws, wd = self.q_coef
        z = w0 + ws * float(score) + wd * math.log(max(float(d2), 1e-12))
        return 1.0 / (1.0 + math.exp(-z))

    # parent.observe with the exponent multiplier g -> _gx(oid)
    def observe(self, oid, t, evidence, eps_l=0.0, **info):
        st = self._s[oid]
        if evidence == NONE:
            return
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        g = self._gx(oid)
        S = self._survival(st, g * r)
        if evidence == HIT:
            L1, L0 = 1.0 - self.p_miss, self.p_false
            w = 1.0 - min(max(float(eps_l), 0.0), 1.0)
            L1, L0 = w * L1 + (1.0 - w), w * L0 + (1.0 - w)
        elif evidence == MISS:
            L1, L0 = self.p_miss, 1.0 - self.p_false
        else:
            raise ValueError(f"unknown evidence {evidence!r}")
        pi = st.pi
        w11 = pi * S * L1
        w10 = pi * (1.0 - S) * L0
        w01 = (1.0 - pi) * self.rho * L1
        w00 = (1.0 - pi) * (1.0 - self.rho) * L0
        z = w11 + w10 + w01 + w00
        if z <= 0.0:
            return
        w11, w10, w01, w00 = w11 / z, w10 / z, w01 / z, w00 / z
        if self.env_alpha is not None:
            self._env_expect(st, r)
            self._env_O += w10
        if self.learn_rate:
            st.a += g * r * w11
            st.b += w10
        st.pi = w11 + w01
        st.t = float(t)
        st.last_evidence = evidence

    def relocate(self, oid, t, score=None, d2=None):
        st = self._s[oid]
        if self.env_alpha is not None:
            self._env_expect(st, max(0.0, (float(t) - st.t) / self.dt_ref))
            self._env_O += st.pi
        if self.relocation_resets_rate:
            m = self._mean(st)
            st.a, st.b = self.kappa0 * m, self.kappa0 * (1.0 - m)
        elif self.learn_rate:
            a1, b1 = st.a, st.b
            kc = self.kappa_reloc
            if kc is not None and a1 + b1 > kc:          # (b) concentration
                s = kc / (a1 + b1)
                a1, b1 = a1 * s, b1 * s
            b1 += self.w_v * st.pi                         # (b) vanish weight
            if self.c_mode != "off" and score is not None and d2 is not None:
                q = self.q_identity(score, d2)             # (c) identity
                m0 = self._mean(st)
                k0 = self.kappa0 if kc is None else min(self.kappa0, kc)
                a2, b2 = k0 * m0, k0 * (1.0 - m0)
                if self.c_mode == "mix1":
                    b2 += self.w_v * st.pi
                a1, b1 = beta_mix(q, a1, b1, a2, b2)
            st.a, st.b = a1, b1
        st.pi = self.pi_reloc                              # (a)
        st.t = float(t)
        st.last_evidence = HIT
        self._moved.add(oid)                               # (b) h_mult

    def presence(self, oid, t):
        st = self._s[oid]
        r = max(0.0, (float(t) - st.t) / self.dt_ref) * self._gx(oid)
        return st.pi * self._survival(st, r) + (1.0 - st.pi) * self.rho

    def remove(self, oid):
        super().remove(oid)
        self._moved.discard(oid)


def prior_lams(streams, K, tau, k):
    """{scene: {oid: rate_fn(emb of the CREATE event)}} for fixed NBR params."""
    nbr = te.NBR["prior"]
    out = {}
    for sc, st in streams.items():
        d = {}
        for oid, ob in st["objects"].items():
            for ev in ob["events"]:
                if ev["ev"] == lo.CREATE and ev.get("emb") is not None:
                    d[oid] = float(nbr.rate(ev["emb"], K=K, tau=tau, k=k))
        out[sc] = d
    return out


def make_x(family, params, x, scene=None):
    """te.make_model for 'hpf_nbr', 'pf_nbr', 'pf' with RelocHPF."""
    p = dict(params)
    kw = {**x, "q_coef": Q["coef"]}
    if family in ("hpf_nbr", "pf_nbr"):
        nbr = te.NBR["prior"]
        K, tau, k = p.pop("K"), p.pop("tau"), p.pop("k_nbr")
        fn = functools.partial(nbr.rate, K=K, tau=tau, k=k)
        pl = (G.get("lam", {}).get((K, tau, k), {}).get(scene)
              if scene is not None else None)
        return RelocHPF(default_lambda=nbr.lam0, use_class_prior=False,
                        rate_fn=fn, learn_rate=(family == "hpf_nbr"),
                        prior_lam=pl, **p, **kw)
    if family == "pf":
        days = p.pop("lifetime_days")
        return RelocHPF(default_lambda=1.0 / (days * DAY), learn_rate=False,
                        use_class_prior=False, **p, **kw)
    raise ValueError(family)


# ── Replay ───────────────────────────────────────────────────────────────────

def apply_x(model, oid, label, ev):
    """lifelong_objects.apply_event, but a RELOCATE passes score / d2 to
    models that use them."""
    if ev["ev"] == lo.RELOCATE and getattr(model, "wants_reloc_info", False):
        model.relocate(oid, ev["t"], score=ev.get("score"), d2=ev.get("d2"))
    else:
        lo.apply_event(model, oid, label, ev)


def _per_k(stream):
    per_k = [[] for _ in stream["times"]]
    for oid, ob in stream["objects"].items():
        for ev in ob["events"]:
            per_k[ev["k"]].append((oid, ob["class_label"], ev))
    return per_k


def replay_x(stream, model, name):
    """lifelong_objects.replay with apply_x (full prediction rows)."""
    times, gt = stream["times"], stream.get("gt")
    base = {"scene": stream["scene"], "split": stream["split"],
            "model": name}
    rows = []
    for k, items in enumerate(_per_k(stream)):
        for oid, label, ev in items:
            apply_x(model, oid, label, ev)
        for oid, label, ev in items:
            rows.append(lo._prediction_row(base, model, oid, label, ev, k,
                                           times, gt))
    return rows


def prepare(streams):
    """Per scene: events per session and the labelled queries (oid, y)."""
    P = {}
    for sc in sorted(streams):
        st = streams[sc]
        times, gt = st["times"], st["gt"]
        per_k = _per_k(st)
        lab = []
        for k, items in enumerate(per_k):
            L = []
            for oid, _, ev in items:
                _, nxt = lo.gt_labels(gt, ev["gt_id"], ev["pos_session"], k)
                if nxt is not None:
                    L.append((oid, float(nxt)))
            lab.append(L)
        P[sc] = (times, per_k, lab)
    return P


def lean_nll(prep_sc, model):
    """NLL of one scene, identical to pm.nll on replay rows (the event
    dispatch is apply_x inlined; the filter ignores the extra info)."""
    times, per_k, lab = prep_sc
    rx = getattr(model, "wants_reloc_info", False)
    we = getattr(model, "wants_embedding", False)
    ps, ys = [], []
    for k, items in enumerate(per_k):
        for oid, label, ev in items:
            e, t = ev["ev"], ev["t"]
            if e == HIT:
                model.observe(oid, t, HIT, eps_l=ev["eps_l"])
            elif e == MISS:
                model.observe(oid, t, MISS)
            elif e == lo.CREATE:
                if we:
                    model.create(oid, label, t, emb=ev.get("emb"))
                else:
                    model.create(oid, label, t)
                model.observe(oid, t, HIT, eps_l=ev["eps_l"])
            elif e == lo.RELOCATE:
                if rx:
                    model.relocate(oid, t, score=ev.get("score"),
                                   d2=ev.get("d2"))
                else:
                    model.relocate(oid, t)
            else:
                model.observe(oid, t, NONE)
        if lab[k]:
            tn = times[k + 1]
            for oid, y in lab[k]:
                ps.append(model.presence(oid, tn))
                ys.append(y)
    p = np.clip(np.asarray(ps, np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(ys, np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


# ── Vectorised replay: one variant x, all inner configs at once ─────────────
# The event times are the same for every config, so each object's state is
# (pi, a, b) as arrays over the inner configs plus a shared (t, moved, m).
# Checked against the scalar RelocHPF (itself checked against the official
# replay) on a random sample of configs in main.

M_CLIP, DT_REF = 1e-4, 86400.0


def _clipm(v):
    return np.minimum(np.maximum(v, M_CLIP), 1.0 - M_CLIP)


def build_spec(name, inner, lam_tables):
    """Arrays of the inner grid for vec_nll (filter defaults otherwise)."""
    for p in inner:
        assert not set(p) & {"dt_ref", "rho", "pi_new", "m_clip",
                             "env_alpha", "relocation_resets_rate"}
    V = len(inner)
    spec = {"p_miss": np.array([float(p["p_miss"]) for p in inner]),
            "p_false": np.array([float(p["p_false"]) for p in inner])}
    nbr = te.NBR["prior"]
    if name in ("hiper", "pf_nbr"):
        key = (inner[0]["K"], inner[0]["tau"], inner[0]["k_nbr"])
        assert all((p["K"], p["tau"], p["k_nbr"]) == key for p in inner)
        spec["lam"] = lam_tables[key]
        spec["m_default"] = float(_clipm(math.exp(-nbr.lam0 * DT_REF)))
    else:
        spec["lam"] = None
        spec["m_default"] = _clipm(np.array(          # == prior_mean()
            [math.exp(-(1.0 / (float(p["lifetime_days"]) * DAY)) * DT_REF)
             for p in inner]))
    spec["learn"] = name == "hiper"
    spec["kappa0"] = (np.array([float(p["kappa0"]) for p in inner])
                      if name == "hiper" else np.full(V, 4.0))
    return spec


def _vsurv(st, r, learn):
    if r <= 0.0:
        return 1.0
    if not learn:
        return st[5] ** r
    a, b = st[1], st[2]
    return np.exp(gammaln(a + r) + gammaln(a + b) - gammaln(a)
                  - gammaln(a + b + r))


def vec_nll(prep_sc, spec, x, scene, q_coef):
    """Per-config NLL (V,) of one scene for variant x (== lean_nll of
    RelocHPF for every inner config)."""
    times, per_k, lab = prep_sc
    pmiss, pfalse, k0 = spec["p_miss"], spec["p_false"], spec["kappa0"]
    learn, mdef = spec["learn"], spec["m_default"]
    lamd = spec["lam"][scene] if spec["lam"] is not None else None
    V = len(pmiss)
    pi_reloc, w_v = float(x["pi_reloc"]), float(x["w_v"])
    kc = None if x["kappa_reloc"] is None else float(x["kappa_reloc"])
    hm, cmode = float(x["h_mult"]), x["c_mode"]
    S = {}       # oid -> [pi, a, b, t, moved, m]

    def observe(st, t, hit, eps_l):
        r = max(0.0, (float(t) - st[3]) / DT_REF)
        g = hm if (hm != 1.0 and st[4]) else 1.0
        Sv = _vsurv(st, g * r, learn)
        if hit:
            w = 1.0 - min(max(float(eps_l), 0.0), 1.0)
            L1 = w * (1.0 - pmiss) + (1.0 - w)
            L0 = w * pfalse + (1.0 - w)
        else:
            L1, L0 = pmiss, 1.0 - pfalse
        pi = st[0]
        w11 = pi * Sv * L1
        w10 = pi * (1.0 - Sv) * L0
        w00 = (1.0 - pi) * L0                    # rho = 0: w01 = 0
        z = w11 + w10 + w00
        if not np.all(z > 0.0):
            raise FloatingPointError("z <= 0: per-config t would diverge")
        w11, w10 = w11 / z, w10 / z
        if learn:
            st[1] = st[1] + g * r * w11
            st[2] = st[2] + w10
        st[0] = w11
        st[3] = float(t)

    tot, n = np.zeros(V), 0
    for k, items in enumerate(per_k):
        for oid, label, ev in items:
            e, t = ev["ev"], ev["t"]
            if e == HIT:
                observe(S[oid], t, True, ev["eps_l"])
            elif e == MISS:
                observe(S[oid], t, False, 0.0)
            elif e == lo.CREATE:
                own = None
                if lamd is not None and ev.get("emb") is not None:
                    lam = lamd.get(oid)
                    if lam is not None:
                        own = min(max(math.exp(-float(lam) * DT_REF),
                                      M_CLIP), 1.0 - M_CLIP)
                m = own if own is not None else mdef
                S[oid] = [np.ones(V), k0 * m, k0 * (1.0 - m), float(t),
                          False, m]
                observe(S[oid], t, True, ev["eps_l"])
            elif e == lo.RELOCATE:
                st = S[oid]
                if learn:
                    a1, b1 = st[1], st[2]
                    if kc is not None:
                        s = np.where(a1 + b1 > kc, kc / (a1 + b1), 1.0)
                        a1, b1 = a1 * s, b1 * s
                    b1 = b1 + w_v * st[0]
                    if cmode != "off":
                        w0, ws, wd = q_coef
                        zq = (w0 + ws * float(ev["score"]) + wd *
                              math.log(max(float(ev["d2"]), 1e-12)))
                        q = 1.0 / (1.0 + math.exp(-zq))
                        m0 = st[5]
                        kk = k0 if kc is None else np.minimum(k0, kc)
                        a2, b2 = kk * m0, kk * (1.0 - m0)
                        if cmode == "mix1":
                            b2 = b2 + w_v * st[0]
                        s1, s2 = a1 + b1, a2 + b2
                        m1, m2 = a1 / s1, a2 / s2
                        v1 = m1 * (1.0 - m1) / (s1 + 1.0)
                        v2 = m2 * (1.0 - m2) / (s2 + 1.0)
                        mu = q * m1 + (1.0 - q) * m2
                        e2 = q * (v1 + m1 * m1) + (1.0 - q) * (v2 + m2 * m2)
                        var = np.maximum(e2 - mu * mu, 1e-12)
                        kap = np.maximum(mu * (1.0 - mu) / var - 1.0, 1e-6)
                        a1, b1 = mu * kap, (1.0 - mu) * kap
                    st[1], st[2] = a1, b1
                st[0] = np.full(V, pi_reloc)
                st[3] = float(t)
                st[4] = True
            # NONE: no update
        if lab[k]:
            tn = times[k + 1]
            for oid, y in lab[k]:
                st = S[oid]
                g = hm if (hm != 1.0 and st[4]) else 1.0
                r = max(0.0, (float(tn) - st[3]) / DT_REF) * g
                p = st[0] * _vsurv(st, r, learn)
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                tot += -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
                n += 1
    return tot / n


def _init(prep, coef, lam=None, spec=None):
    G["prep"] = prep
    G["lam"] = lam or {}
    G["spec"] = spec or {}
    Q["coef"] = coef


def _vjob(args):
    xi, name, x = args
    P = G["prep"]
    return xi, np.stack([vec_nll(P[sc], G["spec"][name], x, sc, Q["coef"])
                         for sc in sorted(P)], axis=1)


def _job(args):
    i, family, params, x = args
    P = G["prep"]
    return i, [lean_nll(P[sc], make_x(family, params, x, scene=sc))
               for sc in sorted(P)]


# ── Identity model (TRAIN + TRAIN2 association rows only) ────────────────────

def reloc_rows(split):
    rows = [r for r in ec.load("assoc", split)
            if r["relocated_pred"] and r["matched_oid"] is not None
            and not r["tentative"]]
    X = np.array([[r["score"], math.log(max(r["d2"], 1e-12))] for r in rows])
    y = np.array([r["matched_gt_id"] == r["gt_id"] for r in rows], float)
    return X, y


def fit_identity():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    D = {sp: reloc_rows(sp) for sp in ("train", "train2")}
    cross = {}
    for fit_on, test_on in (("train", "train2"), ("train2", "train")):
        m = LogisticRegression(C=1e4, max_iter=2000).fit(*D[fit_on])
        X, y = D[test_on]
        q = m.predict_proba(X)[:, 1]
        base = np.full_like(y, D[fit_on][1].mean())
        cross[f"fit_{fit_on}_test_{test_on}"] = {
            "n_test": int(len(y)), "logloss": float(log_loss(y, q)),
            "logloss_const": float(log_loss(y, base)),
            "auroc": float(roc_auc_score(y, q))}
    X = np.vstack([D["train"][0], D["train2"][0]])
    y = np.concatenate([D["train"][1], D["train2"][1]])
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    coef = (float(m.intercept_[0]), float(m.coef_[0][0]),
            float(m.coef_[0][1]))
    q = m.predict_proba(X)[:, 1]
    return coef, {"features": ["score", "ln_d2"], "coef_w0_wscore_wlnd2":
                  coef, "n_rows": int(len(y)),
                  "p_correct_train": float(y.mean()),
                  "q_quantiles_train": [float(v) for v in np.quantile(
                      q, [0, .1, .25, .5, .75, .9, 1])],
                  "cross_split_check": cross}


# ── Grids ────────────────────────────────────────────────────────────────────

def x_grid(fixed_rate):
    """Variant grid; fixed-rate filters only see pi_reloc and h_mult."""
    if fixed_rate:
        return [{**REF_X, "pi_reloc": a, "h_mult": h}
                for a in PI_RELOC for h in H_MULT]
    return [dict(pi_reloc=a, w_v=w, kappa_reloc=kc, h_mult=h, c_mode=c)
            for a in PI_RELOC for w in W_V for kc in KAPPA_RELOC
            for h in H_MULT for c in C_MODE]


def family_grid(name, m):
    det = te.grid(te.DET_GRID)
    if name == "hiper":
        base = {k: v for k, v in m["ours_params"].items()
                if k not in ("p_miss", "p_false", "kappa0")}
        inner = [{**base, **d, "kappa0": k} for d in det
                 for k in te.KAPPA_FINAL]
        return "hpf_nbr", inner, x_grid(False)
    if name == "pf":
        inner = [{"tau_pi": m["pf_params"]["tau_pi"], **d,
                  "lifetime_days": L} for d in det
                 for L in te.LIFETIME_GRID_DAYS]
        return "pf", inner, x_grid(True)
    if name == "pf_nbr":
        base = {k: v for k, v in m["pf_nbr_params"].items()
                if k not in ("p_miss", "p_false")}
        return "pf_nbr", [{**base, **d} for d in det], x_grid(True)
    raise ValueError(name)


def _xkey(x):
    return json.dumps(x, sort_keys=True)


def run_grid(pool, name, inner, xs, log, ckpt=None, resume=False):
    """Per-scene NLL of every (variant, inner) config, variant-major.
    ``ckpt``: partial-results file, saved every 100 variants; with
    ``resume`` it is reused when the variant and inner lists are
    identical (main deletes it after a complete run)."""
    V = len(inner)
    cfg = [(x, p) for x, p in itertools.product(xs, inner)]
    out = np.full((len(cfg), len(G["prep"])), np.nan)
    done = np.zeros(len(xs), bool)
    xk = np.array([_xkey(x) for x in xs])
    ik = np.array([_xkey(p) for p in inner])
    if resume and ckpt is not None and ckpt.exists():
        z = np.load(ckpt)
        if (z["variant"].shape == xk.shape and (z["variant"] == xk).all()
                and z["inner"].shape == ik.shape and (z["inner"] == ik).all()
                and z["nll"].shape == out.shape):
            out[:], done[:] = z["nll"], z["done"]
            log(f"  {name}: resumed {int(done.sum())}/{len(xs)} variants "
                f"from {ckpt.name}")

    def save():
        if ckpt is not None:
            np.savez(ckpt, nll=out, done=done, variant=xk, inner=ik)

    t0 = time.time()
    jobs = [(i, name, x) for i, x in enumerate(xs) if not done[i]]
    for n, (i, v) in enumerate(pool.imap_unordered(_vjob, jobs)):
        out[i * V:(i + 1) * V] = v
        done[i] = True
        if (n + 1) % 100 == 0:
            log(f"  {name}: {n + 1}/{len(jobs)} variants "
                f"{time.time() - t0:.0f}s")
            save()
    assert done.all() and np.isfinite(out).all()
    save()
    log(f"  {name}: {len(cfg)} configs in {time.time() - t0:.0f}s")
    return cfg, out


def check_vec(name, family, inner, xs, n, rng, log):
    """Vectorised vs scalar RelocHPF lean NLL on n random configs."""
    P = G["prep"]
    worst = 0.0
    picks = [xs[0], xs[-1]] + [xs[i] for i in rng.choice(len(xs),
                                                         size=n, replace=True)]
    for x in picks:
        vi = int(rng.integers(len(inner)))
        vec = np.mean([vec_nll(P[sc], G["spec"][name], x, sc, Q["coef"])[vi]
                       for sc in sorted(P)])
        sca = np.mean(_job((0, family, inner[vi], x))[1])
        worst = max(worst, abs(vec - sca))
    log(f"check vectorised vs scalar {name}: {len(picks)} configs, "
        f"max |diff| {worst:.2e}")
    assert worst < 1e-9, worst
    return worst


def _sel(cfg, nll, mask, cols=None):
    """Index of the best config among mask (mean NLL over cols)."""
    v = nll[:, cols].mean(1) if cols is not None else nll.mean(1)
    idx = np.flatnonzero(mask)
    return int(idx[np.argmin(v[idx])])


def _mask(cfg, free):
    """Configs whose variant params outside ``free`` are at REF_X."""
    return np.array([all(x[k] == REF_X[k] for k in REF_X if k not in free)
                     for x, _ in cfg])


def two_fold(cfg, nll, mask, scenes):
    A = [i for i in range(len(scenes)) if i % 2 == 0]
    B = [i for i in range(len(scenes)) if i % 2 == 1]
    ia, ib = _sel(cfg, nll, mask, A), _sel(cfg, nll, mask, B)
    held_b = float(nll[ia, B].mean())     # selected on A, scored on B
    held_a = float(nll[ib, A].mean())
    return {"heldout_mean": 0.5 * (held_a + held_b),
            "fold_B_nll_params_from_A": held_b,
            "fold_A_nll_params_from_B": held_a,
            "params_from_A": {**cfg[ia][0], **cfg[ia][1]},
            "params_from_B": {**cfg[ib][0], **cfg[ib][1]}}


def full_scores(streams, family, params, x, name):
    rows = []
    for sc in sorted(streams):
        rows += replay_x(streams[sc], make_x(family, params, x), name)
    return ec.scores(rows), rows


def breakdown(rows, streams):
    """Val NLL share by the event applied at k, and for rows after an
    object's first relocation (scoring diagnostic)."""
    first_rel = {}
    for sc, st in streams.items():
        for oid, ob in st["objects"].items():
            ks = [ev["k"] for ev in ob["events"] if ev["ev"] == "relocate"]
            if ks:
                first_rel[(sc, oid)] = min(ks)
    acc = defaultdict(lambda: [0, 0.0])
    tot = 0.0
    for r in rows:
        if r["label_next"] is None:
            continue
        p = min(max(float(r["pred_presence"]), 1e-6), 1 - 1e-6)
        y = float(r["label_next"])
        l = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        tot += l
        fr = first_rel.get((r["scene"], r["oid"]))
        grp = ("relocate_event" if r["last_evidence"] == "relocate" else
               "after_relocation" if fr is not None and r["k"] > fr
               else "never_relocated_yet")
        for key in (f"{grp}|y={int(y)}", grp):
            acc[key][0] += 1
            acc[key][1] += l
    return {k: {"n": v[0], "nll_sum": v[1], "mean": v[1] / v[0]}
            for k, v in sorted(acc.items())} | {"total_nll_sum": tot}


# ── TRAIN statistics (training labels only; motivates the grids) ────────────

def train_stats(out):
    """Next-session survival after relocations on the TRAIN / TRAIN2 streams
    (training labels are fair to learn from), and the exponential hazard of
    each group (priors._map_rate, interval-censored MLE). Val: event counts
    only (observable fields), no labels."""
    from hiper.ephemerality.priors import _map_rate
    coef, _ = fit_identity()
    Q["coef"] = coef
    probe = RelocHPF(c_mode="mix0", q_coef=coef)
    res = {"identity_coef_train_train2": coef, "splits": {}}
    for split in ("val", "train", "train2"):
        S = ec.load("streams", split)
        cnt = defaultdict(int)
        n_obj = n_rel_obj = 0
        for st in S.values():
            for ob in st["objects"].values():
                n_obj += 1
                n_rel_obj += any(e["ev"] == "relocate" for e in ob["events"])
                for e in ob["events"]:
                    cnt[e["ev"]] += 1
        rec = {"n_scenes": len(S), "n_objects": n_obj,
               "n_objects_relocated": int(n_rel_obj), "events": dict(cnt)}
        if split != "val":
            grp = defaultdict(list)            # group -> [(gap s, y)]
            for st in S.values():
                gt, times = st["gt"], st["times"]
                for ob in st["objects"].values():
                    prev, nrel = None, 0
                    for e in ob["events"]:
                        k = e["k"]
                        _, y = lo.gt_labels(gt, e["gt_id"], e["pos_session"],
                                            k)
                        if e["ev"] == "relocate":
                            nrel += 1
                        if y is not None and e["ev"] != "none":
                            d = times[k + 1] - times[k]
                            tags = [e["ev"] + ("|after_reloc" if nrel else
                                               "|never_reloc")]
                            if e["ev"] == "relocate":
                                ok = prev is not None and prev == e["gt_id"]
                                q = probe.q_identity(e["score"], e["d2"])
                                tags += ["relocate|id_" + ("ok" if ok
                                                           else "wrong"),
                                         "relocate|q<0.5" if q < 0.5
                                         else "relocate|q>=0.5"]
                            for tg in tags:
                                grp[tg].append((d, float(y)))
                        if e["ev"] in ("create", "hit", "relocate"):
                            prev = e["gt_id"]
            g = {}
            for tg, v in sorted(grp.items()):
                d = np.array([a for a, _ in v])
                y = np.array([b for _, b in v])
                surv = float(d[y == 1].sum())
                lam = (_map_rate(surv, d[y == 0], 0.0, 0.0) if surv > 0
                       else float("inf"))
                g[tg] = {"n": int(len(y)), "p_stay_next": float(y.mean()),
                         "hazard_per_day": float(lam * DAY)}
            base = g["hit|never_reloc"]["hazard_per_day"]
            for tg in g:
                g[tg]["hazard_ratio_vs_hit_never_reloc"] = (
                    g[tg]["hazard_per_day"] / base)
            rec["groups"] = g
            # identity label sanity: relocation rows whose GT ids are None
            rows = [r for r in ec.load("assoc", split)
                    if r["relocated_pred"] and r["matched_oid"] is not None
                    and not r["tentative"]]
            rec["assoc_reloc_rows"] = {
                "n": len(rows),
                "gt_id_none": sum(r["gt_id"] is None for r in rows),
                "matched_gt_id_none": sum(r["matched_gt_id"] is None
                                          for r in rows),
                "p_correct": float(np.mean([r["matched_gt_id"] == r["gt_id"]
                                            for r in rows]))}
        res["splits"][split] = rec
        print(split, json.dumps(rec, indent=1), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_stats.json", "w") as f:
        json.dump(res, f, indent=1, default=float)


# ── Independent check on TRAIN2 (never tuned on) ─────────────────────────────

def train2_check(dest):
    """Frozen val-selected configs (results.json) replayed on the TRAIN2
    streams: prior from TRAIN records (as on val), identity model refitted
    on TRAIN rows only so TRAIN2 stays out of sample for (c)."""
    res = json.load(open(dest / "results.json"))
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in ec.load("records", "train")])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    from sklearn.linear_model import LogisticRegression
    Q["coef"] = None
    X, y = reloc_rows("train")
    mdl = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    Q["coef"] = (float(mdl.intercept_[0]), float(mdl.coef_[0][0]),
                 float(mdl.coef_[0][1]))
    streams = ec.load("streams", "train2")
    out = {"split": "train2", "n_scenes": len(streams),
           "identity_coef_train_only": Q["coef"], "families": {}}
    keys = set(REF_X)
    for name, fr in res["families"].items():
        fam = fr["family"]
        per = {}
        for tag, prm in (("ref", fr["reference_on_grid"]["params"]),
                         ("best", fr["best"]["params"])):
            x = {k: prm[k] for k in keys}
            p = {k: v for k, v in prm.items() if k not in keys}
            rows = []
            for sc in sorted(streams):
                rows += replay_x(streams[sc], make_x(fam, p, x), tag)
            per[tag] = (ec.scores(rows), {
                sc: v["nll"] for sc, v in
                te.pm.per_scene_metrics(rows)[tag].items()})
        d = np.array([per["best"][1][s] - per["ref"][1][s]
                      for s in sorted(per["ref"][1])])
        out["families"][name] = {
            "ref": per["ref"][0], "best": per["best"][0],
            "delta_nll": per["best"][0]["nll"] - per["ref"][0]["nll"],
            "scenes_better": int(np.sum(d < -1e-12)),
            "scenes_worse": int(np.sum(d > 1e-12)),
            "best_params": fr["best"]["params"]}
        print(name, {k: round(v, 5) for k, v in per["ref"][0].items()},
              "->", {k: round(v, 5) for k, v in per["best"][0].items()},
              "better/worse scenes", out["families"][name]["scenes_better"],
              out["families"][name]["scenes_worse"], flush=True)
    with open(dest / "results_train2_check.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


def curves(dest):
    """Per-value val NLL of each variant knob from the saved grids: best
    config with the knob at that value and (one_at_a_time) every other
    knob at REF_X, or (others_free) every other knob free. Inner params
    are always re-tuned."""
    out = {}
    for name in ("hiper", "pf", "pf_nbr"):
        z = np.load(dest / f"grid_{name}.npz")
        nll = z["nll"].mean(1)
        xs = [json.loads(s) for s in z["variant"]]
        fam = {}
        for key in REF_X:
            vals = sorted({json.dumps(x[key]) for x in xs})
            if len(vals) < 2:
                continue
            rows = {}
            for v in vals:
                at = np.array([json.dumps(x[key]) == v for x in xs])
                oat = at & np.array([all(x[k] == REF_X[k] for k in REF_X
                                         if k != key) for x in xs])
                rows[v] = {"one_at_a_time": float(nll[oat].min()),
                           "others_free": float(nll[at].min())}
            fam[key] = rows
        out[name] = fam
        for key, rows in fam.items():
            print(name, key, {v: (round(r["one_at_a_time"], 5),
                                  round(r["others_free"], 5))
                              for v, r in rows.items()}, flush=True)
    with open(dest / "curves.json", "w") as f:
        json.dump(out, f, indent=1)


def summary(dest):
    """Extra val diagnostics from grid_*.npz + results.json (+ the TRAIN2
    check when present): relocation effect at the reference's own inner
    params, per-scene deltas with a two-sided sign test, and the margin of
    HiPer over the fixed-rate baselines before / after the idea."""
    from scipy.stats import binomtest
    res = json.load(open(dest / "results.json"))
    out = {"families": {}}
    for name, fr in res["families"].items():
        z = np.load(dest / f"grid_{name}.npz")
        nll, scenes = z["nll"], [str(s) for s in z["scenes"]]
        xs = [json.loads(s) for s in z["variant"]]
        ins = [json.loads(s) for s in z["inner"]]
        mean = nll.mean(1)
        keys = set(REF_X)
        ref_p = {k: v for k, v in fr["reference_on_grid"]["params"].items()
                 if k not in keys}
        best_p = {k: v for k, v in fr["best"]["params"].items()
                  if k not in keys}
        ir = next(i for i in range(len(xs))
                  if ins[i] == ref_p and all(xs[i][k] == REF_X[k]
                                             for k in REF_X))
        ib = next(i for i in range(len(xs))
                  if ins[i] == best_p and all(
                      xs[i][k] == fr["best"]["params"][k] for k in REF_X))
        # best variant with the inner params frozen at the reference's
        at = np.array([ins[i] == ref_p for i in range(len(xs))])
        i_fix = int(np.flatnonzero(at)[np.argmin(mean[at])])
        d = nll[ib] - nll[ir]
        d_fix = nll[i_fix] - nll[ir]

        def sign(dd):
            nb, nw = int(np.sum(dd < -1e-12)), int(np.sum(dd > 1e-12))
            return {"better": nb, "worse": nw,
                    "p_two_sided": float(binomtest(nb, nb + nw).pvalue)
                    if nb + nw else 1.0}
        out["families"][name] = {
            "ref_val_nll": float(mean[ir]), "best_val_nll": float(mean[ib]),
            "delta": float(mean[ib] - mean[ir]), "sign_test": sign(d),
            "per_scene_delta": dict(zip(scenes, map(float, d))),
            "inner_fixed_at_ref": {
                "variant": xs[i_fix], "val_nll": float(mean[i_fix]),
                "delta": float(mean[i_fix] - mean[ir]),
                "sign_test": sign(d_fix)}}
    f = out["families"]
    out["margin_val"] = {
        b: {"before": f[b]["ref_val_nll"] - f["hiper"]["ref_val_nll"],
            "after": f[b]["best_val_nll"] - f["hiper"]["best_val_nll"]}
        for b in ("pf", "pf_nbr")}
    t2 = dest / "results_train2_check.json"
    if t2.exists():
        g = json.load(open(t2))["families"]
        out["margin_train2"] = {
            b: {"before": g[b]["ref"]["nll"] - g["hiper"]["ref"]["nll"],
                "after": g[b]["best"]["nll"] - g["hiper"]["best"]["nll"]}
            for b in ("pf", "pf_nbr")}
    print(json.dumps({k: v for k, v in out.items() if k != "families"},
                     indent=1))
    for name, v in f.items():
        print(name, {k: v[k] for k in ("ref_val_nll", "best_val_nll",
                                       "delta", "sign_test")})
        print("   inner fixed at ref:", v["inner_fixed_at_ref"])
    with open(dest / "summary.json", "w") as fh:
        json.dump(out, fh, indent=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="3 scenes, tiny grids (bug finding only)")
    ap.add_argument("--train2-check", action="store_true",
                    help="replay the val-selected configs on TRAIN2")
    ap.add_argument("--curves", action="store_true",
                    help="per-value NLL curves from the saved val grids")
    ap.add_argument("--train-stats", action="store_true",
                    help="relocation statistics on TRAIN / TRAIN2 labels")
    ap.add_argument("--resume", action="store_true",
                    help="reuse grid_*.partial.npz of an interrupted run")
    ap.add_argument("--summary", action="store_true",
                    help="sign tests, fixed-inner effect, margins")
    args = ap.parse_args()
    out = OUT / ("smoke" if args.smoke else "")
    if args.train_stats:
        return train_stats(OUT)
    if args.summary:
        return summary(out)
    if args.train2_check:
        return train2_check(out)
    if args.curves:
        return curves(out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")       # keep earlier (killed) runs

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} {vars(args)}")
    t_start = time.time()
    m = ec.meta()
    streams = ec.load("streams", "val")
    if args.smoke:
        streams = {k: streams[k] for k in sorted(streams)[:3]}
        g = globals()
        g["PI_RELOC"], g["W_V"] = [0.7, 1.0], [1.0, 4.0]
        g["KAPPA_RELOC"], g["H_MULT"] = [None, 8.0], [1.0, 2.0]
        te.DET_GRID = dict(p_miss=[0.1, 0.2], p_false=[0.01, 0.05])
        te.KAPPA_FINAL = [8, 32]
        te.LIFETIME_GRID_DAYS = [8, 32]
    scenes = sorted(streams)
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in ec.load("records", "train")])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    log(f"val scenes {len(scenes)}")

    # (c) identity model on TRAIN association rows
    coef, ident = fit_identity()
    Q["coef"] = coef
    log(f"identity model {ident['coef_w0_wscore_wlnd2']} "
        f"{json.dumps(ident['cross_split_check'])}")

    prep = prepare(streams)
    lam = {}
    for p in (m["ours_params"], m["pf_nbr_params"]):
        key = (p["K"], p["tau"], p["k_nbr"])
        if key not in lam:
            lam[key] = prior_lams(streams, *key)
    fams = {name: family_grid(name, m) for name in ("hiper", "pf", "pf_nbr")}
    specs = {name: build_spec(name, inner, lam)
             for name, (_, inner, _) in fams.items()}
    _init(prep, coef, lam, specs)

    # Reproduction checks: official replay vs lean replay vs RelocHPF(REF_X)
    # (rate_fn path and precomputed-rate path)
    checks = {}
    for name, fam, p in (("hiper", "hpf_nbr", m["ours_params"]),
                         ("pf", "pf", m["pf_params"]),
                         ("pf_nbr", "pf_nbr", m["pf_nbr_params"])):
        off = ec.run_model(streams, lambda sc: te.make_model(fam, p), name)
        lean_parent = float(np.mean([lean_nll(prep[sc], te.make_model(fam, p))
                                     for sc in scenes]))
        lean_x = float(np.mean([lean_nll(prep[sc], make_x(fam, p, REF_X))
                                for sc in scenes]))
        lean_xp = float(np.mean(_job((0, fam, p, REF_X))[1]))
        full_x, _ = full_scores(streams, fam, p, REF_X, name)
        checks[name] = {"official": off, "lean_parent_nll": lean_parent,
                        "lean_relochpf_ref_nll": lean_x,
                        "lean_relochpf_ref_precomputed_nll": lean_xp,
                        "full_relochpf_ref": full_x}
        log(f"check {name}: official {off['nll']:.6f} lean {lean_parent:.6f}"
            f" RelocHPF(ref) lean {lean_x:.6f} precomp {lean_xp:.6f}"
            f" full {full_x['nll']:.6f}")
        for v in (lean_parent, lean_x, lean_xp, full_x["nll"]):
            assert abs(off["nll"] - v) < 1e-9, (name, off["nll"], v)

    res = {"idea": "w1_relocation", "split": "val",
           "grids": {"PI_RELOC": PI_RELOC, "W_V": W_V,
                     "KAPPA_RELOC": KAPPA_RELOC, "H_MULT": H_MULT,
                     "C_MODE": C_MODE, "DET_GRID": te.DET_GRID,
                     "KAPPA_FINAL": te.KAPPA_FINAL,
                     "LIFETIME_GRID_DAYS": te.LIFETIME_GRID_DAYS},
           "identity_model": ident, "checks": checks, "families": {}}
    rng = np.random.default_rng(0)
    checks["vectorised_vs_scalar_max_abs_diff"] = {
        name: check_vec(name, fam, inner, xs, 3 if args.smoke else 10, rng,
                        log)
        for name, (fam, inner, xs) in fams.items()}

    grids = {}
    with Pool(args.workers, _init, (prep, coef, lam, specs)) as pool:
        for name, (fam, inner, xs) in fams.items():
            log(f"grid {name}: {len(xs)} variants x {len(inner)} inner")
            grids[name] = (fam,) + run_grid(
                pool, name, inner, xs, log,
                ckpt=out / f"grid_{name}.partial.npz", resume=args.resume)

    for name, (fam, cfg, nll) in grids.items():
        mean = nll.mean(1)
        fixed_rate = name != "hiper"
        subsets = ({"ref": (), "a": COMP["a"], "b": COMP["b"],
                    "c": COMP["c"], "ab": COMP["a"] + COMP["b"],
                    "ac": COMP["a"] + COMP["c"], "bc": COMP["b"] + COMP["c"],
                    "abc": COMP["a"] + COMP["b"] + COMP["c"],
                    "b_w_v": ("w_v",), "b_kappa_reloc": ("kappa_reloc",),
                    "b_h_mult": ("h_mult",)}
                   if not fixed_rate else
                   {"ref": (), "a": ("pi_reloc",), "b_h_mult": ("h_mult",),
                    "ab": ("pi_reloc", "h_mult")})
        fam_res = {"family": fam, "n_configs": int(len(cfg)), "subsets": {}}
        for sname, free in subsets.items():
            mk = _mask(cfg, free)
            i = _sel(cfg, nll, mk)
            fam_res["subsets"][sname] = {
                "free": list(free), "n_configs": int(mk.sum()),
                "val_nll": float(mean[i]),
                "variant": cfg[i][0], "inner": cfg[i][1]}
        full = "abc" if not fixed_rate else "ab"
        ib = _sel(cfg, nll, _mask(cfg, subsets[full]))
        best_x = cfg[ib][0]
        # leave-one-component-out from the joint best (other components at
        # their joint-best values, inner params re-tuned)
        loo = {}
        comps = COMP if not fixed_rate else {"a": ("pi_reloc",),
                                             "b_h_mult": ("h_mult",)}
        for cname, keys in comps.items():
            mk = np.array([all(x[k] == (REF_X[k] if k in keys else best_x[k])
                               for k in REF_X) for x, _ in cfg])
            i = _sel(cfg, nll, mk)
            loo[f"without_{cname}"] = {"val_nll": float(mean[i]),
                                       "variant": cfg[i][0],
                                       "inner": cfg[i][1]}
        fam_res["leave_one_out_from_best"] = loo
        # 2-fold over val scenes (even / odd index of sorted names)
        fam_res["two_fold"] = {
            "variant_full_grid": two_fold(cfg, nll, _mask(cfg,
                                                         subsets[full]),
                                          scenes),
            "reference_inner_grid": two_fold(cfg, nll, _mask(cfg, ()),
                                             scenes)}
        # official re-scoring of the reference-on-grid and the best variant
        ir = _sel(cfg, nll, _mask(cfg, ()))
        sc_ref, rows_ref = full_scores(streams, fam, cfg[ir][1], cfg[ir][0],
                                       "ref")
        sc_best, rows_best = full_scores(streams, fam, cfg[ib][1],
                                         cfg[ib][0], "best")
        assert abs(sc_best["nll"] - mean[ib]) < 1e-9
        assert abs(sc_ref["nll"] - mean[ir]) < 1e-9
        per_ref, per_best = nll[ir], nll[ib]
        fam_res["reference_on_grid"] = {"scores": sc_ref,
                                        "params": {**cfg[ir][0],
                                                   **cfg[ir][1]}}
        fam_res["best"] = {"scores": sc_best,
                           "params": {**cfg[ib][0], **cfg[ib][1]},
                           "delta_nll_vs_ref": float(sc_best["nll"]
                                                     - sc_ref["nll"]),
                           "scenes_better": int(np.sum(per_best
                                                       < per_ref - 1e-12)),
                           "scenes_worse": int(np.sum(per_best
                                                      > per_ref + 1e-12)),
                           "per_scene_ref": dict(zip(scenes,
                                                     map(float, per_ref))),
                           "per_scene_best": dict(zip(scenes,
                                                      map(float, per_best)))}
        fam_res["breakdown_ref"] = breakdown(rows_ref, streams)
        fam_res["breakdown_best"] = breakdown(rows_best, streams)
        res["families"][name] = fam_res
        log(f"{name}: ref {sc_ref['nll']:.5f} best {sc_best['nll']:.5f} "
            f"{cfg[ib][0]} {cfg[ib][1]} | 2-fold variant "
            f"{fam_res['two_fold']['variant_full_grid']['heldout_mean']:.5f}"
            f" ref {fam_res['two_fold']['reference_inner_grid']['heldout_mean']:.5f}")
        for sname, v in fam_res["subsets"].items():
            log(f"   subset {sname:14s} {v['val_nll']:.5f} {v['variant']}")
        for sname, v in loo.items():
            log(f"   {sname:14s} {v['val_nll']:.5f} {v['variant']}")
        np.savez_compressed(
            out / f"grid_{name}.npz", nll=nll, scenes=np.array(scenes),
            variant=np.array([_xkey(x) for x, _ in cfg]),
            inner=np.array([_xkey(p) for _, p in cfg]))
    res["seconds"] = round(time.time() - t_start, 1)
    with open(out / "results.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for name in grids:                       # complete: drop checkpoints
        (out / f"grid_{name}.partial.npz").unlink(missing_ok=True)
    log(f"done in {res['seconds']}s -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
