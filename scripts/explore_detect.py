#!/usr/bin/env python3
"""
scripts/explore_detect.py
=========================
VALIDATION-ONLY screen (report 2026-10-23-env-scale): hierarchical
detectability. Each object keeps a Beta belief over its own miss rate,
Beta(kappa_m P_M, kappa_m (1 - P_M)) around the global P_M; a run of misses
that ends in a hit at the same place counts as misses of a present object
(a_m += misses), every hit counts as a detection (b_m += 1); misses that end
in a relocation are left out (the object had left). The object's own miss
rate replaces P_M in its hit / miss likelihoods.

Same grids for the variant and the reference; 2-fold check over val scenes.
Applied to HiPer (DINOv2 key, run-v2 cache), PF + retrieved rates and
PF(lambda) for fairness.

  python scripts/explore_detect.py --workers 12
"""

import argparse
import functools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HIT, MISS, HierarchicalPersistenceFilter)
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import replay  # noqa: E402

DAY = 86400.0
KAPPA_M = [None, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]      # None = reference


class DetectHPF(HierarchicalPersistenceFilter):
    def __init__(self, kappa_m=None, **kw):
        super().__init__(**kw)
        self.kappa_m = kappa_m
        self._d = {}

    def create(self, oid, class_label, t, emb=None):
        if self.wants_embedding:
            super().create(oid, class_label, t, emb=emb)
        else:
            super().create(oid, class_label, t)
        if self.kappa_m is not None:
            pm_ = self.p_miss
            self._d[oid] = [self.kappa_m * pm_, self.kappa_m * (1 - pm_), 0]

    def observe(self, oid, t, evidence, eps_l=0.0, **info):
        d = self._d.get(oid)
        if d is None or evidence not in (HIT, MISS):
            return super().observe(oid, t, evidence, eps_l=eps_l, **info)
        save = self.p_miss
        self.p_miss = d[0] / (d[0] + d[1])
        try:
            super().observe(oid, t, evidence, eps_l=eps_l, **info)
        finally:
            self.p_miss = save
        if evidence == MISS:
            d[2] += 1
        else:
            d[0] += d[2]
            d[1] += 1.0
            d[2] = 0

    def relocate(self, oid, t):
        super().relocate(oid, t)
        if oid in self._d:
            self._d[oid][2] = 0


_G = {}


def factory(fam, p, kappa_m):
    p = dict(p)
    if fam in ("hpf_nbr", "pf_nbr"):
        nbr = _G["prior"]
        fn = functools.partial(nbr.rate, K=p.pop("K"), tau=p.pop("tau"),
                               k=p.pop("k_nbr"))
        p.pop("key", None)
        return DetectHPF(kappa_m=kappa_m, default_lambda=nbr.lam0,
                         use_class_prior=False, rate_fn=fn,
                         learn_rate=(fam == "hpf_nbr"), **p)
    L = p.pop("lifetime_days")
    return DetectHPF(kappa_m=kappa_m, default_lambda=1.0 / (L * DAY),
                     use_class_prior=False, learn_rate=False, **p)


def job(args):
    fam, p, km = args
    per = {}
    for sc in sorted(_G["streams"]):
        rows = replay(_G["streams"][sc], factory(fam, p, km), "x")
        per[sc] = pm.per_scene_metrics(rows)["x"][sc]["nll"]
    return fam, json.dumps(p, sort_keys=True), km, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    m = ec.meta()
    _G["streams"] = ec.load("streams")
    _G["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in ec.load("records", "train")])
    det = te.grid(te.DET_GRID)
    fams = {
        "ours": ("hpf_nbr", [{**{k: v for k, v in m["ours_params"].items()
                                 if k not in ("p_miss", "p_false", "kappa0")},
                              **d, "kappa0": k}
                             for d in det for k in te.KAPPA_FINAL]),
        "pf_nbr": ("pf_nbr", [{**{k: v for k, v in m["pf_nbr_params"].items()
                                  if k not in ("p_miss", "p_false")}, **d}
                              for d in det]),
        "pf": ("pf", [{"lifetime_days": L, "tau_pi": 0.2, **d} for d in det
                      for L in te.LIFETIME_GRID_DAYS]),
    }
    jobs = [(fam, p, km) for name, (fam, ps) in fams.items() for p in ps
            for km in KAPPA_M]
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs, chunksize=8)
    scenes = sorted(_G["streams"])
    A, B = scenes[0::2], scenes[1::2]
    out = {}
    for name, (fam, _) in fams.items():
        for variant in ("reference", "detect"):
            cand = [r for r in res if r[0] == fam and
                    ((r[2] is None) == (variant == "reference"))]
            full = min(cand, key=lambda r: np.mean(list(r[3].values())))
            def held(tr, te_):
                b = min(cand, key=lambda r: np.mean([r[3][s] for s in tr]))
                return np.mean([b[3][s] for s in te_])
            out[f"{name}/{variant}"] = {
                "val_nll": float(np.mean(list(full[3].values()))),
                "params": full[1], "kappa_m": full[2],
                "cv2": float((held(A, B) + held(B, A)) / 2)}
            print(name, variant, {k: (round(v, 4) if isinstance(v, float)
                                      else v) for k, v in
                                  out[f"{name}/{variant}"].items()},
                  flush=True)
    d = Path(ec.CACHE) / "detect"
    d.mkdir(exist_ok=True)
    json.dump(out, open(d / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
