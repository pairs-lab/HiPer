#!/usr/bin/env python3
"""
scripts/bench_prior.py
======================
Single-threaded cost of the appearance-retrieved prior: one uncached
NeighbourSurvivalPrior.rate() call at the final.json parameters, on DINOv2
embeddings of validation tracks (run once per new map object).

  OMP_NUM_THREADS=1 python scripts/bench_prior.py --run v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    fin = json.load(open(te.RESULTS / args.run / "final.json"))
    p = fin["families"]["ours"]["params"]
    prior = te.neighbour_prior()
    embs = [t.dino_mean for ss in te.split_sessions("val") for s in ss
            for t in s.tracks if t.dino_mean is not None][:args.n]
    t0 = time.perf_counter()
    for e in embs:
        prior._cache.clear()
        prior.rate(e, K=p["K"], tau=p["tau"], k=p["k_nbr"])
    dt = (time.perf_counter() - t0) / len(embs)
    print(f"retrieval_us {1e6 * dt:.1f} (n {len(embs)}, K {p['K']}, "
          f"records {len(prior.E)}, dim {prior.E.shape[1]}, single thread)")
    # one belief update (observe) of the current filter, 200k random calls
    from hiper.ephemerality.persistence_v2 import (
        HIT, MISS, HierarchicalPersistenceFilter)
    rng = np.random.default_rng(0)
    hp = {k: p[k] for k in ("p_miss", "p_false", "kappa0", "tau_pi")}
    f = HierarchicalPersistenceFilter(default_lambda=1e-6, **hp)
    for i in range(1000):
        f.create(f"o{i}", "x", 0.0)
    ev = rng.random(200000) < 0.8
    oid = rng.integers(0, 1000, 200000)
    t0 = time.perf_counter()
    for j in range(200000):
        f.observe(f"o{oid[j]}", 3600.0 * (j // 1000 + 1),
                  HIT if ev[j] else MISS)
    dt = (time.perf_counter() - t0) / 200000
    print(f"observe_us {1e6 * dt:.2f} (200000 calls, single thread)")


if __name__ == "__main__":
    main()
