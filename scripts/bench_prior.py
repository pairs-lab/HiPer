#!/usr/bin/env python3
"""
scripts/bench_prior.py
======================
Single-threaded runtime of the final method (paper, Sec. IV-E), measured on
one pinned CPU core of an otherwise idle machine:

  map_ms        association and evidence extraction (run_mapping, the call
                timed in tune_and_eval._map_job) per session, over the
                held-out THOR sequences, with the final association
  retrieval_ms  one uncached NeighbourSurvivalPrior.rate() call at the
                final parameters and retrieval key, on the queries of the
                CREATE events of the validation streams (as in the replay),
                run once per new map object
  observe_us    one belief update (HierarchicalPersistenceFilter.observe)

  OMP_NUM_THREADS=1 python scripts/bench_prior.py --run v5 --cpu 8
Output: results/<run>/runtime.json
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.evaluation.lifelong_objects import creation_query  # noqa: E402

HELDOUT = ("test3", "test4", "test5")


def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--cpu", type=int, default=8,
                    help="core to pin the timed parts to")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the untimed validation mapping")
    args = ap.parse_args()
    out = te.RESULTS / args.run
    fin = json.load(open(out / "final.json"))
    p, a = fin["families"]["ours"]["params"], fin["assoc"]
    load0 = os.getloadavg()
    # queries of the new objects of the validation streams (untimed)
    prior = te.neighbour_prior(dino_only=p.get("key", "full") == "dino")
    streams, _, _ = te.map_split_serial(te.VAL, a, args.workers)
    queries = [creation_query(ob["events"][0]) for sc in sorted(streams)
               for ob in streams[sc]["objects"].values()][:args.n]
    os.sched_setaffinity(0, {args.cpu})
    res = {"run": args.run, "cpu": cpu_model(), "pinned_cpu": args.cpu,
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    # retrieval of the prior, uncached, once per new object
    t0 = time.perf_counter()
    for q in queries:
        prior._cache.clear()
        prior.rate(q, K=p["K"], tau=p["tau"], k=p["k_nbr"])
    dt = (time.perf_counter() - t0) / len(queries)
    res["retrieval_ms"] = 1e3 * dt
    res["retrieval"] = {"n": len(queries), "K": p["K"],
                        "key": p.get("key", "full"),
                        "records": len(prior.E), "dim": prior.E.shape[1]}
    print(f"retrieval_ms {1e3 * dt:.3f} {res['retrieval']}", flush=True)
    # one belief update (observe) of the final filter, 200k random calls
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
    res["observe_us"] = 1e6 * (time.perf_counter() - t0) / 200000
    print(f"observe_us {res['observe_us']:.3f} (200000 calls)", flush=True)
    # association and evidence extraction per session, held-out THOR
    secs, n_sess, per_seq = 0.0, 0, []
    for split in HELDOUT:
        for scene, files in te.scene_files(split).items():
            sessions = [te.load_session(fn) for fn in files]
            dino = te.appearance_inputs(sessions, a)
            t0 = time.perf_counter()
            te.run_mapping(sessions, te.assoc_config(a), bye=dino,
                           split=split)
            dt = time.perf_counter() - t0
            secs += dt
            n_sess += len(sessions)
            per_seq.append(1e3 * dt / len(sessions))
    res["map_ms"] = 1e3 * secs / n_sess
    res["map"] = {"splits": list(HELDOUT), "sequences": len(per_seq),
                  "sessions": n_sess,
                  "median_ms_per_session_of_sequences": float(
                      np.median(per_seq)),
                  "min_ms": float(np.min(per_seq)),
                  "max_ms": float(np.max(per_seq))}
    print(f"map_ms {res['map_ms']:.1f} {res['map']}", flush=True)
    res["loadavg_start"], res["loadavg_end"] = load0, os.getloadavg()
    json.dump(res, open(out / "runtime.json", "w"), indent=1)
    print("wrote", out / "runtime.json")


if __name__ == "__main__":
    main()
