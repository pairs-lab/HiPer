#!/usr/bin/env python3
"""
scripts/explore_env.py
======================
VALIDATION-ONLY choice of the environment level's prior strength alpha
(docs/reports/2026-10-19-improve-results, declared before any ObChange run
with the environment level).

Protocol: THOR val streams with the clock rescaled by c in {1/4, 1, 4}
(the same changes happen over 4x shorter or longer gaps, i.e. a world
that changes 4x faster or slower than the training scenes). For each alpha
the frozen final.json parameters of HiPer and PF(lambda) are replayed;
the criterion is the mean NLL over the three clocks.

  python scripts/explore_env.py --run v2
"""

import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.evaluation.lifelong_objects import (  # noqa: E402
    load_class_lambdas, write_jsonl)

CLOCKS = [0.25, 1.0, 4.0]
ALPHAS = [None, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]


def scaled(streams, c):
    out = {}
    for sc, st in streams.items():
        s2 = copy.copy(st)
        s2["times"] = [t * c for t in st["times"]]
        s2["objects"] = {oid: {**ob, "events": [{**ev, "t": ev["t"] * c}
                                                for ev in ob["events"]]}
                         for oid, ob in st["objects"].items()}
        out[sc] = s2
    return out


def job(args):
    c, name, fam, params = args
    te._G["streams"] = te._G["by_clock"][c]
    _, _, rows = te._replay_job((fam, params, name))
    return c, name, params.get("env_alpha"), te._scores(rows)["nll"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = te.RESULTS / args.run
    fin = json.load(open(out / "final.json"))
    lam, lam0 = load_class_lambdas(str(te.PRIORS))
    learned = te.learned_rates()
    te.NBR["prior"] = te.neighbour_prior()
    streams, _, _ = te.map_split_serial("val", fin["assoc"], args.workers)
    te._init_worker(streams, lam, lam0, learned)
    te._G["by_clock"] = {c: scaled(streams, c) for c in CLOCKS}
    fam = fin["families"]
    jobs = []
    for a in ALPHAS:
        for c in CLOCKS:
            for name in ("ours", "pf"):
                p = dict(fam[name]["params"])
                if a is not None:
                    p["env_alpha"] = a
                jobs.append((c, name, fam[name]["family"], p))
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs)
    rows = [{"clock": c, "model": n, "env_alpha": a, "val_nll": v}
            for c, n, a, v in res]
    write_jsonl(rows, str(out / "env_alpha_val.jsonl"))
    for name in ("ours", "pf"):
        print(name)
        for a in ALPHAS:
            v = [r["val_nll"] for r in rows if r["model"] == name
                 and r["env_alpha"] == a]
            byc = {r["clock"]: round(r["val_nll"], 4) for r in rows
                   if r["model"] == name and r["env_alpha"] == a}
            print(f"  alpha={a}: mean {np.mean(v):.4f} {byc}")


if __name__ == "__main__":
    main()
