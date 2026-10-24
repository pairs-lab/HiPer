#!/usr/bin/env python3
"""
scripts/explore_assoc_nll.py
============================
VALIDATION-ONLY: the best configuration of each appearance cue (selected
for association accuracy in `tune`) scored by the downstream val NLL of
HiPer with its frozen parameters (docs/reports/2026-10-19-improve-results).

  python scripts/explore_assoc_nll.py --run v2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.evaluation.lifelong_objects import (  # noqa: E402
    load_class_lambdas, write_jsonl)


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
    op = fin["families"]["ours"]["params"]
    cands = {"selected": fin["assoc"],
             **{k: {**v, "tau_u": te.INF} for k, v in fin["assoc_e3"].items()}}
    res = []
    for name, cfg in cands.items():
        streams, rows, _ = te.map_split_serial("val", cfg, args.workers)
        te._init_worker(streams, lam, lam0, learned)
        _, _, pr = te._replay_job(("hpf_nbr", op, "ours"))
        r = {"config": name, **cfg, "val_assoc_accuracy":
             te.assoc_accuracy(rows), "val_nll_ours": te._scores(pr)["nll"]}
        res.append(r)
        print(name, round(r["val_assoc_accuracy"], 4),
              round(r["val_nll_ours"], 4), flush=True)
    write_jsonl(res, str(out / "assoc_nll_val.jsonl"))


if __name__ == "__main__":
    main()
