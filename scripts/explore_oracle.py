#!/usr/bin/env python3
"""
scripts/explore_oracle.py
=========================
DIAGNOSTIC ONLY (val): how good can a prior get? PF and HiPer whose prior
rate for each map object is the TRUE simulator hazard of the object that
created it (sequence.json, hazard_per_day), detection parameters tuned on
val. Not a method: it reads ground truth.

  python scripts/explore_oracle.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

DAY = 86400.0


def main():
    streams = ec.load("streams")
    truth = ec.load("truth")
    ks = ec.keyed(streams)
    prior = {}
    for sc, st in streams.items():
        for oid, ob in st["objects"].items():
            g = ob["events"][0].get("gt_id")
            h = truth.get((sc, g), 0.0) if g else 0.0
            prior[f"{sc}/{oid}"] = h / DAY
    res = {}
    for kappa in (None, 4, 16, 64, 256):
        best = None
        for d in te.grid(te.DET_GRID):
            if kappa is None:   # PF with the true rate (no learning)
                f = (lambda sc, d=d: te.HierarchicalPersistenceFilter(
                    lambda_table=prior, default_lambda=1e-9,
                    learn_rate=False, tau_pi=0.3, **d))
            else:
                f = (lambda sc, d=d, k=kappa: te.HierarchicalPersistenceFilter(
                    lambda_table=prior, default_lambda=1e-9, kappa0=k,
                    variance_gate=False, tau_pi=0.3, **d))
            s = ec.run_model(ks, f, "oracle")
            if best is None or s["nll"] < best[0]["nll"]:
                best = (s, d)
        name = "pf_true_rate" if kappa is None else f"hiper_true_prior_k{kappa}"
        res[name] = {**best[0], "params": best[1]}
        print(name, {k: round(v, 4) for k, v in best[0].items()}, best[1],
              flush=True)
    json.dump(res, open(ec.CACHE / "oracle_val.json", "w"), indent=1)


if __name__ == "__main__":
    main()
