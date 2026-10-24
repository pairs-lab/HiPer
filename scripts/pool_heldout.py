#!/usr/bin/env python3
"""
scripts/pool_heldout.py
=======================
Pool the held-out THOR-Lifelong sets (each 40 new sequences, evaluated with
the same frozen configuration) into one summary with the structure of
tune_and_eval.eval_split, so make_result_tables can read it as a split.
Sequences are renamed '<split>:<scene>' so the sets never collide; paired
bootstraps run over all pooled sequences (raw models vs 'ours', calibrated
models vs 'ours_cal').

  python scripts/pool_heldout.py --run v5 --splits test3 test4 test5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import write_jsonl  # noqa: E402

METS = ("nll", "brier", "ece", "auroc", "ap_absent", "premature_prune_rate",
        "stale_rate")


def load(path, split, out_split):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["scene"] = f"{split}:{r['scene']}"
            r["split"] = out_split
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5")
    ap.add_argument("--splits", nargs="+",
                    default=["test3", "test4", "test5"])
    ap.add_argument("--out", default="heldout")
    args = ap.parse_args()
    res = te.RESULTS / args.run
    out = res / args.out
    out.mkdir(exist_ok=True)
    rows, arows, abl = [], [], []
    for sp in args.splits:
        rows += load(res / sp / "pred.jsonl", sp, args.out)
        arows += load(res / sp / "assoc.jsonl", sp, args.out)
        abl += load(res / sp / "pred_assoc_ablations.jsonl", sp, args.out)
    models = sorted({r["model"] for r in rows})
    summary = pm.summarize(rows, split=args.out)
    per = pm.per_scene_metrics(rows)
    paired = {}
    for m in models:
        if m in ("ours", "ours_cal") or m.startswith("diag_"):
            continue
        ref = "ours_cal" if m.endswith("_cal") else "ours"
        paired[m] = {met: pm.paired_bootstrap(
            {s: v[met] for s, v in per[ref].items()},
            {s: v[met] for s, v in per[m].items()}) for met in METS}
    for m in ("ours", "ours_cal", "pf_nbr", "pf", "elite", "pf_reloc"):
        if m in models:
            p_, y_ = pm.labeled_arrays([r for r in rows if r["model"] == m])
            pm.write_reliability_csv(pm.reliability_bins(p_, y_),
                                     str(out / f"reliability_{m}.csv"))
    timing = json.load(open(res / args.splits[-1] / "summary.json")).get(
        "timing", {})
    write_jsonl(rows, str(out / "pred.jsonl"))
    with open(out / "summary.json", "w") as f:
        json.dump({"split": args.out, "pooled_splits": args.splits,
                   "persistence": summary, "paired_vs_ours": paired,
                   "association": pm.summarize_association(
                       arows, split=args.out),
                   "assoc_ablations_persistence": pm.summarize(
                       abl + [r for r in rows if r["model"] == "ours"],
                       split=args.out),
                   "timing": timing}, f, indent=1, default=float)
    p = summary
    print("pooled", len({r["scene"] for r in rows}), "sequences")
    for m in ("ours", "ours_cal", "pf_nbr", "pf_nbr_cal", "pf", "pf_cal",
              "pf_reloc", "elite", "pocd"):
        if m in p:
            print(f"  {m:12s} nll={p[m]['nll'][0]:.3f} "
                  f"ece={p[m]['ece'][0]:.3f} auroc={p[m]['auroc'][0]:.3f}")


if __name__ == "__main__":
    main()
