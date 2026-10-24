#!/usr/bin/env python3
"""
scripts/explore_assoc.py
========================
VALIDATION-ONLY exploration of association rules
(docs/reports/2026-10-19-improve-results). Never touches test splits.

On top of the selected configuration (final.json):
  tau_reloc  a match whose Mahalanobis distance exceeds chi2_3(0.99)
             (a relocation claim, supported by appearance only) needs
             score >= tau_reloc instead of tau_match
  margin     the best score of a track must beat its second-best candidate
             by at least `margin` (ratio test; ambiguous tracks stay
             unmatched)
Reported: association metrics and HiPer's val NLL (final.json params).

  python scripts/explore_assoc.py --run main --out improve
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.evaluation import lifelong_objects as lo  # noqa: E402
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402
from hiper.object_encoder import association_v2 as av  # noqa: E402

RULE = {"tau_reloc": None, "margin": 0.0}


def associate_rule(tracks, objects, cfg, pose_cov=None):
    if len(tracks) == 0 or len(objects) == 0:
        return [av.Match(i) for i in range(len(tracks))]
    score, d2, u = av.score_matrix(tracks, objects, cfg, pose_cov)
    valid = score >= cfg.tau_match
    if RULE["tau_reloc"] is not None:
        valid &= (d2 <= cfg.chi2_reloc) | (score >= RULE["tau_reloc"])
    if RULE["margin"] > 0 and score.shape[1] > 1:
        # ratio test: a track whose best candidate does not beat the
        # second-best by `margin` stays unmatched (ambiguous)
        top2 = -np.sort(-score, axis=1)[:, :2]
        valid[(top2[:, 0] - top2[:, 1]) < RULE["margin"], :] = False
    # assign() keeps pairs with score >= tau_match: push invalid ones below
    return av.assign(np.where(valid, score, -1.0), d2, u, cfg)


def _job(args):
    scene, files, cfg, rule = args
    RULE.update(rule)
    lo.associate = associate_rule
    sessions = [load_session(f) for f in files]
    dino = te.appearance_inputs(sessions, cfg)
    stream, rows = lo.run_mapping(sessions, te.assoc_config(cfg), bye=dino,
                                  split="val")
    return scene, stream, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--out", default="improve")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = te.RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    fin = json.load(open(te.RESULTS / args.run / "final.json"))
    a = {**fin["assoc"], "name": "ours"}
    op = fin["families"]["ours"]["params"]
    lam, lam0 = lo.load_class_lambdas(str(te.PRIORS))
    learned = te.learned_rates()
    rules = [{"tau_reloc": None, "margin": 0.0}]
    rules += [{"tau_reloc": t, "margin": 0.0}
              for t in (0.6, 0.7, 0.8, 0.9, 1.0)]
    rules += [{"tau_reloc": None, "margin": m} for m in (0.02, 0.05, 0.1)]
    rules += [{"tau_reloc": t, "margin": m} for t in (0.7, 0.8)
              for m in (0.02, 0.05)]
    res = []
    files = te.scene_files("val")
    for rule in rules:
        with Pool(args.workers) as pool:
            got = pool.map(_job, [(sc, f, a, rule) for sc, f in files.items()])
        streams = {sc: st for sc, st, _ in got}
        rows = [r for _, _, rr in got for r in rr]
        acc = te.assoc_accuracy(rows)
        am = pm.association_metrics(rows)
        te._init_worker(streams, lam, lam0, learned)
        _, _, prow = te._replay_job(("hpf_learned", op, "ours"))
        s = te._scores(prow)
        res.append({**rule, "val_assoc_accuracy": acc,
                    **{k: am[k] for k in ("success", "success_stationary",
                                          "success_relocated",
                                          "success_reappeared",
                                          "novel_false_match")},
                    "val_nll_ours": s["nll"]})
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in res[-1].items()}, flush=True)
    lo.write_jsonl(res, str(out / "assoc_rules_val.jsonl"))


if __name__ == "__main__":
    main()
