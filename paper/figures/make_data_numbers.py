#!/usr/bin/env python3
"""
Dataset statistics quoted in the paper, computed from the saved sessions
(THOR-Lifelong splits and ObChange), written as LaTeX macros to
paper/figures/data_numbers.tex. Also the per-object cost of a belief update
from the timing of the test replay.

  python paper/figures/make_data_numbers.py --run main
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import conftest  # noqa: E402,F401

from hiper.evaluation.object_session import load_session  # noqa: E402

from hiper.evaluation.paths import (  # noqa: E402
    OBCHANGE_SESSIONS as OBC, RESULTS_ROOT, THOR_ROOT as THOR)


def split_stats(files):
    st = Counter()
    clip_ok = []
    scenes = set()
    for f in files:
        s = load_session(f)
        scenes.add(s.scene)
        st["sessions"] += 1
        st["frames"] += int(s.meta.get("n_frames", len(s.T_wc_est)))
        st["tracks"] += len(s.tracks)
        g = {o["gt_id"]: o for o in s.gt_objects}
        for o in s.gt_objects:
            st["moved"] += bool(o.get("moved_since_prev"))
            st["appeared"] += bool(o.get("appeared_since_prev"))
            st["disappeared"] += bool(o.get("disappeared_since_prev"))
            if s.session_index == 0:
                st["objects"] += 1
        for t in s.tracks:
            if t.gt_id in g:
                clip_ok.append(t.clip_label == g[t.gt_id]["type"])
    st["scenes"] = len(scenes)
    return st, (float(np.mean(clip_ok)) if clip_ok else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    out = []

    def put(name, val, nd=0):
        v = f"{val:,.{nd}f}" if isinstance(val, float) else f"{val:,}"
        out.append(f"\\newcommand{{\\{name}}}{{{v.replace(',', '{,}')}}}")

    total = Counter()
    floorplans = set()
    for split, tag in (("train", "Train"), ("train2", "Traintwo"),
                       ("val", "Val"), ("test", "Test"),
                       ("test2", "Testtwo"), ("test3", "Testthree"),
                       ("test4", "Testfour"), ("test5", "Testfive")):
        files = sorted(glob.glob(str(THOR / "sessions" / split /
                                     "*/session_*.npz")))
        if not files:
            continue
        floorplans |= {Path(f).parent.name for f in files}
        st, acc = split_stats(files)
        total.update(st)
        for k in ("scenes", "sessions", "frames", "tracks", "objects",
                  "moved", "appeared", "disappeared"):
            put(f"d{tag}{k.capitalize()}", st[k])
        put(f"d{tag}ClipAcc", 100.0 * acc, 1)
    for k in ("scenes", "sessions", "frames", "tracks"):
        put(f"dAll{k.capitalize()}", total[k])
    # pooled held-out sets (each 40 new sequences, evaluated once)
    held = sorted(f for sp in ("test3", "test4", "test5") for f in glob.glob(
        str(THOR / "sessions" / sp / "*/session_*.npz")))
    if held:
        st, acc = split_stats(held)
        st["scenes"] = len({Path(f).parent.parent.name + Path(f).parent.name
                            for f in held})
        for k in ("scenes", "moved", "appeared", "disappeared", "tracks"):
            put(f"dHeld{k.capitalize()}", st[k])
        put("dHeldClipAcc", 100.0 * acc, 1)
    # sequences = scenes summed over splits (test3 reuses test floor plans
    # with new histories); distinct floor plans
    put("dAllSequences", total["scenes"])
    put("dAllFloorPlans", len(floorplans))
    files = sorted(glob.glob(str(OBC / "*/session_*.npz")))
    st, _ = split_stats(files)
    for k in ("scenes", "sessions", "tracks"):
        put(f"dObc{k.capitalize()}", st[k])
    # per-update cost: test replay time / prediction rows of all models
    res = RESULTS_ROOT / args.run
    split = "test3" if (res / "test3/summary.json").exists() else "test"
    summ = json.load(open(res / split / "summary.json"))
    n_rows = sum(1 for _ in open(res / split / "pred.jsonl"))
    t = summ["timing"]["replay_all_models_s"]
    put("dUpdateUs", 1e6 * t / n_rows, 1)
    put("dMapMs", 1e3 * summ["timing"]["mapping_s_per_session"], 0)
    rscan_numbers(put)
    (ROOT / "paper/figures/data_numbers.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def rscan_numbers(put):
    """3RScan: split sizes, ground-truth transitions and the covariance fit,
    from the conversion reports of scripts/rscan_to_sessions.py and
    scripts/rscan_fit_cov.py (nothing if 3RScan was not converted)."""
    from hiper.evaluation import rscan
    from hiper.evaluation.paths import RSCAN_META, RSCAN_SESSIONS
    reps = sorted(RSCAN_SESSIONS.glob("report_*.json"))
    if not reps or not RSCAN_META.exists():
        return
    index = rscan.load_index(str(RSCAN_META))
    put("dRsEnvsAll", len(index))
    put("dRsScansAll", sum(1 + len(e["rescans"]) for e in index.values()))
    envs = {}
    for r in reps:
        envs.update(json.load(open(r))["envs"])
    for split, tag in (("rscan_train", "Train"), ("rscan_val", "Val"),
                       ("rscan_test", "Test")):
        E = [e for e in envs.values() if e["split"] == split]
        if not E:
            continue
        c = Counter()
        for e in E:
            for t in e["transitions"]:
                c.update(t)
        put(f"dRs{tag}Envs", len(E))
        put(f"dRs{tag}Scans", sum(len(e["scans"]) for e in E))
        put(f"dRs{tag}Tracks", sum(sum(e["tracks"]) for e in E))
        for k in ("present", "moved", "removed", "appeared", "unknown"):
            put(f"dRs{tag}{k.capitalize()}", c.get(k, 0))
        short = sum(len(e["scans"]) <= 3 for e in E)
        put(f"dRs{tag}ShortPct", 100.0 * short / len(E), 0)
    fit = RSCAN_SESSIONS / "cov_fit.json"
    if fit.exists():
        f = json.load(open(fit))
        best = next(g for g in f["grid"] if g["kappa"] == f["kappa"])
        put("dRsKappa", float(f["kappa"]), 3)
        put("dRsKappaMoved", 100.0 * best["moved_flagged"], 0)
        put("dRsKappaStatic", 100.0 * best["static_kept"], 0)
        put("dRsKappaPairs", int(f["pairs"]))


if __name__ == "__main__":
    main()
