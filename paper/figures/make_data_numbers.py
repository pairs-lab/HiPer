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
    for split, tag in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        files = sorted(glob.glob(str(THOR / "sessions" / split /
                                     "*/session_*.npz")))
        st, acc = split_stats(files)
        total.update(st)
        for k in ("scenes", "sessions", "frames", "tracks", "objects",
                  "moved", "appeared", "disappeared"):
            put(f"d{tag}{k.capitalize()}", st[k])
        put(f"d{tag}ClipAcc", 100.0 * acc, 1)
    for k in ("scenes", "sessions", "frames", "tracks"):
        put(f"dAll{k.capitalize()}", total[k])
    files = sorted(glob.glob(str(OBC / "*/session_*.npz")))
    st, _ = split_stats(files)
    for k in ("scenes", "sessions", "tracks"):
        put(f"dObc{k.capitalize()}", st[k])
    # per-update cost: test replay time / prediction rows of all models
    res = RESULTS_ROOT / args.run
    summ = json.load(open(res / "test/summary.json"))
    n_rows = sum(1 for _ in open(res / "test/pred.jsonl"))
    t = summ["timing"]["replay_all_models_s"]
    put("dUpdateUs", 1e6 * t / n_rows, 1)
    put("dMapMs", 1e3 * summ["timing"]["mapping_s_per_session"], 0)
    (ROOT / "paper/figures/data_numbers.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
