#!/usr/bin/env python3
"""
Convert 3RScan environments into SessionData files:
<out>/<split>/<environment id>/session_<kk>.npz (session 0 = reference
scan, then the annotated rescans), plus <out>/report_<splits>.json with the
ground-truth transitions per environment. Protocol and conventions:
src/evaluation/rscan.py.

  python scripts/rscan_to_sessions.py --splits rscan_train rscan_val
  python scripts/rscan_to_sessions.py --splits rscan_test      # once
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import conftest  # noqa: E402,F401  (registers the hiper package)

from hiper.evaluation import paths as hpaths  # noqa: E402
from hiper.evaluation import rscan  # noqa: E402
from hiper.evaluation.object_session import save_session  # noqa: E402


def _prepare(job):
    root, env, split, stride = job
    try:
        return env["reference"], rscan.prepare_env(root, env, split, stride), None
    except Exception:                                   # noqa: BLE001
        return env["reference"], None, traceback.format_exc(limit=3)


def transitions(preps):
    """Ground-truth counts per transition k-1 -> k."""
    out = []
    for p in preps[1:]:
        c = Counter()
        for o in p["gt"]["objects"]:
            if o["present"] is None:
                c["unknown"] += 1
            elif o["moved_since_prev"]:
                c["moved"] += 1
            elif o["disappeared_since_prev"]:
                c["removed"] += 1
            elif o["appeared_since_prev"]:
                c["appeared"] += 1
            elif o["present"]:
                c["present"] += 1
        out.append(dict(c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(hpaths.RSCAN_SCANS))
    ap.add_argument("--meta", default=str(hpaths.RSCAN_META))
    ap.add_argument("--out", default=str(hpaths.RSCAN_SESSIONS))
    ap.add_argument("--splits", nargs="+",
                    default=["rscan_train", "rscan_val"])
    ap.add_argument("--envs", nargs="*", default=None,
                    help="only these environment (reference scan) ids")
    ap.add_argument("--split-as", default=None,
                    help="write the selected environments under this split "
                         "name (smoke tests only)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="drop rescans not downloaded (smoke tests only)")
    ap.add_argument("--frame-stride", type=int, default=rscan.FRAME_STRIDE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--kappa", type=float, default=None,
                    help="CENTROID_EXTENT_FRAC to use (from "
                         "scripts/rscan_fit_cov.py on rscan_train)")
    args = ap.parse_args()
    if args.kappa is not None:
        rscan.CENTROID_EXTENT_FRAC = args.kappa   # workers fork after this

    index = rscan.load_index(args.meta)
    jobs, missing = [], {}
    for env_id, env in sorted(index.items()):
        split = rscan.assign_split(env_id, env["type"])
        if args.envs is not None:
            if env_id not in args.envs:
                continue
        elif split not in args.splits:
            continue
        split = args.split_as or split
        ids = [s for s, _, _ in rscan.env_scans(env)]
        absent = [s for s in ids if not os.path.isdir(rscan.scan_dir(args.root, s))]
        if absent:
            if not args.allow_missing or ids[0] in absent:
                missing[env_id] = absent
                continue
            env = {**env, "rescans": [r for r in env["rescans"]
                                      if r["id"] not in absent]}
        if args.skip_existing and os.path.isdir(os.path.join(args.out, split, env_id)):
            continue
        jobs.append((args.root, env, split, args.frame_stride))
    print(f"{len(jobs)} environments to convert, {len(missing)} with missing "
          "scans", flush=True)

    from hiper.evaluation.thor_tracks import TrackEncoders, build_session
    import cv2
    enc = TrackEncoders(device=args.device)
    report = {"splits": args.splits, "frame_stride": args.frame_stride,
              "missing_scans": missing, "envs": {}, "errors": {}}
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        for n, (env_id, preps, err) in enumerate(
                pool.imap_unordered(_prepare, jobs), 1):
            if err is not None:
                report["errors"][env_id] = err
                print("ERROR", env_id, err.splitlines()[-1], flush=True)
                continue
            for p in preps:
                p["rgb"] = np.stack([cv2.imread(f)[..., ::-1]
                                     for f in p["frame_files"]])
                sess = build_session(p, enc)
                p["rgb"] = None
                save_session(sess, os.path.join(
                    args.out, p["split"], env_id,
                    f"session_{p['k']:02d}.npz"))
            report["envs"][env_id] = {
                "split": preps[0]["split"],
                "scans": [p["meta"]["scan_id"] for p in preps],
                "tracks": [len(p["tracks"]) for p in preps],
                "transitions": transitions(preps)}
            print(f"[{n}/{len(jobs)}] {env_id} sessions={len(preps)} "
                  f"tracks={report['envs'][env_id]['tracks']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    name = "report_" + "_".join(sorted({p[2] for p in jobs}) or ["none"]) + ".json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(report, f, indent=1)
    print("done in", report["seconds"], "s ->", os.path.join(args.out, name))


if __name__ == "__main__":
    main()
