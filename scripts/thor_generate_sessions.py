#!/usr/bin/env python3
"""
scripts/thor_generate_sessions.py
=================================
Generate THOR-Lifelong raw sequences (design.md 'Benchmark: THOR-Lifelong';
code in src/evaluation/thor_world.py). One worker process = one Unity
process (CloudRendering); at most 3 at once on the shared GPU.

Output: <out-root>/<split>/<scene>/{sequence.json, validation.json,
session_<kk>/{gt.json, frames/...}}. A scene with sequence.json is complete
and is skipped unless --overwrite.

Usage
  # smoke test (FloorPlan21, 3 sessions) + report
  python scripts/thor_generate_sessions.py --split val --scenes FloorPlan21 \
      --sessions 3 --out-root <DATA>/raw/_smoke --jobs 1 \
      --report <DATA>/logs/thor/smoke_report.md
  # full split
  python scripts/thor_generate_sessions.py --split val --sessions 8 --jobs 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402  (registers hiper pkg + rospy stub)

from hiper.evaluation.thor_world import (  # noqa: E402
    SPLIT_SCENES,
    SequenceGenerator,
    ThorSim,
    load_json,
    validate_sequence,
)

from hiper.evaluation.paths import THOR_ROOT as DATA_ROOT  # noqa: E402
_SIM = None   # one simulator per worker process


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_scene(job: tuple) -> dict:
    split, scene, seed, n_sessions, out_root, n_waypoints = job
    global _SIM
    out_dir = os.path.join(out_root, split, scene)
    err = ""
    for attempt in range(2):
        try:
            if _SIM is None:
                _SIM = ThorSim()
            t0 = time.time()
            seq = SequenceGenerator(_SIM, split, scene, seed, n_sessions,
                                    out_dir, n_waypoints=n_waypoints).run()
            val = validate_sequence(out_dir)
            with open(os.path.join(out_dir, "validation.json"), "w") as fh:
                json.dump(val, fh)
            print(f"[{split}/{scene}] done in {time.time() - t0:.0f}s "
                  f"valid={val['ok']} failures={len(seq['failures'])}",
                  flush=True)
            return {"scene": scene, "ok": True, "valid": val["ok"],
                    "wall_s": seq["wall_s"], "attempt": attempt}
        except Exception as e:   # noqa: BLE001  (Unity crash/timeout)
            err = f"{type(e).__name__}: {e}"
            print(f"[{split}/{scene}] attempt {attempt} FAILED {err}\n"
                  f"{traceback.format_exc()}", flush=True)
            if _SIM is not None:
                _SIM.stop()
            _SIM = None
    return {"scene": scene, "ok": False, "error": err}


# ── Report ───────────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float)
                                  else str(x))


def scene_summary(seq_dir: str) -> dict:
    seq = load_json(os.path.join(seq_dir, "sequence.json"))
    val = load_json(os.path.join(seq_dir, "validation.json"))
    gts = [load_json(os.path.join(seq_dir, f"session_{k:02d}", "gt.json"))
           for k in range(seq["n_sessions"])]
    names = [sorted(o["gt_id"] for o in g["objects"]) for g in gts]
    ids_changed = 0
    first = {o["gt_id"]: o["object_id"] for o in gts[0]["objects"]}
    for g in gts[1:]:
        for o in g["objects"]:
            if o["present"] and o["object_id"] != first.get(o["gt_id"]):
                ids_changed += 1
    groups = Counter(o["group"] for o in gts[0]["objects"])
    seen = [np.mean([o["seen_frames"] > 0 for o in g["objects"]
                     if o["present"]]) for g in gts]
    return {"seq": seq, "val": val, "gts": gts,
            "names_unique": all(len(n) == len(set(n)) for n in names),
            "names_constant": all(n == names[0] for n in names),
            "n_objects": len(names[0]), "objectid_changes": ids_changed,
            "groups": dict(groups),
            "attached": sum(o["attached_to"] is not None
                            for o in gts[0]["objects"]),
            "seen_frac": seen}


def write_report(seq_dirs, path: str, title: str, cmd: str,
                 t_start: str, t_end: str, wall_s: float) -> None:
    L = [f"# {title}", "",
         f"Written {_now()} by scripts/thor_generate_sessions.py. "
         f"Command: `{cmd}`", "",
         f"Start {t_start} | end {t_end} | wall {wall_s:.0f} s "
         f"(date stamps of this run)", ""]
    tot = Counter()
    for d in seq_dirs:
        if not os.path.exists(os.path.join(d, "sequence.json")):
            L += [f"## {d}: MISSING (no sequence.json)", ""]
            continue
        S = scene_summary(d)
        seq, val = S["seq"], S["val"]
        L += [f"## {seq['split']}/{seq['scene']} (seed {seq['seed']}, "
              f"{seq['n_sessions']} sessions)", ""]
        L += ["| Check | Result | Status |", "|---|---|---|",
              f"| Identity gt_id = Unity name: unique per session | "
              f"{S['names_unique']} | {'PASS' if S['names_unique'] else 'FAIL'} |",
              f"| Same gt_id set in every session | {S['names_constant']} "
              f"({S['n_objects']} objects, Floor excluded) | "
              f"{'PASS' if S['names_constant'] else 'FAIL'} |",
              f"| objectId changed while gt_id constant (present, vs s0) | "
              f"{S['objectid_changes']} object-sessions | info: shows "
              f"objectId is not a stable id |",
              f"| Fixed objects never move (>1 cm or >1 deg) | "
              f"{len(val['fixed_moved'])} violations | "
              f"{'PASS' if not val['fixed_moved'] else 'FAIL'} |",
              f"| Absent (disabled) objects in instance masks | "
              f"{len(val['disabled_in_mask'])} pixels-sets | "
              f"{'PASS' if not val['disabled_in_mask'] else 'FAIL'} |",
              f"| Present objects below the floor (fell through) | "
              f"{len(val.get('present_below_floor', []))} | "
              f"{'PASS' if not val.get('present_below_floor') else 'FAIL'} |",
              f"| Unknown indices in instance PNGs | {val['unknown_index']} "
              f"| {'PASS' if val['unknown_index'] == 0 else 'FAIL'} |",
              f"| GT flags = recomputation from stored poses | "
              f"{len(val['flag_mismatch'])} mismatches | "
              f"{'PASS' if not val['flag_mismatch'] else 'FAIL'} |",
              f"| Duplicates (types, requested -> actual total) | "
              f"{val['duplicates']['types']} "
              f"{val['duplicates']['requested_total']} -> "
              f"{val['duplicates']['actual_total']} | "
              f"{'PASS' if val['duplicates']['requested_total'] == val['duplicates']['actual_total'] else 'FAIL'} |",
              f"| Frames per session | {val['frames']} | "
              f"{'PASS' if min(val['frames']) >= 60 else 'FAIL'} |",
              f"| Top-down render per session (exists; median "
              f"|proj AABB centre - mask centroid| px) | "
              f"{[(t['exists'], None if t['median_err_px'] is None else round(t['median_err_px'], 1)) for t in val['topdown']]}"
              f" at {_fmt((val['topdown'][0] or {}).get('px_per_m'), 0)} px/m"
              f" | {'PASS' if all(t['exists'] and t['ok'] for t in val['topdown']) else 'FAIL'} |",
              ""]
        L += [f"Groups at s0: {S['groups']} (attached fixed parts counted "
              f"as moveable: {S['attached']}). Failures logged: "
              f"{len(seq['failures'])}.", ""]
        L += ["| s | t (days) | gap | frames | teleport fail | moved | "
              "appeared | disappeared | present | seen>=1 frame | "
              "obj drift in session (m) | ms/frame render | wall s | "
              "pose offset xy (m) | yaw (deg) | end drift (m) | "
              "sqrt diag pose_cov (m) |",
              "|" + "---|" * 17]
        for k, (ss, g) in enumerate(zip(seq["sessions"], S["gts"])):
            pe = g["pose_error"]
            dr = np.linalg.norm(np.asarray(pe["drift"])[-1]) \
                if pe["drift"] else 0.0
            sd = np.sqrt(np.diag(np.asarray(g["pose_cov"])))
            L.append(
                f"| {k} | {ss['time'] / 86400:.1f} | "
                f"{_fmt(g['gap_days'], 1)} | {ss['n_frames']} | "
                f"{ss['teleport_failures']} | {ss['moved']} | "
                f"{ss['appeared']} | {ss['disappeared']} | {ss['present']} "
                f"| {S['seen_frac'][k]:.2f} | {ss['drift_m']:.4f} | "
                f"{1000 * ss['render_s'] / max(ss['n_frames'], 1):.0f} | "
                f"{ss['wall_s']:.0f} | "
                f"{np.linalg.norm(pe['offset'][:2]):.3f} | "
                f"{pe['yaw_deg']:.2f} | {dr:.3f} | "
                f"{', '.join(f'{v:.3f}' for v in sd)} |")
            for key in ("moved", "appeared", "disappeared"):
                tot[key] += ss[key]
            tot["frames"] += ss["n_frames"]
            tot["sessions"] += 1
        L += ["", "Intended net change -> actual GT outcome (all sessions):",
              "", "| group:intended->actual | n |", "|---|---|"]
        for key, n in sorted(val["intended_vs_actual"].items()):
            L.append(f"| {key} | {n} |")
        L += ["", "Changes without an intended event (side effects):", "",
              "| group:outcome:reason | n |", "|---|---|"]
        for key, n in sorted(val["side_effects"].items()):
            L.append(f"| {key} | {n} |")
        if seq["failures"]:
            L += ["", "Failures (first 15):", "", "| s | gt_id | what | msg |",
                  "|---|---|---|---|"]
            for f in seq["failures"][:15]:
                L.append(f"| {f['session']} | {f['gt_id']} | {f['what']} | "
                         f"{f['msg'][:80]} |")
        L.append("")
    L += ["## Totals", "", f"{dict(tot)}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def topdown_check_image(session_dir: str, out_path: str,
                        n_objects: int = 3) -> list:
    """Annotate the top-down render: cross = AABB centre projected with
    gt.json 'world_xy_to_pixel', circle = centroid of the object's top-down
    instance mask. Returns [(type, du, dv)] for the drawn objects."""
    import cv2
    gt = load_json(os.path.join(session_dir, "gt.json"))
    td = gt["topdown"]
    img = cv2.imread(os.path.join(session_dir, td["image"]))
    inst = cv2.imread(os.path.join(session_dir, td["inst_image"]),
                      cv2.IMREAD_UNCHANGED)
    A = np.asarray(td["world_xy_to_pixel"])
    objs = {o["gt_id"]: o for o in gt["objects"]}
    out = []
    for gid, typ, npx, du, dv in td["check"]["largest"][:n_objects]:
        idx = int(next(i for i, n in td["instances"].items() if n == gid))
        vv, uu = np.nonzero(inst == idx)
        cu, cv_ = uu.mean() + 0.5, vv.mean() + 0.5
        x, y = objs[gid]["position"][:2]
        u, v, _ = A @ np.array([x, y, 1.0])
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255),
                       cv2.MARKER_CROSS, 28, 3)
        cv2.circle(img, (int(cu), int(cv_)), 14, (0, 255, 0), 3)
        cv2.putText(img, f"{typ} d=({u - cu:.1f},{v - cv_:.1f})px",
                    (int(u) + 16, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        out.append((typ, round(float(u - cu), 2), round(float(v - cv_), 2)))
    cv2.putText(img, "red cross: AABB centre via world_xy_to_pixel; "
                "green circle: mask centroid", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out


def gpu_mem() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_SCENES))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--waypoints", type=int, default=12)
    ap.add_argument("--out-root", default=str(DATA_ROOT / "raw"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", default=None,
                    help="write a markdown report of the scenes here")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--topdown-check", default=None,
                    help="write an annotated top-down check image of the "
                         "first scene's session 0 here")
    args = ap.parse_args(argv)
    if args.jobs > 3:
        raise SystemExit("at most 3 Unity processes on the shared GPU")
    scenes = args.scenes or SPLIT_SCENES[args.split]
    t0, t_start = time.time(), _now()
    print(f"start {t_start} split={args.split} scenes={len(scenes)} "
          f"sessions={args.sessions} jobs={args.jobs} gpu={gpu_mem()}",
          flush=True)
    jobs = []
    for sc in scenes:
        d = os.path.join(args.out_root, args.split, sc)
        if args.report_only:
            continue
        if os.path.exists(os.path.join(d, "sequence.json")) and \
                not args.overwrite:
            print(f"skip {sc} (complete)", flush=True)
            continue
        jobs.append((args.split, sc, args.seed, args.sessions, args.out_root,
                     args.waypoints))
    results = []
    if jobs:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=max(1, args.jobs),
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_scene, j) for j in jobs]
            for fu in as_completed(futs):
                results.append(fu.result())
                print(f"progress {len(results)}/{len(jobs)} gpu={gpu_mem()}",
                      flush=True)
    t_end = _now()
    wall = time.time() - t0
    summ = {"split": args.split, "start": t_start, "end": t_end,
            "wall_s": round(wall, 1), "results": results,
            "argv": sys.argv if argv is None else list(argv)}
    log_dir = DATA_ROOT / "logs" / "thor"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.out_root).name
    with open(log_dir / f"gen_{tag}_{args.split}_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    if args.report:
        write_report([os.path.join(args.out_root, args.split, s)
                      for s in scenes], args.report,
                     f"THOR-Lifelong generation report ({args.split})",
                     " ".join(sys.argv if argv is None else argv),
                     t_start, t_end, wall)
    if args.topdown_check:
        res = topdown_check_image(os.path.join(
            args.out_root, args.split, scenes[0], "session_00"),
            args.topdown_check)
        print(f"topdown check {args.topdown_check}: {res}", flush=True)
    bad = [r for r in results if not r.get("ok")]
    print(f"end {t_end} wall={wall:.0f}s done={len(results) - len(bad)} "
          f"failed={[r['scene'] for r in bad]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
