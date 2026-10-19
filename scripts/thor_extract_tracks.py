#!/usr/bin/env python3
"""
scripts/thor_extract_tracks.py
==============================
Convert generated THOR-Lifelong raw sessions into SessionData files
(src/evaluation/thor_tracks.py):

  <raw-root>/<split>/<scene>/session_<kk>/ -> <out-root>/<split>/<scene>/session_<kk>.npz

Only scenes with sequence.json (complete) are converted; existing .npz are
skipped unless --overwrite. The CPU stage runs in --workers processes, the
GPU stage (CLIP ViT-B/32 quickgelu + DINOv2 ViT-B/14) in this process.

--report writes a markdown validation report (CLIP zero-shot accuracy,
track coverage, SessionData.visible() sanity, end-to-end mapping check).

Usage
  python scripts/thor_extract_tracks.py --splits val train test --workers 6
  python scripts/thor_extract_tracks.py --splits val --report-only \
      --report <DATA>/logs/thor/extract_report.md --e2e-scenes FloorPlan21 FloorPlan221
"""

import os

# single-threaded BLAS: numpy matmuls spawn threads that thrash on a
# loaded machine (CPU stage 25 s -> 9 s per session, 2026-10-19)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import (  # noqa: E402
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402  (registers hiper pkg + rospy stub)

from hiper.evaluation.object_session import (  # noqa: E402
    load_session,
    save_session,
)
from hiper.evaluation.thor_tracks import (  # noqa: E402
    TrackEncoders,
    build_session,
    prepare_session,
)

from hiper.evaluation.paths import THOR_ROOT as DATA_ROOT  # noqa: E402


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _prep(job):
    split, scene, k, raw_dir = job
    t0 = time.time()
    try:
        out = prepare_session(raw_dir, split, scene, k, keep_rgb=True)
        out["cpu_s"] = time.time() - t0
        return out
    except Exception as e:   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(), "split": split,
                "scene": scene, "k": k}


def list_jobs(raw_root: Path, out_root: Path, splits, scenes, overwrite):
    jobs = []
    for sp in splits:
        d = raw_root / sp
        if not d.is_dir():
            continue
        for sc in sorted(os.listdir(d)):
            if scenes and sc not in scenes:
                continue
            seq = d / sc / "sequence.json"
            if not seq.exists():
                continue
            n = json.load(open(seq))["n_sessions"]
            for k in range(n):
                out = out_root / sp / sc / f"session_{k:02d}.npz"
                if out.exists() and not overwrite:
                    continue
                jobs.append((sp, sc, k, str(d / sc / f"session_{k:02d}")))
    return jobs


# ── Report ───────────────────────────────────────────────────────────────────

def session_stats(path: str) -> dict:
    s = load_session(path)
    gt = {g["gt_id"]: g for g in s.gt_objects}
    tracked = {t.gt_id for t in s.tracks}
    rows = []
    for t in s.tracks:
        g = gt[t.gt_id]
        top = [lab for lab, _ in s.meta["clip_top5"][str(t.track_id)]]
        rows.append({"group": g["group"], "type": g["type"],
                     "label": t.clip_label, "top1": t.clip_label == g["type"],
                     "top5": g["type"] in top, "n_obs": t.n_obs,
                     "conf": t.clip_label_conf, "disp": t.clip_disp})
    present = [g for g in s.gt_objects if g["present"]]
    vis_tr = s.visible(np.array([t.centroid for t in s.tracks])) \
        if s.tracks else np.zeros(0, bool)
    never = [g for g in present if g["gt_id"] not in tracked]
    vis_never = s.visible(np.array([g["position"] for g in never])) \
        if never else np.zeros(0, bool)
    # centroid error vs GT AABB centre (estimated frame vs GT frame)
    err = [float(np.linalg.norm(np.asarray(t.centroid) -
                                np.asarray(gt[t.gt_id]["position"])))
           for t in s.tracks]
    return {"rows": rows, "n_tracks": len(s.tracks),
            "n_present": len(present),
            "present_tracked": {grp: [sum(g["gt_id"] in tracked for g in present
                                          if g["group"] == grp),
                                      sum(g["group"] == grp for g in present)]
                                for grp in ("fixed", "moveable",
                                            "pickupable")},
            "vis_tracked": [int(vis_tr.sum()), len(vis_tr)],
            "vis_never": [int(vis_never.sum()), len(vis_never)],
            "never_types": Counter(g["type"] for g in never),
            "centroid_err": err, "n_frames": s.meta.get("n_frames"),
            "depth_shape": list(s.depth_stack.shape)}


def _f3(x):
    return "-" if x is None else f"{x:.3f}"


def unmoved_shift(files) -> dict:
    """Distance between the track centroids of the same GT object in
    consecutive sessions when the GT says it did not move (measurement
    noise: partial views, pose error of both sessions)."""
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.parent, []).append(f)
    small, large = [], []
    for sc, fs in by_scene.items():
        prev = None
        for f in sorted(fs):
            s = load_session(str(f))
            cur = {t.gt_id: t.centroid for t in s.tracks}
            if prev is not None:
                g = {x["gt_id"]: x for x in s.gt_objects}
                for gid, c in cur.items():
                    if gid in prev and g[gid]["present"] and not \
                            g[gid]["moved_since_prev"] and not \
                            g[gid]["appeared_since_prev"]:
                        d = float(np.linalg.norm(np.asarray(c) -
                                                 np.asarray(prev[gid])))
                        (large if max(g[gid]["aabb_size"]) > 1.0
                         else small).append(d)
            prev = cur
    out = {}
    for k, v in (("small (max AABB side <= 1 m)", small),
                 ("large (> 1 m)", large)):
        if v:
            out[k] = (len(v), float(np.median(v)), float(np.percentile(v, 90)),
                      float(np.mean(np.asarray(v) > 0.5)))
    return out


def appearance_separability(files, max_scenes: int = 20) -> dict:
    """Cosine of track embeddings: same GT object in consecutive sessions
    vs different GT objects in the same session (association headroom)."""
    by_scene = {}
    for f in files:
        by_scene.setdefault(f.parent, []).append(f)
    out = {}
    for key in ("clip_mean", "dino_mean"):
        same, diff = [], []
        for sc in sorted(by_scene)[:max_scenes]:
            prev = None
            for f in sorted(by_scene[sc]):
                s = load_session(str(f))
                E = {t.gt_id: np.asarray(getattr(t, key), np.float64)
                     for t in s.tracks if getattr(t, key) is not None}
                ids = sorted(E)
                if len(ids) > 1:
                    M = np.array([E[i] for i in ids])
                    C = M @ M.T
                    diff += C[np.triu_indices(len(ids), 1)].tolist()
                if prev is not None:
                    same += [float(E[i] @ prev[i]) for i in ids if i in prev]
                prev = E
        if same and diff:
            out[key] = {"same_median": float(np.median(same)),
                        "same_p10": float(np.percentile(same, 10)),
                        "diff_median": float(np.median(diff)),
                        "diff_p90": float(np.percentile(diff, 90)),
                        "diff_p99": float(np.percentile(diff, 99))}
    return out


def e2e_check(out_root: Path, split: str, scenes, priors: str) -> list:
    from hiper.ephemerality.persistence_v2 import (
        HierarchicalPersistenceFilter,
    )
    from hiper.evaluation.lifelong_objects import (
        load_class_lambdas,
        replay,
        run_mapping,
    )
    from hiper.object_encoder.association_v2 import AssocConfig
    lam, lam0 = load_class_lambdas(priors)
    res = []
    for sc in scenes:
        files = sorted((out_root / split / sc).glob("session_*.npz"))
        sessions = [load_session(str(f)) for f in files]
        t0 = time.time()
        stream, assoc = run_mapping(sessions, AssocConfig(name="clip_only",
                                                          w=1.0),
                                    split=split)
        rows = replay(stream, HierarchicalPersistenceFilter(
            lambda_table=lam, default_lambda=lam0))
        ev = Counter(e["ev"] for ob in stream["objects"].values()
                     for e in ob["events"])
        lab = [r["label_next"] for r in rows if r["label_next"] is not None]
        mapped = [a for a in assoc if a["gt_mapped_before"]]
        novel = [a for a in assoc if not a["gt_mapped_before"]]
        rel = [a for a in assoc if a["relocated_pred"]]
        res.append({
            "scene": sc, "n_sessions": len(sessions),
            "n_objects": len(stream["objects"]), "events": dict(ev),
            "pred_rows": len(rows), "label_rows": len(lab),
            "frac_label1": float(np.mean(lab)) if lab else None,
            "assoc_rows": len(assoc),
            "reid_success_mapped": float(np.mean(
                [a["matched_gt_id"] == a["gt_id"] for a in mapped]))
            if mapped else None,
            "novel_false_match": float(np.mean(
                [a["matched_oid"] is not None for a in novel]))
            if novel else None,
            "n_novel_rows": len(novel), "n_relocated_pred": len(rel),
            "relocated_pred_gt_moved": sum(bool(a["gt_moved"]) for a in rel),
            "pred_presence_mean": float(np.mean(
                [r["pred_presence"] for r in rows
                 if r["pred_presence"] is not None])),
            "seconds": round(time.time() - t0, 1)})
    return res


def write_report(out_root: Path, splits, path: str, e2e_scenes, priors,
                 extra: dict) -> None:
    L = [f"# THOR-Lifelong track extraction report", "",
         f"Written {_now()} by scripts/thor_extract_tracks.py "
         f"(sessions under {out_root}).", ""]
    for k, v in extra.items():
        L.append(f"- {k}: {v}")
    L.append("")
    for sp in splits:
        files = sorted((out_root / sp).glob("*/session_*.npz"))
        if not files:
            continue
        st = [session_stats(str(f)) for f in files]
        rows = [r for s in st for r in s["rows"]]
        n_sc = len({f.parent.name for f in files})
        L += [f"## Split {sp}: {n_sc} scenes, {len(files)} sessions, "
              f"{sum(s['n_tracks'] for s in st)} tracks", ""]
        nt = np.array([s["n_tracks"] for s in st])
        L += [f"Tracks per session: mean {nt.mean():.1f}, min {nt.min()}, "
              f"max {nt.max()}. Frames per session: "
              f"{sorted(Counter(s['n_frames'] for s in st).items())}. "
              f"depth_stack {st[0]['depth_shape']}.", ""]
        L += ["| Group | tracks | CLIP top-1 | CLIP top-5 | present GT "
              "objects with a track |", "|---|---|---|---|---|"]
        for grp in ("fixed", "moveable", "pickupable", "all"):
            rr = [r for r in rows if grp == "all" or r["group"] == grp]
            a = sum(s["present_tracked"][g][0] for s in st
                    for g in s["present_tracked"]
                    if grp == "all" or g == grp)
            b = sum(s["present_tracked"][g][1] for s in st
                    for g in s["present_tracked"]
                    if grp == "all" or g == grp)
            if not rr:
                continue
            L.append(f"| {grp} | {len(rr)} | "
                     f"{np.mean([r['top1'] for r in rr]):.3f} | "
                     f"{np.mean([r['top5'] for r in rr]):.3f} | "
                     f"{a}/{b} = {a / max(b, 1):.3f} |")
        vt = np.sum([s["vis_tracked"] for s in st], axis=0)
        vn = np.sum([s["vis_never"] for s in st], axis=0)
        ce = np.concatenate([s["centroid_err"] for s in st])
        L += ["", "| visible() sanity | value |", "|---|---|",
              f"| tracked objects whose own centroid is visible() | "
              f"{vt[0]}/{vt[1]} = {vt[0] / max(vt[1], 1):.3f} |",
              f"| present GT objects never segmented, reported visible() "
              f"at their GT position | {vn[0]}/{vn[1]} = "
              f"{vn[0] / max(vn[1], 1):.3f} |",
              f"| track centroid - GT AABB centre (m): median / p90 | "
              f"{np.median(ce):.3f} / {np.percentile(ce, 90):.3f} |", ""]
        never = Counter()
        for s in st:
            never.update(s["never_types"])
        L += ["| unmoved object, centroid shift between consecutive "
              "sessions | n | median (m) | p90 (m) | frac > 0.5 m |",
              "|---|---|---|---|---|"]
        for k, (n, med, p90, f5) in unmoved_shift(files).items():
            L.append(f"| {k} | {n} | {med:.3f} | {p90:.3f} | {f5:.3f} |")
        L.append("")
        sep = appearance_separability(files)
        if sep:
            L += ["| embedding cosine | same object, consecutive sessions: "
                  "median / p10 | different objects, same session: median "
                  "/ p90 / p99 |", "|---|---|---|"]
            for k, v in sep.items():
                L.append(f"| {k} | {v['same_median']:.3f} / "
                         f"{v['same_p10']:.3f} | {v['diff_median']:.3f} / "
                         f"{v['diff_p90']:.3f} / {v['diff_p99']:.3f} |")
            L.append("")
        L += [f"Never-segmented present objects, top types: "
              f"{never.most_common(12)}", ""]
        conf = Counter((r["type"], r["label"]) for r in rows
                       if not r["top1"])
        L += [f"Most frequent CLIP confusions (GT type -> label): "
              f"{conf.most_common(12)}", ""]
        by_type = defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r["top1"])
        worst = sorted(((np.mean(v), len(v), t) for t, v in by_type.items()
                        if len(v) >= 10))[:10]
        L += ["Lowest top-1 types (n >= 10): " + ", ".join(
            f"{t} {a:.2f} (n={n})" for a, n, t in worst), ""]
    if e2e_scenes:
        res = e2e_check(out_root, "val", e2e_scenes, priors)
        L += ["## End-to-end check (val): run_mapping AssocConfig(w=1.0, "
              "CLIP only) + replay HierarchicalPersistenceFilter "
              "(class priors from config/class_priors_indoor.yaml)", "",
              "| scene | sessions | map objects | events | pred rows | "
              "label_next=1 | re-ID success (mapped) | novel false match "
              "(n novel) | predicted relocations (GT moved) | "
              "mean pred_presence |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            L.append(f"| {r['scene']} | {r['n_sessions']} | {r['n_objects']}"
                     f" | {r['events']} | {r['pred_rows']} | "
                     f"{r['frac_label1']:.3f} (n={r['label_rows']}) | "
                     f"{_f3(r['reid_success_mapped'])} | "
                     f"{_f3(r['novel_false_match'])} ({r['n_novel_rows']}) | "
                     f"{r['n_relocated_pred']} "
                     f"({r['relocated_pred_gt_moved']}) | "
                     f"{r['pred_presence_mean']:.3f} |")
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--raw-root", default=str(DATA_ROOT / "raw"))
    ap.add_argument("--out-root", default=str(DATA_ROOT / "sessions"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--e2e-scenes", nargs="*", default=[])
    ap.add_argument("--priors", default=str(
        _PKG_ROOT / "config" / "class_priors_indoor.yaml"))
    args = ap.parse_args(argv)
    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    t0, t_start = time.time(), _now()
    jobs = [] if args.report_only else list_jobs(
        raw_root, out_root, args.splits, args.scenes, args.overwrite)
    print(f"start {t_start} jobs={len(jobs)}", flush=True)
    timings, failed = [], []
    if jobs:
        enc = TrackEncoders()
        import multiprocessing as mp
        saver = ThreadPoolExecutor(max_workers=2)
        pending = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            # bounded window: finished results (with RGB frames) must not
            # pile up while the GPU stage is the bottleneck
            todo = list(jobs)
            live = set()
            i = -1

            def results():
                while todo or live:
                    while todo and len(live) < 2 * args.workers:
                        live.add(ex.submit(_prep, todo.pop(0)))
                    done = next(as_completed(live))
                    live.discard(done)
                    yield done.result()

            for prep in results():
                i += 1
                if "error" in prep:
                    failed.append(prep)
                    print(f"FAIL {prep['split']}/{prep['scene']} "
                          f"s{prep['k']}: {prep['error']}", flush=True)
                    continue
                t1 = time.time()
                sess = build_session(prep, enc)
                out = out_root / prep["split"] / prep["scene"] / \
                    f"session_{prep['k']:02d}.npz"
                pending.append(saver.submit(save_session, sess, str(out)))
                timings.append({"split": prep["split"],
                                "scene": prep["scene"], "k": prep["k"],
                                "cpu_s": round(prep["cpu_s"], 2),
                                "gpu_s": round(time.time() - t1, 2),
                                "tracks": len(sess.tracks)})
                if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                    print(f"{i + 1}/{len(jobs)} {_now()} last "
                          f"{prep['split']}/{prep['scene']} s{prep['k']} "
                          f"tracks={len(sess.tracks)} "
                          f"gpu={time.time() - t1:.1f}s", flush=True)
        for f in pending:
            f.result()
        saver.shutdown()
    t_end = _now()
    wall = time.time() - t0
    log_dir = DATA_ROOT / "logs" / "thor"
    log_dir.mkdir(parents=True, exist_ok=True)
    if jobs:
        with open(log_dir / f"extract_{'_'.join(args.splits)}_summary.json",
                  "w") as fh:
            json.dump({"start": t_start, "end": t_end, "wall_s": wall,
                       "n_jobs": len(jobs), "failed": failed,
                       "timings": timings, "argv": sys.argv}, fh, indent=1)
    if args.report:
        extra = {"run": f"start {t_start}, end {t_end}, wall {wall:.0f} s, "
                        f"jobs {len(jobs)}, failed {len(failed)}"}
        write_report(out_root, args.splits, args.report, args.e2e_scenes,
                     args.priors, extra)
    print(f"end {t_end} wall={wall:.0f}s done={len(timings)} "
          f"failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
