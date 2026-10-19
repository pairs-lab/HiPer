#!/usr/bin/env python3
"""
Submission video (RA-L: one anonymized mp4). Every panel is drawn from the
saved runs; nothing is staged. Segments:
  1 title  2 problem (one scene changing across sessions)  3 idea (belief
  of two mugs, figures/data/hpf_belief.csv)  4 THOR-Lifelong: ego views +
  map decisions of ours vs a baseline per session  5 ObChange real robot
  6 results (summary.json)  7 closing.

  python scripts/make_video.py --run main --split test --out paper/video.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper/figures"))
import conftest  # noqa: E402,F401

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import make_qualitative as mq  # noqa: E402
import tune_and_eval as te  # noqa: E402
from hiper.evaluation import obchange as ob  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402

W, H, FPS = 12.8, 7.2, 24
BG = "#111111"
TITLE = ("Which Objects Stay? Learning Hierarchical Persistence Beliefs\n"
         "for Lifelong Open-Vocabulary Object Maps")


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    return fig


def grab(fig) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def text_card(lines, sub=None):
    fig = new_fig()
    fig.text(0.5, 0.58, lines, ha="center", va="center", color="white",
             fontsize=26, weight="bold")
    if sub:
        fig.text(0.5, 0.32, sub, ha="center", va="center", color="#bbbbbb",
                 fontsize=16)
    return grab(fig)


def caption(fig, text):
    fig.text(0.5, 0.035, text, ha="center", va="center", color="white",
             fontsize=15)


def draw_map(ax, img, pts, title, counts=None):
    ax.imshow(img)
    for (u, v, o, com) in pts:
        mk, col, sz = mq.STYLE[o]
        ax.scatter(u, v, marker=mk, c=col, s=sz * 3, linewidths=1.6)
        if com:
            ax.scatter(u, v, marker="o", facecolors="none",
                       edgecolors="#56B4E9", s=150, linewidths=1.3)
    ax.set_title(title, color="white", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    if counts:
        ax.text(0.02, 0.04, counts, transform=ax.transAxes, color="white",
                fontsize=12, bbox=dict(facecolor="black", alpha=0.6,
                                       edgecolor="none"))


def legend_strip(fig):
    handles = [plt.Line2D([], [], marker=mq.STYLE[o][0],
                          color=mq.STYLE[o][1], linestyle="none",
                          markersize=9, label=mq.LABEL[o])
               for o in ("present", "gone", "stale", "premature",
                         "pruned_ok")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=5,
                     fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.07))
    for t in leg.get_texts():
        t.set_color("white")


def thor_segment(args, sel, scene):
    """Per session: ego flipbook (left) + ours vs baseline maps (right)."""
    files = te.scene_files(args.split)[scene]
    sessions = [load_session(f) for f in files]
    pos = mq.mapped_positions(sessions, {**sel["assoc"], "name": "ours"})
    pred = te.RESULTS / args.run / args.split / "pred.jsonl"
    rows = mq.load_rows(pred, scene, {"ours", args.baseline})
    raw = mq.DATA / "raw" / args.split / scene
    frames = []
    for k in range(len(sessions)):
        gt = json.load(open(raw / f"session_{k:02d}/gt.json"))
        Hk = np.array(gt["topdown"]["world_xy_to_pixel"])
        top = np.asarray(Image.open(raw / f"session_{k:02d}/topdown.png"))
        panels = {}
        for m in ("ours", args.baseline):
            oc = mq.outcomes([r for r in rows if r["model"] == m], k)
            pts, cnt = [], {"premature": 0, "stale": 0}
            for oid, (o, com) in oc.items():
                if oid in pos[k]:
                    u, v, _ = Hk @ np.r_[pos[k][oid], 1.0]
                    pts.append((u, v, o, com))
                    cnt[o] = cnt.get(o, 0) + 1
            panels[m] = (pts, f"premature {cnt['premature']}  "
                              f"stale {cnt['stale']}")
        ego = sorted((raw / f"session_{k:02d}/frames").glob("*_rgb.jpg"))
        pick = ego[:: max(1, len(ego) // 8)][:8]
        day = sessions[k].time / 86400.0
        for f in pick:
            fig = new_fig()
            ax0 = fig.add_axes([0.01, 0.30, 0.32, 0.45])
            ax0.imshow(np.asarray(Image.open(f)))
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_title(f"robot view, session {k} (day {day:.1f})",
                          color="white", fontsize=14)
            for i, m in enumerate(("ours", args.baseline)):
                ax = fig.add_axes([0.35 + 0.325 * i, 0.15, 0.31, 0.72])
                draw_map(ax, top, panels[m][0], mq.NAMES.get(m, m),
                         panels[m][1])
            legend_strip(fig)
            caption(fig, "Simulated rooms (AI2-THOR): objects move or "
                         "vanish between visits")
            img = grab(fig)
            frames += [img] * int(FPS * 0.45)
    return frames


def belief_segment():
    """Animate figures/data/hpf_belief.csv (two mugs, same class prior)."""
    import csv
    rows = list(csv.DictReader(open(ROOT / "paper/figures/data/"
                                    "hpf_belief.csv")))
    d = np.array([float(r["day"]) for r in rows])
    a = np.array([float(r["a_pred"]) for r in rows])
    b = np.array([float(r["b_pred"]) for r in rows])
    p = np.array([float(r["pf_pred"]) for r in rows])
    frames = []
    for n in range(1, len(d) + 1):
        fig = new_fig()
        ax = fig.add_axes([0.1, 0.18, 0.8, 0.62], facecolor="#1c1c1c")
        ax.plot(d[:n], a[:n], "o-", color="#56B4E9", lw=2.5,
                label="HiPer: mug that stays")
        ax.plot(d[:n], b[:n], "s-", color="#E69F00", lw=2.5,
                label="HiPer: mug moved every 2nd day")
        ax.plot(d[:n], p[:n], "--", color="#bbbbbb", lw=2,
                label="fixed class rate: both mugs")
        ax.set_xlim(0.5, d[-1] + 0.5)
        ax.set_ylim(0.35, 1.0)
        ax.set_xlabel("day", color="white", fontsize=14)
        ax.set_ylabel("P(still there tomorrow)", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(loc="lower left", fontsize=13, facecolor="#1c1c1c",
                  labelcolor="white")
        fig.text(0.5, 0.9, "A CLIP class label gives the prior; each object's "
                 "own history refines it", ha="center", color="white",
                 fontsize=18)
        frames += [grab(fig)] * int(FPS * 0.5)
    return frames


def obchange_segment(args, sel, room):
    """Real robot: plane clouds seen from above, ours vs baseline."""
    res = te.RESULTS / args.run / "obchange_loro" / "pred.jsonl"
    scene = f"obchange_{room}"
    files = te.scene_files("obchange")[room]
    sessions = [load_session(f) for f in files]
    pos = mq.mapped_positions(sessions, {**sel["assoc"], "name": "ours"})
    rows = mq.load_rows(res, scene, {"ours", args.baseline})
    visits = ob.list_visits(str(te.OBC.parent / "extracted"), room)
    frames = []
    for k, vd in enumerate(visits):
        v = ob.load_visit(vd, k)
        xyz = np.vstack([p["xyz"] for p in v.planes.values()])
        rgb = np.vstack([p["rgb"] for p in v.planes.values()])
        keep = (xyz[:, 2] > 0.25)          # tables and objects, not floor
        xyz, rgb = xyz[keep][::7], rgb[keep][::7]
        fig = new_fig()
        for i, m in enumerate(("ours", args.baseline)):
            ax = fig.add_axes([0.04 + 0.48 * i, 0.15, 0.44, 0.72],
                              facecolor="black")
            ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=0.4, linewidths=0)
            oc = mq.outcomes([r for r in rows if r["model"] == m], k)
            cnt = {"premature": 0, "stale": 0}
            for oid, (o, com) in oc.items():
                if oid in pos[k]:
                    x, y = pos[k][oid]
                    mk, col, sz = mq.STYLE[o]
                    ax.scatter(x, y, marker=mk, c=col, s=sz * 5,
                               linewidths=2)
                    cnt[o] = cnt.get(o, 0) + 1
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{mq.NAMES.get(m, m)} | visit {k}: premature "
                         f"{cnt['premature']}, stale {cnt['stale']}",
                         color="white", fontsize=14)
        legend_strip(fig)
        caption(fig, "Real robot (ObChange, Toyota HSR); class rates "
                     "learned on the other rooms")
        frames += [grab(fig)] * int(FPS * 2.5)
    return frames


def results_segment(args):
    names = ["ours", "pf_learned", "pf", "pocd", "elite", "draft_v1"]
    frames = []
    for split, label in (("test", "Simulated test scenes (20)"),
                         ("obchange_loro", "Real robot, leave-one-room-out "
                          "(5 rooms)")):
        path = te.RESULTS / args.run / split / "summary.json"
        if not path.exists():
            continue
        s = json.load(open(path))["persistence"]
        fig = new_fig()
        for i, (met, lab) in enumerate((("nll", "NLL (lower better)"),
                                        ("ece", "ECE (lower better)"))):
            ax = fig.add_axes([0.07 + 0.47 * i, 0.2, 0.42, 0.6],
                              facecolor="#1c1c1c")
            vals = [s[n][met][0] for n in names]
            errs = [s[n][met][1] for n in names]
            ax.bar(range(len(names)), vals, yerr=errs, capsize=4,
                   color=["#56B4E9"] + ["#888888"] * (len(names) - 1))
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([mq.NAMES.get(n, n) for n in names],
                               rotation=25, color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.set_title(lab, color="white", fontsize=15)
        fig.text(0.5, 0.9, f"{label}: mean $\\pm$ std over scenes",
                 ha="center", color="white", fontsize=18)
        frames += [grab(fig)] * int(FPS * 5)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", default=None,
                    help="default: the qualitative figure's scene")
    ap.add_argument("--baseline", default="pf_learned")
    ap.add_argument("--room", default="SmallRoom")
    ap.add_argument("--out", default=str(ROOT / "paper/video.mp4"))
    args = ap.parse_args()
    sel = json.load(open(te.RESULTS / args.run / "final.json"))
    qual = ROOT / "paper/figures/qualitative_prob.json"
    scene = args.scene or (json.load(open(qual))["scene"] if qual.exists()
                           else mq.choose_scene(te.RESULTS / args.run /
                                                args.split / "pred.jsonl",
                                                "ours", args.baseline))
    frames = [text_card(TITLE, "HiPer  |  RA-L submission video (anonymized)")] \
        * (FPS * 4)
    frames += [text_card("A lifelong object map must decide which objects "
                         "are still there.",
                         "Mugs move daily, counters never: a class prior "
                         "helps, but objects of one class differ.")] \
        * (FPS * 5)
    frames += belief_segment()
    frames += thor_segment(args, sel, scene)
    if (te.RESULTS / args.run / "obchange_loro" / "pred.jsonl").exists():
        frames += obchange_segment(args, sel, args.room)
    frames += results_segment(args)
    frames += [text_card("Code and data: anonymized link in the paper")] \
        * (FPS * 3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(args.out, fps=FPS, codec="libx264",
                           quality=None, ffmpeg_params=[
                               "-crf", "24", "-preset", "slow"],
                           pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    print(f"{args.out}: {len(frames) / FPS:.1f} s, "
          f"{Path(args.out).stat().st_size / 1e6:.1f} MB, scene {scene}")


if __name__ == "__main__":
    main()
