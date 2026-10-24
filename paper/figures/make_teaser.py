#!/usr/bin/env python3
"""
Page-1 teaser: what the robot sees in four sessions of one THOR-Lifelong
test scene (top) and the predicted probability that three objects are
still in place at the next session, HiPer vs PF(lambda) (bottom).

Nothing is hand-picked:
  scene    the test scene with the median per-scene NLL difference
           HiPer - PF(lambda) (same rule as make_qualitative.py);
  objects  for each story (stays / is moved / disappears) the pickupable
           or moveable object with the most pixels seen over all sessions
           (ground truth only, independent of any model);
  frames   per session, the frame showing most of the three objects.
Right: per-sequence NLL of HiPer vs the baseline on every evaluated
sequence (paper/figures/data/per_sequence.csv, written by
make_extra_results.py; run that first).

  python paper/figures/make_teaser.py --run v5 --split heldout
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper/figures"))
import conftest  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

import figstyle  # noqa: E402
import make_qualitative as mq  # noqa: E402
import tune_and_eval as te  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402

STORY_COLORS = {"stays": "#1b7837", "moved": "#2166ac",
                "disappears": "#b2182b"}
OURS_COLOR = "#0072B2"
PER_SEQ = ROOT / "paper/figures/data/per_sequence.csv"
CMAP = plt.get_cmap("RdYlGn")


def humanize(t):
    import re
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", t).lower()


def stories(sessions, gain=None):
    """{story: gt_id}: the most visible candidate per story (ground truth
    only), or, with ``gain`` ({gt_id: HiPer's Brier gain over the
    baseline}), the candidate with the largest gain among those at least as
    visible as the story's median candidate (a best case)."""
    hist = {}
    for k, s in enumerate(sessions):
        for o in s.gt_objects:
            h = hist.setdefault(o["gt_id"], {"type": o["type"],
                                             "group": o.get("group"),
                                             "present": [], "moved": [],
                                             "pix": 0})
            h["present"].append(bool(o["present"]))
            h["moved"].append(bool(o.get("moved_since_prev")))
            h["pix"] += int(o.get("seen_pixels") or 0)
    n = len(sessions)
    cand = {"stays": [], "moved": [], "disappears": []}
    for g, h in hist.items():
        if h["group"] not in ("pickupable", "moveable") or \
                len(h["present"]) != n or not h["present"][0]:
            continue
        if all(h["present"]) and not any(h["moved"]):
            cand["stays"].append((h["pix"], g))
        elif all(h["present"]) and any(h["moved"]):
            cand["moved"].append((h["pix"], g))
        elif not any(h["moved"]):
            # leaves for good: present, then absent until the end
            first_gone = h["present"].index(False)
            if not any(h["present"][first_gone:]):
                cand["disappears"].append((h["pix"], g))
    out = {}
    for st, c in cand.items():
        if not c:
            continue
        if gain is None:
            out[st] = max(c)[1]
            continue
        med = float(np.median([px for px, _ in c]))
        vis = [(gain.get(g, -np.inf), px, g) for px, g in c if px >= med]
        out[st] = max(vis)[2]
    return out, hist


def most_changes(pred: Path) -> str:
    """Results-independent rule: the evaluated sequence with the most
    ground-truth change events (moves, removals, reappearances); ties by
    name. Sequence ids may be '<split>:<scene>' (pooled held-out sets)."""
    ids = sorted({json.loads(l)["scene"] for l in open(pred)})
    best = None
    for sid in ids:
        sp, sc = sid.split(":", 1) if ":" in sid else (None, sid)
        n = 0
        for f in te.scene_files(sp)[sc]:
            for o in load_session(f).gt_objects:
                n += (bool(o.get("moved_since_prev"))
                      + bool(o.get("disappeared_since_prev"))
                      + bool(o.get("appeared_since_prev")))
        if best is None or n > best[0]:
            best = (n, sid)
    return best[1]


def draw_per_sequence(ax, baseline, scene, label):
    """HiPer vs baseline NLL for every sequence; below the diagonal means
    HiPer is better. Returns (wins, n)."""
    import csv
    rows = list(csv.DictReader(open(PER_SEQ)))
    x = np.array([float(r[f"{baseline}_nll"]) for r in rows])
    y = np.array([float(r["ours_nll"]) for r in rows])
    win = y < x
    lo, hi = min(x.min(), y.min()) * 0.85, max(x.max(), y.max()) * 1.15
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=OURS_COLOR,
                    alpha=0.08, lw=0)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.7, ls="--")
    ax.scatter(x[~win], y[~win], s=7, c="0.55", lw=0, zorder=3)
    ax.scatter(x[win], y[win], s=7, c=OURS_COLOR, lw=0, zorder=3)
    me = [i for i, r in enumerate(rows) if r["seq"] == scene]
    if me:
        ax.scatter(x[me], y[me], s=55, marker="*", c="#F0E442",
                   edgecolors="black", lw=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [t for t in (0.1, 0.2, 0.5, 1.0) if lo <= t <= hi]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        a.set_major_formatter(matplotlib.ticker.FixedFormatter(
            [f"{t:g}" for t in ticks]))
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(f"{label} NLL", fontsize=6.5, labelpad=1)
    ax.set_ylabel("HiPer NLL", fontsize=6.5, labelpad=1)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(labelsize=5.5, pad=1, length=2)
    ax.set_title(f"HiPer better in {win.sum()}/{len(win)}\nheld-out "
                 "sequences", fontsize=7, pad=2, color=OURS_COLOR,
                 fontweight="bold")
    return int(win.sum()), len(win)


def best_frame(frames_dir: Path, gids):
    best, best_key = None, (-1, -1)
    for js in sorted(frames_dir.glob("*.json")):
        d = json.load(open(js))
        inst = {v["gt_id"]: int(i) for i, v in d["instances"].items()}
        vis = [g for g in gids if g in inst
               and d["instances"][str(inst[g])]["pixels"] >= 150]
        key = (len(vis), sum(d["instances"][str(inst[g])]["pixels"]
                             for g in vis))
        if key > best_key:
            best, best_key = (js, inst), key
    return best


def series(rows, gid, model, n):
    p = [np.nan] * n
    y = [np.nan] * n
    for r in rows:
        if r["model"] != model or r["gt_id"] != gid:
            continue
        if r["pred_presence"] is None or r["label_next"] is None:
            continue
        k = r["k"]
        p[k] = r["pred_presence"] if np.isnan(p[k]) else \
            max(p[k], r["pred_presence"])
        y[k] = r["label_next"]
    return np.array(p), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--scene-rule", choices=["events", "median", "best"],
                    default="events",
                    help="events: most ground-truth changes (independent of "
                         "any result); median: median NLL difference; best: "
                         "HiPer's largest smallest NLL margin over --best-over")
    ap.add_argument("--best-over", nargs="+", default=["pf", "elite", "pocd"])
    ap.add_argument("--object-rule", choices=["visibility", "gain"],
                    default="visibility",
                    help="gain: per story the object with HiPer's largest "
                         "Brier gain over --baseline (among well-visible "
                         "ones)")
    ap.add_argument("--columns", nargs="+", type=int, default=[0, 2, 4, 6])
    ap.add_argument("--baseline", default="pf")
    ap.add_argument("--out", default=str(ROOT / "paper/figures/teaser"))
    args = ap.parse_args()
    res = te.RESULTS / args.run / args.split
    pred = res / "pred.jsonl"
    if args.scene:
        scene = args.scene
    elif args.scene_rule == "events":
        scene = most_changes(pred)
    elif args.scene_rule == "best":
        scene = mq.choose_scene(pred, "ours", args.baseline, "nll", "best",
                                others=args.best_over)
    else:
        scene = mq.choose_scene(pred, "ours", args.baseline, "nll")
    # pooled held-out sets name sequences '<split>:<scene>'
    split, sname = (scene.split(":", 1) if ":" in scene
                    else (args.split, scene))
    files = te.scene_files(split)[sname]
    sessions = sorted((load_session(f) for f in files),
                      key=lambda s: s.session_index)
    n = len(sessions)
    days = [s.time / 86400.0 for s in sessions]
    rows = mq.load_rows(pred, scene, ["ours", args.baseline])
    gain = None
    if args.object_rule == "gain":
        se = {m: {} for m in ("ours", args.baseline)}
        for r in rows:
            if r["pred_presence"] is None or r["label_next"] is None:
                continue
            se[r["model"]].setdefault(r["gt_id"], []).append(
                (r["pred_presence"] - r["label_next"]) ** 2)
        gain = {g: float(np.mean(se[args.baseline][g]) - np.mean(v))
                for g, v in se["ours"].items() if g in se[args.baseline]}
    chosen, hist = stories(sessions, gain)
    raw = te.THOR.parent / "raw" / split / sname

    plt.rcParams.update({"font.size": 7})
    figstyle.use_latex_font()
    fig = plt.figure(figsize=(7.16, 2.75))
    ncol = len(args.columns)
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.45, 1.0], hspace=0.32,
                          wspace=0.04, left=0.075, right=0.955, top=0.94,
                          bottom=0.13)
    gids = list(chosen.values())
    for c, k in enumerate(args.columns):
        ax = fig.add_subplot(gs[0, c])
        js, inst = best_frame(raw / f"session_{k:02d}" / "frames", gids)
        stem = js.name[:-5]
        rgb = np.asarray(Image.open(js.parent / f"{stem}_rgb.jpg"))
        ids = np.asarray(Image.open(js.parent / f"{stem}_inst.png"))
        img = rgb.astype(float) / 255.0
        ax.imshow(img)
        placed = []
        for st, g in chosen.items():
            if g not in inst:
                continue
            m = ids == inst[g]
            if m.sum() < 30:
                continue
            p, _ = series(rows, g, "ours", n)
            col = STORY_COLORS[st]
            edge = m & ~ndimage.binary_erosion(m, iterations=3)
            ov = np.zeros((*m.shape, 4))
            ov[edge] = matplotlib.colors.to_rgba(col)
            ax.imshow(ov)
            fill = np.zeros((*m.shape, 4))
            fill[m] = matplotlib.colors.to_rgba(col, 0.25)
            ax.imshow(fill)
            yy, xx = np.nonzero(m)
            txt = humanize(hist[g]["type"])
            if np.isfinite(p[k]):
                txt += f" {p[k]:.2f}"
            H, W = m.shape
            # label box in pixels (approx.); shift down until it is free
            w_px, h_px = 15 * len(txt), 40
            x = float(np.clip(xx.mean(), 120, W - 120))
            y = float(np.clip(yy.min() - 6, 30, H - 10))
            while any(abs(x - px) < (w_px + pw) / 2 and abs(y - py) < h_px
                      for px, py, pw in placed) and y < H - 10:
                y += h_px
            placed.append((x, y, w_px))
            ax.text(x, y, txt,
                    color="white", fontsize=5.5, ha="center", va="bottom",
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.12", fc=col, ec="none",
                              alpha=0.9))
        ax.set_title(f"session {k} (day {days[k]:.1f})", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=[2.2, 0.95, 0.85],
                                  wspace=0.06)
    ax = fig.add_subplot(bottom[0])
    lax = fig.add_subplot(bottom[1])
    lax.axis("off")
    wins = draw_per_sequence(fig.add_subplot(bottom[2]), args.baseline,
                             scene, r"PF($\lambda$)" if args.baseline == "pf"
                             else args.baseline)
    n = n - 1                      # the last session has no next session
    x = np.arange(n)
    for st, g in chosen.items():
        col = STORY_COLORS[st]
        p_o, y = series(rows, g, "ours", n)
        p_b, _ = series(rows, g, args.baseline, n)
        lab = f"{humanize(hist[g]['type'])} ({st})"
        ax.plot(x, p_o, "-", color=col, lw=1.8, marker="o", ms=2.8,
                label=lab)
        ax.plot(x, p_b, "--", color=col, lw=0.9, alpha=0.6)
        ok = np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=22, facecolors=np.where(
            y[ok] > 0.5, col, "white"), edgecolors=col, zorder=4, lw=1.0)
    ax.plot([], [], "k-", lw=1.8, label="HiPer (ours)")
    ax.plot([], [], "k--", lw=1.0, label=r"PF($\lambda$)")
    ax.scatter([], [], s=22, facecolors="k", edgecolors="k",
               label="truth: still there")
    ax.scatter([], [], s=22, facecolors="white", edgecolors="k",
               label="truth: gone")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} ({d:.1f} d)" for k, d in zip(x, days)],
                       fontsize=6)
    ax.set_xlim(-0.2, n - 0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("P(still there\nnext session)", fontsize=7)
    ax.set_xlabel("session (day)", fontsize=7, labelpad=1)
    ax.grid(alpha=0.25, lw=0.5)
    h, lb = ax.get_legend_handles_labels()
    lax.legend(h, lb, fontsize=6, loc="center left", frameon=False,
               handlelength=2.4, borderaxespad=0.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    with open(f"{args.out}.json", "w") as f:
        json.dump({"run": args.run, "split": args.split, "scene": scene,
                   "objects": chosen, "columns": args.columns,
                   "per_sequence_wins": wins,
                   "baseline": args.baseline,
                   "rule": (args.scene or
                            {"events": "most ground-truth change events",
                             "median": "median per-scene nll difference "
                                       f"ours - {args.baseline}",
                             "best": "best case: largest smallest nll margin "
                                     f"of ours over {args.best_over}"}
                            [args.scene_rule]) + (
                       "; objects by GT visibility"
                       if args.object_rule == "visibility" else
                       "; objects: largest Brier gain over "
                       f"{args.baseline} among candidates at least as visible "
                       "as the story's median")},
                  f, indent=1)
    print("teaser", scene, chosen)


if __name__ == "__main__":
    main()
