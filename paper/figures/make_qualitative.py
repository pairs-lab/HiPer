#!/usr/bin/env python3
"""
Qualitative comparison (paper figure): one THOR-Lifelong scene, several
sessions (columns) x persistence models (rows). Each panel shows the
ground-truth top-down render of that session with the map objects at their
mapped positions, marked by the model's decision outcome (first-decision
semantics, as in persistence_metrics.maintenance_metrics):
  green dot     kept, object still there
  orange tri    kept, object gone (stale once gone >= 2 sessions)
  red x         pruned while still there (premature prune)
  grey x        pruned after it left (correct prune)
  blue ring     committed as a static landmark
The scene is chosen by a fixed rule unless --scene is given: the test
scene with the median per-scene difference between the first two models
in 0.5 (premature_prune_rate + stale_rate). Nothing is hand-picked.

  python paper/figures/make_qualitative.py --run main --split test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import conftest  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import figstyle  # noqa: E402
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import ObjectMapper  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402
import tune_and_eval as te  # noqa: E402

DATA = te.THOR.parent
NAMES = {"ours": "HiPer", "pf_learned": "PF, CLIP class rates",
         "pf_reloc": r"PF($\lambda$), reloc.-aware", "pf_nbr": "PF, retrieved rates",
         "pf": r"PF($\lambda$)", "pf_class": "PF, hand-set prior",
         "elite": "ELite (scalar)", "pocd": "POCD-style",
         "draft_v1": "Fuse-then-decay", "always": "Never forget",
         "ours_hand_prior": "HiPer, hand-set prior"}
STYLE = {  # outcome -> (marker, colour, size)
    "present": ("o", "#009E73", 12), "gone": ("^", "#E69F00", 16),
    "stale": ("^", "#D55E00", 20), "premature": ("x", "#CC0000", 26),
    "pruned_ok": ("x", "#7F7F7F", 12)}
LABEL = {"present": "kept, still there", "gone": "kept, left (< 2 sessions)",
         "stale": "kept, stale (left >= 2 sessions)",
         "premature": "pruned while still there", "pruned_ok": "pruned after it left"}


def load_rows(path: Path, scene: str, models):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["scene"] == scene and r["model"] in models:
                rows.append(r)
    return rows


def choose_scene(pred_path: Path, a: str, b: str, metric: str = "nll",
                 rule: str = "median", min_labelled: int = 0,
                 others=None):
    """Rule 'median': the scene with the median per-scene difference a - b
    in ``metric`` (for 'decision': 0.5 (premature + stale)). Rule 'best':
    the scene where a's smallest margin over the models ``others`` (default
    [b]) is largest, i.e. a beats the strongest of them by the most. Only
    scenes with >= ``min_labelled`` labelled predictions of a; ties by
    name."""
    others = list(others or [b])
    rows = []
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            if r["model"] in {a, b, *others}:
                rows.append(r)
    per = pm.per_scene_metrics(rows)
    n_lab: Dict[str, int] = {}
    for r in rows:
        if r["model"] == a and r["label_next"] is not None:
            n_lab[r["scene"]] = n_lab.get(r["scene"], 0) + 1

    def crit(m):
        if metric != "decision":
            return m[metric]
        p = m["premature_prune_rate"]
        s = m["stale_rate"]
        return 0.5 * ((0.0 if p != p else p) + (0.0 if s != s else s))
    ok = [s for s in per[a] if n_lab.get(s, 0) >= min_labelled]
    if rule == "best":
        margin = sorted((-min(crit(per[o][s]) - crit(per[a][s])
                              for o in others), s) for s in ok)
        return margin[0][1]
    diffs = sorted((crit(per[a][s]) - crit(per[b][s]), s) for s in ok)
    return diffs[len(diffs) // 2][1]


def mapped_positions(sessions, cfg_dict):
    """{k: {oid: (x, y)}} after each session: a deterministic re-run of the
    evaluated mapping (same association config and appearance inputs)."""
    bye = te.appearance_inputs(sessions, cfg_dict)
    mapper = ObjectMapper(te.assoc_config(cfg_dict))
    out = {}
    for k, s in enumerate(sorted(sessions, key=lambda s: s.session_index)):
        mapper.step(s, k, (bye or {}).get(s.session_index))
        out[k] = {oid: o.position[:2].copy()
                  for oid, o in mapper.objects.items()}
    return out


def outcomes(rows, k):
    """{oid: (outcome, committed)} at session k for one model's rows."""
    hist = {}
    for r in rows:
        hist.setdefault(r["oid"], []).append(r)
    out = {}
    for oid, rs in hist.items():
        rs.sort(key=lambda r: r["k"])
        if rs[0]["k"] > k:
            continue
        upto = [r for r in rs if r["k"] <= k]
        first_prune = next((r for r in upto if r["pruned"]), None)
        committed = any(r["committed"] for r in upto) and first_prune is None
        if first_prune is not None:
            o = "premature" if first_prune["gt_present_now"] else "pruned_ok"
        else:
            now = upto[-1]["gt_present_now"]
            run = 0
            for r in reversed(upto):
                if r["gt_present_now"]:
                    break
                run += 1
            o = "present" if now else ("stale" if run >= 2 else "gone")
        out[oid] = (o, committed)
    return out


def confident_error(p, y):
    """P > 0.9 for an object that leaves, P < 0.1 for one that stays."""
    return (y == 0 and p > 0.9) or (y == 1 and p < 0.1)


def draw_prob(ax, rows, k, pos, Hk, to_panel):
    """Objects coloured by predicted presence at session k+1; a black ring
    marks objects whose ground-truth label_next is 0 (they leave), a cross
    a confident error. Returns (NLL over the session's labelled
    predictions, number of confident errors)."""
    cmap = plt.cm.RdYlGn
    ps, ys, n_err = [], [], 0
    for r in rows:
        if r["k"] != k or r["pred_presence"] is None:
            continue
        p = float(r["pred_presence"])
        if r["label_next"] is not None:
            ps.append(p)
            ys.append(r["label_next"])
            n_err += confident_error(p, r["label_next"])
        if r["oid"] not in pos[k]:
            continue
        u, v, _ = Hk @ np.r_[pos[k][r["oid"]], 1.0]
        u, v = to_panel(u, v)
        ax.scatter(u, v, s=30, c=[cmap(p)],
                   edgecolors="black" if r["label_next"] == 0 else "white",
                   linewidths=1.4 if r["label_next"] == 0 else 0.5, zorder=3)
        if r["label_next"] is not None and confident_error(p, r["label_next"]):
            ax.scatter(u, v, marker="x", s=70, c="#CC0000", linewidths=1.6,
                       zorder=4)
    nll = pm.nll(np.array(ps), np.array(ys)) if ps else float("nan")
    return nll, n_err


def main():
    figstyle.use_latex_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--models", nargs="*",
                    default=["ours", "pf_learned", "pf", "pocd", "elite"])
    ap.add_argument("--sessions", nargs="*", type=int, default=[2, 4, 7])
    ap.add_argument("--out", default=str(ROOT / "paper/figures/qualitative_thor"))
    ap.add_argument("--rule-models", nargs=2, default=None,
                    help="pair for the scene rule (default: first two)")
    ap.add_argument("--scene-rule", choices=["median", "best"],
                    default="median",
                    help="best: the scene where the first rule model gains "
                         "most over the second (a best case, say so)")
    ap.add_argument("--min-labelled", type=int, default=0,
                    help="scene rule: only scenes with this many labelled "
                         "predictions of the first rule model")
    ap.add_argument("--sessions-rule", choices=["fixed", "most-leaving"],
                    default="fixed",
                    help="most-leaving: the len(--sessions) sessions with "
                         "the most objects that leave before the next one "
                         "(ground truth only)")
    ap.add_argument("--mode", choices=["decision", "prob"], default="decision",
                    help="prob: colour = predicted presence at the next "
                         "session, black ring = object leaves before it")
    args = ap.parse_args()

    res = te.RESULTS / args.run
    sel = json.load(open(res / "final.json"))
    pred_path = res / args.split / "pred.jsonl"
    rule_pair = (args.rule_models or args.models[:2])
    rule_metric = "nll" if args.mode == "prob" else "decision"
    scene = args.scene or choose_scene(pred_path, rule_pair[0], rule_pair[1],
                                       rule_metric, args.scene_rule,
                                       args.min_labelled,
                                       [m for m in args.models
                                        if m != rule_pair[0]])
    # pooled held-out sets name sequences '<split>:<scene>'
    split, sname = (scene.split(":", 1) if ":" in scene
                    else (args.split, scene))
    files = te.scene_files(split)[sname]
    sessions = [load_session(f) for f in files]
    pos = mapped_positions(sessions, {**sel["assoc"], "name": "ours"})
    rows = load_rows(pred_path, scene, set(args.models))
    if args.sessions_rule == "most-leaving":
        leave = {}
        for r in rows:
            if r["model"] == rule_pair[0] and r["label_next"] == 0:
                leave[r["k"]] = leave.get(r["k"], 0) + 1
        args.sessions = sorted(sorted(leave, key=lambda k: (-leave[k], k))
                               [:len(args.sessions)])
    raw = DATA / "raw" / split / sname
    allxy = np.array([p for k in pos for p in pos[k].values()])
    backgrounds = {}

    def background(k):
        """(top-down image, world-xy -> pixel H) of session k."""
        if k in backgrounds:
            return backgrounds[k]
        if split.startswith("rscan"):
            # 3RScan: each session's own scan, coloured from its frames
            from hiper.evaluation import rscan
            from hiper.evaluation.paths import RSCAN_META, RSCAN_SCANS
            if "index" not in backgrounds:
                backgrounds["index"] = rscan.load_index(str(RSCAN_META))
            sess = sorted(sessions, key=lambda s: s.session_index)[k]
            WT = rscan.scan_world_transform(
                backgrounds["index"], str(RSCAN_SCANS), sname,
                sess.meta["scan_id"])
            b = (*(allxy.min(0) - 0.8), *(allxy.max(0) + 0.8))
            backgrounds[k] = rscan.topdown(
                str(RSCAN_SCANS / sess.meta["scan_id"]), WT, b)
        else:
            gt_ = json.load(open(raw / f"session_{k:02d}/gt.json"))
            backgrounds[k] = (
                np.asarray(Image.open(raw / f"session_{k:02d}/topdown.png")),
                np.array(gt_["topdown"]["world_xy_to_pixel"]))
        return backgrounds[k]

    # crop to the mapped objects (+ margin), same window in every panel
    H = background(0)[1]
    uv = (H @ np.c_[allxy, np.ones(len(allxy))].T).T[:, :2]
    lo = np.floor(uv.min(0) - 40).astype(int).clip(0)
    hi = np.ceil(uv.max(0) + 40).astype(int)
    rotate = (hi - lo)[1] > (hi - lo)[0]       # show tall crops landscape

    def to_panel(u, v):
        x, y = u - lo[0], v - lo[1]
        return ((hi - lo)[1] - 1 - y, x) if rotate else (x, y)

    nr, nc = len(args.models), len(args.sessions)
    w, h = (hi - lo) if not rotate else (hi - lo)[::-1]
    pw = 7.0 / nc
    fig, axes = plt.subplots(nr, nc, figsize=(7.16, nr * pw * h / w + 0.45),
                             squeeze=False)
    plt.rcParams.update({"font.size": 7})
    times = [s.time / 86400.0 for s in sorted(sessions,
                                              key=lambda s: s.session_index)]
    # whole-sequence NLL per model (all sessions), shown in the row labels
    seq_nll = {}
    for m in args.models:
        lab = [x for x in rows if x["model"] == m
               and x["pred_presence"] is not None
               and x["label_next"] is not None]
        seq_nll[m] = pm.nll(np.array([x["pred_presence"] for x in lab]),
                            np.array([x["label_next"] for x in lab]))
    panel = {}
    for c, k in enumerate(args.sessions):
        img, Hk = background(k)
        img = img[lo[1]:hi[1], lo[0]:hi[0]]
        if rotate:
            img = np.rot90(img, k=-1)
        for r, m in enumerate(args.models):
            ax = axes[r, c]
            ax.imshow(img, alpha=0.75)
            if args.mode == "prob":
                panel[(m, k)] = draw_prob(
                    ax, [x for x in rows if x["model"] == m], k, pos, Hk,
                    to_panel)
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
                ax.set_xticks([])
                ax.set_yticks([])
                if m == "ours":           # our row framed in its colour
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#0072B2")
                        sp.set_linewidth(2.0)
                if c == 0:
                    lab = f"{NAMES.get(m, m)}\nseq. NLL {seq_nll[m]:.3f}"
                    ax.set_ylabel(lab, fontsize=7, color="#0072B2"
                                  if m == "ours" else "black",
                                  fontweight="bold" if m == "ours" else
                                  "normal")
                if r == 0:
                    ax.set_title(f"after session {k} (day {times[k]:.1f})",
                                 fontsize=7)
                continue
            oc = outcomes([x for x in rows if x["model"] == m], k)
            counts = {"premature": 0, "stale": 0}
            for oid, (o, com) in oc.items():
                if oid not in pos[k]:
                    continue
                u, v, _ = Hk @ np.r_[pos[k][oid], 1.0]
                u, v = to_panel(u, v)
                mk, col, sz = STYLE[o]
                ax.scatter(u, v, marker=mk, c=col, s=sz, linewidths=1.2)
                if com:
                    ax.scatter(u, v, marker="o", facecolors="none",
                               edgecolors="#0072B2", s=40, linewidths=0.8)
                if o in counts:
                    counts[o] += 1
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.02, 0.03, f"premature {counts['premature']} | "
                    f"stale {counts['stale']}", transform=ax.transAxes,
                    fontsize=6, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.0,
                              edgecolor="none"))
            if c == 0:
                ax.set_ylabel(NAMES.get(m, m), fontsize=7)
            if r == 0:
                ax.set_title(f"session {k} (day {times[k]:.1f})", fontsize=7)
    if args.mode == "prob":
        # per panel: NLL of that session's predictions (best in the column
        # in bold) and the number of confident errors
        for c, k in enumerate(args.sessions):
            best = min(panel[(m, k)][0] for m in args.models)
            for r, m in enumerate(args.models):
                nll, ne = panel[(m, k)]
                axes[r, c].text(
                    0.02, 0.03, f"NLL {nll:.3f}  |  {ne} confident err.",
                    transform=axes[r, c].transAxes, fontsize=6,
                    color="white", fontweight="bold"
                    if abs(nll - best) < 5e-4 else "normal",
                    bbox=dict(facecolor="#0072B2" if m == "ours" else "black",
                              alpha=0.7, pad=1.0, edgecolor="none"))
        sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                   norm=plt.Normalize(0.0, 1.0))
        # colour bar only; colour, ring and cross are explained in the caption
        bottom = 0.32 / fig.get_figheight()
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=0.15, w_pad=0.15)
        cax = fig.add_axes([0.35, 0.5 * bottom, 0.3, 0.22 * bottom])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6, pad=1)
        fig.savefig(args.out + ".pdf", dpi=300)
        fig.savefig(args.out + ".png", dpi=150)
        with open(args.out + ".json", "w") as f:
            json.dump({"scene": scene, "mode": "prob", "models": args.models,
                       "seq_nll": seq_nll,
                       "panel": {f"{m}@{k}": {"nll": v[0], "conf_err": v[1]}
                                 for (m, k), v in panel.items()},
                       "sessions": args.sessions, "run": args.run,
                       "split": args.split, "rule": None if args.scene else
                       (f"best: largest smallest {rule_metric} margin of "
                        f"{rule_pair[0]} over "
                        f"{[m for m in args.models if m != rule_pair[0]]}"
                        if args.scene_rule == "best" else
                        f"median per-scene {rule_metric} difference "
                        f"{rule_pair[0]} - {rule_pair[1]}")
                       + (f", >= {args.min_labelled} labelled predictions"
                          if args.min_labelled else ""),
                       "sessions_rule": args.sessions_rule},
                      f, indent=1)
        print("scene", scene, "->", args.out + ".pdf")
        return
    handles = [plt.Line2D([], [], marker=STYLE[o][0], color=STYLE[o][1],
                          linestyle="none", markersize=5, label=LABEL[o])
               for o in ("present", "gone", "stale", "premature",
                         "pruned_ok")]
    handles.append(plt.Line2D([], [], marker="o", markerfacecolor="none",
                              color="#0072B2", linestyle="none",
                              markersize=6, label="committed"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6,
               frameon=False)
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1),
                     h_pad=0.15, w_pad=0.15)
    fig.savefig(args.out + ".pdf", dpi=300)
    fig.savefig(args.out + ".png", dpi=150)
    with open(args.out + ".json", "w") as f:
        json.dump({"scene": scene, "rule": None if args.scene else
                   f"{args.scene_rule} per-scene {rule_metric} difference "
                   f"{rule_pair[0]} - {rule_pair[1]}",
                   "models": args.models, "sessions": args.sessions,
                   "run": args.run, "split": args.split}, f, indent=1)
    print("scene", scene, "->", args.out + ".pdf")


if __name__ == "__main__":
    main()
