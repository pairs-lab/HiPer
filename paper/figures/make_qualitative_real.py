#!/usr/bin/env python3
"""
Qualitative comparison on real robot data (ObChange, leave-one-room-out
predictions): one room, robot visits (columns) x persistence models (rows).
Each panel shows the reconstructed table/shelf planes from above with the
map objects at their mapped positions, coloured by the predicted
probability of still being there at the next visit; a black ring marks an
object that actually leaves before it. The per-panel number is the Brier
score of that model at that visit (lower is better).

Room rule unless --room is given: the room with the median per-room NLL
difference HiPer - PF(lambda) (--room-rule median), or HiPer's best case
(--room-rule best: the room where HiPer's smallest NLL margin over the
other shown models is largest; say so in the caption).

  python paper/figures/make_qualitative_real.py --run v2
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

import figstyle  # noqa: E402
import make_qualitative as mq  # noqa: E402
import tune_and_eval as te  # noqa: E402
from hiper.evaluation import obchange as ob  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402

NAMES = {"ours": "HiPer", "pf": r"PF($\lambda$)", "elite": "ELite",
         "pocd": "POCD-style", "pf_nbr": "PF, retrieved",
         "pf_learned": "PF, CLIP class rates",
         "pf_reloc": r"PF($\lambda$), reloc.-aware"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2")
    ap.add_argument("--room", default=None)
    ap.add_argument("--room-rule", choices=["median", "best"],
                    default="median")
    ap.add_argument("--models", nargs="+",
                    default=["ours", "pf", "elite", "pocd"])
    ap.add_argument("--out", default=str(ROOT / "paper/figures/"
                                         "qualitative_real"))
    args = ap.parse_args()
    pred = te.RESULTS / args.run / "obchange_loro" / "pred.jsonl"
    scene = (f"obchange_{args.room}" if args.room else
             mq.choose_scene(pred, "ours", "pf", "nll", args.room_rule,
                             others=[m for m in args.models if m != "ours"]))
    room = scene.replace("obchange_", "")
    fin = json.load(open(te.RESULTS / args.run / "final.json"))
    files = te.scene_files("obchange")[room]
    sessions = sorted((load_session(f) for f in files),
                      key=lambda s: s.session_index)
    pos = mq.mapped_positions(sessions, {**fin["assoc"], "name": "ours"})
    rows = mq.load_rows(pred, scene, set(args.models))
    visits = ob.list_visits(str(te.OBC.parent / "extracted"), room)
    ks = [k for k in range(len(visits) - 1)]
    clouds = []
    for k in ks:
        v = ob.load_visit(visits[k], k)
        xyz = np.vstack([p["xyz"] for p in v.planes.values()])
        rgb = np.vstack([p["rgb"] for p in v.planes.values()])
        keep = xyz[:, 2] > 0.25
        clouds.append((xyz[keep][::5], np.clip(rgb[keep][::5], 0, 1)))
    allxy = np.vstack([c[0][:, :2] for c in clouds])
    lo, hi = np.percentile(allxy, 0.5, 0), np.percentile(allxy, 99.5, 0)
    pad = 0.15
    plt.rcParams.update({"font.size": 7})
    figstyle.use_latex_font()
    nr, nc = len(args.models), len(ks)
    fig, axes = plt.subplots(nr, nc, figsize=(7.16, 0.98 * nr + 0.3),
                             squeeze=False)
    cmap = plt.cm.RdYlGn
    for i, m in enumerate(args.models):
        mrows = [r for r in rows if r["model"] == m]
        for j, k in enumerate(ks):
            ax = axes[i, j]
            xyz, rgb = clouds[j]
            ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb * 0.55 + 0.45, s=0.15,
                       linewidths=0, rasterized=True)
            sq = []
            for r in mrows:
                if r["k"] != k or r["pred_presence"] is None \
                        or r["oid"] not in pos[k]:
                    continue
                x, y = pos[k][r["oid"]]
                p = float(r["pred_presence"])
                leaves = r["label_next"] == 0
                ax.scatter(x, y, s=22, c=[cmap(p)],
                           edgecolors="black" if leaves else "white",
                           linewidths=1.3 if leaves else 0.4, zorder=3)
                if r["label_next"] is not None:
                    sq.append((p - r["label_next"]) ** 2)
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if sq:
                ax.text(0.03, 0.04, f"Brier {np.mean(sq):.2f}",
                        transform=ax.transAxes, fontsize=5.5,
                        bbox=dict(fc="white", ec="none", alpha=0.8,
                                  pad=0.8))
            if i == 0:
                ax.set_title(f"visit {k} $\\rightarrow$ {k + 1}", fontsize=7,
                             pad=2)
            if j == 0:
                ax.set_ylabel(NAMES.get(m, m), fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.02,
                        wspace=0.03, hspace=0.05)
    cax = fig.add_axes([0.9, 0.25, 0.015, 0.55])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=5.5)   # colour and ring: see the caption
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    with open(f"{args.out}.json", "w") as f:
        json.dump({"run": args.run, "room": room, "models": args.models,
                   "rule": None if args.room else
                   ("best: largest smallest nll margin of ours over "
                    f"{[m for m in args.models if m != 'ours']}"
                    if args.room_rule == "best" else
                    "median per-room nll difference ours - pf")},
                  f, indent=1)
    print("qualitative_real", room)


if __name__ == "__main__":
    main()
