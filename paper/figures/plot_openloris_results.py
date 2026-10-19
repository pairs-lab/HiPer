#!/usr/bin/env python3
"""
paper/figures/plot_openloris_results.py
=======================================
Plot the OpenLORIS-Scene lifelong evaluation results figure from the
JSON written by scripts/run_openloris_eval.py.

This script intentionally refuses to run without real results — the
manuscript must not contain fabricated experimental data.

Usage:
  # 1. run the experiments (one JSON per scene)
  python3 scripts/run_openloris_eval.py --data-root /data/openloris \
      --scene office --out paper/figures/results_office.json
  # 2. plot
  python3 paper/figures/plot_openloris_results.py \
      paper/figures/results_office.json [results_corridor.json ...]

Output: fig_openloris_results.pdf/.png — per-session ATE RMSE bars and
C-rate lines, one panel per scene.
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figstyle import C, TEXT_W, save, use_ieee_style  # noqa: E402


def main(paths):
    if not paths:
        print(__doc__)
        sys.exit(1)
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))

    use_ieee_style()
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(min(TEXT_W, 2.4 * n), 1.9),
                             squeeze=False)
    for ax, run in zip(axes[0], runs):
        sessions = [r["session"].split("-")[-1] for r in run["sessions"]]
        ate = [r["ate_rmse"] for r in run["sessions"]]
        cr = [r["c_rate"] for r in run["sessions"]]
        x = np.arange(len(sessions))

        ax.bar(x, ate, width=0.62, color=C["blue"], label="ATE RMSE (m)")
        ax.set_xticks(x)
        ax.set_xticklabels(sessions)
        ax.set_xlabel("session")
        ax.set_ylabel("ATE RMSE (m)", color=C["blue"])
        ax.tick_params(axis="y", colors=C["blue"])
        ax.set_title(f"{run['scene']} ({run['mode']})", fontsize=7)

        ax2 = ax.twinx()
        ax2.plot(x, cr, "-o", ms=2.6, color=C["red"], label="C-rate")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("C-rate", color=C["red"])
        ax2.tick_params(axis="y", colors=C["red"])

    fig.tight_layout()
    save(fig, str(_HERE / "fig_openloris_results"))


if __name__ == "__main__":
    main(sys.argv[1:])
