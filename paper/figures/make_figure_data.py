#!/usr/bin/env python3
"""
paper/figures/make_figure_data.py
=================================
Export the CSV data tables consumed by the TikZ/pgfplots manuscript
figures. All values are computed by the *shipped* hiper code
(fuse_and_decay, GlobalEphemeralityBPF, parameter defaults), so the
figures stay truthful to the implementation.

Usage:  python3 paper/figures/make_figure_data.py
Output: paper/figures/data/*.csv
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402  (registers hiper pkg + rospy stub)

from hiper.common.params import EphemeralityParams  # noqa: E402
from hiper.ephemerality.global_bpf import (  # noqa: E402
    GlobalEphemeralityBPF,
    fuse_and_decay,
)

DAY = 86400.0
WEEK = 7 * DAY
DATA = _HERE / "data"
DATA.mkdir(exist_ok=True)


def export_decay():
    """epsilon(dt) per class — fusing with eps_l = 0.5 is the Bayes
    identity, so fuse_and_decay isolates the survival decay (Eq. 8)."""
    priors = EphemeralityParams().class_priors
    cols = ["person", "bicycle", "car", "bench", "tree", "building"]
    eps0 = 0.10
    t = np.logspace(np.log10(60), np.log10(10 * 365 * DAY), 240)

    rows = ["t " + " ".join(cols)]
    for ti in t:
        vals = [fuse_and_decay(eps0, 0.5, priors[c], ti) for c in cols]
        rows.append(f"{ti:.6g} " + " ".join(f"{v:.5f}" for v in vals))
    (DATA / "decay.csv").write_text("\n".join(rows) + "\n")
    # characteristic-lifetime markers (1/lambda, eps(1/lambda))
    rows = ["t eps cls"]
    for c in cols:
        lam = priors[c]
        rows.append(f"{1/lam:.6g} "
                    f"{fuse_and_decay(eps0, 0.5, lam, 1/lam):.5f} {c}")
    (DATA / "decay_marks.csv").write_text("\n".join(rows) + "\n")
    print("  wrote decay.csv, decay_marks.csv")


def export_beta_belief():
    """Beta posterior across 10 weekly sessions via the real BPF."""
    priors = EphemeralityParams().class_priors
    scenarios = {
        "bld": dict(cls="building", seen=lambda s: True, eps_l=0.10),
        "bench": dict(cls="bench", seen=lambda s: s % 2 == 1, eps_l=0.30),
        "person": dict(cls="person", seen=lambda s: s <= 2, eps_l=0.80),
    }
    n_sessions = 10
    series = {}
    for key, sc in scenarios.items():
        bpf = GlobalEphemeralityBPF(default_lambda=1e-4,
                                    class_priors=priors)
        eps, lo, hi, var = [], [], [], []
        for s in range(1, n_sessions + 1):
            t_now = s * WEEK
            if sc["seen"](s):
                bpf.update("o", sc["cls"], sc["eps_l"], t_now)
            else:
                bpf.mark_not_observed("o", t_now)
            st = bpf.get_all_states()["o"]
            e = 1.0 - st.persistence_mean()
            sd = float(np.sqrt(st.persistence_var()))
            eps.append(e)
            lo.append(max(0.0, e - sd))
            hi.append(min(1.0, e + sd))
            var.append(st.persistence_var())
        series[key] = (eps, lo, hi, var)

    header = ["s"]
    for key in series:
        header += [f"{key}_eps", f"{key}_lo", f"{key}_hi", f"{key}_var"]
    rows = [" ".join(header)]
    for i in range(n_sessions):
        row = [str(i + 1)]
        for eps, lo, hi, var in series.values():
            row += [f"{eps[i]:.5f}", f"{lo[i]:.5f}", f"{hi[i]:.5f}",
                    f"{var[i]:.6f}"]
        rows.append(" ".join(row))
    (DATA / "beta_belief.csv").write_text("\n".join(rows) + "\n")
    print("  wrote beta_belief.csv")


def export_zscale():
    """Synthetic tree cloud, original + z-scaled with the augmentor's
    alpha = h/(h+10) (Eq. zscale); x-z profile, decimated."""
    from hiper.object_encoder.uav_augment import UAVPointCloudAugmentor
    aug = UAVPointCloudAugmentor(altitude_jitter_range=5.0)
    alpha = aug._alt_range / (aug._alt_range + 10.0)

    rng = np.random.default_rng(3)
    n, n_t = 900, 180
    trunk = np.stack([rng.normal(0, 0.07, n_t),
                      rng.normal(0, 0.07, n_t),
                      rng.uniform(0.0, 1.6, n_t)], axis=1)
    n_c = n - n_t
    phi = rng.uniform(0, 2 * np.pi, n_c)
    costh = rng.uniform(-1, 1, n_c)
    r = 1.05 * rng.uniform(0.55, 1.0, n_c) ** (1 / 3)
    sinth = np.sqrt(1 - costh ** 2)
    canopy = np.stack([r * sinth * np.cos(phi),
                       r * sinth * np.sin(phi),
                       2.4 + 0.85 * r * costh], axis=1)
    pts = np.concatenate([trunk, canopy])
    pts -= pts.mean(axis=0)

    for name, sz in [("orig", 1.0), ("lo", 1 - alpha), ("hi", 1 + alpha)]:
        p = pts.copy()
        p[:, 2] *= sz
        p -= p.mean(axis=0)
        rows = ["x z"] + [f"{x:.4f} {z:.4f}" for x, z in
                          zip(p[:, 0], p[:, 2])]
        (DATA / f"zscale_{name}.csv").write_text("\n".join(rows) + "\n")
    print(f"  wrote zscale_*.csv (alpha = {alpha:.3f})")


if __name__ == "__main__":
    print("Exporting figure data from shipped code:")
    export_decay()
    export_beta_belief()
    export_zscale()
    print("Done.")
