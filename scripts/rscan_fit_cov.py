#!/usr/bin/env python3
"""
Fit the 3RScan centroid-covariance term CENTROID_EXTENT_FRAC (kappa, see
src/evaluation/rscan.py) on the TRAINING split, then optionally rewrite the
track covariances of converted sessions with it.

A scan reconstructs only part of an object, so a static object's centroid
moves between scans. For every pair of tracks of the same object in
consecutive sessions of rscan_train whose ground truth is known (static or
moved), the association's relocation test flags a move when
d^2 = delta^T (R_a + R_b + P_a + P_b)^-1 delta > chi2_3(0.99), with
R = cov_base + diag((kappa * extent)^2). kappa is the grid value with the
best balanced accuracy of that test (ties: the smaller kappa).

  python scripts/rscan_fit_cov.py                      # fit on rscan_train
  python scripts/rscan_fit_cov.py --apply rscan_train rscan_val rscan_test
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import conftest  # noqa: E402,F401

from hiper.evaluation import paths as hpaths  # noqa: E402
from hiper.evaluation.object_session import (  # noqa: E402
    load_session, save_session)

KAPPA_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
CHI2_3_99 = 11.3449


def base_cov(t, kappa_stored):
    return np.asarray(t.centroid_cov, float) - np.diag(
        (kappa_stored * np.asarray(t.extent, float)) ** 2)


def pairs(split_dir):
    """(delta, base cov a, base cov b, extent a, extent b, pose covs, moved)
    for consecutive-session tracks of one object with known ground truth."""
    out = []
    for env in sorted(glob.glob(os.path.join(split_dir, "*"))):
        S = [load_session(f) for f in sorted(glob.glob(
            os.path.join(env, "session_*.npz")))]
        for a, b in zip(S, S[1:]):
            ka = float(a.meta.get("centroid_extent_frac", 0.0))
            kb = float(b.meta.get("centroid_extent_frac", 0.0))
            gt = {g["gt_id"]: g for g in b.gt_objects}
            ta = {t.gt_id: t for t in a.tracks}
            for t in b.tracks:
                g = gt.get(t.gt_id)
                if t.gt_id not in ta or g is None or g["present"] is not True:
                    continue
                prev = {x["gt_id"]: x for x in a.gt_objects}.get(t.gt_id)
                if prev is None or prev["present"] is not True:
                    continue
                s = ta[t.gt_id]
                out.append((np.asarray(t.centroid, float) -
                            np.asarray(s.centroid, float),
                            base_cov(s, ka), base_cov(t, kb),
                            np.asarray(s.extent, float),
                            np.asarray(t.extent, float),
                            np.asarray(a.pose_cov) + np.asarray(b.pose_cov),
                            bool(g["moved_since_prev"])))
    return out


def fit(prs):
    rows = []
    for k in KAPPA_GRID:
        flag, moved = [], []
        for d, ca, cb, ea, eb, P, mv in prs:
            C = ca + cb + np.diag((k * ea) ** 2 + (k * eb) ** 2) + P
            flag.append(float(d @ np.linalg.solve(C, d)) > CHI2_3_99)
            moved.append(mv)
        flag, moved = np.array(flag), np.array(moved)
        tpr = flag[moved].mean() if moved.any() else float("nan")
        tnr = (~flag[~moved]).mean() if (~moved).any() else float("nan")
        rows.append({"kappa": k, "moved_flagged": float(tpr),
                     "static_kept": float(tnr),
                     "balanced_acc": float(np.nanmean([tpr, tnr]))})
    best = max(rows, key=lambda r: (r["balanced_acc"], -r["kappa"]))
    return best["kappa"], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(hpaths.RSCAN_SESSIONS))
    ap.add_argument("--fit-split", default="rscan_train")
    ap.add_argument("--kappa", type=float, default=None,
                    help="skip fitting, use this value")
    ap.add_argument("--apply", nargs="*", default=[],
                    help="rewrite the track covariances of these splits")
    args = ap.parse_args()
    if args.kappa is None:
        prs = pairs(os.path.join(args.root, args.fit_split))
        kappa, rows = fit(prs)
        n_mv = sum(p[-1] for p in prs)
        print(f"{len(prs)} pairs ({n_mv} moved) in {args.fit_split}")
        for r in rows:
            print(r)
        with open(os.path.join(args.root, "cov_fit.json"), "w") as f:
            json.dump({"fit_split": args.fit_split, "pairs": len(prs),
                       "moved": int(n_mv), "grid": rows, "kappa": kappa},
                      f, indent=1)
    else:
        kappa = args.kappa
    print("kappa =", kappa)
    for sp in args.apply:
        n = 0
        for f in sorted(glob.glob(os.path.join(args.root, sp, "*",
                                               "session_*.npz"))):
            s = load_session(f)
            k0 = float(s.meta.get("centroid_extent_frac", 0.0))
            for t in s.tracks:
                t.centroid_cov = (base_cov(t, k0) + np.diag(
                    (kappa * np.asarray(t.extent, float)) ** 2)
                    ).astype(np.float32)
            s.meta["centroid_extent_frac"] = kappa
            save_session(s, f)
            n += 1
        print(f"{sp}: rewrote {n} sessions")


if __name__ == "__main__":
    main()
