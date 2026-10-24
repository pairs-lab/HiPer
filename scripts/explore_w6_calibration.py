#!/usr/bin/env python3
"""
scripts/explore_w6_calibration.py
=================================
Idea w6 (VAL-ONLY exploration): post-hoc recalibration of the predicted
presence, fitted on TRAINING replays only, applied to HiPer AND to every
baseline with the same variant set and tuning budget.

  python scripts/explore_w6_calibration.py            # full declared grid
  python scripts/explore_w6_calibration.py --smoke    # frozen configs only

Protocol (declared before any result was seen)
  Families and grids (presence only; decision thresholds do not change the
  open-loop presence, so they are fixed at the frozen values):
    ours    hpf_nbr  DET_GRID x KAPPA_FINAL, variance_gate False,
                     NBR fixed at the frozen (K=300, tau=0.05, k_nbr=1)  112
    pf_nbr  pf_nbr   DET_GRID, NBR fixed as above                         16
    pf      pf       DET_GRID x LIFETIME_GRID_DAYS (its reference grid)   192
    elite   elite    ELITE_GRID (its reference grid)                       75
    pocd    pocd     POCD_GRID (its reference grid, hand class table)       6
  For every config: replay val, train and train2 with the same params.
    * nbr families on TRAIN use a leave-one-scene-out prior (train records
      minus the replayed scene): with the full train prior a train object
      retrieves its own record (in-sample, over-confident). train2 and val
      use the reference prior (all train records), as in run v2.
  Calibration variants = method x fit set (+ 'none'):
    method  temp       q = s(a z)                 z = logit(p), p clipped 1e-6
            platt      q = s(a z + b)
            platt_ev   + one-hot last evidence (create/miss/relocate/none;
                       base hit)
            platt_evx  + evidence-specific slopes (z x one-hot)
            platt_evgk platt_ev + log(gap days) + session index k
            iso        isotonic in p, output clipped to [1e-3, 1 - 1e-3]
            iso2       isotonic in p, output clipped to [1e-2, 1 - 1e-2]
            (logistic maps: sklearn LogisticRegression, L2, C = 1)
    fit set tr   train replays (labels label_next of TRAIN)
            tr2  train + train2 replays
  Selection: val NLL (mean over scenes), jointly over config x variant.
  'none' with this grid must reproduce the frozen val NLLs.
  2-fold check (sorted val scenes, A = even index, B = odd index): select on
  one fold, score the other; reported for 'none' (reference), for the best
  variant with its config re-selected per fold, and for the joint
  (config x variant) selection.

Leakage: calibrators see only train/train2 predictions and TRAIN labels; val
labels are used only for scoring and the declared selection; no truth_*.pkl
or true_hazard is read.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # 3 worker processes, no BLAS threads

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as ec  # noqa: E402
import tune_and_eval as te  # noqa: E402

from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import replay  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

SLUG = "w6_calibration"
OUT = te.RESULTS / "explore_cache" / SLUG
DAY = 86400.0
EPS = 1e-6
EVS = ("create", "miss", "relocate", "none")          # one-hot, base = hit
METHODS = ("temp", "platt", "platt_ev", "platt_evx", "platt_evgk", "iso",
           "iso2")
FITSETS = ("tr", "tr2")
VARIANTS = ("none",) + tuple(f"{m}@{f}" for f in FITSETS for m in METHODS)
NBR_FIXED = {"K": 300, "tau": 0.05, "k_nbr": 1.0}
NBR_FAMS = ("hpf_nbr", "pf_nbr")
NAMES = ("ours", "pf_nbr", "pf", "elite", "pocd")
BASELINES = ("pf_nbr", "pf", "elite", "pocd")

G = {}          # streams, priors (set in the parent before the Pool forks)


# ── Setup ────────────────────────────────────────────────────────────────────

def prior_of(records):
    return te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]} for r in records])


def setup():
    meta = ec.meta()
    fin = json.load(open(te.RESULTS / "v2" / "final.json"))
    streams = {sp: ec.load("streams", sp) for sp in ("val", "train",
                                                     "train2")}
    recs = ec.load("records", "train")
    ref = prior_of(recs)
    loso = {sc: prior_of([r for r in recs if r["scene"] != sc])
            for sc in streams["train"]}
    assert all(len(p.E) < len(ref.E) for p in loso.values())
    # POCD's static/dynamic split reads the hand class table (as in the
    # frozen run v2, cmd_tune_final); no other family here reads lam.
    lam, lam0 = te.load_class_lambdas(str(te.PRIORS))
    te._G.update(lam=lam, lam0=lam0, learned=({}, 1e-6))
    te.NBR["prior"] = ref
    G.update(streams=streams, ref=ref, loso=loso)
    frozen = {
        "ours": ("hpf_nbr", meta["ours_params"]),
        "pf_nbr": ("pf_nbr", meta["pf_nbr_params"]),
        "pf": ("pf", meta["pf_params"]),
        "elite": ("elite", fin["families"]["elite"]["params"]),
        "pocd": ("pocd", fin["families"]["pocd"]["params"])}
    frozen_nll = {n: fin["families"][n]["val_nll"] for n in NAMES}
    return frozen, frozen_nll


def grids(frozen):
    det = te.grid(te.DET_GRID)
    tp = lambda n, k: {k: frozen[n][1][k]}  # noqa: E731  decision threshold
    return {
        "ours": ("hpf_nbr", [{**d, "kappa0": k, "variance_gate": False,
                              **NBR_FIXED, **tp("ours", "tau_pi")}
                             for d in det for k in te.KAPPA_FINAL]),
        "pf_nbr": ("pf_nbr", [{**d, **NBR_FIXED, **tp("pf_nbr", "tau_pi")}
                              for d in det]),
        "pf": ("pf", [{**d, "lifetime_days": L, **tp("pf", "tau_pi")}
                      for d in det for L in te.LIFETIME_GRID_DAYS]),
        "elite": ("elite", [{**g, **tp("elite", "tau_e")}
                            for g in te.grid(te.ELITE_GRID)]),
        "pocd": ("pocd", [{**g, **tp("pocd", "tau_remove")}
                          for g in te.grid(te.POCD_GRID)]),
    }


# ── Replays ──────────────────────────────────────────────────────────────────

def replay_split(fam, params, split):
    rows = []
    for sc in sorted(G["streams"][split]):
        if fam in NBR_FAMS:
            te.NBR["prior"] = (G["loso"][sc] if split == "train"
                               else G["ref"])
        rows += replay(G["streams"][split][sc], te.make_model(fam, params),
                       "m")
    te.NBR["prior"] = G["ref"]
    lab = [r for r in rows if r["label_next"] is not None]
    p = np.array([r["pred_presence"] for r in lab], np.float64)
    if not np.isfinite(p).all():
        raise ValueError(f"non-finite presence {fam} {params}")
    return {"p": np.clip(p, 0.0, 1.0),
            "y": np.array([r["label_next"] for r in lab], np.float64),
            "ev": np.array([r["last_evidence"] for r in lab]),
            "k": np.array([r["k"] for r in lab], np.float64),
            "gap": np.array([(r["t_next"] - r["t_k"]) / DAY for r in lab]),
            "scene": np.array([r["scene"] for r in lab])}


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


# ── Calibrators ──────────────────────────────────────────────────────────────

def logit(p):
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q) - np.log1p(-q)


def features(d, method):
    z = logit(d["p"])[:, None]
    oh = np.stack([(d["ev"] == e).astype(float) for e in EVS], 1)
    if method in ("temp", "platt"):
        return z
    if method == "platt_ev":
        return np.hstack([z, oh])
    if method == "platt_evx":
        return np.hstack([z, oh, z * oh])
    if method == "platt_evgk":
        return np.hstack([z, oh, np.log(d["gap"])[:, None], d["k"][:, None]])
    raise ValueError(method)


def fit_calibrator(method, d):
    """Returns (predict(d) -> q, description)."""
    if method in ("iso", "iso2"):
        e = 1e-3 if method == "iso" else 1e-2
        ir = IsotonicRegression(y_min=e, y_max=1.0 - e, increasing=True,
                                out_of_bounds="clip").fit(d["p"], d["y"])
        return (lambda x: np.clip(ir.predict(x["p"]), e, 1.0 - e),
                {"n_blocks": int(len(ir.X_thresholds_))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(C=1.0, fit_intercept=(method != "temp"),
                                max_iter=5000, tol=1e-8)
        lr.fit(features(d, method), d["y"])
    return (lambda x: lr.predict_proba(features(x, method))[:, 1],
            {"coef": [float(c) for c in lr.coef_.ravel()],
             "intercept": float(np.ravel(lr.intercept_)[0])
             if method != "temp" else 0.0})


EDGES = np.linspace(0.0, 1.0, pm.N_BINS + 1)


def fast_metrics(q, y):
    """Exact vectorised equivalents of pm.nll, pm.brier, pm.ece (15
    equal-width bins) and pm.auroc (average ranks = roc_auc_score); checked
    against pm on the selected configs (results.json 'metric_check')."""
    qc = np.clip(q, pm.NLL_EPS, 1.0 - pm.NLL_EPS)
    nll = float(-np.mean(y * np.log(qc) + (1.0 - y) * np.log1p(-qc)))
    brier = float(np.mean((q - y) ** 2))
    idx = np.clip(np.searchsorted(EDGES, q, side="right") - 1, 0,
                  pm.N_BINS - 1)
    sp = np.bincount(idx, weights=q, minlength=pm.N_BINS)
    sy = np.bincount(idx, weights=y, minlength=pm.N_BINS)
    ece = float(np.abs(sp - sy).sum() / len(q))
    n1 = float(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        auc = float("nan")
    else:
        r = rankdata(q)
        auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return nll, brier, ece, auc


def scene_metrics(d, q, scenes):
    return {sc: fast_metrics(q[d["scene"] == sc], d["y"][d["scene"] == sc])
            for sc in scenes}


def scene_metrics_pm(d, q, scenes):
    out = {}
    for sc in scenes:
        m = d["scene"] == sc
        out[sc] = (pm.nll(q[m], d["y"][m]), pm.brier(q[m], d["y"][m]),
                   pm.ece(q[m], d["y"][m]), pm.auroc(q[m], d["y"][m]))
    return out


def mean_scene_nll(d, q):
    return float(np.mean([pm.nll(q[d["scene"] == s], d["y"][d["scene"] == s])
                          for s in np.unique(d["scene"])]))


def summarise(per):
    a = np.array(list(per.values()), np.float64)
    return {"nll": float(np.mean(a[:, 0])), "brier": float(np.mean(a[:, 1])),
            "ece": float(np.mean(a[:, 2])),
            "auroc": float(np.nanmean(a[:, 3])),
            "scene_nll": {s: float(v[0]) for s, v in per.items()}}


def evaluate(fam, params, detail=False):
    d = {sp: replay_split(fam, params, sp) for sp in ("val", "train",
                                                      "train2")}
    fits = {"tr": d["train"], "tr2": concat(d["train"], d["train2"])}
    scenes = sorted(G["streams"]["val"])
    res = {"none": summarise(scene_metrics(d["val"], d["val"]["p"], scenes))}
    extra = {"train_nll_raw": mean_scene_nll(d["train"], d["train"]["p"]),
             "train2_nll_raw": mean_scene_nll(d["train2"], d["train2"]["p"])}
    det, gap = {}, 0.0
    for fs, dd in fits.items():
        for m in METHODS:
            pred, desc = fit_calibrator(m, dd)
            q = pred(d["val"])
            v = f"{m}@{fs}"
            res[v] = summarise(scene_metrics(d["val"], q, scenes))
            if detail:
                det[v] = {"calibrator": desc,
                          "fit_nll_before": mean_scene_nll(dd, dd["p"]),
                          "fit_nll_after": mean_scene_nll(dd, pred(dd)),
                          "val_by_evidence": by_evidence(d["val"], q),
                          "val_pm": strip(summarise(
                              scene_metrics_pm(d["val"], q, scenes)))}
                gap = max(gap, max(abs(det[v]["val_pm"][k] - res[v][k])
                                   for k in ("nll", "brier", "ece",
                                             "auroc")))
    if detail:
        det["none"] = {"val_by_evidence": by_evidence(d["val"],
                                                      d["val"]["p"])}
        det["reliability_none"] = pm.reliability_bins(d["val"]["p"],
                                                      d["val"]["y"])
        det["metric_check_max_abs_diff"] = gap
    return res, extra, det


def by_evidence(d, q):
    """Pooled val NLL contribution per (last evidence, label)."""
    qq = np.clip(q, pm.NLL_EPS, 1.0 - pm.NLL_EPS)
    ll = -(d["y"] * np.log(qq) + (1 - d["y"]) * np.log1p(-qq))
    out = {}
    for e in ("create", "hit", "miss", "relocate", "none"):
        for y in (0, 1):
            m = (d["ev"] == e) & (d["y"] == y)
            if m.any():
                out[f"{e}|y={y}"] = {"n": int(m.sum()),
                                     "sum_nll": float(ll[m].sum()),
                                     "mean_q": float(q[m].mean())}
    out["total"] = {"n": int(len(ll)), "sum_nll": float(ll.sum())}
    return out


def _job(args):
    name, fam, params = args
    t0 = time.time()
    res, extra, _ = evaluate(fam, params)
    return name, fam, params, res, extra, time.time() - t0


# ── Selection ────────────────────────────────────────────────────────────────

def best_of(entries, variant, scenes=None):
    """(entry, nll) minimising the mean NLL of `variant` over `scenes`."""
    def score(e):
        sn = e["res"][variant]["scene_nll"]
        return float(np.mean([sn[s] for s in (scenes or sorted(sn))]))
    e = min(entries, key=score)
    return e, score(e)


def fold_mean(e, variant, scenes):
    sn = e["res"][variant]["scene_nll"]
    return float(np.mean([sn[s] for s in scenes]))


def cv2(entries, variants, scenes):
    """2-fold over val scenes: select (config, variant in `variants`) on
    one fold, score on the other."""
    A, B = scenes[0::2], scenes[1::2]
    out = {}
    for tag, sel, held in (("A->B", A, B), ("B->A", B, A)):
        cands = [(best_of(entries, v, sel), v) for v in variants]
        (e, s_sel), v = min(cands, key=lambda c: c[0][1])
        out[tag] = {"variant": v, "params": e["params"], "sel_nll": s_sel,
                    "heldout_nll": fold_mean(e, v, held)}
    out["mean_heldout_nll"] = 0.5 * (out["A->B"]["heldout_nll"]
                                     + out["B->A"]["heldout_nll"])
    return out


def strip(r):
    return {k: v for k, v in r.items() if k != "scene_nll"}


def analyse(entries_by, scenes):
    fams = {}
    for n, ents in entries_by.items():
        per_v = {}
        for v in VARIANTS:
            e, s = best_of(ents, v)
            per_v[v] = {"params": e["params"], **strip(e["res"][v])}
        cal = [v for v in VARIANTS if v != "none"]
        bv = min(cal, key=lambda v: per_v[v]["nll"])
        fams[n] = {
            "n_configs": len(ents),
            "before": per_v["none"],
            "best_calibrated": {"variant": bv, **per_v[bv]},
            "per_variant": per_v,
            "cv2": {"none": cv2(ents, ["none"], scenes),
                    "best_variant_fixed": cv2(ents, [bv], scenes),
                    "joint_all_variants": cv2(ents, list(VARIANTS), scenes),
                    "joint_calibrated_only": cv2(ents, cal, scenes)}}
    # Rankings and HiPer's margin (baseline NLL - HiPer NLL; > 0 = HiPer
    # better) before and after
    def rank(get):
        return sorted(((n, get(n)) for n in fams), key=lambda x: x[1])

    rk = {"before": rank(lambda n: fams[n]["before"]["nll"]),
          "after_best_each": rank(
              lambda n: fams[n]["best_calibrated"]["nll"])}
    for v in VARIANTS[1:]:
        rk[f"same_{v}"] = rank(lambda n: fams[n]["per_variant"][v]["nll"])
    marg = {}
    for key, get in (
            ("before", lambda n: fams[n]["before"]["nll"]),
            ("after_best_each", lambda n: fams[n]["best_calibrated"]["nll"]),
            *[(f"same_{v}", (lambda n, v=v: fams[n]["per_variant"][v]["nll"]))
              for v in VARIANTS[1:]]):
        marg[key] = {b: get(b) - get("ours") for b in BASELINES}
    marg["cv2_before"] = {b: fams[b]["cv2"]["none"]["mean_heldout_nll"]
                          - fams["ours"]["cv2"]["none"]["mean_heldout_nll"]
                          for b in BASELINES}
    marg["cv2_after_joint_calibrated"] = {
        b: fams[b]["cv2"]["joint_calibrated_only"]["mean_heldout_nll"]
        - fams["ours"]["cv2"]["joint_calibrated_only"]["mean_heldout_nll"]
        for b in BASELINES}
    return fams, rk, marg


def extra_analysis(entries_by, scenes):
    """Post-hoc, from the grid only: paired bootstrap over val scenes of
    (baseline NLL - HiPer NLL), > 0 = HiPer better; scene wins; per-variant
    2-fold held-out NLL of every family."""
    fams, _, _ = analyse(entries_by, scenes)
    hv = fams["ours"]["best_calibrated"]["variant"]

    def scene_nll(n, v):
        e, _ = best_of(entries_by[n], v)
        return e["res"][v]["scene_nll"]

    keys = {"before": lambda n: "none",
            "after_best_each": lambda n: fams[n]["best_calibrated"]["variant"],
            f"same_{hv}": lambda n: hv}
    paired = {}
    for key, var in keys.items():
        o = scene_nll("ours", var("ours"))
        paired[key] = {}
        for b in BASELINES:
            s = scene_nll(b, var(b))
            paired[key][b] = {
                **pm.paired_bootstrap(s, o),
                "hiper_wins": int(sum(s[x] > o[x] for x in scenes)),
                "n_scenes": len(scenes)}
    cv_var = {n: {v: cv2(entries_by[n], [v], scenes)["mean_heldout_nll"]
                  for v in VARIANTS} for n in entries_by}
    return {"hiper_selected_variant": hv, "paired_bootstrap_margin": paired,
            "cv2_per_variant": cv_var,
            "selected_params": {n: {"before": fams[n]["before"]["params"],
                                    "after": fams[n]["best_calibrated"]
                                    ["params"]} for n in fams}}


def diag_always():
    """DIAGNOSTIC (not a selection): the calibrators applied to the constant
    'always' model (p = 1 - 1e-6), i.e. what the calibrator's own features
    (evidence type, log gap, k) reach without any persistence model."""
    setup()
    res, extra, _ = evaluate("always", {})
    out = {v: strip(r) for v, r in res.items()}
    with open(OUT / "diag_always.json", "w") as f:
        json.dump({"note": diag_always.__doc__, "val": out, **extra}, f,
                  indent=1, default=float)
    for v, r in out.items():
        print(f"{v:14s} nll {r['nll']:.5f} brier {r['brier']:.5f} "
              f"ece {r['ece']:.5f} auroc {r['auroc']:.5f}")
    print("->", OUT / "diag_always.json")


def from_grid():
    entries = {}
    for line in open(OUT / "grid.jsonl"):
        r = json.loads(line)
        entries.setdefault(r["name"], []).append(
            {"params": r["params"], "res": r["res"]})
    scenes = sorted(next(iter(entries.values()))[0]["res"]["none"]
                    ["scene_nll"])
    out = extra_analysis(entries, scenes)
    with open(OUT / "analysis_extra.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("HiPer variant", out["hiper_selected_variant"])
    for key, d in out["paired_bootstrap_margin"].items():
        print(key, {b: (round(v["diff"], 5), round(v["ci_lo"], 5),
                        round(v["ci_hi"], 5), round(v["p_value"], 4),
                        f"{v['hiper_wins']}/{v['n_scenes']}")
                    for b, v in d.items()})
    print("2-fold held-out NLL per variant")
    print("variant        " + " ".join(f"{n:>8s}" for n in entries))
    for v in VARIANTS:
        print(f"{v:14s} " + " ".join(
            f"{out['cv2_per_variant'][n][v]:8.5f}" for n in entries))
    print("->", OUT / "analysis_extra.json")


# ── Post-hoc simplify + independent check (added on resume, 2026-10-20) ──────
# Declared before running, AFTER the main grid was seen (so: diagnostic, never
# a selection of the headline). Same grids and fit sets as the main protocol.
#   platt_miss@F        Platt (2 params) fitted only on rows whose last
#                       evidence is MISS; identity q = p on every other row
#   platt_evx_nomiss@F  platt_evx fitted on the non-MISS rows; identity on MISS
#   val rows also recompute none and platt_evx@tr2 (must match grid.jsonl)
# Independent checks that never read val, at each family's frozen config
# (and HiPer's platt_evx@tr2-selected config):
#   train2_from_train  every METHOD fitted on TRAIN only, scored on TRAIN2
#                      (40 scenes disjoint from train and val)
#   crossfit5          5-fold cross-fit over the 60 training scenes (train +
#                      train2; fold = sorted scene index mod 5): fit on 4
#                      folds, score the held-out fold
#   split_stats        per split: rows, P(y=1), MISS rows, P(y=1 | MISS)

SIMP_VAL = ("none", "platt_evx@tr2", "platt_miss@tr", "platt_miss@tr2",
            "platt_evx_nomiss@tr2")
SIMP_T2 = ("none",) + tuple(f"{m}@tr" for m in METHODS) + ("platt_miss@tr",)
CF_VARIANTS = ("none",) + METHODS + ("platt_miss",)
CF_K = 5


def fit_masked(method, d, mask_fn):
    """Calibrator fitted only on rows with mask_fn(d); q = p elsewhere."""
    m = mask_fn(d)
    pred, desc = fit_calibrator(method, {k: v[m] for k, v in d.items()})

    def f(x):
        q = x["p"].copy()
        mx = mask_fn(x)
        if mx.any():
            q[mx] = pred({k: v[mx] for k, v in x.items()})
        return q
    return f, desc


def _is_miss(d):
    return d["ev"] == "miss"


def _not_miss(d):
    return d["ev"] != "miss"


def simp_calibrator(v, fits):
    meth, fs = v.split("@")
    if meth == "platt_miss":
        return fit_masked("platt", fits[fs], _is_miss)
    if meth == "platt_evx_nomiss":
        return fit_masked("platt_evx", fits[fs], _not_miss)
    return fit_calibrator(meth, fits[fs])


def evaluate_simplify(fam, params):
    """Val NLL etc. of the SIMP_VAL variants (calibrators fitted on train /
    train + train2 only)."""
    d = {sp: replay_split(fam, params, sp) for sp in ("val", "train",
                                                      "train2")}
    fits = {"tr": d["train"], "tr2": concat(d["train"], d["train2"])}
    scenes = sorted(G["streams"]["val"])
    out = {}
    for v in SIMP_VAL:
        q = d["val"]["p"] if v == "none" else \
            simp_calibrator(v, fits)[0](d["val"])
        out[v] = summarise(scene_metrics(d["val"], q, scenes))
    return out


def _cf_predict(v, fit, held):
    if v == "platt_miss":
        return fit_masked("platt", fit, _is_miss)[0](held)
    return fit_calibrator(v, fit)[0](held)


def independent_checks(fam, params):
    """Checks that never read val labels (see the section header)."""
    d = {sp: replay_split(fam, params, sp) for sp in ("train", "train2")}
    t2s = sorted(G["streams"]["train2"])
    t2 = {}
    for v in SIMP_T2:
        assert v == "none" or v.endswith("@tr")   # never fitted on train2
        q = d["train2"]["p"] if v == "none" else \
            simp_calibrator(v, {"tr": d["train"]})[0](d["train2"])
        t2[v] = strip(summarise(scene_metrics(d["train2"], q, t2s)))
    pool = concat(d["train"], d["train2"])
    scenes = sorted(np.unique(pool["scene"]))
    fold = {s: i % CF_K for i, s in enumerate(scenes)}
    fid = np.array([fold[s] for s in pool["scene"]])
    cf = {}
    for v in CF_VARIANTS:
        q = pool["p"].copy()
        if v != "none":
            for f in range(CF_K):
                q[fid == f] = _cf_predict(
                    v, {k: x[fid != f] for k, x in pool.items()},
                    {k: x[fid == f] for k, x in pool.items()})
        cf[v] = strip(summarise(scene_metrics(pool, q, scenes)))
    return {"train2_from_train": t2, "crossfit5": cf,
            "n_scenes": {"train2": len(t2s), "crossfit": len(scenes)}}


def split_stats(fam, params):
    out = {}
    for sp in ("val", "train", "train2"):
        d = replay_split(fam, params, sp)
        m = d["ev"] == "miss"
        out[sp] = {"rows": int(len(d["y"])), "p_y1": float(d["y"].mean()),
                   "miss_rows": int(m.sum()),
                   "p_y1_given_miss": float(d["y"][m].mean()),
                   "mean_pred_given_miss": float(d["p"][m].mean()),
                   "mean_scene_nll_raw": mean_scene_nll(d, d["p"])}
    return out


def _simp_job(args):
    name, fam, params = args
    t0 = time.time()
    return name, fam, params, evaluate_simplify(fam, params), time.time() - t0


def simplify(workers):
    """Run the post-hoc simplify variants over every family's grid, then
    summarise: best-config val NLL and 2-fold held-out NLL per variant,
    HiPer margins under the same variant, the train->train2 check at each
    family's frozen config, and HiPer's own gain bootstrapped over scenes."""
    assert workers <= 3
    t_start = time.time()
    frozen, _ = setup()
    for sp in ("val", "train", "train2"):
        replay_split("hpf_nbr", frozen["ours"][1], sp)
    gr = grids(frozen)
    jobs = [(n, f, p) for n, (f, ps) in gr.items() for p in ps]
    print(f"setup+warm {time.time() - t_start:.1f}s jobs {len(jobs)}",
          flush=True)
    ent = {n: [] for n in gr}
    with Pool(workers) as pool, open(OUT / "simplify.jsonl", "w") as fg:
        for i, (n, f, p, res, dt) in enumerate(
                pool.imap_unordered(_simp_job, jobs, chunksize=2)):
            ent[n].append({"params": p, "res": res})
            fg.write(json.dumps({"name": n, "family": f, "params": p,
                                 "val": res, "sec": dt}) + "\n")
            if i % 50 == 0:
                print(f"[{i + 1}/{len(jobs)}] {n} ({dt:.1f}s, "
                      f"{time.time() - t_start:.0f}s total)", flush=True)
    print(f"grid done {time.time() - t_start:.0f}s; independent checks",
          flush=True)
    summary = summarise_simplify(ent, frozen, gr)
    summary["seconds"] = round(time.time() - t_start, 1)
    with open(OUT / "simplify.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print_simplify(summary)
    print(f"done {time.time() - t_start:.0f}s ->", OUT / "simplify.json",
          flush=True)


def summarise_simplify(ent, frozen, gr):
    scenes = sorted(G["streams"]["val"])
    main = {}
    for line in open(OUT / "grid.jsonl"):
        r = json.loads(line)
        main[(r["name"], json.dumps(r["params"], sort_keys=True))] = r["res"]
    fams, repro = {}, 0.0
    for n, es in ent.items():
        for e in es:      # recomputed variants must equal the main grid
            g = main[(n, json.dumps(e["params"], sort_keys=True))]
            for v in ("none", "platt_evx@tr2"):
                repro = max(repro, abs(e["res"][v]["nll"] - g[v]["nll"]))
        per_v = {}
        for v in SIMP_VAL:
            e, _ = best_of(es, v)
            per_v[v] = {"params": e["params"], **strip(e["res"][v]),
                        "cv2_heldout_nll": cv2(es, [v], scenes)
                        ["mean_heldout_nll"]}
        fams[n] = {"val": per_v,
                   "independent_at_frozen_config":
                       independent_checks(gr[n][0], frozen[n][1])}
        print("  checks", n, flush=True)
    sel = best_of(ent["ours"], "platt_evx@tr2")[0]["params"]
    fams["ours"]["independent_at_platt_evx_tr2_selected_config"] = {
        "params": sel, **independent_checks("hpf_nbr", sel)}
    margins = {v: {b: fams[b]["val"][v]["nll"] - fams["ours"]["val"][v]["nll"]
                   for b in BASELINES} for v in SIMP_VAL}
    cf_margins = {v: {b: fams[b]["independent_at_frozen_config"]["crossfit5"]
                      [v]["nll"] - fams["ours"]["independent_at_frozen_config"]
                      ["crossfit5"][v]["nll"] for b in BASELINES}
                  for v in CF_VARIANTS}
    # HiPer's own gain over val scenes (none at its selected config vs the
    # headline variant at its selected config, and both at the frozen one)
    o = ent["ours"]
    fz = next(e for e in o if json.dumps(e["params"], sort_keys=True)
              == json.dumps(frozen["ours"][1], sort_keys=True))
    sel, _ = best_of(o, "platt_evx@tr2")
    a, b = best_of(o, "none")[0]["res"]["none"]["scene_nll"], \
        sel["res"]["platt_evx@tr2"]["scene_nll"]
    a2, b2 = fz["res"]["none"]["scene_nll"], \
        fz["res"]["platt_evx@tr2"]["scene_nll"]
    gain = {
        "selected_configs": {**pm.paired_bootstrap(a, b),
                             "scenes_improved": int(sum(b[s] < a[s]
                                                        for s in scenes))},
        "frozen_config": {**pm.paired_bootstrap(a2, b2),
                          "scenes_improved": int(sum(b2[s] < a2[s]
                                                     for s in scenes))},
        "note": "diff = mean over val scenes of NLL(none) - NLL(platt_evx@tr2)"
                "; > 0 = calibration better"}
    return {"note": "post-hoc diagnostic (declared after the main grid)",
            "variants_val": SIMP_VAL, "variants_train2": SIMP_T2,
            "variants_crossfit": CF_VARIANTS, "crossfit_k": CF_K,
            "recomputed_vs_grid_max_abs_diff": repro, "families": fams,
            "hiper_margin_same_variant_val": margins,
            "hiper_margin_crossfit5_frozen": cf_margins,
            "hiper_self_gain": gain,
            "split_stats_ours_frozen": split_stats(*frozen["ours"])}


def print_simplify(s):
    F = s["families"]
    print(f"recomputed none/platt_evx@tr2 vs grid.jsonl: max abs diff "
          f"{s['recomputed_vs_grid_max_abs_diff']:.2e}")
    print("\n== val NLL (best config each) | 2-fold held-out NLL ==")
    print("variant              " + " ".join(f"{n:>17s}" for n in F))
    for v in SIMP_VAL:
        print(f"{v:20s} " + " ".join(
            f"{F[n]['val'][v]['nll']:8.5f}|{F[n]['val'][v]['cv2_heldout_nll']:8.5f}"
            for n in F))
    for chk, vs in (("train2_from_train", SIMP_T2),
                    ("crossfit5", CF_VARIANTS)):
        print(f"\n== independent check {chk} (frozen config; never reads "
              f"val) NLL ==")
        print("variant              " + " ".join(f"{n:>8s}" for n in F))
        for v in vs:
            print(f"{v:20s} " + " ".join(
                f"{F[n]['independent_at_frozen_config'][chk][v]['nll']:8.5f}"
                for n in F))
    sel = F["ours"]["independent_at_platt_evx_tr2_selected_config"]
    print("\nours at its platt_evx@tr2-selected config", sel["params"])
    for chk in ("train2_from_train", "crossfit5"):
        print(f"  {chk}:", {v: round(r["nll"], 5) for v, r in sel[chk].items()})
    print("\nHiPer margin (baseline - HiPer), same variant, val:")
    for v, m in s["hiper_margin_same_variant_val"].items():
        print(f"  {v:20s}", {b: round(x, 5) for b, x in m.items()})
    print("HiPer margin (baseline - HiPer), same variant, crossfit5:")
    for v, m in s["hiper_margin_crossfit5_frozen"].items():
        print(f"  {v:20s}", {b: round(x, 5) for b, x in m.items()})
    print("HiPer self gain:", json.dumps(s["hiper_self_gain"], default=float))
    print("split stats (ours frozen):",
          json.dumps(s["split_stats_ours_frozen"], default=float))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="frozen configs only (bug finding)")
    ap.add_argument("--from-grid", action="store_true",
                    help="no replays: extra analysis of grid.jsonl")
    ap.add_argument("--diag-always", action="store_true",
                    help="diagnostic: calibrators on the constant model")
    ap.add_argument("--simplify", action="store_true",
                    help="post-hoc: MISS-only Platt ablation + train->train2"
                         " check (needs grid.jsonl)")
    args = ap.parse_args()
    if args.from_grid:
        return from_grid()
    if args.diag_always:
        return diag_always()
    if args.simplify:
        OUT.mkdir(parents=True, exist_ok=True)
        return simplify(args.workers)
    assert args.workers <= 3
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    frozen, frozen_nll = setup()
    # warm the retrieved-prior caches once (workers inherit them by fork)
    for sp in ("val", "train", "train2"):
        replay_split("hpf_nbr", frozen["ours"][1], sp)
    print(f"setup+warm {time.time() - t_start:.1f}s", flush=True)
    gr = grids(frozen)
    if args.smoke:
        gr = {n: (f, [p]) for n, (f, p) in frozen.items()}
    jobs = [(n, f, p) for n, (f, ps) in gr.items() for p in ps]
    print("jobs", {n: len(ps) for n, (_, ps) in gr.items()}, flush=True)
    entries = {n: [] for n in gr}
    tag = "_smoke" if args.smoke else ""
    with Pool(args.workers) as pool, \
            open(OUT / f"grid{tag}.jsonl", "w") as fg:
        for i, (n, f, p, res, extra, dt) in enumerate(
                pool.imap_unordered(_job, jobs, chunksize=2)):
            entries[n].append({"params": p, "res": res, **extra})
            fg.write(json.dumps({"name": n, "family": f, "params": p,
                                 "res": res, **extra, "sec": dt}) + "\n")
            if i % 25 == 0 or args.smoke:
                print(f"[{i + 1}/{len(jobs)}] {n} {p} none="
                      f"{res['none']['nll']:.5f} best_cal="
                      f"{min(res[v]['nll'] for v in VARIANTS[1:]):.5f} "
                      f"({dt:.1f}s, {time.time() - t_start:.0f}s total)",
                      flush=True)
    scenes = sorted(G["streams"]["val"])
    fams, rk, marg = analyse(entries, scenes)
    # 'none' must reproduce the frozen selection
    repro = {n: {"frozen_val_nll": frozen_nll[n],
                 "grid_none_best_nll": fams[n]["before"]["nll"],
                 "match": abs(frozen_nll[n] - fams[n]["before"]["nll"]) < 1e-9}
             for n in fams}
    print("reproduce", json.dumps(repro), flush=True)
    # details (calibrator coefficients, per-evidence NLL) for each family's
    # selected uncalibrated and calibrated configs
    details = {}
    for n in fams:
        f = gr[n][0]
        bc = fams[n]["best_calibrated"]
        _, _, det = evaluate(f, bc["params"], detail=True)
        details[n] = {"best_calibrated_config": {
            "params": bc["params"],
            "metric_check_max_abs_diff": det["metric_check_max_abs_diff"],
            "selected_variant": bc["variant"],
            "selected": det[bc["variant"]], "none": det["none"],
            "all_variants_calibrators": {v: det[v]["calibrator"]
                                         for v in VARIANTS[1:]},
            "fit_nll": {v: [det[v]["fit_nll_before"], det[v]["fit_nll_after"]]
                        for v in VARIANTS[1:]}}}
    out = {
        "slug": SLUG, "smoke": args.smoke,
        "protocol": {
            "grids": {n: {"family": f, "n_configs": len(ps)}
                      for n, (f, ps) in gr.items()},
            "nbr_fixed": NBR_FIXED, "variants": VARIANTS,
            "methods": METHODS, "fit_sets": {"tr": "train (LOSO prior for "
                                             "nbr families)",
                                             "tr2": "train + train2"},
            "selection": "val NLL mean over scenes, joint config x variant",
            "cv2": "sorted val scenes, A = even index, B = odd index",
            "pocd_class_table": "hand table config/class_priors_indoor.yaml "
                                "(as in run v2)"},
        "reproduce_frozen": repro,
        "families": fams, "ranking": rk, "hiper_margin": marg,
        "details": details,
        "seconds": round(time.time() - t_start, 1)}
    with open(OUT / f"results{tag}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print_summary(fams, rk, marg)
    print(f"done {time.time() - t_start:.0f}s ->", OUT / f"results{tag}.json",
          flush=True)


def print_summary(fams, rk, marg):
    print("\n== val NLL / Brier / ECE / AUROC, before -> best calibrated ==")
    for n, r in fams.items():
        b, a = r["before"], r["best_calibrated"]
        print(f"{n:7s} before {b['nll']:.5f} {b['brier']:.5f} {b['ece']:.5f}"
              f" {b['auroc']:.5f} | after[{a['variant']}] {a['nll']:.5f} "
              f"{a['brier']:.5f} {a['ece']:.5f} {a['auroc']:.5f}")
    print("\n== per variant val NLL (best config each) ==")
    print("variant        " + " ".join(f"{n:>8s}" for n in fams))
    for v in VARIANTS:
        print(f"{v:14s} " + " ".join(
            f"{fams[n]['per_variant'][v]['nll']:8.5f}" for n in fams))
    print("\n== 2-fold mean held-out NLL ==")
    for n, r in fams.items():
        c = r["cv2"]
        print(f"{n:7s} none {c['none']['mean_heldout_nll']:.5f}  "
              f"best_variant_fixed {c['best_variant_fixed']['mean_heldout_nll']:.5f}"
              f"  joint_all {c['joint_all_variants']['mean_heldout_nll']:.5f}"
              f"  joint_cal {c['joint_calibrated_only']['mean_heldout_nll']:.5f}")
    print("\nranking before:", [(n, round(v, 5)) for n, v in rk["before"]])
    print("ranking after (best each):",
          [(n, round(v, 5)) for n, v in rk["after_best_each"]])
    print("HiPer margin before:", {k: round(v, 5) for k, v in
                                   marg["before"].items()})
    print("HiPer margin after (best each):",
          {k: round(v, 5) for k, v in marg["after_best_each"].items()})


if __name__ == "__main__":
    main()
