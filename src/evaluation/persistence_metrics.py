"""
hiper.evaluation.persistence_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Metrics of the THOR-Lifelong benchmark (design.md, 'Evaluation'). Inputs
are rows as plain dicts with the fields of design.md 'Row schemas'.

E2 persistence, from prediction rows (one per map object per session k).
  p = pred_presence, y = label_next, on rows whose label_next is not null:
  calibration  ECE (15 equal-width bins, or equal-mass bins), MCE,
               reliability bins (lo, hi, n, mean_pred, frac_pos)
  scores       Brier, NLL (p clipped to [1e-6, 1 - 1e-6]),
               MAE = mean |p - y| over queries, the persistence MAE of
               PreSIST (arXiv:2607.04057, Sec. V-B), which follows Perpetua
  ranking      AUROC, average precision (present or absent as positive)
  decisions    balanced accuracy and F1, "present" iff p >= threshold
  maintenance  premature prunes, stale objects, commit precision/recall
               (see maintenance_metrics)
E3 association, from association rows (one per track per session k >= 1):
  success over previously mapped GT objects, split stationary /
  relocated / reappeared, and the false-match rate on novel objects.
Statistics over scenes (N = scenes): mean +- std (ddof=1), percentile
bootstrap CI (seeded) and a paired bootstrap for model differences.
A per-scene value that is undefined (one label class, empty denominator)
is NaN and is left out of that metric's aggregate; n_scenes says how many
scenes remain.
"""

import math
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_BINS = 15
NLL_EPS = 1e-6
Stat = Tuple[float, float, float, float, int]   # mean, std, ci_lo, ci_hi, n_scenes


def _is_null(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))


def _flag(v) -> bool:
    """Row boolean (bool or 0/1); None and NaN read as False."""
    return False if _is_null(v) else bool(v)


def _ratio(a: float, b: float) -> float:
    return a / b if b else math.nan


# ── Binary predictions ───────────────────────────────────────────────────────

def _binary(p, y) -> Tuple[np.ndarray, np.ndarray]:
    """Check p in [0, 1] (1e-9 slack, then clipped) and y in {0, 1}."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"p and y differ in length: {p.size} vs {y.size}")
    if p.size and (not np.isfinite(p).all() or p.min() < -1e-9
                   or p.max() > 1.0 + 1e-9):
        raise ValueError("predictions must be finite and in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels must be 0 or 1")
    return np.clip(p, 0.0, 1.0), y


def _equal_mass_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Edges of equal-mass bins: the prediction of rank r goes to bin
    floor(r n_bins / n); tied predictions share the bin of their first copy,
    so fewer than n_bins bins can result. The last edge may repeat."""
    s = np.sort(p)
    rank = np.searchsorted(s, s, side="left")
    b = (rank * n_bins) // s.size
    start = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    return np.append(s[start], s[-1])


def reliability_bins(p, y, n_bins: int = N_BINS,
                     adaptive: bool = False) -> List[dict]:
    """
    Reliability-diagram bins [{lo, hi, n, mean_pred, frac_pos}]. A bin
    holds lo <= p < hi; the last bin also holds p == hi. Equal-width bins
    on [0, 1] (empty bins kept, with NaN means), or equal-mass bins with
    adaptive=True.
    """
    p, y = _binary(p, y)
    if adaptive:
        if p.size == 0:
            return []
        edges = _equal_mass_edges(p, n_bins)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0,
                  edges.size - 2)
    bins = []
    for b in range(edges.size - 1):
        m = idx == b
        n = int(m.sum())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "mean_pred": float(p[m].mean()) if n else math.nan,
                     "frac_pos": float(y[m].mean()) if n else math.nan})
    return bins


def _bin_gaps(bins: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    full = [b for b in bins if b["n"] > 0]
    n = np.array([b["n"] for b in full], dtype=np.float64)
    gap = np.array([abs(b["mean_pred"] - b["frac_pos"]) for b in full])
    return n, gap


def ece(p, y, n_bins: int = N_BINS, adaptive: bool = False) -> float:
    """Expected calibration error, sum_b (n_b / N) |mean_pred_b - frac_pos_b|."""
    n, gap = _bin_gaps(reliability_bins(p, y, n_bins, adaptive))
    return float((n * gap).sum() / n.sum()) if n.size else math.nan


def mce(p, y, n_bins: int = N_BINS) -> float:
    """Maximum calibration error over the non-empty equal-width bins."""
    n, gap = _bin_gaps(reliability_bins(p, y, n_bins))
    return float(gap.max()) if n.size else math.nan


def write_reliability_csv(bins: Sequence[dict], path: str) -> None:
    """Write bins for \\pgfplotstableread[col sep=comma]; empty bins carry
    'nan', which pgfplots discards by default."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("lo,hi,n,mean_pred,frac_pos\n")
        for b in bins:
            f.write(f"{b['lo']:.6f},{b['hi']:.6f},{b['n']:d},"
                    f"{b['mean_pred']:.6f},{b['frac_pos']:.6f}\n")


def brier(p, y) -> float:
    p, y = _binary(p, y)
    return float(np.mean((p - y) ** 2)) if p.size else math.nan


def nll(p, y, eps: float = NLL_EPS) -> float:
    """Mean negative log-likelihood, p clipped to [eps, 1 - eps]."""
    p, y = _binary(p, y)
    if p.size == 0:
        return math.nan
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def mae(p, y) -> float:
    """Mean |p - y| between presence probability and 0/1 existence."""
    p, y = _binary(p, y)
    return float(np.mean(np.abs(p - y))) if p.size else math.nan


def auroc(p, y) -> float:
    """ROC AUC; NaN when y holds a single class."""
    p, y = _binary(p, y)
    if p.size == 0 or y.min() == y.max():
        return math.nan
    return float(roc_auc_score(y, p))


def average_precision(p, y, positive: int = 1) -> float:
    """Average precision (PR-AUC) of class ``positive``: 1 = present
    (score p), 0 = absent (score 1 - p). NaN without positives."""
    p, y = _binary(p, y)
    if positive == 0:
        p, y = 1.0 - p, 1.0 - y
    if y.sum() == 0:
        return math.nan
    return float(average_precision_score(y, p))


def balanced_accuracy(p, y, threshold: float = 0.5) -> float:
    """Mean recall over the classes present in y, "present" iff
    p >= threshold. With one class it is that class's recall (sklearn)."""
    p, y = _binary(p, y)
    pred = p >= threshold
    recalls = [np.mean(pred[m] == want)
               for m, want in ((y == 1.0, True), (y == 0.0, False)) if m.any()]
    return float(np.mean(recalls)) if recalls else math.nan


def f1(p, y, threshold: float = 0.5, positive: int = 1) -> float:
    """F1 of class ``positive`` (1 = present, 0 = absent), "present" iff
    p >= threshold. NaN when there is no true and no predicted positive."""
    p, y = _binary(p, y)
    pred, true = p >= threshold, y == 1.0
    if positive == 0:
        pred, true = ~pred, ~true
    tp = int(np.sum(pred & true))
    denom = 2 * tp + int(np.sum(pred & ~true)) + int(np.sum(~pred & true))
    return 2.0 * tp / denom if denom else math.nan


def binary_metrics(p, y, n_bins: int = N_BINS,
                   threshold: float = 0.5) -> Dict[str, float]:
    """All E2 prediction metrics of one set of (p, y)."""
    p, y = _binary(p, y)
    return {
        "n_labeled": int(p.size),
        "ece": ece(p, y, n_bins),
        "ece_adaptive": ece(p, y, n_bins, adaptive=True),
        "mce": mce(p, y, n_bins),
        "brier": brier(p, y),
        "nll": nll(p, y),
        "mae": mae(p, y),
        "auroc": auroc(p, y),
        "ap_present": average_precision(p, y, positive=1),
        "ap_absent": average_precision(p, y, positive=0),
        "bacc": balanced_accuracy(p, y, threshold),
        "f1_present": f1(p, y, threshold, positive=1),
        "f1_absent": f1(p, y, threshold, positive=0),
    }


def labeled_arrays(rows: Iterable[Mapping]) -> Tuple[np.ndarray, np.ndarray]:
    """(pred_presence, label_next) of the rows whose label_next is not null."""
    lab = [r for r in rows if not _is_null(r["label_next"])]
    p = np.array([float(r["pred_presence"]) for r in lab], dtype=np.float64)
    y = np.array([float(r["label_next"]) for r in lab], dtype=np.float64)
    return p, y


# ── Map maintenance (prediction rows) ────────────────────────────────────────

def _object_histories(rows: Iterable[Mapping]) -> Dict[tuple, List[Mapping]]:
    """Rows per map object (scene, model, oid), sorted by session k."""
    hist = defaultdict(list)
    for r in rows:
        hist[(r["scene"], r["model"], r["oid"])].append(r)
    for key, rs in hist.items():
        rs.sort(key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in rs]
        if len(set(ks)) != len(ks):
            raise ValueError(f"duplicate session rows for object {key}")
    return hist


def maintenance_metrics(rows: Iterable[Mapping],
                        stale_after: int = 2) -> Dict[str, float]:
    """
    Map-maintenance quality from prediction rows. The replay is open loop
    (decisions recorded, not applied); a deployed map acts on an object's
    FIRST prune and FIRST commit decision, so those are scored.
      premature prune  the first prune comes at a session with
                       gt_present_now = 1; premature_prune_rate =
                       premature_prunes / n_pruned (objects ever pruned).
      departed         gt_present_now = 0 at >= stale_after consecutive
                       sessions (k contiguous) ending at some k.
      stale            departed at k and not pruned at or before k;
                       stale_rate = stale_objects / n_departed,
                       stale_sessions counts the (object, k) pairs.
      commit precision a first commit at k_c is correct iff gt_present_now
                       = 1 at k_c and label_next = 1 at every labelled
                       k >= k_c (label_next also catches a move that the map
                       later fixed by relocation); a commit without a
                       labelled session k >= k_c is not scored.
      commit recall    share of static objects that are ever committed;
                       static = gt_present_now = 1 and label_next = 1 on
                       every row, with at least one labelled row.
    """
    c = dict.fromkeys(("n_objects", "n_pruned", "premature_prunes",
                       "n_departed", "stale_objects", "stale_sessions",
                       "n_commits", "correct_commits", "n_static",
                       "static_committed"), 0)
    for rs in _object_histories(rows).values():
        ks = [int(r["k"]) for r in rs]
        known_now = [not _is_null(r["gt_present_now"]) for r in rs]
        present = [_flag(r["gt_present_now"]) for r in rs]
        pruned = [_flag(r["pruned"]) for r in rs]
        committed = [_flag(r["committed"]) for r in rs]
        labels = [None if _is_null(r["label_next"]) else float(r["label_next"])
                  for r in rs]
        c["n_objects"] += 1
        if any(pruned) and known_now[pruned.index(True)]:
            c["n_pruned"] += 1
            c["premature_prunes"] += int(present[pruned.index(True)])
        run, gone, departed, stale = 0, False, False, 0
        for j in range(len(rs)):
            contiguous = j > 0 and ks[j] == ks[j - 1] + 1
            if not known_now[j]:       # unknown state (3RScan): breaks a run
                run = 0
                gone = gone or pruned[j]
                continue
            run = 0 if present[j] else (run + 1 if contiguous else 1)
            gone = gone or pruned[j]
            if run >= stale_after:
                departed = True
                stale += int(not gone)
        c["n_departed"] += int(departed)
        c["stale_objects"] += int(stale > 0)
        c["stale_sessions"] += stale
        if any(committed):
            i = committed.index(True)
            later = [v for v in labels[i:] if v is not None]
            if later:
                c["n_commits"] += 1
                c["correct_commits"] += int(present[i] and min(later) == 1.0)
        known = [v for v in labels if v is not None]
        if known and all(p for p, kn in zip(present, known_now) if kn) \
                and min(known) == 1.0:
            c["n_static"] += 1
            c["static_committed"] += int(any(committed))
    c["premature_prune_rate"] = _ratio(c["premature_prunes"], c["n_pruned"])
    c["stale_rate"] = _ratio(c["stale_objects"], c["n_departed"])
    c["commit_precision"] = _ratio(c["correct_commits"], c["n_commits"])
    c["commit_recall"] = _ratio(c["static_committed"], c["n_static"])
    return c


def prediction_metrics(rows: Iterable[Mapping], n_bins: int = N_BINS,
                       threshold: float = 0.5,
                       stale_after: int = 2) -> Dict[str, float]:
    """binary_metrics on the labelled rows plus maintenance_metrics."""
    rows = list(rows)
    p, y = labeled_arrays(rows)
    return {**binary_metrics(p, y, n_bins, threshold),
            **maintenance_metrics(rows, stale_after)}


# ── Association (E3) ─────────────────────────────────────────────────────────

_ASSOC = ("success", "success_stationary", "success_relocated",
          "success_reappeared", "novel_false_match")


def association_metrics(rows: Iterable[Mapping],
                        count_tentative: bool = True) -> Dict[str, float]:
    """
    Re-identification quality from association rows. A track is associated
    iff matched_oid is not null; with count_tentative=False a tentative
    match counts as no match.
      success            associated and matched_gt_id == gt_id, over tracks
                         with gt_mapped_before; split by GT state:
                         reappeared if gt_reappeared, else relocated if
                         gt_moved, else stationary.
      novel_false_match  share of tracks without gt_mapped_before that are
                         associated (their success is "no match").
    """
    hits = defaultdict(list)
    for r in rows:
        matched = not _is_null(r["matched_oid"]) and \
            (count_tentative or not _flag(r["tentative"]))
        if _flag(r["gt_mapped_before"]):
            ok = matched and not _is_null(r["gt_id"]) and \
                r["matched_gt_id"] == r["gt_id"]
            kind = ("reappeared" if _flag(r["gt_reappeared"]) else
                    "relocated" if _flag(r["gt_moved"]) else "stationary")
            hits["success"].append(ok)
            hits[f"success_{kind}"].append(ok)
        else:
            hits["novel_false_match"].append(matched)
    out = {m: float(np.mean(hits[m])) if hits[m] else math.nan for m in _ASSOC}
    out.update({"n_mapped": len(hits["success"]),
                "n_stationary": len(hits["success_stationary"]),
                "n_relocated": len(hits["success_relocated"]),
                "n_reappeared": len(hits["success_reappeared"]),
                "n_novel": len(hits["novel_false_match"])})
    return out


# ── Statistics over scenes ───────────────────────────────────────────────────

def _finite(values: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(values), dtype=np.float64)
    return v[np.isfinite(v)]


def _boot_means(v: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of n_boot resamples (with replacement) of the values v."""
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)


def bootstrap_ci(values: Iterable[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap (1 - alpha) CI of the mean of per-scene values
    (non-finite values dropped first)."""
    v = _finite(values)
    if v.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(_boot_means(v, n_boot, seed),
                           [50.0 * alpha, 100.0 - 50.0 * alpha])
    return float(lo), float(hi)


def aggregate(values: Iterable[float], n_boot: int = 10000,
              alpha: float = 0.05, seed: int = 0) -> Stat:
    """(mean, std with ddof=1, ci_lo, ci_hi, n) over the finite values;
    std is NaN for n < 2."""
    v = _finite(values)
    if v.size == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    std = float(v.std(ddof=1)) if v.size > 1 else math.nan
    lo, hi = bootstrap_ci(v, n_boot, alpha, seed)
    return float(v.mean()), std, lo, hi, int(v.size)


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float],
                     n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> Dict[str, float]:
    """
    Paired bootstrap of mean_s (a_s - b_s) over scenes s; a and b map
    scene -> value of one metric for two models and must have the same
    scenes. Scenes where either value is NaN are dropped. Returns diff,
    ci_lo, ci_hi, p_value (two-sided: the smallest level at which the
    percentile CI excludes 0, i.e. 2 min(P*(d <= 0), P*(d >= 0)), capped
    at 1) and n_scenes.
    """
    if set(a) != set(b):
        raise ValueError("models were run on different scenes: "
                         f"{sorted(set(a) ^ set(b))}")
    d = _finite([a[s] - b[s] for s in sorted(a)])
    if d.size == 0:
        return {"diff": math.nan, "ci_lo": math.nan, "ci_hi": math.nan,
                "p_value": math.nan, "n_scenes": 0}
    means = _boot_means(d, n_boot, seed)
    lo, hi = np.percentile(means, [50.0 * alpha, 100.0 - 50.0 * alpha])
    p = min(1.0, 2.0 * min(np.mean(means <= 0.0), np.mean(means >= 0.0)))
    return {"diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(p), "n_scenes": int(d.size)}


# ── Per-scene tables and summaries ───────────────────────────────────────────

def _per_scene(rows: Iterable[Mapping], key: str,
               fn: Callable[[List[Mapping]], Dict[str, float]]
               ) -> Dict[str, Dict[str, Dict[str, float]]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r[key], r["scene"])].append(r)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for g, scene in sorted(groups):
        out.setdefault(g, {})[scene] = fn(groups[(g, scene)])
    return out


def per_scene_metrics(rows: Iterable[Mapping], n_bins: int = N_BINS,
                      threshold: float = 0.5, stale_after: int = 2
                      ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{model: {scene: prediction_metrics}} from prediction rows."""
    return _per_scene(rows, "model", lambda rs: prediction_metrics(
        rs, n_bins, threshold, stale_after))


def per_scene_association(rows: Iterable[Mapping],
                          count_tentative: bool = True
                          ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{config: {scene: association_metrics}} from association rows."""
    return _per_scene(rows, "config", lambda rs: association_metrics(
        rs, count_tentative))


def _one_split(rows: Iterable[Mapping], split: Optional[str]) -> List[Mapping]:
    rows = list(rows)
    if split is not None:
        return [r for r in rows if r.get("split") == split]
    splits = {r.get("split") for r in rows}
    if len(splits) > 1:
        raise ValueError(f"rows mix splits {sorted(map(str, splits))}; "
                         "pass split=")
    return rows


def _summarize(per_scene: Mapping[str, Mapping[str, Mapping[str, float]]],
               n_boot: int, alpha: float, seed: int
               ) -> Dict[str, Dict[str, Stat]]:
    scene_sets = {g: frozenset(v) for g, v in per_scene.items()}
    if len(set(scene_sets.values())) > 1:
        raise ValueError("groups cover different scenes: " + ", ".join(
            f"{g}: {len(s)}" for g, s in scene_sets.items()))
    out = {}
    for g, by_scene in per_scene.items():
        scenes = sorted(by_scene)
        names = list(by_scene[scenes[0]])
        out[g] = {m: aggregate([by_scene[s][m] for s in scenes],
                               n_boot, alpha, seed) for m in names}
    return out


def summarize(rows: Iterable[Mapping], split: Optional[str] = None,
              n_bins: int = N_BINS, threshold: float = 0.5,
              stale_after: int = 2, n_boot: int = 10000, alpha: float = 0.05,
              seed: int = 0) -> Dict[str, Dict[str, Stat]]:
    """
    {model: {metric: (mean, std, ci_lo, ci_hi, n_scenes)}} from prediction
    rows of several models, aggregated over scenes. Rows must come from one
    split (or pass split=), and every model must cover the same scenes.
    """
    rows = _one_split(rows, split)
    return _summarize(per_scene_metrics(rows, n_bins, threshold, stale_after),
                      n_boot, alpha, seed)


def summarize_association(rows: Iterable[Mapping], count_tentative: bool = True,
                          split: Optional[str] = None, n_boot: int = 10000,
                          alpha: float = 0.05, seed: int = 0
                          ) -> Dict[str, Dict[str, Stat]]:
    """{config: {metric: (mean, std, ci_lo, ci_hi, n_scenes)}} from
    association rows, aggregated over scenes (same rules as summarize)."""
    rows = _one_split(rows, split)
    return _summarize(per_scene_association(rows, count_tentative),
                      n_boot, alpha, seed)
