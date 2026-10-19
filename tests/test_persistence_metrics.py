"""
tests/test_persistence_metrics.py
=================================
Unit tests for hiper.evaluation.persistence_metrics: calibration and
scores against hand-computed values, perfect and constant predictors,
bootstrap determinism, and the premature / stale / commit and association
logic on hand-built row tables.
"""

import csv
import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from hiper.evaluation.persistence_metrics import (
    aggregate,
    association_metrics,
    auroc,
    average_precision,
    balanced_accuracy,
    binary_metrics,
    bootstrap_ci,
    brier,
    ece,
    f1,
    labeled_arrays,
    mae,
    maintenance_metrics,
    mce,
    nll,
    paired_bootstrap,
    per_scene_metrics,
    reliability_bins,
    summarize,
    summarize_association,
    write_reliability_csv,
)

approx = pytest.approx

# Hand example: 15 bins of width 1/15; 0.91 and 0.93 share bin 13
# ([0.8667, 0.9333)), 0.25 is in bin 3, 0.62 in bin 9.
P4 = [0.91, 0.93, 0.25, 0.62]
Y4 = [1, 0, 0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Binary metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestHandComputed:
    def test_reliability_bins(self):
        bins = reliability_bins(P4, Y4)
        assert len(bins) == 15
        assert [b["n"] for b in bins] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                                          0, 0, 0, 2, 0]
        b13 = bins[13]
        assert b13["lo"] == approx(13 / 15) and b13["hi"] == approx(14 / 15)
        assert b13["mean_pred"] == approx(0.92) and b13["frac_pos"] == 0.5
        assert math.isnan(bins[0]["mean_pred"]) and math.isnan(bins[0]["frac_pos"])

    def test_ece_mce(self):
        # gaps: bin 13 |0.92 - 0.5| = 0.42 (n=2), bin 3 0.25, bin 9 0.38
        assert ece(P4, Y4) == approx((2 * 0.42 + 0.25 + 0.38) / 4)   # 0.3675
        assert mce(P4, Y4) == approx(0.42)

    def test_adaptive_ece(self):
        # 2 equal-mass bins: {0.25, 0.62} (mean 0.435, 1/2 positive) and
        # {0.91, 0.93} (mean 0.92, 1/2 positive)
        assert ece(P4, Y4, n_bins=2, adaptive=True) == approx(
            (2 * 0.065 + 2 * 0.42) / 4)                               # 0.2425
        # more bins than points: one point per bin, so ECE = MAE
        assert ece(P4, Y4, adaptive=True) == approx(1.65 / 4)
        # equal-width and equal-mass differ: [0.1, 0.2 | 0.3, 0.4], y 0 0 1 1
        p, y = [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]
        assert ece(p, y, n_bins=2) == approx(0.25)
        assert ece(p, y, n_bins=2, adaptive=True) == approx((0.3 + 1.3) / 4)

    def test_adaptive_bins_never_split_ties(self):
        bins = reliability_bins([0.5, 0.5, 0.5, 0.9], [1, 0, 1, 1], n_bins=4,
                                adaptive=True)
        assert [b["n"] for b in bins] == [3, 1]
        assert bins[0]["frac_pos"] == approx(2 / 3)
        bins = reliability_bins([0.1, 0.5, 0.5, 0.9], [0, 1, 1, 1], n_bins=2,
                                adaptive=True)
        assert [b["n"] for b in bins] == [3, 1]

    def test_brier_nll_mae(self):
        assert brier(P4, Y4) == approx(
            (0.09 ** 2 + 0.93 ** 2 + 0.25 ** 2 + 0.38 ** 2) / 4)      # 0.269975
        assert nll(P4, Y4) == approx(-(math.log(0.91) + math.log(0.07)
                                       + math.log(0.75) + math.log(0.62)) / 4)
        assert mae(P4, Y4) == approx((0.09 + 0.93 + 0.25 + 0.38) / 4)  # 0.4125

    def test_ranking(self):
        # positive/negative pairs won: (0.91 > 0.25), (0.62 > 0.25) -> 2/4
        assert auroc(P4, Y4) == approx(0.5)
        # present: ranked 0.93(-) 0.91(+) 0.62(+) 0.25(-) -> (1/2 + 2/3) / 2
        assert average_precision(P4, Y4, positive=1) == approx(7 / 12)
        # absent, score 1-p: 0.75(+) 0.38(-) 0.09(-) 0.07(+) -> (1 + 2/4) / 2
        assert average_precision(P4, Y4, positive=0) == approx(0.75)

    def test_threshold_decisions(self):
        # predicted present: [T, T, F, T]
        assert balanced_accuracy(P4, Y4) == approx((1.0 + 0.5) / 2)
        assert f1(P4, Y4, positive=1) == approx(2 * 2 / (2 * 2 + 1 + 0))
        assert f1(P4, Y4, positive=0) == approx(2 * 1 / (2 * 1 + 0 + 1))
        # threshold is inclusive: p == threshold reads as present
        assert balanced_accuracy([0.5, 0.2], [1, 0], threshold=0.5) == 1.0

    def test_nll_clip(self):
        assert nll([0.0], [1]) == approx(-math.log(1e-6))
        assert nll([1.0], [1]) == approx(-math.log1p(-1e-6))


class TestPerfectAndConstant:
    def test_perfect_predictor(self):
        y = [1, 0, 1, 1, 0, 1]
        m = binary_metrics(np.array(y, float), y)
        for k in ("ece", "ece_adaptive", "mce", "brier", "mae"):
            assert m[k] == approx(0.0, abs=1e-12), k
        for k in ("auroc", "ap_present", "ap_absent", "bacc", "f1_present",
                  "f1_absent"):
            assert m[k] == approx(1.0), k
        assert m["nll"] == approx(-math.log1p(-1e-6))
        assert m["n_labeled"] == 6

    def test_constant_predictor(self):
        y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]          # base rate 0.4
        m = binary_metrics([0.7] * 10, y)
        assert m["ece"] == approx(0.3)
        assert m["ece_adaptive"] == approx(0.3)
        assert m["mce"] == approx(0.3)
        assert m["brier"] == approx(0.4 * 0.3 ** 2 + 0.6 * 0.7 ** 2)
        assert m["nll"] == approx(-(0.4 * math.log(0.7) + 0.6 * math.log(0.3)))
        assert m["mae"] == approx(0.4 * 0.3 + 0.6 * 0.7)
        assert m["auroc"] == approx(0.5)
        assert m["ap_present"] == approx(0.4)
        assert m["bacc"] == approx(0.5)
        assert m["f1_present"] == approx(2 * 4 / (2 * 4 + 6 + 0))
        assert m["f1_absent"] == 0.0                 # never predicts absent
        assert math.isnan(f1([0.7], [1], positive=0))  # no true/pred. absent

    def test_calibrated_constant(self):
        y = [1, 1, 0, 0, 0]
        assert ece([0.4] * 5, y) == approx(0.0, abs=1e-12)
        assert brier([0.4] * 5, y) == approx(0.4 * 0.6)


class TestEdgeCases:
    def test_extreme_probabilities_are_binned(self):
        bins = reliability_bins([0.0, 1.0], [0, 1])
        assert bins[0]["n"] == 1 and bins[-1]["n"] == 1
        assert sum(b["n"] for b in bins) == 2

    def test_single_class(self):
        p, y = [0.2, 0.7, 0.9], [1, 1, 1]
        assert math.isnan(auroc(p, y))
        assert math.isnan(average_precision(p, y, positive=0))
        assert average_precision(p, y, positive=1) == approx(1.0)
        assert balanced_accuracy(p, y) == approx(2 / 3)   # recall of class 1

    def test_empty(self):
        m = binary_metrics([], [])
        assert m["n_labeled"] == 0
        assert all(math.isnan(v) for k, v in m.items() if k != "n_labeled")

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            brier([1.2], [1])
        with pytest.raises(ValueError):
            brier([float("nan")], [1])
        with pytest.raises(ValueError):
            brier([0.5], [2])
        with pytest.raises(ValueError):
            brier([0.5, 0.5], [1])

    def test_against_sklearn(self):
        rng = np.random.default_rng(3)
        p = rng.random(200)
        y = (rng.random(200) < 0.7).astype(int)
        pred = (p >= 0.5).astype(int)
        assert balanced_accuracy(p, y) == approx(balanced_accuracy_score(y, pred))
        assert f1(p, y, positive=1) == approx(f1_score(y, pred))
        assert f1(p, y, positive=0) == approx(f1_score(1 - y, 1 - pred))

    def test_reliability_csv(self, tmp_path):
        path = tmp_path / "rel" / "hpf.csv"
        write_reliability_csv(reliability_bins(P4, Y4), str(path))
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == ["lo", "hi", "n", "mean_pred", "frac_pos"]
        assert len(rows) == 15
        assert rows[13]["n"] == "2" and float(rows[13]["mean_pred"]) == approx(0.92)
        assert rows[0]["mean_pred"] == "nan"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction rows: map maintenance
# ─────────────────────────────────────────────────────────────────────────────

def _row(oid, k, present, label, pruned=0, committed=0, p=0.9, scene="s1",
         model="hpf", split="test"):
    return {"scene": scene, "split": split, "model": model, "oid": oid,
            "k": k, "t_k": 86400.0 * k, "t_next": 86400.0 * (k + 1),
            "class_label": "mug", "gt_id": f"gt_{oid}",
            "gt_group": "pickupable", "pred_presence": p, "rate_mean": 0.9,
            "rate_var": 0.001, "label_next": label, "gt_present_now": present,
            "pruned": pruned, "committed": committed, "last_evidence": "hit"}


def _object(oid, present, labels, pruned, committed, k0=0, **kw):
    return [_row(oid, k0 + i, *v, **kw)
            for i, v in enumerate(zip(present, labels, pruned, committed))]


Z5 = [0, 0, 0, 0, 0]
N = None
TABLE = (
    # A static, committed at k=2 -> correct commit
    _object("A", [1, 1, 1, 1, 1], [1, 1, 1, 1, N], Z5, [0, 0, 1, 1, 1])
    # B static, first prune at k=1 while present -> premature (later un-prune ignored)
    + _object("B", [1, 1, 1, 1, 1], [1, 1, 1, 1, N], [0, 1, 1, 0, 0], Z5)
    # C leaves at k=2, pruned at k=3 when absent 2 sessions -> correct, not stale
    + _object("C", [1, 1, 0, 0, 0], [1, 0, 0, 0, N], [0, 0, 0, 1, 1], Z5)
    # D leaves at k=1, never pruned -> stale at k=2,3,4; commit at k=0 wrong
    + _object("D", [1, 0, 0, 0, 0], [0, 0, 0, 0, N], Z5, [1, 1, 0, 0, 0])
    # E moved between k=1 and k=2, relocated by the map; commit at k=1 wrong
    + _object("E", [1, 1, 1, 1, 1], [1, 0, 1, 1, N], Z5, [0, 1, 1, 1, 1])
    # F absent at k=1 and k=3 only (never 2 in a row) -> not departed
    + _object("F", [1, 0, 1, 0, 1], [0, 1, 0, 1, N], Z5, Z5)
    # G created at the last session and committed -> commit not scored
    + _object("G", [1], [N], [0], [1], k0=4)
    # H static from k=2, pruned at k=4 while present -> premature
    + _object("H", [1, 1, 1], [1, 1, N], [0, 0, 1], [0, 0, 0], k0=2)
    # I pruned at k=0 while present (premature), leaves later -> not stale
    + _object("I", [1, 1, 0, 0, 0], [1, 0, 0, 0, N], [1, 1, 1, 1, 1], Z5)
    # J pruned at k=1 while present, un-pruned, leaves at k=2 -> the first
    # prune removed it, so it is not stale at k=3, 4
    + _object("J", [1, 1, 0, 0, 0], [1, 0, 0, 0, N], [0, 1, 0, 0, 0], Z5)
)


class TestMaintenance:
    def test_hand_table(self):
        m = maintenance_metrics(TABLE)
        assert m["n_objects"] == 10
        assert (m["n_pruned"], m["premature_prunes"]) == (5, 4)      # BCHIJ; BHIJ
        assert m["premature_prune_rate"] == approx(0.8)
        assert (m["n_departed"], m["stale_objects"]) == (4, 1)       # C D I J; D
        assert m["stale_sessions"] == 3
        assert m["stale_rate"] == approx(1 / 4)
        assert (m["n_commits"], m["correct_commits"]) == (3, 1)      # A D E; A
        assert m["commit_precision"] == approx(1 / 3)
        assert (m["n_static"], m["static_committed"]) == (3, 1)      # A B H; A
        assert m["commit_recall"] == approx(1 / 3)

    def test_stale_after_one(self):
        m = maintenance_metrics(TABLE, stale_after=1)
        assert m["n_departed"] == 5                                   # C D F I J
        assert m["stale_objects"] == 3                                # C D F
        assert m["stale_sessions"] == 1 + 4 + 2

    def test_row_order_and_bool_types_do_not_matter(self):
        rows = [dict(r, pruned=bool(r["pruned"]), committed=bool(r["committed"]))
                for r in reversed(TABLE)]
        np.testing.assert_equal(maintenance_metrics(rows),
                                maintenance_metrics(TABLE))

    def test_absence_run_needs_contiguous_sessions(self):
        rows = [_row("X", 1, 0, 0), _row("X", 3, 0, N)]               # k=2 missing
        assert maintenance_metrics(rows)["n_departed"] == 0
        rows = [_row("X", 1, 0, 0), _row("X", 2, 0, N)]
        assert maintenance_metrics(rows)["stale_objects"] == 1

    def test_never_pruned_and_never_committed(self):
        m = maintenance_metrics(_object("A", [1, 1], [1, N], [0, 0], [0, 0]))
        assert m["premature_prunes"] == 0
        assert math.isnan(m["premature_prune_rate"])
        assert math.isnan(m["commit_precision"])
        assert m["commit_recall"] == 0.0

    def test_duplicate_rows_raise(self):
        with pytest.raises(ValueError):
            maintenance_metrics([_row("A", 0, 1, 1), _row("A", 0, 1, 1)])

    def test_labeled_arrays_skip_null(self):
        rows = [_row("A", 0, 1, 1, p=0.8), _row("A", 1, 1, N, p=0.7),
                _row("B", 0, 1, float("nan"), p=0.6), _row("B", 1, 0, 0, p=0.1)]
        p, y = labeled_arrays(rows)
        assert list(p) == [0.8, 0.1] and list(y) == [1.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# Association rows (E3)
# ─────────────────────────────────────────────────────────────────────────────

def _arow(track, gt, mapped, matched_oid, matched_gt, tentative=0, moved=0,
          reappeared=0, scene="s1", config="full"):
    return {"scene": scene, "split": "test", "config": config, "k": 1,
            "track_id": track, "gt_id": gt, "matched_oid": matched_oid,
            "matched_gt_id": matched_gt, "score": 0.8, "d2": 1.0,
            "tentative": tentative, "relocated_pred": 0,
            "gt_mapped_before": mapped, "gt_moved": moved,
            "gt_reappeared": reappeared}


ASSOC = [
    _arow(1, "a", 1, "o1", "a"),                      # stationary, correct
    _arow(2, "b", 1, "o2", "c"),                      # stationary, wrong object
    _arow(3, "d", 1, "o3", "d", tentative=1, moved=1),  # relocated, tentative
    _arow(4, "e", 1, None, None, reappeared=1),       # reappeared, missed
    _arow(5, "f", 0, None, None),                     # novel, correctly new
    _arow(6, "g", 0, "o7", "x", tentative=1),         # novel, tentative match
    _arow(7, "h", 0, "o8", "y"),                      # novel, false match
    _arow(8, "k", 1, "o9", "k", moved=1, reappeared=1),  # reappeared wins
]


class TestAssociation:
    def test_with_tentative(self):
        m = association_metrics(ASSOC, count_tentative=True)
        assert m["success"] == approx(3 / 5)
        assert m["success_stationary"] == approx(1 / 2)
        assert m["success_relocated"] == approx(1.0)
        assert m["success_reappeared"] == approx(1 / 2)
        assert m["novel_false_match"] == approx(2 / 3)
        assert (m["n_mapped"], m["n_stationary"], m["n_relocated"],
                m["n_reappeared"], m["n_novel"]) == (5, 2, 1, 2, 3)

    def test_without_tentative(self):
        m = association_metrics(ASSOC, count_tentative=False)
        assert m["success"] == approx(2 / 5)
        assert m["success_relocated"] == approx(0.0)
        assert m["novel_false_match"] == approx(1 / 3)

    def test_summary_per_config(self):
        rows = ASSOC + [dict(r, config="clip_only") for r in ASSOC]
        s = summarize_association(rows, count_tentative=False, n_boot=200)
        assert set(s) == {"full", "clip_only"}
        assert s["full"]["success"][0] == approx(0.4)
        assert s["full"]["success"][4] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Statistics over scenes
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrap:
    V = [0.12, 0.30, 0.25, 0.41, 0.18, 0.22, 0.35, 0.29, 0.15, 0.33,
         0.27, 0.21, 0.38, 0.19, 0.24, 0.31, 0.26, 0.17, 0.36, 0.28]

    def test_seeded_and_deterministic(self):
        assert bootstrap_ci(self.V, seed=7) == bootstrap_ci(self.V, seed=7)
        assert bootstrap_ci(self.V, seed=7) != bootstrap_ci(self.V, seed=8)

    def test_ci_brackets_mean(self):
        lo, hi = bootstrap_ci(self.V)
        assert lo < np.mean(self.V) < hi

    def test_ci_level(self):
        # percentile CI of a mean ~ mean +- z SE, SE = std(ddof=0) / sqrt(n)
        se = np.std(self.V) / math.sqrt(len(self.V))
        for alpha, z in ((0.05, 1.96), (0.10, 1.645)):
            lo, hi = bootstrap_ci(self.V, n_boot=20000, alpha=alpha)
            assert (hi - lo) / 2 == approx(z * se, rel=0.05)

    def test_constant_values(self):
        lo, hi = bootstrap_ci([0.3] * 6)
        assert lo == approx(0.3) and hi == approx(0.3)

    def test_aggregate(self):
        mean, std, lo, hi, n = aggregate([1.0, 2.0, 4.0, float("nan")])
        assert (mean, n) == (approx(7 / 3), 3)                     # NaN dropped
        assert std == approx(math.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2
                                        + (4 - 7 / 3) ** 2) / 2))  # ddof = 1
        assert (lo, hi) == bootstrap_ci([1.0, 2.0, 4.0])
        assert aggregate([0.5])[:2] == (0.5, approx(float("nan"), nan_ok=True))
        assert aggregate([float("nan")])[4] == 0

    def test_paired_identical(self):
        a = {f"s{i}": v for i, v in enumerate(self.V)}
        r = paired_bootstrap(a, a)
        assert (r["diff"], r["ci_lo"], r["ci_hi"], r["p_value"]) == (0, 0, 0, 1)
        assert r["n_scenes"] == 20

    def test_paired_shift(self):
        a = {f"s{i}": v for i, v in enumerate(self.V)}
        b = {s: v - 0.05 for s, v in a.items()}
        r = paired_bootstrap(a, b, seed=1)
        assert r["diff"] == approx(0.05)
        assert r["ci_lo"] == approx(0.05) and r["ci_hi"] == approx(0.05)
        assert r["p_value"] == 0.0
        # input order does not change the result
        assert paired_bootstrap(dict(reversed(list(a.items()))), b, seed=1) == r

    def test_paired_requires_same_scenes(self):
        with pytest.raises(ValueError):
            paired_bootstrap({"s1": 0.1, "s2": 0.2}, {"s1": 0.1})


class TestSummarize:
    @staticmethod
    def _rows(model, preds, scene):
        # two objects x three sessions; labels 1 1 - and 1 0 -
        labels = [1, 1, N, 1, 0, N]
        present = [1, 1, 1, 1, 1, 0]
        return [_row(oid, k, present[i], labels[i], p=preds[i], scene=scene,
                     model=model)
                for i, (oid, k) in enumerate([("A", 0), ("A", 1), ("A", 2),
                                              ("B", 0), ("B", 1), ("B", 2)])]

    def test_structure_and_values(self):
        perfect = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        rows = []
        for scene in ("s1", "s2", "s3"):
            rows += self._rows("oracle", perfect, scene)
            rows += self._rows("const", [0.5] * 6, scene)
        s = summarize(rows, n_boot=500, seed=0)
        assert set(s) == {"oracle", "const"}
        mean, std, lo, hi, n = s["oracle"]["brier"]
        assert (mean, std, lo, hi, n) == (approx(0.0), approx(0.0), approx(0.0),
                                          approx(0.0), 3)
        assert s["const"]["brier"][0] == approx(0.25)
        assert s["const"]["ece"][0] == approx(abs(0.5 - 3 / 4))
        assert s["oracle"]["auroc"][0] == approx(1.0)
        ps = per_scene_metrics(rows)
        assert ps["const"]["s2"]["brier"] == approx(0.25)
        np.testing.assert_equal(
            summarize(list(reversed(rows)), n_boot=500, seed=0), s)

    def test_undefined_scene_values_are_counted_out(self):
        rows = self._rows("m", [0.9] * 6, "s1")
        rows += [dict(r, label_next=None if r["label_next"] is None else 1,
                      scene="s2") for r in self._rows("m", [0.9] * 6, "s2")]
        s = summarize(rows, n_boot=100)
        assert s["m"]["auroc"][4] == 1          # s2 has one label class
        assert s["m"]["brier"][4] == 2

    def test_mixed_splits_and_scene_sets_raise(self):
        rows = self._rows("m", [0.9] * 6, "s1") + [
            dict(r, split="val") for r in self._rows("m", [0.9] * 6, "s2")]
        with pytest.raises(ValueError):
            summarize(rows)
        assert summarize(rows, split="val", n_boot=50)["m"]["brier"][4] == 1
        rows = self._rows("m1", [0.9] * 6, "s1") + self._rows("m2", [0.9] * 6, "s2")
        with pytest.raises(ValueError):
            summarize(rows)
