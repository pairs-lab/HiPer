"""Tests for the appearance-retrieved survival prior (priors.py)."""

import math

import numpy as np
import pytest

from hiper.ephemerality.priors import (
    DAY, NeighbourSurvivalPrior, _map_rate, learn_class_rates,
    weighted_map_rate)


def _rec(e, S_days, gaps_days):
    e = np.asarray(e, float)
    return {"emb": e / np.linalg.norm(e), "S": S_days * DAY,
            "gaps": [g * DAY for g in gaps_days]}


def test_unit_weights_equal_class_estimator():
    S = np.array([3.0, 5.0]) * DAY
    gaps = np.array([1.0, 2.0, 0.5]) * DAY
    lam0 = 1.0 / (4 * DAY)
    for k in (0.0, 1.0, 4.0):
        if k == 0.0:
            ref = _map_rate(float(S.sum()), gaps, 0.0, 0.0)
            got = weighted_map_rate(np.ones(2), S, gaps, np.ones(3), 1e-12,
                                    lam0)
        else:
            ref = _map_rate(float(S.sum()), gaps, k, k / lam0)
            got = weighted_map_rate(np.ones(2), S, gaps, np.ones(3), k, lam0)
        assert got == pytest.approx(ref, rel=1e-6)


def test_hard_neighbourhood_reproduces_learned_class_rate():
    # two classes with orthogonal embeddings: retrieving exactly the K
    # objects of class A with unit weights gives A's learned rate
    recs = ([_rec([1, 0, 0], 4.0, [1.0]) for _ in range(3)] +
            [_rec([0, 1, 0], 10.0, []) for _ in range(3)])
    prior = NeighbourSurvivalPrior(recs)
    # class A: summed exposure 3 x 4 days, three 1-day vanish gaps
    S_A = 3 * 4.0 * DAY
    gaps_A = np.array([1.0, 1.0, 1.0]) * DAY
    ref = _map_rate(S_A, gaps_A, 1.0, 1.0 / prior.lam0)
    got = prior.rate([1, 0, 0], K=3, tau=None, k=1.0)
    assert got == pytest.approx(ref, rel=1e-6)


def test_pooled_rate_matches_learn_class_rates_pooled():
    # one scene, 3 sessions one day apart; object a survives, b vanishes
    def gt(present_a, present_b):
        return [{"gt_id": "a", "type": "Mug", "present": present_a},
                {"gt_id": "b", "type": "Mug", "present": present_b}]
    hist = [[(0.0, gt(True, True)), (DAY, gt(True, False)),
             (2 * DAY, gt(True, False))]]
    _, lam0 = learn_class_rates(hist)
    recs = [_rec([1, 0], 2.0, []), _rec([0, 1], 0.0, [1.0])]
    assert NeighbourSurvivalPrior(recs).lam0 == pytest.approx(lam0, rel=1e-6)


def test_similar_long_lived_neighbours_give_lower_rate():
    recs = ([_rec([1, 0.05 * i, 0], 30.0, []) for i in range(5)] +
            [_rec([0, 1, 0.05 * i], 1.0, [0.5, 0.5]) for i in range(5)])
    p = NeighbourSurvivalPrior(recs)
    lam_stay = p.rate([1, 0, 0], K=5, tau=0.05, k=1.0)
    lam_go = p.rate([0, 1, 0], K=5, tau=0.05, k=1.0)
    assert lam_stay < p.lam0 < lam_go


def test_missing_embedding_falls_back_to_pooled():
    p = NeighbourSurvivalPrior([_rec([1, 0], 2.0, [1.0])])
    assert p.rate(np.zeros(2)) == p.lam0
    assert p.rate([math.nan, 0.0]) == p.lam0


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        NeighbourSurvivalPrior([])


# ── Environment level of the filter (persistence_v2.env_alpha) ──────────────

def _run_env(alpha, vanish):
    from hiper.ephemerality.persistence_v2 import (
        HIT, MISS, HierarchicalPersistenceFilter)
    f = HierarchicalPersistenceFilter(default_lambda=1.0 / (10 * DAY),
                                      use_class_prior=False, kappa0=8,
                                      p_miss=0.1, p_false=0.01,
                                      env_alpha=alpha)
    for i in range(30):
        oid = f"o{i}"
        f.create(oid, "x", 0.0)
        f.observe(oid, 0.0, HIT)
        f.observe(oid, DAY, MISS if vanish else HIT)
    return f


def test_env_gamma_off_is_one():
    assert _run_env(None, True).env_gamma() == 1.0


def test_env_gamma_rises_when_objects_vanish_faster_than_expected():
    assert _run_env(2.0, True).env_gamma() > 3.0


def test_env_gamma_falls_when_objects_stay():
    assert _run_env(2.0, False).env_gamma() < 1.0


def test_env_gamma_scales_presence():
    from hiper.ephemerality.persistence_v2 import HIT
    f = _run_env(2.0, True)
    g = _run_env(None, True)
    for m in (f, g):
        m.create("new", "x", 2 * DAY)
        m.observe("new", 2 * DAY, HIT)
    assert f.presence("new", 3 * DAY) < g.presence("new", 3 * DAY)


# ── Composite retrieval key (DINOv2 + CLIP + geometry) ─────────────────────

def _crec(e, c, geo, S_days, gaps_days):
    r = _rec(e, S_days, gaps_days)
    c = np.asarray(c, float)
    r["clip"] = c / np.linalg.norm(c)
    r["geo"] = np.asarray(geo, float)
    return r


def test_zero_weights_equal_dino_only_retrieval():
    from hiper.ephemerality.priors import geo_features
    rng = np.random.default_rng(0)
    recs = [_crec(rng.normal(size=6), rng.normal(size=4),
                  geo_features(rng.random(3) + 0.1, rng.random(),
                               rng.random() * 0.1 + 1e-3),
                  rng.random() * 10, list(rng.random(rng.integers(0, 3))))
            for _ in range(40)]
    old = NeighbourSurvivalPrior(recs)
    new = NeighbourSurvivalPrior(recs, clip_weight=0.0, geo_weight=0.0)
    for _ in range(5):
        q = rng.normal(size=6)
        assert new.rate({"dino": q, "clip": rng.normal(size=4)}, K=10,
                        tau=0.05, k=1.0) == pytest.approx(
            old.rate(q, K=10, tau=0.05, k=1.0), rel=1e-12)


def test_clip_and_geometry_change_the_neighbours():
    from hiper.ephemerality.priors import geo_features
    # same DINOv2 key; CLIP and size separate a long-lived from a short-lived
    # group, so only the composite key can tell them apart
    big = geo_features([1.0, 1.0, 1.0], 0.5, 0.2)
    small = geo_features([0.05, 0.05, 0.05], 0.9, 0.001)
    rng = np.random.default_rng(1)
    noise = lambda: [1, 0] + 0.05 * rng.normal(size=2)  # noqa: E731
    recs = ([_crec(noise(), [1, 0], big, 60.0, []) for _ in range(5)] +
            [_crec(noise(), [0, 1], small, 1.0, [0.5, 0.5])
             for _ in range(5)])
    p = NeighbourSurvivalPrior(recs, clip_weight=0.25, geo_weight=1.0)
    lam_big = p.rate({"dino": [1, 0], "clip": [1, 0], "geo": big}, K=5,
                     tau=0.05, k=1.0)
    lam_small = p.rate({"dino": [1, 0], "clip": [0, 1], "geo": small}, K=5,
                       tau=0.05, k=1.0)
    assert lam_big < p.lam0 < lam_small


def test_missing_geometry_dims_are_skipped():
    from hiper.ephemerality.priors import geo_features
    g = geo_features([0.2, 0.1, 0.3], 0.8, 0.0)       # no mask area
    assert np.isnan(g[4]) and np.isfinite(g[:4]).all()
    recs = [_crec([1, 0], [1, 0], geo_features([0.2, 0.1, 0.3], 0.8, 0.01),
                  5.0, [1.0]), _crec([0, 1], [0, 1],
                                     geo_features([2, 2, 2], 0.1, 0.5),
                                     5.0, [])]
    p = NeighbourSurvivalPrior(recs, clip_weight=0.25, geo_weight=1.0)
    d = p.distance(np.array([1.0, 0.0]), np.array([1.0, 0.0]), g)
    assert np.isfinite(d).all() and d[0] < d[1]


def test_relocation_hazard_multiplier_for_fixed_rate_filter():
    from hiper.ephemerality.persistence_v2 import (
        HIT, HierarchicalPersistenceFilter)
    def run(mult):
        f = HierarchicalPersistenceFilter(default_lambda=1 / (10 * DAY),
                                          use_class_prior=False,
                                          learn_rate=False,
                                          reloc_hazard_mult=mult)
        f.create("o", "x", 0.0)
        f.observe("o", 0.0, HIT)
        f.relocate("o", DAY)
        return f.presence("o", 2 * DAY)
    assert run(3.0) < run(1.0)
    assert run(1.0) == pytest.approx(math.exp(-0.1), rel=1e-9)


def test_creation_query_from_events():
    from hiper.evaluation.lifelong_objects import creation_query
    e = np.ones(3, np.float32)
    assert creation_query({"emb": e}) is e
    q = creation_query({"emb": e, "emb_clip": e, "geo": np.zeros(5)})
    assert set(q) == {"dino", "clip", "geo"}


# ── Post-hoc calibration (evaluation/calibration.py) ───────────────────────

def test_calibration_fixes_overconfident_misses():
    from hiper.evaluation import calibration as cal
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(2000):
        ev = "miss" if rng.random() < 0.3 else "hit"
        y = int(rng.random() < (0.4 if ev == "miss" else 0.9))
        p = 0.02 if ev == "miss" else 0.9          # misses far too confident
        rows.append({"model": "m", "pred_presence": p, "label_next": y,
                     "last_evidence": ev})
    c = cal.fit(rows)
    out = cal.apply(rows, c)
    assert out[0]["model"] == "m_cal"
    q_miss = np.mean([r["pred_presence"] for r in out
                      if r["last_evidence"] == "miss"])
    assert 0.3 < q_miss < 0.5


def test_calibration_identity_when_one_class():
    from hiper.evaluation import calibration as cal
    rows = [{"model": "m", "pred_presence": 0.7, "label_next": 1,
             "last_evidence": "hit"}] * 5
    out = cal.apply(rows, cal.fit(rows))
    assert out[0]["pred_presence"] == 0.7
