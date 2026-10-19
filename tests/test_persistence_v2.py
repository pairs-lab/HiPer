"""
tests/test_persistence_v2.py
============================
Unit tests for hiper.ephemerality.persistence_v2.HierarchicalPersistenceFilter.

The fixed-rate limit (learn_rate=False) is checked against
  - the closed-form exponential persistence filter of Rosen et al.,
    "Towards lifelong feature-based mapping in semi-static environments",
    ICRA 2016 (implemented below), and
  - Rosen's official pure-Python code, github.com/david-m-rosen/Persistence-Filter
    at commit ROSEN_COMMIT, cloned to <workspace>/third_party/Persistence-Filter
    (or $ROSEN_PF_DIR). That test is skipped when the clone is absent.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hiper.ephemerality.persistence_v2 import (
    HIT, MISS, NONE, HierarchicalPersistenceFilter)

DAY = 86400.0
TABLE = {c: 1.0 / (d * DAY) for c, d in
         {"Apple": 1, "Pillow": 7, "Book": 14, "Box": 30, "Sofa": 365}.items()}

ROSEN_COMMIT = "1c7c812ec9b295429b0e539812e4d5d98c9b19b7"
ROSEN_DIR = Path(os.environ.get(
    "ROSEN_PF_DIR",
    Path(__file__).resolve().parents[3] / "third_party" / "Persistence-Filter"))


def _hpf(**kw):
    return HierarchicalPersistenceFilter(lambda_table=TABLE, **kw)


# ── Rosen's exponential persistence filter ───────────────────────────────────

def _rosen_closed_form(lam, t0, times, ys, p_m, p_f, t_pred):
    """
    p(X_t = 1 | y_1:N) for t >= t_N, survival prior S(t) = exp(-lam (t - t0)).
    T in [t_k, t_k+1) means present at t_1..t_k and gone at t_k+1..t_N:
      p(y) = sum_k prod_{i<=k} L1(y_i) prod_{i>k} L0(y_i) (S(t_k) - S(t_k+1)),
      p(X_t = 1 | y) = prod_i L1(y_i) S(t) / p(y).
    """
    S = [1.0] + [math.exp(-lam * (t - t0)) for t in times] + [0.0]
    L1 = [(1.0 - p_m) if y else p_m for y in ys]
    L0 = [p_f if y else (1.0 - p_f) for y in ys]
    evidence = sum(math.prod(L1[:k]) * math.prod(L0[k:]) * (S[k] - S[k + 1])
                   for k in range(len(ys) + 1))
    return math.prod(L1) * math.exp(-lam * (t_pred - t0)) / evidence


def _random_sequences(n_seq=300, seed=0):
    """Lifetimes 0.5 d .. 10 y (log-uniform), gaps 0.1-7 d, P_M, P_F random."""
    rng = np.random.default_rng(seed)
    for _ in range(n_seq):
        days = float(np.exp(rng.uniform(np.log(0.5), np.log(3650.0))))
        p_m = float(rng.uniform(0.05, 0.3))
        p_f = float(rng.uniform(0.01, 0.2))
        n = int(rng.integers(1, 12))
        t0 = float(rng.uniform(0.0, 10.0)) * DAY
        times = (t0 + np.cumsum(rng.uniform(0.1, 7.0, n)) * DAY).tolist()
        ys = (rng.random(n) < 0.6).tolist()
        t_pred = times[-1] + float(rng.uniform(0.0, 7.0)) * DAY
        none_mask = (rng.random(n) < 0.3).tolist()
        yield 1.0 / (days * DAY), p_m, p_f, t0, times, ys, t_pred, none_mask


def _run_fixed_rate(lam, p_m, p_f, t0, times, ys, none_mask):
    """HPF(learn_rate=False) on the sequence; NONE inserted before some steps."""
    f = HierarchicalPersistenceFilter(lambda_table={"c": lam},
                                      learn_rate=False,
                                      p_miss=p_m, p_false=p_f)
    f.create("o", "c", t0)
    for t, y, none_first in zip(times, ys, none_mask):
        if none_first:
            f.observe("o", t - 0.05 * DAY, NONE)
        f.observe("o", t, HIT if y else MISS)
    return f


def test_fixed_rate_equals_rosen_closed_form():
    err = 0.0
    for lam, p_m, p_f, t0, times, ys, t_pred, nm in _random_sequences():
        f = _run_fixed_rate(lam, p_m, p_f, t0, times, ys, nm)
        for t in (times[-1], t_pred):
            ref = _rosen_closed_form(lam, t0, times, ys, p_m, p_f, t)
            err = max(err, abs(f.presence("o", t) - ref))
    assert err < 1e-12


def _git_head(path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@pytest.fixture(scope="module")
def rosen_filter_cls():
    """Rosen's official PersistenceFilter class (pinned commit)."""
    py = ROSEN_DIR / "python"
    if not (py / "PersistenceFilter.py").exists():
        pytest.skip(f"Rosen's Persistence-Filter not found at {ROSEN_DIR}: git "
                    "clone https://github.com/david-m-rosen/Persistence-Filter "
                    f"there and check out {ROSEN_COMMIT}")
    head = _git_head(ROSEN_DIR)
    if head is not None and head != ROSEN_COMMIT:
        pytest.skip(f"{ROSEN_DIR} is at {head}; this test is pinned to "
                    f"{ROSEN_COMMIT}")
    pytest.importorskip("scipy")             # persistence_filter_utils needs it
    # PersistenceFilter.py imports persistence_filter_utils by its bare name.
    spec = importlib.util.spec_from_file_location(
        "persistence_filter_utils", py / "persistence_filter_utils.py")
    utils = importlib.util.module_from_spec(spec)
    sys.modules["persistence_filter_utils"] = utils
    try:
        spec.loader.exec_module(utils)
        spec = importlib.util.spec_from_file_location(
            "rosen_persistence_filter", py / "PersistenceFilter.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("persistence_filter_utils", None)
    return mod.PersistenceFilter


def test_fixed_rate_equals_rosen_official_code(rosen_filter_cls):
    err = 0.0
    for lam, p_m, p_f, t0, times, ys, t_pred, nm in _random_sequences():
        f = _run_fixed_rate(lam, p_m, p_f, t0, times, ys, nm)
        ref = rosen_filter_cls(lambda t, lam=lam: -lam * t,
                               initialization_time=t0)
        for t, y in zip(times, ys):
            ref.update(bool(y), t, p_m, p_f)
        for t in (times[-1], t_pred):
            err = max(err, abs(f.presence("o", t) - float(ref.predict(t))))
    assert err < 1e-12


# ── Per-object rate learning ─────────────────────────────────────────────────

def test_learning_raises_persistence_of_reobserved_short_lived_object():
    learn, fixed = _hpf(), _hpf(learn_rate=False)
    for f in (learn, fixed):
        f.create("o", "Apple", 0.0)            # 1-day class: E[p] = e^-1
    prior_next = learn.presence("o", DAY)
    prev = prior_next
    for d in range(1, 11):                     # seen at the same place daily
        for f in (learn, fixed):
            f.observe("o", d * DAY, HIT)
        nxt = learn.presence("o", (d + 1) * DAY)
        assert nxt > prev
        prev = nxt
    assert learn.rate_mean("o") > fixed.rate_mean("o") + 0.3
    assert prev > fixed.presence("o", 11 * DAY) + 0.3
    assert prev > prior_next + 0.3


def test_learning_lowers_persistence_after_relocations():
    learn, fixed = _hpf(), _hpf(learn_rate=False)
    for f in (learn, fixed):
        f.create("o", "Sofa", 0.0)             # 1-year class
    prev = learn.rate_mean("o")
    for d in range(1, 11):                     # moved to a new place daily
        for f in (learn, fixed):
            f.relocate("o", d * DAY)
        assert learn.rate_mean("o") < prev
        prev = learn.rate_mean("o")
    assert fixed.rate_mean("o") == pytest.approx(math.exp(-1.0 / 365.0))
    assert learn.presence("o", 11 * DAY) < fixed.presence("o", 11 * DAY) - 0.5


def test_learning_lowers_rate_after_repeated_vanishes():
    learn, fixed = _hpf(), _hpf(learn_rate=False)
    for f in (learn, fixed):
        f.create("o", "Sofa", 0.0)
    for d, ev in enumerate([HIT, MISS, MISS] * 6, start=1):
        for f in (learn, fixed):
            f.observe("o", d * DAY, ev)
    assert learn.rate_mean("o") < fixed.rate_mean("o") - 0.02


@pytest.mark.parametrize("cls", sorted(TABLE))
def test_rate_variance_shrinks_with_evidence(cls):
    f = _hpf()
    f.create("o", cls, 0.0)
    v = [f.rate_var("o")]
    for d in range(1, 21):
        f.observe("o", d * DAY, HIT)
        v.append(f.rate_var("o"))
    assert all(b < a for a, b in zip(v, v[1:]))
    assert v[-1] < v[0] / 10.0


def test_rate_variance_shrinks_with_repeated_relocations():
    f = _hpf()
    f.create("o", "Sofa", 0.0)
    v = {}
    for d in range(1, 31):
        f.relocate("o", d * DAY)
        v[d] = f.rate_var("o")
    assert v[30] < v[10] < v[3]


# ── NONE and the fixed-rate / no-class-prior limits ──────────────────────────

@pytest.mark.parametrize("learn_rate", [True, False])
def test_none_leaves_state_unchanged(learn_rate):
    f, twin = _hpf(learn_rate=learn_rate), _hpf(learn_rate=learn_rate)
    for g in (f, twin):
        g.create("o", "Book", 0.0)
        g.observe("o", 1.0 * DAY, HIT)
        g.observe("o", 3.0 * DAY, MISS)
    before = f.export("o")
    for t in (4.0, 9.0, 2.0):                  # any time, even an earlier one
        f.observe("o", t * DAY, NONE, eps_l=0.7, n_obs=12)
        assert f.export("o") == before
    for t in (3.0, 5.0, 20.0):
        assert f.presence("o", t * DAY) == twin.presence("o", t * DAY)
    f.observe("o", 6.0 * DAY, HIT)
    twin.observe("o", 6.0 * DAY, HIT)
    assert f.export("o") == twin.export("o")


def test_no_class_prior_uses_default_lambda():
    lam = 1.0 / (30.0 * DAY)
    f = HierarchicalPersistenceFilter(lambda_table=TABLE, default_lambda=lam,
                                      use_class_prior=False)
    for cls in ("Apple", "Sofa", "never_seen"):
        assert f.class_lambda(cls) == lam
        assert f.prior_mean(cls) == pytest.approx(math.exp(-lam * DAY))
    for oid, cls in (("apple", "Apple"), ("sofa", "Sofa")):
        f.create(oid, cls, 0.0)
        f.observe(oid, 2.0 * DAY, HIT)
    ea, es = f.export("apple"), f.export("sofa")
    assert (ea["pi"], ea["a"], ea["b"]) == (es["pi"], es["a"], es["b"])
    assert f.presence("apple", 9 * DAY) == f.presence("sofa", 9 * DAY)
    with_prior = HierarchicalPersistenceFilter(lambda_table=TABLE,
                                               default_lambda=lam)
    assert with_prior.class_lambda("Apple") == TABLE["Apple"]
    assert with_prior.class_lambda("never_seen") == lam


def test_fixed_rate_names_and_single_lambda_limit():
    lam = 1.0 / (30.0 * DAY)
    pf = HierarchicalPersistenceFilter(lambda_table=TABLE, default_lambda=lam,
                                       learn_rate=False, use_class_prior=False)
    assert pf.name == "pf"
    assert _hpf(learn_rate=False).name == "pf_class"
    assert _hpf().name == "hpf"
    pf.create("o", "Apple", 0.0)               # class ignored: PF(lambda)
    times, ys = [1.0 * DAY, 2.5 * DAY, 6.0 * DAY], [True, False, True]
    for t, y in zip(times, ys):
        pf.observe("o", t, HIT if y else MISS)
    ref = _rosen_closed_form(lam, 0.0, times, ys, 0.1, 0.05, 8.0 * DAY)
    assert pf.presence("o", 8.0 * DAY) == pytest.approx(ref, abs=1e-14)
    assert pf.rate_var("o") == 0.0


# ── Prune and commit gates ───────────────────────────────────────────────────

@pytest.mark.parametrize("gate", [True, False])
def test_prune_after_miss_ignores_variance_gate(gate):
    f = _hpf(variance_gate=gate)
    f.create("o", "Apple", 0.0)
    f.observe("o", DAY, MISS)
    assert f.presence("o", DAY) < f.tau_pi
    assert f.rate_var("o") >= f.tau_v          # rate still uncertain
    assert f.should_prune("o", DAY)


def test_variance_gate_keeps_object_that_only_decayed():
    on, off = _hpf(), _hpf(variance_gate=False)
    for f in (on, off):
        f.create("o", "Apple", 0.0)
        f.observe("o", 0.0, HIT)
    t = 5.0 * DAY                              # no evidence for 5 days
    assert on.presence("o", t) < on.tau_pi
    assert on.rate_var("o") >= on.tau_v
    assert not on.should_prune("o", t)         # prior decay alone: keep
    assert off.should_prune("o", t)


def test_variance_gate_prunes_decayed_object_with_confident_rate():
    f = _hpf(kappa0=100.0)                     # confident class prior
    f.create("o", "Apple", 0.0)
    assert f.rate_var("o") < f.tau_v
    assert not f.should_prune("o", 1.0 * DAY)  # presence e^-1 >= tau_pi
    assert f.presence("o", 3.0 * DAY) < f.tau_pi
    assert f.should_prune("o", 3.0 * DAY)


def test_fixed_rate_prunes_on_presence_alone():
    f = _hpf(learn_rate=False)
    f.create("o", "Apple", 0.0)
    assert not f.should_prune("o", 2.0 * DAY)  # e^-2 = 0.135 >= 0.1
    assert f.should_prune("o", 3.0 * DAY)      # e^-3 = 0.050 <  0.1


def test_commit_gate():
    on, off = _hpf(), _hpf(variance_gate=False)
    for f in (on, off):
        f.create("o", "Book", 0.0)             # E[p] = 0.93, Var = 0.0128
    assert on.rate_mean("o") >= on.tau_commit
    assert on.rate_var("o") >= on.tau_v
    assert not on.is_committed("o", 0.0)       # mean high, rate uncertain
    assert off.is_committed("o", 0.0)
    assert not off.is_committed("o", 60.0 * DAY)   # presence below 0.5
    on.observe("o", DAY, HIT)
    assert on.rate_var("o") < on.tau_v
    assert on.is_committed("o", DAY)
    short = _hpf(variance_gate=False)
    short.create("o", "Apple", 0.0)            # E[p] = 0.37 < tau_commit
    assert not short.is_committed("o", 0.0)


# ── Export / load ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("learn_rate", [True, False])
def test_export_load_round_trip(learn_rate):
    f = _hpf(learn_rate=learn_rate)
    f.create("o", "Book", 0.0)
    f.observe("o", 1.0 * DAY, HIT, eps_l=0.2)
    f.observe("o", 3.0 * DAY, MISS)
    f.relocate("o", 4.0 * DAY)
    f.observe("o", 6.0 * DAY, MISS)
    state = json.loads(json.dumps(f.export("o")))
    g = _hpf(learn_rate=learn_rate)
    g.load("o", state)
    assert "o" in g
    for t in (6.0, 7.5, 30.0):
        t *= DAY
        assert g.presence("o", t) == f.presence("o", t)
        assert g.should_prune("o", t) == f.should_prune("o", t)
        assert g.is_committed("o", t) == f.is_committed("o", t)
    assert g.rate_mean("o") == f.rate_mean("o")
    assert g.rate_var("o") == f.rate_var("o")
    for h in (f, g):
        h.observe("o", 8.0 * DAY, HIT)
        h.observe("o", 9.0 * DAY, MISS)
    assert g.export("o") == f.export("o")
    g.remove("o")
    assert "o" not in g
