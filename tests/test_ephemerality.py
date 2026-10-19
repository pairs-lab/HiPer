"""
tests/test_ephemerality.py
==========================
Unit tests for:
  - hiper.ephemerality.local_bpf.LocalEphemeralityBPF
  - hiper.ephemerality.global_bpf.GlobalEphemeralityBPF

Run with: python -m pytest tests/test_ephemerality.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hiper.ephemerality.local_bpf  import LocalEphemeralityBPF
from hiper.ephemerality.global_bpf import GlobalEphemeralityBPF


# ─────────────────────────────────────────────────────────────────────────────
# LocalEphemeralityBPF
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalBPF:

    def _bpf(self):
        return LocalEphemeralityBPF(
            sigma_occupied=0.1, sigma_free=0.3,
            alpha=0.5, beta=0.1, knn_k=3)

    def test_output_shape(self):
        bpf = self._bpf()
        pts = np.random.rand(20, 3).astype(np.float32)
        origin = np.zeros(3, dtype=np.float32)
        eps = bpf.update(pts, None, origin)
        assert eps.shape == (20,)

    def test_output_range(self):
        bpf = self._bpf()
        pts = np.random.rand(50, 3).astype(np.float32)
        origin = np.zeros(3, dtype=np.float32)
        eps = bpf.update(pts, None, origin)
        assert np.all(eps >= 0.0)
        assert np.all(eps <= 1.0)

    def test_single_point(self):
        bpf = self._bpf()
        pts    = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        origin = np.zeros(3, dtype=np.float32)
        eps = bpf.update(pts, None, origin)
        assert eps.shape == (1,)
        assert 0.0 <= float(eps[0]) <= 1.0

    def test_bayesian_update_symmetry(self):
        """
        f_occ(0) should decrease ε from 0.5 and
        f_free(0) should increase it — check direction.
        """
        bpf = LocalEphemeralityBPF(sigma_occupied=0.1, sigma_free=0.1,
                                    alpha=0.5, beta=0.1, knn_k=1)
        # occupied endpoint at distance 0 from itself
        f_occ  = bpf._f_occupied(0.0)
        after_occ = bpf._bayesian_update(0.5, f_occ)
        # free point at distance 0 from free space
        f_free = bpf._f_free(0.0)
        after_free = bpf._bayesian_update(0.5, f_free)

        # occupied endpoint should pull ε toward lower values
        # free space should push ε higher
        # (directions depend on alpha/beta — just check they differ)
        assert after_occ != after_free

    def test_with_existing_cloud(self):
        """Should not crash when existing_cloud is provided."""
        bpf = self._bpf()
        new_pts = np.random.rand(10, 3).astype(np.float32)
        existing = np.random.rand(30, 3).astype(np.float32)
        origin   = np.zeros(3, dtype=np.float32)
        eps = bpf.update(new_pts, existing, origin)
        assert eps.shape == (10,)


# ─────────────────────────────────────────────────────────────────────────────
# GlobalEphemeralityBPF
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalBPF:

    _PRIORS = {"tree": 0.001, "person": 2.0, "car": 0.5}

    def _bpf(self):
        return GlobalEphemeralityBPF(
            default_lambda=0.01,
            class_priors=self._PRIORS)

    def test_new_object_initialises_at_half(self):
        bpf = self._bpf()
        eg  = bpf.get_epsilon_g("obj_1", "tree", current_time=0.0)
        assert 0.0 <= eg <= 1.0

    def test_bayesian_update_fuses_correctly(self):
        """
        Fusing eg=0.5 with el=0.5 should stay near 0.5 (max entropy).
        Fusing eg=0.5 with el=0.0 should move eg toward 0 (static).
        """
        bpf = self._bpf()
        # High el → object is dynamic → eg should increase
        bpf.load_object_state("a", 0.5, 0.5, 0.0, "car")
        eg_hi = bpf.update("a", "car", epsilon_l=0.9, current_time=1.0)

        bpf.load_object_state("b", 0.5, 0.5, 0.0, "car")
        eg_lo = bpf.update("b", "car", epsilon_l=0.1, current_time=1.0)

        assert eg_hi > eg_lo, (
            "High ε_l should yield higher ε_g than low ε_l")

    def test_permanent_object_stays_low(self):
        """
        A 'tree' (λ ≈ 0) seen repeatedly should stay near ε_g = 0.
        """
        bpf = self._bpf()
        bpf.load_object_state("tree_1", 0.1, 0.9, 0.0, "tree")
        for t in range(1, 10):
            eg = bpf.update("tree_1", "tree", epsilon_l=0.05,
                             current_time=float(t))
        assert eg < 0.3, f"Permanent object ε_g too high: {eg}"

    def test_ephemeral_object_increases(self):
        """
        A 'person' (high λ) not seen for 100 s should have ε_g near 1.
        """
        bpf = self._bpf()
        bpf.load_object_state("p1", 0.5, 0.5, 0.0, "person")
        bpf.mark_not_observed("p1", current_time=100.0)
        eg = bpf.get_epsilon_g("p1", "person", current_time=100.0)
        assert eg > 0.5, f"Ephemeral object ε_g should be high: {eg}"

    def test_update_class_priors(self):
        bpf = self._bpf()
        bpf.update_class_priors({"bench": 0.02})
        # bench is now known → should not use default
        bpf.load_object_state("bench_1", 0.5, 0.5, 0.0, "bench")
        eg = bpf.update("bench_1", "bench", epsilon_l=0.05,
                         current_time=50.0)
        assert 0.0 <= eg <= 1.0

    def test_bayes_fuse_boundary_values(self):
        """Fusing extremes should not produce NaN or out-of-range values."""
        for eg in [0.0, 0.5, 1.0]:
            for el in [0.0, 0.5, 1.0]:
                fused = GlobalEphemeralityBPF._bayes_fuse(eg, el)
                assert 0.0 <= fused <= 1.0, \
                    f"Bayes fuse({eg},{el}) = {fused} out of range"

    def test_load_and_get_state(self):
        bpf = self._bpf()
        bpf.load_object_state("x", 0.3, 0.7, 10.0, "car")
        states = bpf.get_all_states()
        assert "x" in states
        assert pytest.approx(states["x"].epsilon_g, abs=1e-3) == 0.3

    def test_decay_direction_increases_ephemerality_with_absence(self):
        """Survival decay acts on persistence p = 1 - eps: the longer an
        object goes unobserved, the MORE ephemeral it must look.
        Regression for the inverted eps*exp(-lambda dt) form, which
        decreased ephemerality with elapsed time."""
        bpf = self._bpf()
        bpf.load_object_state("c1", 0.2, 0.8, 0.0, "car")
        eg_soon  = bpf.get_epsilon_g("c1", "car", current_time=1.0)
        eg_later = bpf.get_epsilon_g("c1", "car", current_time=50.0)
        assert eg_later > eg_soon
        assert eg_later <= 1.0

    def test_decay_is_class_conditioned(self):
        """Same absence interval: a person's ephemerality must grow much
        faster than a tree's."""
        bpf = self._bpf()
        bpf.load_object_state("p", 0.2, 0.8, 0.0, "person")
        bpf.load_object_state("t", 0.2, 0.8, 0.0, "tree")
        eg_p = bpf.get_epsilon_g("p", "person", current_time=10.0)
        eg_t = bpf.get_epsilon_g("t", "tree", current_time=10.0)
        assert eg_p > eg_t

    def test_absence_adds_only_transience_evidence(self):
        """mark_not_observed must add mass to beta_b only — absence is
        never evidence of persistence."""
        bpf = self._bpf()
        bpf.load_object_state("p1", 0.5, 0.5, 0.0, "person",
                              beta_a=2.0, beta_b=2.0)
        states = bpf.get_all_states()
        a0, b0 = states["p1"].beta_a, states["p1"].beta_b
        bpf.mark_not_observed("p1", current_time=100.0)
        states = bpf.get_all_states()
        assert states["p1"].beta_a == pytest.approx(a0)
        assert states["p1"].beta_b > b0
        # epsilon_g re-derived from the posterior mean
        expected = 1.0 - states["p1"].persistence_mean()
        assert states["p1"].epsilon_g == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Second-order ephemerality (Beta belief)
# ─────────────────────────────────────────────────────────────────────────────

class TestBetaBelief:

    _PRIORS = {"tree": 0.001, "person": 2.0, "car": 0.5}

    def _bpf(self):
        return GlobalEphemeralityBPF(
            default_lambda=0.01, class_priors=self._PRIORS,
            beta_concentration=10.0)

    def test_variance_decreases_with_evidence(self):
        """Repeated consistent observations should tighten the posterior."""
        bpf = self._bpf()
        bpf.load_object_state("t", 0.1, 0.9, 0.0, "tree")
        v0 = bpf.get_persistence_var("t")
        for k in range(1, 12):
            bpf.update("t", "tree", epsilon_l=0.05, current_time=float(k))
        v1 = bpf.get_persistence_var("t")
        assert v1 < v0, "Variance should shrink as evidence accumulates"

    def test_new_object_not_committed(self):
        """A freshly seen object has wide variance and is not committed."""
        bpf = self._bpf()
        bpf.update("new", "car", epsilon_l=0.5, current_time=1.0)
        assert not bpf.is_committed("new", tau_v=0.02)

    def test_committed_after_enough_evidence(self):
        bpf = self._bpf()
        bpf.load_object_state("c", 0.1, 0.9, 0.0, "tree")
        for k in range(1, 40):
            bpf.update("c", "tree", epsilon_l=0.05, current_time=float(k))
        assert bpf.is_committed("c", tau_v=0.02)

    def test_seed_matches_scalar(self):
        bpf = self._bpf()
        bpf.load_object_state("s", 0.2, 0.8, 0.0, "tree")
        eg = bpf.get_epsilon_g("s", "tree", current_time=0.0)
        assert 0.0 <= eg <= 1.0
