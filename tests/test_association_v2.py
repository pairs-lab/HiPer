"""
tests/test_association_v2.py
============================
Unit tests for association v2 (hiper.object_encoder.association_v2):
the score terms, per-pair modality renormalisation, the ablation switches,
the gated one-to-one Hungarian assignment, the tentative and relocation
flags, and pose inflation of the Mahalanobis distance.
"""

import math

import numpy as np
import pytest

from hiper.object_encoder.association_v2 import (
    CHI2_3_99,
    AssocConfig,
    AssocSet,
    assign,
    associate,
    mahalanobis,
    score_matrix,
)

E1 = np.array([1.0, 0.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0, 0.0])
E60 = np.array([0.6, 0.8, 0.0, 0.0])      # cos(E1, E60) = 0.6


def _set(pos, cov=0.01, u=0.0, clip=None, bye=None) -> AssocSet:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    n = len(pos)
    return AssocSet(
        pos=pos, cov=np.repeat((cov * np.eye(3))[None], n, axis=0),
        u=np.broadcast_to(np.asarray(u, dtype=np.float64), (n,)).copy(),
        clip=None if clip is None else np.atleast_2d(clip).astype(float),
        bye=None if bye is None else np.atleast_2d(bye).astype(float))


class TestScore:
    def test_score_matches_formula(self):
        """score = gamma * app + w_geo * exp(-d2 / 2), every term by hand."""
        tr = _set([0, 0, 0], cov=0.01, u=0.1, clip=E1, bye=E1)
        ob = _set([0.1, 0, 0], cov=0.02, u=0.2, clip=E60, bye=E1)
        P = 0.03 * np.eye(3)
        cfg = AssocConfig(w=0.5, w_geo=0.5, sigma_u=1.0)
        score, d2, u_sum = score_matrix(tr, ob, cfg, P)
        d2_ref = 0.1 ** 2 / (0.01 + 0.02 + 0.03)
        app = 0.5 * 1.0 + 0.5 * 0.6
        ref = math.exp(-0.3 / 2.0) * app + 0.5 * math.exp(-d2_ref / 2.0)
        assert d2[0, 0] == pytest.approx(d2_ref, rel=1e-6)
        assert u_sum[0, 0] == pytest.approx(0.3)
        assert score[0, 0] == pytest.approx(ref, rel=1e-6)

    def test_no_pose_inflation_ignores_session_cov(self):
        tr = _set([0, 0, 0], cov=0.01)
        ob = _set([0.1, 0, 0], cov=0.02)
        cfg = AssocConfig(pose_inflation=False)
        _, d2, _ = score_matrix(tr, ob, cfg, 0.03 * np.eye(3))
        assert d2[0, 0] == pytest.approx(0.01 / 0.03, rel=1e-6)

    def test_mahalanobis_full_covariance(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 3))
        B = rng.normal(size=(3, 3))
        tr = AssocSet(pos=rng.normal(size=(1, 3)), cov=(A @ A.T)[None],
                      u=np.zeros(1))
        ob = AssocSet(pos=rng.normal(size=(1, 3)), cov=(B @ B.T)[None],
                      u=np.zeros(1))
        P = 0.1 * np.eye(3)
        diff = tr.pos[0] - ob.pos[0]
        ref = diff @ np.linalg.inv(A @ A.T + B @ B.T + P) @ diff
        assert mahalanobis(tr, ob, P)[0, 0] == pytest.approx(ref, rel=1e-6)

    def test_missing_modality_renormalised_per_pair(self):
        """Object 0 has no BYE: its pair falls back to CLIP alone, while
        object 1 keeps the BYE+CLIP ensemble."""
        tr = _set([0, 0, 0], clip=E1, bye=E1)
        ob = _set([[0, 0, 0], [0, 0, 0]], clip=[E60, E60],
                  bye=[np.zeros(4), E2])
        cfg = AssocConfig(w=0.5, w_geo=0.0, use_gamma=False)
        score, _, _ = score_matrix(tr, ob, cfg)
        assert score[0, 0] == pytest.approx(0.6)               # CLIP only
        assert score[0, 1] == pytest.approx(0.5 * 0.0 + 0.5 * 0.6)

    @pytest.mark.parametrize("w, ref", [(1.0, 0.6), (0.0, 0.0)])
    def test_single_modality_ablations(self, w, ref):
        """w = 1: CLIP only (BYE ignored); w = 0: BYE only (CLIP ignored)."""
        tr = _set([0, 0, 0], clip=E1, bye=E1)
        ob = _set([0, 0, 0], clip=E60, bye=E2)
        cfg = AssocConfig(w=w, w_geo=0.0, use_gamma=False)
        assert score_matrix(tr, ob, cfg)[0][0, 0] == pytest.approx(ref)

    def test_bye_only_without_bye_leaves_geometry(self):
        tr = _set([0, 0, 0], clip=E1)
        ob = _set([0, 0, 0], clip=E1)
        cfg = AssocConfig(w=0.0, w_geo=0.5)
        assert score_matrix(tr, ob, cfg)[0][0, 0] == pytest.approx(0.5)

    def test_gamma_switch(self):
        tr = _set([0, 0, 0], u=0.3, clip=E1)
        ob = _set([0, 0, 0], u=0.3, clip=E1)
        on = AssocConfig(w_geo=0.0, sigma_u=0.5)
        off = AssocConfig(w_geo=0.0, sigma_u=0.5, use_gamma=False)
        assert score_matrix(tr, ob, on)[0][0, 0] == \
            pytest.approx(math.exp(-0.6 / 0.5))
        assert score_matrix(tr, ob, off)[0][0, 0] == pytest.approx(1.0)


class TestAssignment:
    def test_tentative_gate_and_switch(self):
        tr = _set([0, 0, 0], u=0.4, clip=E1)
        ob = _set([0, 0, 0], u=0.2, clip=E1)
        m_on = associate(tr, ob, AssocConfig(tau_u=0.5, tau_match=0.1))[0]
        m_off = associate(tr, ob, AssocConfig(tau_u=0.5, tau_match=0.1,
                                              use_tau_u=False))[0]
        assert m_on.obj == 0 and m_on.tentative
        assert m_off.obj == 0 and not m_off.tentative

    def test_relocation_flag_at_chi2(self):
        """Same appearance: matched either way; relocated iff d2 > 11.34."""
        cfg = AssocConfig(pose_inflation=False)
        tr = _set([[0, 0, 0], [5.0, 0, 0]], cov=0.005, clip=[E1, E2])
        near = math.sqrt(0.9 * CHI2_3_99 * 0.01)   # d2 = 0.9 * chi2
        far = math.sqrt(1.1 * CHI2_3_99 * 0.01)    # d2 = 1.1 * chi2
        ob = _set([[near, 0, 0], [5.0 + far, 0, 0]], cov=0.005,
                  clip=[E1, E2])
        m = associate(tr, ob, cfg)
        assert [x.obj for x in m] == [0, 1]
        assert m[0].d2 == pytest.approx(0.9 * CHI2_3_99, rel=1e-6)
        assert not m[0].relocated and m[1].relocated

    def test_hungarian_one_to_one_geometry_breaks_ties(self):
        """Duplicates (same appearance): each track takes the object at its
        own position, and a third look-alike cannot share a match."""
        ob = _set([[-0.5, 0, 2], [0.5, 0, 2]], cov=0.001, clip=[E1, E1])
        tr = _set([[0.5, 0, 2], [-0.5, 0, 2], [0.0, 0, 3.5]], cov=0.001,
                  clip=[E1, E1, E1])
        m = associate(tr, ob, AssocConfig())
        assert [x.obj for x in m] == [1, 0, -1]

    def test_pairs_below_threshold_never_displace_a_match(self):
        """Ungated max-sum would take the two sub-threshold pairs (1.48)
        and leave no match; gating keeps the valid pair (0, 0)."""
        score = np.array([[0.80, 0.74], [0.74, -1.0]])
        z = np.zeros_like(score)
        m = assign(score, z, z, AssocConfig(tau_match=0.75))
        assert (m[0].obj, m[1].obj) == (0, -1)
        assert m[0].score == pytest.approx(0.80)

    def test_empty_inputs(self):
        cfg = AssocConfig()
        assert associate(_set(np.zeros((0, 3))), _set([0, 0, 0]), cfg) == []
        m = associate(_set([[0, 0, 0], [1, 0, 0]]), _set(np.zeros((0, 3))),
                      cfg)
        assert [x.obj for x in m] == [-1, -1]
