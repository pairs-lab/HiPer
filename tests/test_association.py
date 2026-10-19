"""
tests/test_association.py
=========================
Unit tests for the score-level VLM-ensemble association
(hiper.object_encoder.association): Eq. (13) score fusion,
gamma uncertainty down-weighting, Hungarian one-to-one assignment,
and the tau_u tentative-match gate.
"""

import numpy as np
import pytest

from hiper.object_encoder.association import ensemble_associate


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _bank(dim, n, rng):
    rows = rng.normal(size=(n, dim)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class TestEnsembleScoring:
    def test_score_level_fusion_matches_eq13(self):
        """s_ens must equal (1-w) s_BYE + w s_CLIP exactly — embeddings of
        different dimensionality are never mixed at the feature level."""
        rng = np.random.default_rng(0)
        bye_bank = _bank(8, 1, rng)
        clip_bank = _bank(16, 1, rng)
        # query identical to bank in BYE space, orthogonal in CLIP space
        bye_q = bye_bank.copy()
        clip_q = _bank(16, 1, rng)
        clip_q -= (clip_q @ clip_bank.T) * clip_bank   # orthogonalise
        clip_q /= np.linalg.norm(clip_q)

        res = ensemble_associate(
            bye_queries=bye_q, clip_queries=clip_q,
            bank_ids=["a"], bye_bank=bye_bank, clip_bank=clip_bank,
            w=0.5, tau_u=10.0, score_threshold=0.0)
        # s_BYE = 1, s_CLIP = 0  ->  ensemble = 0.5
        assert res[0].score == pytest.approx(0.5, abs=1e-5)

    def test_mixed_dims_do_not_crash(self):
        """Regression: 256-d BYE + 512-d CLIP previously crashed when
        fused at the feature level."""
        rng = np.random.default_rng(1)
        res = ensemble_associate(
            bye_queries=_bank(256, 3, rng), clip_queries=_bank(512, 3, rng),
            bank_ids=["a", "b"], bye_bank=_bank(256, 2, rng),
            clip_bank=_bank(512, 2, rng),
            w=0.5, score_threshold=-1.0)
        assert len(res) == 3

    def test_missing_modality_renormalises(self):
        rng = np.random.default_rng(2)
        clip_bank = _bank(16, 1, rng)
        res = ensemble_associate(
            bye_queries=None, clip_queries=clip_bank.copy(),
            bank_ids=["a"], bye_bank=None, clip_bank=clip_bank,
            w=0.5, tau_u=10.0, score_threshold=0.0)
        # CLIP carries the full weight: perfect match scores 1, not 0.5
        assert res[0].score == pytest.approx(1.0, abs=1e-5)


class TestHungarian:
    def test_one_to_one_assignment(self):
        """Two near-identical queries cannot both claim the same object."""
        rng = np.random.default_rng(3)
        bank = _bank(16, 2, rng)
        queries = np.stack([bank[0], bank[0] * 0.99 + bank[1] * 0.01])
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        res = ensemble_associate(
            bye_queries=None, clip_queries=queries,
            bank_ids=["a", "b"], bye_bank=None, clip_bank=bank,
            w=0.5, score_threshold=-1.0, tau_u=10.0)
        matched = [r.object_id for r in res]
        assert sorted(matched) == ["a", "b"]

    def test_below_threshold_unmatched(self):
        rng = np.random.default_rng(4)
        bank = _bank(16, 1, rng)
        query = _bank(16, 1, rng)
        query -= (query @ bank.T) * bank
        query /= np.linalg.norm(query)
        res = ensemble_associate(
            bye_queries=None, clip_queries=query,
            bank_ids=["a"], bye_bank=None, clip_bank=bank,
            score_threshold=0.75)
        assert res[0].object_id == ""
        assert not res[0].is_confident

    def test_empty_bank_returns_unmatched(self):
        rng = np.random.default_rng(5)
        res = ensemble_associate(
            bye_queries=None, clip_queries=_bank(16, 2, rng),
            bank_ids=[], bye_bank=None, clip_bank=None)
        assert len(res) == 2
        assert all(r.object_id == "" for r in res)


class TestUncertaintyGate:
    def test_gamma_downweights_uncertain_candidates(self):
        """Of two equally similar candidates, the one with lower u_j wins."""
        rng = np.random.default_rng(6)
        emb = _bank(16, 1, rng)[0]
        bank = np.stack([emb, emb])
        res = ensemble_associate(
            bye_queries=None, clip_queries=emb[np.newaxis],
            bank_ids=["uncertain", "confident"],
            bye_bank=None, clip_bank=bank,
            bank_uncertainties=np.array([0.5, 0.0]),
            sigma_u=0.25, tau_u=10.0, score_threshold=0.0)
        assert res[0].object_id == "confident"

    def test_tau_u_marks_tentative(self):
        """A match whose combined uncertainty exceeds tau_u is tentative
        and must not be treated as confident (no eps_g update)."""
        rng = np.random.default_rng(7)
        emb = _bank(16, 1, rng)[0]
        res = ensemble_associate(
            bye_queries=None, clip_queries=emb[np.newaxis],
            bank_ids=["a"], bye_bank=None, clip_bank=emb[np.newaxis],
            bank_uncertainties=np.array([0.1]),
            query_uncertainties=np.array([0.45]),
            sigma_u=1000.0,        # disable gamma so the score stays high
            tau_u=0.5, score_threshold=0.5)
        assert res[0].object_id == "a"
        assert res[0].is_tentative
        assert not res[0].is_confident

    def test_confident_when_below_tau_u(self):
        rng = np.random.default_rng(8)
        emb = _bank(16, 1, rng)[0]
        res = ensemble_associate(
            bye_queries=None, clip_queries=emb[np.newaxis],
            bank_ids=["a"], bye_bank=None, clip_bank=emb[np.newaxis],
            bank_uncertainties=np.array([0.1]),
            sigma_u=1000.0,        # isolate the tau_u gate from gamma
            tau_u=0.5, score_threshold=0.5)
        assert res[0].is_confident
        assert not res[0].is_tentative
