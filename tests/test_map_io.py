"""
tests/test_map_io.py
====================
Unit tests for:
  - hiper.map_io.object_record.ObjectRecord
  - hiper.map_io.ltm_io.LTMIO
  - hiper.map_io.sparsifier.KLSparsifier

Run with: python -m pytest tests/test_map_io.py -v
"""

import sys
import tempfile
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hiper.map_io.object_record import ObjectRecord
from hiper.map_io.ltm_io        import LTMIO
from hiper.map_io.sparsifier     import KLSparsifier


# ─────────────────────────────────────────────────────────────────────────────
# ObjectRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectRecord:

    def _rec(self, oid="test_obj", embed_dim=32) -> ObjectRecord:
        rec = ObjectRecord()
        rec.object_id    = oid
        rec.class_label  = "tree"
        rec.position     = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rec.axes         = np.array([0.5, 0.5, 1.0],  dtype=np.float32)
        rec.orientation  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        rec.embedding    = np.random.randn(embed_dim).astype(np.float32)
        rec.epsilon_g    = 0.3
        rec.epsilon_l    = 0.1
        rec.persistence_prob = 0.7
        rec.observation_count = 10
        rec.first_seen   = 1000.0
        rec.last_seen    = 2000.0
        return rec

    def test_numpy_round_trip(self):
        rec  = self._rec()
        d    = rec.to_numpy_dict()
        rec2 = ObjectRecord.from_numpy_dict(d)
        assert rec2.object_id    == rec.object_id
        assert rec2.class_label  == rec.class_label
        np.testing.assert_allclose(rec2.position, rec.position, atol=1e-6)
        np.testing.assert_allclose(rec2.embedding, rec.embedding, atol=1e-6)
        assert abs(rec2.epsilon_g - rec.epsilon_g) < 1e-5
        assert rec2.observation_count == rec.observation_count

    def test_default_fields_valid(self):
        rec = ObjectRecord()
        assert rec.object_id != ""
        assert rec.epsilon_g == 0.5
        assert rec.position.shape == (3,)
        assert rec.axes.shape == (3,)
        assert rec.orientation.shape == (4,)

    def test_no_embedding_round_trip(self):
        rec = self._rec()
        rec.embedding = None
        d    = rec.to_numpy_dict()
        rec2 = ObjectRecord.from_numpy_dict(d)
        assert rec2.embedding is None

    def test_inflate_uses_ros_translation_block(self):
        """ROS pose covariance is ordered (x, y, z, rotX, rotY, rotZ):
        inflation must use the top-left translation block, not the
        rotation block."""
        rec = ObjectRecord()
        base = rec.position_covariance.copy()
        cov6 = np.zeros((6, 6))
        cov6[:3, :3] = np.eye(3) * 0.5    # translation
        cov6[3:, 3:] = np.eye(3) * 99.0   # rotation (must be ignored)
        rec.inflate_position_covariance(cov6)
        assert rec.position_covariance[0] == pytest.approx(base[0] + 0.5)
        assert rec.position_covariance[5] == pytest.approx(base[5] + 0.5)

    def test_update_beta_absence_transience_only(self):
        rec = ObjectRecord()
        rec.beta_a, rec.beta_b = 3.0, 1.0
        rec.update_beta_absence(0.4)
        assert rec.beta_a == pytest.approx(3.0)
        assert rec.beta_b == pytest.approx(1.4)
        assert rec.epsilon_g == pytest.approx(1.4 / 4.4)

    def test_unique_ids_by_default(self):
        ids = {ObjectRecord().object_id for _ in range(100)}
        assert len(ids) == 100   # all unique


# ─────────────────────────────────────────────────────────────────────────────
# LTMIO
# ─────────────────────────────────────────────────────────────────────────────

class TestLTMIO:

    def _objects(self, n=5):
        objs = {}
        for i in range(n):
            rec = ObjectRecord()
            rec.object_id      = f"obj_{i:03d}"
            rec.class_label    = ["tree","bench","car","person","building"][i%5]
            rec.position       = np.random.randn(3).astype(np.float32)
            rec.embedding      = np.random.randn(64).astype(np.float32)
            rec.epsilon_g      = float(np.random.uniform(0.1, 0.9))
            rec.observation_count = int(np.random.randint(1, 20))
            rec.last_seen      = float(1000 + i * 100)
            objs[rec.object_id] = rec
        return objs

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            objs = self._objects(10)
            priors = {"tree": 0.001, "car": 0.5}

            path = io.save(objs, session_id=3, class_priors=priors)
            assert os.path.exists(path)

            loaded = io.load(path)
            assert loaded is not None
            assert loaded["session_id"] == 3
            assert len(loaded["objects"]) == 10
            assert loaded["class_priors"]["tree"] == pytest.approx(0.001)

    def test_position_preserved_after_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            objs = self._objects(3)
            path = io.save(objs, session_id=0, class_priors={})
            loaded_objs = io.load(path)["objects"]
            for oid, orig in objs.items():
                loaded = loaded_objs[oid]
                np.testing.assert_allclose(
                    loaded.position, orig.position, atol=1e-5)

    def test_embedding_preserved_after_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            objs = self._objects(2)
            path = io.save(objs, session_id=0, class_priors={})
            loaded_objs = io.load(path)["objects"]
            for oid, orig in objs.items():
                np.testing.assert_allclose(
                    loaded_objs[oid].embedding,
                    orig.embedding, atol=1e-5)

    def test_empty_map_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            path = io.save({}, session_id=0, class_priors={})
            loaded = io.load(path)
            assert loaded is not None
            assert len(loaded["objects"]) == 0

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io = LTMIO(save_dir=tmpdir)
            assert io.load("/nonexistent/path.npz") is None

    def test_load_latest_via_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            io.save(self._objects(2), session_id=1, class_priors={})
            io.save(self._objects(4), session_id=2, class_priors={})
            loaded = io.load_latest()
            assert loaded is not None
            assert loaded["session_id"] == 2
            assert len(loaded["objects"]) == 4

    def test_default_path_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            io   = LTMIO(save_dir=tmpdir)
            path = io.default_path(session_id=7)
            assert "0007" in path
            assert path.endswith(".npz")


# ─────────────────────────────────────────────────────────────────────────────
# KLSparsifier
# ─────────────────────────────────────────────────────────────────────────────

class TestKLSparsifier:

    def _objects(self):
        objs = {}
        specs = [
            ("perm",  0.05, 20),   # permanent, well-observed → kept
            ("eph",   0.98, 15),   # highly ephemeral         → pruned
            ("rare",  0.3,   2),   # too few observations     → pruned
            ("good",  0.4,  10),   # moderate ε_g, enough obs → kept
        ]
        for oid, eg, n_obs in specs:
            rec = ObjectRecord()
            rec.object_id         = oid
            rec.class_label       = "tree"
            rec.epsilon_g         = eg
            rec.observation_count = n_obs
            rec.position_covariance = np.ones(6, dtype=np.float32) * 0.5
            # Seed a confident Beta posterior consistent with the scalar eps_g
            # so the variance gate does not defer pruning (kappa = 200).
            rec.seed_beta_from_scalar(eg, concentration=200.0)
            objs[oid] = rec
        return objs

    def test_pruning_highly_ephemeral(self):
        sp   = KLSparsifier(prune_eg_threshold=0.95, min_observations=3)
        objs = self._objects()
        kept = sp.sparsify(objs)
        assert "eph"  not in kept, "Ephemeral object should be pruned"
        assert "perm" in kept

    def test_pruning_under_observed(self):
        sp   = KLSparsifier(prune_eg_threshold=0.95, min_observations=3)
        objs = self._objects()
        # Grace period: an under-observed object first seen this session
        # (sessions_absent == 0) is retained...
        kept = sp.sparsify(objs)
        assert "rare" in kept, "New under-observed object gets a grace period"
        assert "good" in kept
        # ...but is pruned once it has been absent for a session without
        # accumulating enough observations.
        objs = self._objects()
        objs["rare"].sessions_absent = 1
        kept = sp.sparsify(objs)
        assert "rare" not in kept, "Under-observed absent object is pruned"

    def test_pruning_absent_sessions(self):
        sp   = KLSparsifier(prune_eg_threshold=0.95, min_observations=3,
                            absent_sessions_threshold=3)
        objs = self._objects()
        objs["good"].sessions_absent = 3
        kept = sp.sparsify(objs)
        assert "good" not in kept, \
            "Object absent 3 consecutive sessions is pruned (rule iii)"
        assert "perm" in kept

    def test_shrink_does_not_compound(self):
        sp  = KLSparsifier(min_observations=1)
        rec = ObjectRecord()
        rec.epsilon_g = 0.2
        rec.observation_count = 4
        rec.position_covariance = np.ones(6, dtype=np.float32)
        kept = sp.sparsify({"a": rec})
        cov_first = kept["a"].position_covariance.copy()
        assert cov_first[0] == pytest.approx(0.5)   # 1/sqrt(4)
        # Re-sparsifying with no new observations must be a no-op
        kept = sp.sparsify(kept)
        assert kept["a"].position_covariance[0] == pytest.approx(0.5)
        # With 16 total observations the prior is 1/sqrt(16) of the
        # ORIGINAL, not 1/sqrt(4) of the already-shrunk value
        kept["a"].observation_count = 16
        kept = sp.sparsify(kept)
        assert kept["a"].position_covariance[0] == pytest.approx(0.25)

    def test_belief_state_round_trip(self):
        """The unified belief state (Beta posterior, embedding
        uncertainty, Welford stats, absence counters) must survive the
        npz save/load cycle — losing it would silently reset the
        cross-session uncertainty propagation."""
        rec = ObjectRecord()
        rec.object_id = "obj1"
        rec.class_label = "tree"
        rec.embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)
        rec.observation_count = 7
        rec.beta_a, rec.beta_b = 5.5, 1.5
        rec.sessions_absent = 2
        rec.last_sparsify_count = 4
        rng = np.random.default_rng(0)
        for _ in range(4):
            rec.update_embedding_uncertainty(
                rng.normal(size=8).astype(np.float32))
        u_before = rec.embedding_uncertainty
        assert u_before > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            io = LTMIO(save_dir=tmpdir)
            path = io.save({"obj1": rec}, session_id=1, class_priors={})
            loaded = io.load(path)["objects"]["obj1"]

        assert loaded.beta_a == pytest.approx(5.5)
        assert loaded.beta_b == pytest.approx(1.5)
        assert loaded.embedding_uncertainty == pytest.approx(u_before,
                                                             rel=1e-5)
        assert loaded.sessions_absent == 2
        assert loaded.last_sparsify_count == 4
        # Welford accumulation continues seamlessly after reload
        assert loaded._emb_count == 4
        loaded.update_embedding_uncertainty(
            rng.normal(size=8).astype(np.float32))
        assert loaded._emb_count == 5

    def test_old_map_without_belief_fields_loads(self):
        """Maps written before the uncertainty upgrade lack the belief
        arrays; loading must fall back to defaults, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            io = LTMIO(save_dir=tmpdir)
            rec = ObjectRecord()
            rec.object_id = "legacy"
            path = io.save({"legacy": rec}, session_id=0, class_priors={})
            # Strip the belief arrays to simulate an old-format file
            data = dict(np.load(path, allow_pickle=True))
            for key in ["beta_a", "beta_b", "embedding_uncertainties",
                        "emb_means", "emb_m2s", "emb_counts",
                        "sessions_absent", "last_sparsify_counts"]:
                data.pop(key, None)
            np.savez_compressed(path, **data)
            loaded = io.load(path)["objects"]["legacy"]
        assert loaded.beta_a == 1.0 and loaded.beta_b == 1.0
        assert loaded.embedding_uncertainty == 0.0
        assert loaded.sessions_absent == 0

    def test_covariance_shrinks_with_observations(self):
        sp   = KLSparsifier(min_observations=1)
        rec  = ObjectRecord()
        rec.epsilon_g         = 0.2
        rec.observation_count = 100
        rec.position_covariance = np.ones(6, dtype=np.float32)
        kept = sp.sparsify({"a": rec})
        assert "a" in kept
        # After 100 observations, covariance should be much smaller than 1
        assert kept["a"].position_covariance.max() < 0.5

    def test_empty_input(self):
        sp   = KLSparsifier()
        kept = sp.sparsify({})
        assert len(kept) == 0
