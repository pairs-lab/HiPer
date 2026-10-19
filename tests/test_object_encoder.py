"""
tests/test_object_encoder.py
=============================
Unit tests for:
  - hiper.object_encoder.bye_encoder.BYEEncoder
  - hiper.object_encoder.memory_bank.ObjectMemoryBank
  - hiper.object_encoder.uav_augment.UAVPointCloudAugmentor

Run with: python -m pytest tests/test_object_encoder.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hiper.object_encoder.memory_bank import ObjectMemoryBank
from hiper.object_encoder.uav_augment import UAVPointCloudAugmentor
from hiper.object_encoder.bye_encoder  import BYEEncoder


# ─────────────────────────────────────────────────────────────────────────────
# BYEEncoder  (stub mode — no GPU/PyTorch required)
# ─────────────────────────────────────────────────────────────────────────────

class TestBYEEncoder:

    def _enc(self, backbone="pointnet"):
        return BYEEncoder(embed_dim=64, backbone=backbone,
                          num_points=64, knn_k=5, device="cpu")

    # ── encode() ─────────────────────────────────────────────────────────────

    def test_encode_returns_correct_shape(self):
        enc = self._enc()
        pts = np.random.rand(200, 3).astype(np.float32)
        emb = enc.encode(pts)
        assert emb.shape == (64,), f"Expected (64,) got {emb.shape}"

    def test_encode_output_is_finite(self):
        enc = self._enc()
        pts = np.random.rand(100, 3).astype(np.float32)
        emb = enc.encode(pts)
        assert np.all(np.isfinite(emb))

    def test_encode_empty_cloud_does_not_crash(self):
        enc = self._enc()
        pts = np.zeros((0, 3), dtype=np.float32)
        emb = enc.encode(pts)
        assert emb.shape == (64,)

    def test_encode_single_point(self):
        enc = self._enc()
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        emb = enc.encode(pts)
        assert emb.shape == (64,)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def test_voxel_downsample_reduces_dense_cloud(self):
        # 5000 points packed tightly → fewer after 0.01 m voxel
        pts = (np.random.rand(5000, 3).astype(np.float32) * 0.005)
        out = BYEEncoder._voxel_downsample(pts, voxel_size=0.01)
        assert out.shape[0] < pts.shape[0]
        assert out.shape[1] == 3

    def test_voxel_downsample_empty_returns_empty(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        out = BYEEncoder._voxel_downsample(pts, voxel_size=0.01)
        assert out.shape == (0, 3)

    def test_fps_returns_correct_count(self):
        pts     = np.random.rand(300, 3).astype(np.float32)
        sampled = BYEEncoder._fps(pts, n=64)
        assert sampled.shape == (64, 3)

    def test_fps_unique_indices(self):
        """FPS should return distinct points (no duplicates at init)."""
        pts     = np.random.rand(200, 3).astype(np.float32)
        sampled = BYEEncoder._fps(pts, n=50)
        # All rows should be distinct
        unique  = np.unique(sampled, axis=0)
        assert unique.shape[0] == 50

    def test_preprocess_large_cloud_returns_n_points(self):
        enc = self._enc()
        pts = np.random.rand(2000, 3).astype(np.float32)
        out = enc._preprocess(pts)
        assert out.shape == (64, 3)

    def test_preprocess_small_cloud_pads_to_n_points(self):
        enc = self._enc()
        pts = np.random.rand(10, 3).astype(np.float32)
        out = enc._preprocess(pts)
        assert out.shape == (64, 3)

    def test_preprocess_exact_n_returns_n_points(self):
        enc = self._enc()
        pts = np.random.rand(64, 3).astype(np.float32)
        out = enc._preprocess(pts)
        assert out.shape == (64, 3)


# ─────────────────────────────────────────────────────────────────────────────
# ObjectMemoryBank
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectMemoryBank:

    def _bank(self, embed_dim=32, knn_k=3):
        return ObjectMemoryBank(embed_dim=embed_dim, knn_k=knn_k)

    def _emb(self, seed: int, dim: int = 32) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        v    = rng.standard_normal(dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    # ── Basic operations ──────────────────────────────────────────────────────

    def test_empty_query_returns_empty_string_and_zero(self):
        bank = self._bank()
        oid, score = bank.query(self._emb(0))
        assert oid == ""
        assert score == 0.0

    def test_add_increments_len(self):
        bank = self._bank()
        bank.add("a", self._emb(1), "tree")
        assert len(bank) == 1
        bank.add("b", self._emb(2), "bench")
        assert len(bank) == 2

    def test_add_same_id_twice_does_not_grow(self):
        bank = self._bank()
        bank.add("x", self._emb(1), "tree")
        bank.add("x", self._emb(2), "tree")   # update, not new entry
        assert len(bank) == 1

    def test_clear_resets_bank(self):
        bank = self._bank()
        bank.add("a", self._emb(1), "tree")
        bank.clear()
        assert len(bank) == 0
        oid, score = bank.query(self._emb(0))
        assert oid == "" and score == 0.0

    # ── Query ─────────────────────────────────────────────────────────────────

    def test_query_returns_best_match(self):
        """The stored embedding should be its own best match."""
        bank = self._bank()
        emb  = self._emb(42)
        bank.add("correct", emb.copy(), "car")
        bank.add("wrong",   self._emb(999), "person")
        oid, score = bank.query(emb)
        assert oid == "correct"
        assert score > 0.95

    def test_query_score_in_unit_range(self):
        bank = self._bank()
        for i in range(5):
            bank.add(f"obj_{i}", self._emb(i * 7), "tree")
        _, score = bank.query(self._emb(0))
        assert -1.0 <= score <= 1.0

    def test_query_topk_length(self):
        bank = self._bank()
        for i in range(10):
            bank.add(f"obj_{i}", self._emb(i * 17), "tree")
        topk = bank.query_topk(self._emb(0), k=3)
        assert len(topk) == 3

    def test_query_topk_descending_scores(self):
        bank = self._bank()
        for i in range(10):
            bank.add(f"obj_{i}", self._emb(i * 17), "tree")
        topk   = bank.query_topk(self._emb(0), k=5)
        scores = [s for _, s in topk]
        assert scores == sorted(scores, reverse=True)

    def test_query_topk_capped_at_bank_size(self):
        bank = self._bank()
        bank.add("only_one", self._emb(1), "bench")
        topk = bank.query_topk(self._emb(0), k=10)
        assert len(topk) == 1

    # ── Introspection ─────────────────────────────────────────────────────────

    def test_get_embedding_round_trip(self):
        bank = self._bank()
        emb  = self._emb(7)
        bank.add("obj_7", emb, "bench")
        retrieved = bank.get_embedding("obj_7")
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, emb, atol=1e-6)

    def test_get_embedding_missing_returns_none(self):
        bank = self._bank()
        assert bank.get_embedding("nonexistent") is None

    def test_object_ids_matches_added(self):
        bank = self._bank()
        bank.add("alpha", self._emb(1), "tree")
        bank.add("beta",  self._emb(2), "car")
        assert set(bank.object_ids()) == {"alpha", "beta"}

    def test_large_bank_query_correct(self):
        """Even with 100 objects, the exact embedding should be found."""
        bank = self._bank(embed_dim=64, knn_k=5)
        target_emb = self._emb(999, dim=64)
        bank.add("target", target_emb.copy(), "building")
        for i in range(99):
            bank.add(f"noise_{i}", self._emb(i, dim=64), "tree")
        oid, score = bank.query(target_emb)
        assert oid == "target"
        assert score > 0.99


# ─────────────────────────────────────────────────────────────────────────────
# UAVPointCloudAugmentor
# ─────────────────────────────────────────────────────────────────────────────

class TestUAVAugmentor:

    def _aug(self, **kw):
        defaults = dict(
            altitude_jitter_range=5.0,
            angle_jitter_deg=30.0,
            position_jitter_m=0.03,
            grazing_pitch_deg=15.0,
            topdown_mask_prob=0.0,   # off by default for most tests
        )
        defaults.update(kw)
        return UAVPointCloudAugmentor(**defaults)

    def _pts(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3)).astype(np.float32)
        return (pts - pts.mean(0)).astype(np.float32)

    # ── Output shape & dtype ──────────────────────────────────────────────────

    def test_output_ndim_is_two(self):
        aug = self._aug()
        out = aug.augment(self._pts(200))
        assert out.ndim == 2

    def test_output_has_three_columns(self):
        aug = self._aug()
        out = aug.augment(self._pts(200))
        assert out.shape[1] == 3

    def test_output_is_float32(self):
        aug = self._aug()
        out = aug.augment(self._pts(200))
        assert out.dtype == np.float32

    def test_output_is_zero_centred(self):
        aug = self._aug()
        out = aug.augment(self._pts(300))
        np.testing.assert_allclose(out.mean(0), np.zeros(3), atol=1e-5)

    def test_topdown_mask_reduces_points(self):
        """With prob=1.0, top-down masking must remove some points."""
        aug = self._aug(topdown_mask_prob=1.0,
                        altitude_jitter_range=0.0,
                        angle_jitter_deg=0.0,
                        position_jitter_m=0.0,
                        grazing_pitch_deg=0.0)
        pts = self._pts(200)
        out = aug.augment(pts)
        # Masking keeps top 70% by z → fewer than 200
        assert out.shape[0] < pts.shape[0]
        assert out.shape[0] >= 5   # at least some survive

    def test_topdown_mask_off_keeps_all(self):
        aug = self._aug(topdown_mask_prob=0.0,
                        altitude_jitter_range=0.0,
                        angle_jitter_deg=0.0,
                        position_jitter_m=0.0,
                        grazing_pitch_deg=0.0)
        pts = self._pts(100)
        out = aug.augment(pts)
        assert out.shape[0] == 100

    # ── Rotation ─────────────────────────────────────────────────────────────

    def test_rotation_preserves_pairwise_distances(self):
        """
        A pure rotation must not change the Euclidean distance between
        any pair of points. We check this for a random sample of pairs.
        """
        aug = UAVPointCloudAugmentor(
            altitude_jitter_range=0.0,
            angle_jitter_deg=45.0,
            position_jitter_m=0.0,
            grazing_pitch_deg=0.0,
            topdown_mask_prob=0.0,
        )
        pts = self._pts(50)
        out = aug.augment(pts)   # same number of points (no masking)
        assert out.shape[0] == 50

        # Pairwise distance matrix should be approximately equal
        d_before = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        d_after  = np.linalg.norm(out[:, None] - out[None, :], axis=-1)
        np.testing.assert_allclose(d_before, d_after, atol=1e-4)

    def test_rotate_axis_x_90_degrees(self):
        """
        Rotating the point (0, 1, 0) by 90° around X should give (0, 0, 1).
        _rotate_axis expects (N, 3) arrays.
        """
        pts = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        out = UAVPointCloudAugmentor._rotate_axis(pts, np.pi / 2, "x")
        np.testing.assert_allclose(out[0], [0.0, 0.0, 1.0], atol=1e-5)

    def test_rotate_axis_y_90_degrees(self):
        """Rotating (1, 0, 0) by 90° around Y should give (0, 0, -1)."""
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = UAVPointCloudAugmentor._rotate_axis(pts, np.pi / 2, "y")
        np.testing.assert_allclose(out[0], [0.0, 0.0, -1.0], atol=1e-5)

    def test_rotate_axis_z_90_degrees(self):
        """Rotating (1, 0, 0) by 90° around Z should give (0, 1, 0)."""
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = UAVPointCloudAugmentor._rotate_axis(pts, np.pi / 2, "z")
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-5)

    def test_rotate_axis_zero_angle_is_identity(self):
        pts = self._pts(20)
        out = UAVPointCloudAugmentor._rotate_axis(pts, 0.0, "x")
        np.testing.assert_allclose(out, pts, atol=1e-6)

    def test_altitude_jitter_changes_z_scale(self):
        """
        Altitude jitter uses z-axis SCALING (not translation) so the effect
        survives re-centering. Different runs should produce different z-stds.
        """
        aug = UAVPointCloudAugmentor(
            altitude_jitter_range=10.0,   # large range → strong effect
            angle_jitter_deg=0.0,
            position_jitter_m=0.0,
            grazing_pitch_deg=0.0,
            topdown_mask_prob=0.0,
        )
        pts  = self._pts(200)
        stds = []
        for seed in range(30):
            np.random.seed(seed)
            out = aug.augment(pts.copy())
            stds.append(float(out[:, 2].std()))

        # With large altitude_jitter_range the z-std should vary across runs
        assert max(stds) - min(stds) > 0.05, (
            f"z-std range={max(stds)-min(stds):.4f} "
            "should vary with altitude jitter"
        )

    def test_altitude_jitter_zero_leaves_z_unchanged(self):
        """altitude_jitter_range=0 must not alter z at all."""
        aug = UAVPointCloudAugmentor(
            altitude_jitter_range=0.0,
            angle_jitter_deg=0.0,
            position_jitter_m=0.0,
            grazing_pitch_deg=0.0,
            topdown_mask_prob=0.0,
        )
        pts = self._pts(100)
        out = aug.augment(pts.copy())
        np.testing.assert_allclose(out[:, 2], pts[:, 2], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-gated association (memory bank)
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyGatedAssociation:

    def _bank(self):
        from hiper.object_encoder.memory_bank import ObjectMemoryBank
        return ObjectMemoryBank(embed_dim=8, knn_k=3, sigma_u=0.25)

    def test_high_uncertainty_candidate_downweighted(self):
        """A high-uncertainty stored object should lose to a confident one
        even with identical embeddings."""
        bank = self._bank()
        e = np.zeros(8, dtype=np.float32)
        e[0] = 1.0
        bank.add("confident", e, "tree", uncertainty=0.0)
        bank.add("uncertain", e, "tree", uncertainty=1.0)
        match_id, _ = bank.query(e, query_uncertainty=0.0)
        assert match_id == "confident"

    def test_gating_preserves_match_when_equal_uncertainty(self):
        bank = self._bank()
        e1 = np.zeros(8, dtype=np.float32)
        e1[0] = 1.0
        e2 = np.zeros(8, dtype=np.float32)
        e2[1] = 1.0
        bank.add("a", e1, "tree", uncertainty=0.1)
        bank.add("b", e2, "car",  uncertainty=0.1)
        mid, _ = bank.query(e1, query_uncertainty=0.0)
        assert mid == "a"

    def test_empty_bank_returns_blank(self):
        bank = self._bank()
        mid, score = bank.query(np.ones(8, dtype=np.float32))
        assert mid == "" and score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-uncertainty tracking and pose inflation (ObjectRecord)
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordUncertainty:

    def _rec(self):
        from hiper.map_io.object_record import ObjectRecord
        return ObjectRecord()

    def test_embedding_uncertainty_grows_with_drift(self):
        rec = self._rec()
        rng = np.random.default_rng(0)
        base = rng.standard_normal(16).astype(np.float32)
        rec.update_embedding_uncertainty(base)
        u_first = rec.embedding_uncertainty
        for _ in range(8):
            rec.update_embedding_uncertainty(
                base + rng.standard_normal(16).astype(np.float32))
        assert rec.embedding_uncertainty > u_first

    def test_pose_inflation_increases_covariance(self):
        rec = self._rec()
        before = rec._cov_vec_to_mat(rec.position_covariance).trace()
        rec.inflate_position_covariance(np.eye(3) * 0.5)
        after = rec._cov_vec_to_mat(rec.position_covariance).trace()
        assert after > before

    def test_belief_state_round_trips(self):
        from hiper.map_io.object_record import ObjectRecord
        rec = self._rec()
        rec.seed_beta_from_scalar(0.3, concentration=20.0)
        rec.update_embedding_uncertainty(np.ones(16, dtype=np.float32))
        rec.update_embedding_uncertainty(np.ones(16, dtype=np.float32) * 2)
        d = rec.to_numpy_dict()
        rec2 = ObjectRecord.from_numpy_dict(d)
        assert abs(rec2.beta_a - rec.beta_a) < 1e-4
        assert abs(rec2.beta_b - rec.beta_b) < 1e-4
        assert abs(rec2.embedding_uncertainty - rec.embedding_uncertainty) < 1e-4
