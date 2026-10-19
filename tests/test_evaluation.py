"""
tests/test_evaluation.py
========================
Unit tests for the OpenLORIS-Scene evaluation stack:
trajectory metrics, dataset reader, and the multi-session
lifelong runner (using the real LTM/BPF/sparsifier machinery).
"""

import os

import numpy as np
import pytest

from hiper.evaluation.trajectory_metrics import (
    associate_timestamps,
    ate_rmse,
    correct_rate,
    integrate_relative_poses,
    umeyama_alignment,
)
from hiper.evaluation.openloris import (
    discover_scene_sessions,
    load_sequence,
    read_odometry,
    read_trajectory,
)
from hiper.evaluation.lifelong_runner import (
    LifelongEvaluator,
    Observation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestAssociation:
    def test_exact_match(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.0, 1.0, 2.0])
        ia, ib = associate_timestamps(a, b, max_dt=0.01)
        assert list(ia) == [0, 1, 2]
        assert list(ib) == [0, 1, 2]

    def test_offset_within_tolerance(self):
        a = np.array([0.0, 1.0, 2.0])
        b = a + 0.005
        ia, ib = associate_timestamps(a, b, max_dt=0.02)
        assert len(ia) == 3

    def test_no_match_outside_tolerance(self):
        ia, ib = associate_timestamps(
            np.array([0.0]), np.array([1.0]), max_dt=0.02)
        assert len(ia) == 0

    def test_each_timestamp_used_once(self):
        a = np.array([0.0, 0.001, 0.002])
        b = np.array([0.0])
        ia, ib = associate_timestamps(a, b, max_dt=0.05)
        assert len(ia) == 1


class TestUmeyama:
    def test_identity(self):
        pts = np.random.default_rng(0).normal(size=(50, 3))
        R, t, s = umeyama_alignment(pts, pts)
        assert np.allclose(R, np.eye(3), atol=1e-9)
        assert np.allclose(t, 0, atol=1e-9)
        assert s == 1.0

    def test_recovers_rigid_transform(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(100, 3))
        theta = 0.7
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]])
        t_true = np.array([1.0, -2.0, 0.5])
        dst = (R_true @ src.T).T + t_true
        R, t, s = umeyama_alignment(src, dst)
        assert np.allclose(R, R_true, atol=1e-8)
        assert np.allclose(t, t_true, atol=1e-8)

    def test_recovers_scale(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(100, 3))
        dst = 2.5 * src
        R, t, s = umeyama_alignment(src, dst, with_scale=True)
        assert abs(s - 2.5) < 1e-8


class TestATE:
    def test_zero_for_identical(self):
        pts = np.random.default_rng(3).normal(size=(30, 3))
        assert ate_rmse(pts, pts) < 1e-12

    def test_alignment_removes_offset(self):
        pts = np.random.default_rng(4).normal(size=(30, 3))
        shifted = pts + np.array([5.0, 0.0, 0.0])
        assert ate_rmse(shifted, pts, align=True) < 1e-9
        assert ate_rmse(shifted, pts, align=False) == pytest.approx(5.0)

    def test_empty_returns_inf(self):
        assert ate_rmse(np.zeros((0, 3)), np.zeros((0, 3))) == float("inf")


class TestCorrectRate:
    def test_perfect_tracking(self):
        t = np.linspace(0, 10, 101)
        xyz = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        assert correct_rate(t, xyz, t, xyz) == pytest.approx(1.0)

    def test_penalises_missing_estimates(self):
        gt_t = np.linspace(0, 10, 101)
        gt_xyz = np.stack([gt_t, np.zeros_like(gt_t),
                           np.zeros_like(gt_t)], axis=1)
        # estimator only covers the first half
        half = len(gt_t) // 2
        cr = correct_rate(gt_t[:half], gt_xyz[:half], gt_t, gt_xyz)
        assert 0.4 < cr < 0.6

    def test_penalises_large_errors(self):
        t = np.linspace(0, 10, 101)
        gt = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        est = gt.copy()
        est[:, 1] += np.linspace(0, 5.0, 101)   # growing drift
        cr = correct_rate(t, est, t, gt, tau_pos=0.3, align=False)
        assert cr < 0.5

    def test_no_estimates(self):
        t = np.linspace(0, 10, 11)
        gt = np.zeros((11, 3))
        assert correct_rate(np.zeros(0), np.zeros((0, 3)), t, gt) == 0.0


class TestIntegration:
    def test_chains_transforms(self):
        dT = np.eye(4)
        dT[0, 3] = 1.0   # 1 m forward each step
        traj = integrate_relative_poses([dT] * 5)
        assert traj.shape == (6, 4, 4)
        assert traj[-1][0, 3] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# OpenLORIS reader
# ─────────────────────────────────────────────────────────────────────────────

def _write_sequence(seq_dir, n_frames=10, t0=1000.0, with_twist=True):
    """Create a minimal synthetic OpenLORIS package directory."""
    os.makedirs(os.path.join(seq_dir, "color"), exist_ok=True)
    color_lines, depth_lines, odom_lines, gt_lines = [], [], [], []
    for i in range(n_frames):
        t = t0 + 0.1 * i
        color_lines.append(f"{t:.6f} color/{i:04d}.png")
        depth_lines.append(f"{t:.6f} aligned_depth/{i:04d}.png")
        x = 0.5 * i * 0.1   # 0.5 m/s along x
        if with_twist:
            odom_lines.append(
                f"{t:.6f} {x} 0 0 0 0 0 1 0.5 0 0 0 0 0")
        else:
            odom_lines.append(f"{t:.6f} {x} 0 0 0 0 0 1")
        gt_lines.append(f"{t:.6f} {x} 0 0 0 0 0 1")
    for fname, lines in [("color.txt", color_lines),
                         ("aligned_depth.txt", depth_lines),
                         ("odom.txt", odom_lines),
                         ("groundtruth.txt", gt_lines)]:
        with open(os.path.join(seq_dir, fname), "w") as f:
            f.write("# header comment\n")
            f.write("\n".join(lines) + "\n")


class TestOpenLorisReader:
    def test_load_sequence(self, tmp_path):
        seq_dir = str(tmp_path / "office1-1")
        _write_sequence(seq_dir, n_frames=10)
        seq = load_sequence(seq_dir)
        assert seq.name == "office1-1"
        assert len(seq.frames) == 10
        assert seq.frames[0]["depth_path"] is not None
        assert seq.frames[3]["velocity"] == pytest.approx(0.5)
        assert len(seq.gt_t) == 10
        assert seq.gt_xyz[-1][0] == pytest.approx(0.45)

    def test_velocity_finite_difference_fallback(self, tmp_path):
        seq_dir = str(tmp_path / "office1-2")
        _write_sequence(seq_dir, n_frames=10, with_twist=False)
        seq = load_sequence(seq_dir)
        # finite-difference of 0.05 m per 0.1 s = 0.5 m/s
        assert seq.frames[5]["velocity"] == pytest.approx(0.5, abs=1e-6)

    def test_stride(self, tmp_path):
        seq_dir = str(tmp_path / "office1-3")
        _write_sequence(seq_dir, n_frames=10)
        seq = load_sequence(seq_dir, stride=2)
        assert len(seq.frames) == 5

    def test_trajectory_parsers_skip_comments(self, tmp_path):
        seq_dir = str(tmp_path / "x1-1")
        _write_sequence(seq_dir, n_frames=4)
        t, xyz, quat = read_trajectory(
            os.path.join(seq_dir, "groundtruth.txt"))
        assert len(t) == 4 and xyz.shape == (4, 3) and quat.shape == (4, 4)
        ot, oxyz, ospeed = read_odometry(os.path.join(seq_dir, "odom.txt"))
        assert ospeed[0] == pytest.approx(0.5)

    def test_discover_scene_sessions(self, tmp_path):
        for name in ["office1-2", "office1-10", "office1-1",
                     "corridor1-1", "notaseq", "office1-3"]:
            os.makedirs(tmp_path / name)
        (tmp_path / "afile.txt").write_text("x")
        found = discover_scene_sessions(str(tmp_path), "office")
        names = [os.path.basename(p) for p in found]
        assert names == ["office1-1", "office1-2", "office1-3", "office1-10"]
        assert discover_scene_sessions(str(tmp_path), "corridor") != []


# ─────────────────────────────────────────────────────────────────────────────
# Lifelong runner
# ─────────────────────────────────────────────────────────────────────────────

def _make_evaluator(tmp_path, **kw):
    defaults = dict(
        map_dir=str(tmp_path / "maps"),
        class_priors={"building": 1e-7, "person": 2e-3},
        min_observations=1,
        tau_v=0.02,
    )
    defaults.update(kw)
    return LifelongEvaluator(**defaults)


def _static_obs(position, label="building", eps_l=0.1, stamp=0.0, dim=8):
    rng = np.random.default_rng(42)
    emb = rng.normal(size=dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return Observation(position=np.array(position, dtype=np.float32),
                       embedding=emb, class_label=label,
                       epsilon_l=eps_l, stamp=stamp)


class TestLifelongRunner:
    def test_session_odometry_metrics(self, tmp_path):
        seq_dir = str(tmp_path / "office1-1")
        _write_sequence(seq_dir, n_frames=20)
        seq = load_sequence(seq_dir)
        ev = _make_evaluator(tmp_path)
        res = ev.run_session(seq, pose_source="odom")
        # odom == gt in the synthetic sequence
        assert res.ate == pytest.approx(0.0, abs=1e-9)
        assert res.c_rate == pytest.approx(1.0)
        assert res.n_frames == 20

    def test_map_persists_across_sessions(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        # Session 1: ingest a static object
        ev.start_session()
        oid = ev.ingest(_static_obs([1.0, 2.0, 0.0], stamp=100.0))
        ev.ingest(_static_obs([1.0, 2.0, 0.0], stamp=101.0))
        ev.end_session(current_time=102.0)

        # Session 2 (fresh evaluator → must reload from disk)
        ev2 = _make_evaluator(tmp_path)
        ev2.start_session()
        assert oid in ev2._objects
        rec = ev2._objects[oid]
        assert rec.observation_count == 2
        assert rec.class_label == "building"

    def test_persistent_object_epsilon_g_decreases(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        eps_history = []
        for s in range(4):
            ev.start_session()
            t0 = 100.0 * (s + 1)
            oid = ev.ingest(_static_obs([0.0, 0.0, 0.0],
                                        eps_l=0.05, stamp=t0))
            ev.end_session(current_time=t0 + 1.0)
            eps_history.append(ev._objects[oid].epsilon_g)
        # consistently observed static object becomes confidently persistent
        assert eps_history[-1] < 0.3
        assert eps_history[-1] < eps_history[0]

    def test_association_merges_repeated_observations(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        ev.start_session()
        oid1 = ev.ingest(_static_obs([0.0, 0.0, 0.0]))
        oid2 = ev.ingest(_static_obs([0.1, 0.0, 0.0]))   # same object
        oid3 = ev.ingest(_static_obs([10.0, 0.0, 0.0]))  # different object
        assert oid1 == oid2
        assert oid3 != oid1

    def test_pose_covariance_inflation_applied(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        ev.start_session()
        obs = _static_obs([0.0, 0.0, 0.0])
        obs.pose_covariance = np.eye(3) * 0.5
        oid = ev.ingest(obs)
        cov_after = ev._objects[oid].position_covariance
        # diagonal terms inflated by 0.5 over the 1.0 default
        assert cov_after[0] == pytest.approx(1.5)
        assert cov_after[3] == pytest.approx(1.5)
        assert cov_after[5] == pytest.approx(1.5)

    def test_embedding_uncertainty_accumulates(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        ev.start_session()
        rng = np.random.default_rng(7)
        base = rng.normal(size=8).astype(np.float32)
        base /= np.linalg.norm(base)
        oid = None
        for _ in range(5):
            emb = base + 0.1 * rng.normal(size=8).astype(np.float32)
            emb /= np.linalg.norm(emb)
            obs = Observation(position=np.zeros(3, dtype=np.float32),
                              embedding=emb, class_label="chair",
                              epsilon_l=0.5, stamp=1.0)
            oid = ev.ingest(obs)
        assert ev._objects[oid].embedding_uncertainty > 0.0

    def test_vio_pose_source_requires_net(self, tmp_path):
        seq_dir = str(tmp_path / "office1-1")
        _write_sequence(seq_dir)
        seq = load_sequence(seq_dir)
        ev = _make_evaluator(tmp_path)
        with pytest.raises(ValueError):
            ev.run_session(seq, pose_source="vio", vio=None)
