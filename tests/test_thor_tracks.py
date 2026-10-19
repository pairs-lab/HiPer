"""Unit tests for hiper.evaluation.thor_tracks and clip_vocab (no GPU,
no simulator, no model weights)."""

import json

import numpy as np
import pytest

from hiper.evaluation.clip_vocab import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    aggregate_label,
    humanize,
    label_embeddings,
    label_probs,
    prompts_for,
    thor_types,
)
from hiper.evaluation.object_session import load_session, save_session
from hiper.evaluation.thor_tracks import (
    aggregate_centroids,
    build_session,
    downsample_depth,
    extent_of,
    frame_observations,
    scale_K,
    square_crop_box,
    unit_mean,
    voxel_downsample,
)
from hiper.evaluation.thor_world import intrinsics


# ── clip_vocab ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key,words", [
    ("AlarmClock", "alarm clock"), ("TVStand", "tv stand"), ("CD", "cd"),
    ("BasketBall", "basket ball"), ("Floor", "floor"),
    ("ToiletPaperHanger", "toilet paper hanger"),
    ("PaperTowelRoll", "paper towel roll")])
def test_humanize(key, words):
    assert humanize(key) == words


def test_model_constant_and_prompts():
    assert CLIP_MODEL == "ViT-B-32-quickgelu" and CLIP_PRETRAINED == "openai"
    assert prompts_for("AlarmClock") == ["a photo of a alarm clock.",
                                         "a alarm clock in a room."]


def test_thor_types_from_file(tmp_path):
    p = tmp_path / "types.json"
    p.write_text(json.dumps({"types": {"Mug": {}, "AlarmClock": {},
                                       "Floor": {}}}))
    assert thor_types(str(p)) == ["AlarmClock", "Floor", "Mug"]
    assert thor_types(str(p), exclude=["Floor"]) == ["AlarmClock", "Mug"]
    q = tmp_path / "list.json"
    q.write_text(json.dumps(["b", "a"]))
    assert thor_types(str(q)) == ["a", "b"]


def test_label_probs_and_aggregate():
    text = np.eye(3)
    img = np.array([[10.0, 0, 0], [0.9, 0.1, 0]])
    p = label_probs(img, text, logit_scale=100.0)
    assert np.allclose(p.sum(1), 1) and p[0, 0] > 0.99
    label, conf, top = aggregate_label(p, ["A", "B", "C"], k=2)
    assert label == "A" and conf == pytest.approx(p[:, 0].mean())
    assert [t[0] for t in top] == ["A", "B"]
    lab2, _, _ = label_embeddings(np.array([[0, 0, 1.0]]), text,
                                  ["A", "B", "C"])
    assert lab2 == "C"


# ── geometry helpers ─────────────────────────────────────────────────────────

def test_square_crop_box():
    x0, y0, x1, y1 = square_crop_box(100, 100, 140, 120, 640, 480)
    assert x1 - x0 == pytest.approx(40 * 1.3) == pytest.approx(y1 - y0)
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((120, 110))
    # near the border: shifted inside the image, size kept
    x0, y0, x1, y1 = square_crop_box(0, 0, 50, 50, 640, 480)
    assert x0 == 0 and y0 == 0 and x1 == pytest.approx(65)
    # larger than the image height: centred vertically (zero padding)
    x0, y0, x1, y1 = square_crop_box(0, 0, 600, 480, 640, 480)
    assert y1 - y0 == pytest.approx(780) and (y0 + y1) / 2 == \
        pytest.approx(240) and x0 <= 0


def test_voxel_downsample_means():
    P = np.array([[0.001, 0.001, 0.001, 1, 0, 0],
                  [0.003, 0.002, 0.004, 0, 1, 0],
                  [0.5, 0.5, 0.5, 0, 0, 1]])
    V = voxel_downsample(P, 0.01)
    assert len(V) == 2
    V = V[np.argsort(V[:, 0])]
    assert np.allclose(V[0], [0.002, 0.0015, 0.0025, 0.5, 0.5, 0])
    assert np.allclose(V[1], P[2])
    assert voxel_downsample(np.zeros((0, 6))).shape == (0, 6)


def test_downsample_depth_block_median_ignores_invalid():
    d = np.zeros((8, 8))
    d[:4, :4] = np.arange(16).reshape(4, 4) + 1.0     # 1..16 -> median 8.5
    d[:4, 4:] = 2.0
    d[0, 4] = 0.0                                      # one invalid pixel
    m = downsample_depth(d, 4)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(8.5) and m[0, 1] == pytest.approx(2.0)
    assert m[1, 0] == 0 and m[1, 1] == 0               # all invalid
    K = scale_K(intrinsics(640, 480), 4)
    assert np.allclose(K, [[60, 0, 80], [0, 60, 60], [0, 0, 1]])


def test_frame_observations_backprojection():
    W, H = 40, 30
    K = intrinsics(W, H, 90.0)                         # f = 15, c = (20, 15)
    depth = np.full((H, W), 2.0)
    inst = np.zeros((H, W), np.uint16)
    inst[10:20, 20:30] = 3                             # 100 px object
    inst[0:2, 0:2] = 5                                 # 4 px: below 0.5 %
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[inst == 3] = (255, 0, 0)
    obs = frame_observations(rgb, depth, inst, np.eye(4), K,
                             stride=2, min_area_frac=0.005, min_points=3)
    assert set(obs) == {3}
    o = obs[3]
    assert o["pixels"] == 100 and o["bbox"] == (20.0, 10.0, 30.0, 20.0)
    # stride-2 pixels: cols 20..28, rows 10..18 -> centres +0.5
    u = (np.arange(20, 30, 2) + 0.5 - 20) / 15 * 2
    v = (np.arange(10, 20, 2) + 0.5 - 15) / 15 * 2
    assert o["centroid"] == pytest.approx([np.median(u), np.median(v), 2.0])
    assert o["points"].shape == (25, 6)
    assert np.allclose(o["points"][:, 3:], [1, 0, 0])
    depth[inst == 3] = 0.0                             # no valid depth
    assert frame_observations(rgb, depth, inst, np.eye(4), K, stride=2,
                              min_area_frac=0.005) == {}


def test_aggregate_centroids_and_unit_mean():
    C = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    c, cov = aggregate_centroids(C, [1, 3])
    assert np.allclose(c, [0.75, 0, 0])
    assert cov[0, 0] == pytest.approx(0.5 / 2 + 1e-4)
    assert cov[1, 1] == pytest.approx(1e-4)
    c1, cov1 = aggregate_centroids(C[:1], [5])
    assert np.allclose(cov1, 1e-4 * np.eye(3))
    m, disp = unit_mean(np.array([[2.0, 0], [0, 3.0]]))
    assert np.allclose(m, [2 ** -0.5, 2 ** -0.5]) and disp == \
        pytest.approx(0.5)
    m, disp = unit_mean(np.array([[1.0, 0], [5.0, 0]]))
    assert disp == pytest.approx(0.0)
    P = np.random.default_rng(0).uniform(0, 1, (1000, 6))
    assert np.allclose(extent_of(P), [0.9, 0.9, 0.9], atol=0.05)


# ── SessionData assembly + save/load round trip ─────────────────────────────

class _StubClip:
    types = ["Apple", "Mug"]

    def label(self, E):
        return label_embeddings(E[:, :2], np.eye(2), self.types)


class _StubEnc:
    clip = _StubClip()

    def encode(self, raw_dir, n_frames, frames, crops, rgb=None):
        n = len(frames)
        ce = np.zeros((n, 512), np.float32)
        ce[:, 1] = 1.0                                  # -> 'Mug'
        de = np.zeros((n, 768), np.float32)
        de[:, 0] = 1.0
        return ce, de


def test_build_session_round_trip(tmp_path):
    K = intrinsics(640, 480)
    T = np.tile(np.eye(4), (2, 1, 1))
    pts = np.random.default_rng(1).uniform(0, 1, (50, 6)).astype(np.float32)
    prep = {
        "split": "val", "scene": "FloorPlan21", "k": 3,
        "raw_dir": "/nonexistent", "K": K, "T_wc_est": T, "T_wc_gt": T,
        "depth_stack": np.ones((2, 120, 160), np.float16), "n_frames": 2,
        "gt": {"time": 86400.0, "gap_days": 1.0,
               "pose_cov": (0.01 * np.eye(3)).tolist(),
               "objects": [{"gt_id": "Mug_1", "type": "Mug",
                            "group": "pickupable", "present": True,
                            "position": [0, 0, 1], "moved_since_prev": False,
                            "appeared_since_prev": False}]},
        "tracks": [{"gt_id": "Mug_1", "centroid": np.array([0, 0, 1.0]),
                    "centroid_cov": 1e-4 * np.eye(3), "n_obs": 2,
                    "pixel_area": 0.01, "points": pts,
                    "extent": extent_of(pts), "frames": [0, 1],
                    "crops": [(0, 0, 10, 10), (5, 5, 20, 20)]}]}
    sess = build_session(prep, _StubEnc())
    tr = sess.tracks[0]
    assert tr.clip_label == "Mug" and tr.gt_id == "Mug_1"
    assert tr.clip_disp == pytest.approx(0.0) and tr.eps_l == 0.0
    assert np.allclose(sess.K, scale_K(K))
    path = tmp_path / "val" / "FloorPlan21" / "session_03.npz"
    save_session(sess, str(path))
    back = load_session(str(path))
    assert back.scene == "FloorPlan21" and back.session_index == 3
    assert back.time == 86400.0 and back.meta["split"] == "val"
    assert back.meta["clip_top5"]["0"][0][0] == "Mug"
    b = back.tracks[0]
    assert b.gt_id == "Mug_1" and b.n_obs == 2 and b.clip_label == "Mug"
    assert np.allclose(b.centroid, tr.centroid)
    assert np.allclose(b.clip_mean, tr.clip_mean)
    assert np.allclose(b.dino_mean, tr.dino_mean)
    assert np.allclose(b.points, pts, atol=1e-3)       # float16 storage
    assert back.depth_stack.shape == (2, 120, 160)
    assert np.allclose(back.T_wc_gt, T) and back.gt_objects[0]["type"] == \
        "Mug"
    # visible(): a point 1 m in front of camera 0 (depth 1 m everywhere)
    assert back.visible(np.array([[0.0, 0.0, 1.0]]), min_frames=2)[0]
