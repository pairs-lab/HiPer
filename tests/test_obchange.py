"""
tests/test_obchange.py
Unit tests for the ObChange adapter (parsing, ground-truth transitions,
rendering and the virtual plane cameras used by SessionData.visible).
"""

import numpy as np
import pytest

from hiper.evaluation import obchange as ob
from hiper.evaluation.object_session import SessionData


def test_pcd_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(50, 3)).astype(np.float32)
    rgb = rng.random((50, 3)).astype(np.float32)
    path = str(tmp_path / "c.pcd")
    ob.write_pcd_binary(path, xyz, rgb)
    out = ob.read_pcd(path)
    assert np.allclose(out["xyz"], xyz, atol=1e-6)
    assert np.allclose(out["rgb"], np.round(rgb * 255) / 255, atol=1e-6)


def test_gt_anno_floor_flag(tmp_path):
    path = tmp_path / "x_GT.anno"
    path.write_text("025_mug 1 2 3 false\n024_bowl 4 5 true\n001_can 7 8\n")
    objs = ob.read_gt_anno(str(path))
    assert [o["name"] for o in objs] == ["025_mug", "024_bowl", "001_can"]
    assert objs[0]["on_floor"] is False and objs[1]["on_floor"] is True
    assert objs[2]["on_floor"] is None
    assert list(objs[0]["indices"]) == [1, 2, 3]


def _visit(k, objs):
    return ob.Visit(k, f"scene{k}", {}, {
        n: ob.VisitObject(n, 0, np.tile(np.asarray(c, float), (10, 1)),
                          np.zeros((10, 3), np.float32))
        for n, c in objs.items()})


def test_gt_transitions_labels():
    v0 = _visit(0, {"mug": [0, 0, 0], "bowl": [1, 0, 0], "can": [2, 0, 0]})
    v1 = _visit(1, {"mug": [0.1, 0, 0], "bowl": [1.5, 0, 0],
                    "box": [3, 0, 0]})
    tr = ob.gt_transitions([v0, v1])[1]
    assert tr["mug"]["present"] and not tr["mug"]["moved_since_prev"]
    assert tr["bowl"]["moved_since_prev"]                 # 0.5 m >= 0.2 m
    assert tr["can"]["disappeared_since_prev"] and not tr["can"]["present"]
    assert tr["box"]["appeared_since_prev"]
    assert tr["can"]["centroid"] == [2.0, 0.0, 0.0]       # last known


def test_render_zbuffer_keeps_nearest():
    K = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    T = ob.look_at(np.array([0.0, 0.0, 2.0]), np.zeros(3))  # looking down
    near = np.array([[0.0, 0.0, 0.5]])
    far = np.array([[0.0, 0.0, 0.0]])
    img, depth = ob.render_points(
        np.vstack([far, near]), np.array([[1, 0, 0], [0, 1, 0]], float),
        T, K, 64, 64, radius=0)
    v, u = np.argwhere(depth > 0)[0]
    assert depth[v, u] == pytest.approx(1.5)
    assert np.allclose(img[v, u], [0, 1, 0])


def test_plane_camera_visibility():
    """A position on the plane is visible; one off the plane footprint is
    not (NaN depth); an occluded one (under the surface) is not."""
    g = np.linspace(-0.5, 0.5, 60)
    xx, yy = np.meshgrid(g, g)
    plane = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.7)], 1)
    hull = np.array([[-0.5, -0.5, 0.7], [0.5, 0.5, 0.7], [0.5, -0.5, 0.7],
                     [-0.5, 0.5, 0.7]])          # deliberately unordered
    table = {"center": np.array([0.0, 0.0, 0.7]), "height": 0.7,
             "points": hull, "plane": np.array([0.0, 0.0, 1.0, -0.7])}
    T, K, depth = ob.plane_camera(table, plane)
    sess = SessionData(scene="t", session_index=0, time=0.0, tracks=[],
                       T_wc_est=np.array([T] * 3), K=K,
                       depth_stack=np.array([depth] * 3, np.float16),
                       pose_cov=np.zeros((3, 3)))
    vis = sess.visible(np.array([[0.1, 0.1, 0.75],     # on the table
                                 [3.0, 3.0, 0.75],     # far off
                                 [0.0, 0.0, 0.30]]))   # below the surface
    assert list(vis) == [True, False, False]
