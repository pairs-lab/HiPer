"""3RScan adapter (src/evaluation/rscan.py) on synthetic data in the
dataset's formats, plus the unknown-state handling it relies on."""

import json
import os

import numpy as np
import pytest

from hiper.evaluation import rscan
from hiper.evaluation.lifelong_objects import gt_labels, gt_tables
from hiper.evaluation.object_session import SessionData
from hiper.evaluation import persistence_metrics as pm
from hiper.ephemerality.priors import exposure_counts


def _rot_z(deg, t=(0, 0, 0)):
    a = np.deg2rad(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = t
    return T


def test_mat4_is_column_major():
    T = _rot_z(30, (1.0, 2.0, 3.0))
    assert np.allclose(rscan.mat4(T.T.reshape(-1).tolist()), T)


def test_load_index_and_splits(tmp_path):
    T = _rot_z(10, (0.1, 0.2, 0.0))
    M = _rot_z(0, (0.5, 0.0, 0.0))
    meta = [
        {"reference": "envA", "type": "train", "ambiguity": [],
         "scans": [{"reference": "scanA1", "transform": T.T.ravel().tolist(),
                    "rigid": [{"instance_reference": 3, "instance_rescan": 3,
                               "symmetry": 0,
                               "transform": M.T.ravel().tolist()}],
                    "removed": [5], "nonrigid": [7]}]},
        {"reference": "envT", "type": "test", "ambiguity": [],
         "scans": [{"reference": "scanT1", "rigid": [1, 2]}]}]
    p = tmp_path / "3RScan.json"
    p.write_text(json.dumps(meta))
    idx = rscan.load_index(str(p))
    a = idx["envA"]["rescans"][0]
    assert np.allclose(a["T"], T) and np.allclose(a["rigid"][3], M)
    assert a["removed"] == {5} and a["nonrigid"] == {7}
    t = idx["envT"]["rescans"][0]
    assert t["T"] is None and t["rigid"] == {}       # hidden test labels
    assert rscan.assign_split("x", "validation") == "rscan_test"
    assert rscan.assign_split("x", "test") is None
    ids = [f"env{i}" for i in range(400)]
    sp = [rscan.assign_split(i, "train") for i in ids]
    assert sp == [rscan.assign_split(i, "train") for i in ids]   # stable
    frac = sp.count("rscan_val") / len(sp)
    assert 0.12 < frac < 0.28 and set(sp) == {"rscan_train", "rscan_val"}


def test_env_ground_truth_rules():
    W = np.eye(4)
    W[2, 3] = -0.5                         # reference floor at z = 0.5
    T = _rot_z(90, (1.0, 0.0, 0.0))       # rescan -> reference scan
    ref_scan = {1: np.array([1.0, 0.0, 1.0]), 2: np.array([0.0, 2.0, 1.0]),
                3: np.array([3.0, 0.0, 1.0]), 4: np.array([0.0, 0.0, 1.5]),
                6: np.array([2.0, 2.0, 1.0])}
    ref_cent = {i: (W @ np.r_[c, 1.0])[:3] for i, c in ref_scan.items()}
    WT = W @ T
    # object 2 moved by 1 m along x in reference-scan coordinates, then
    # expressed in rescan coordinates (M includes the scan alignment)
    move = _rot_z(0, (1.0, 0.0, 0.0))
    M2 = np.linalg.inv(T) @ move
    small = np.linalg.inv(T) @ _rot_z(0, (0.05, 0.0, 0.0))
    s1 = {"W": W, "W_T": WT, "rigid": {2: M2, 6: small}, "removed": {3},
          "annotated": {1, 2, 6, 9}, "cent": {9: np.array([5.0, 5.0, 0.3])}}
    # second rescan: 3 is back (not removed, annotated), 2 back in place,
    # 1 not annotated (unknown), 9 gone from the annotation (unknown)
    s2 = {"W": W, "W_T": WT, "rigid": {}, "removed": set(),
          "annotated": {2, 3, 4, 6}, "cent": {}}
    gt = rscan.env_ground_truth(ref_cent, [{}, s1, s2])
    g0, g1, g2 = gt
    assert g0[1]["present"] is True and g0[9]["present"] is False
    assert g1[1]["present"] is True and not g1[1]["moved_since_prev"]
    assert g1[2]["moved_since_prev"]
    assert np.allclose(g1[2]["position"], ref_cent[2] + [1.0, 0.0, 0.0])
    assert g1[3]["present"] is False and g1[3]["disappeared_since_prev"]
    assert g1[4]["present"] is None                 # not annotated: unknown
    assert g1[6]["present"] and not g1[6]["moved_since_prev"]  # 5 cm < 20
    assert g1[9]["present"] and g1[9]["appeared_since_prev"]
    assert g2[1]["present"] is None and g2[9]["present"] is None
    assert g2[2]["moved_since_prev"]                # back to its place
    assert g2[3]["present"] and g2[3]["appeared_since_prev"]
    assert g2[4]["present"] is True and not g2[4]["appeared_since_prev"]


def _cube(c, s, oid, nyu):
    """Closed axis-aligned cube (8 vertices, 12 triangles)."""
    o = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                 float)
    V = c + (o - 0.5) * s
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5],
                  [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                  [1, 5, 7], [1, 7, 3]])
    return V, F, np.full(8, oid), np.full(8, nyu)


def test_render_labels_occlusion_and_upright():
    # camera at the origin looking along +x (world z up); columns are the
    # OpenCV camera axes in the world: right = -y, down = -z, forward = +x
    R = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    V1, F1, L1, _ = _cube(np.array([3.0, 0.0, 0.0]), 0.5, 5, 39)
    V2, F2, L2, _ = _cube(np.array([5.0, 0.0, 0.0]), 2.0, 6, 39)
    V = np.vstack([V1, V2])
    F = np.vstack([F1, F2 + 8])
    L = np.concatenate([L1, L2])
    P, PL = rscan.surface_samples(V, F, L, spacing=0.02, seed=0)
    K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    lab, depth = rscan.render_labels(P, PL, T_wc, K, 128, 96)
    assert lab[48, 64] == 5                     # near cube in front
    assert np.isclose(depth[48, 64], 2.75, atol=0.05)
    assert (lab == 6).sum() > 0 and lab[48, 64] != 6
    assert rscan.upright_k(T_wc) == 0           # world up = image up
    rolled = T_wc.copy()                        # camera rolled by 90 deg
    rolled[:3, :3] = R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    k = rscan.upright_k(rolled)
    assert k in (1, 3)
    up = rolled[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    assert (k == 1) == (up[0] > 0)


def test_read_label_ply_and_structure(tmp_path):
    V1, F1, L1, N1 = _cube(np.array([0.0, 0.0, 0.25]), 0.5, 5, 39)
    V2, F2, L2, N2 = _cube(np.array([0.0, 0.0, -0.05]), 0.1, 2, 2)  # floor
    V = np.vstack([V1, V2])
    F = np.vstack([F1, F2 + 8])
    L = np.concatenate([L1, L2])
    N = np.concatenate([N1, N2])
    lines = ["ply", "format ascii 1.0", f"element vertex {len(V)}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green",
             "property uchar blue", "property ushort objectId",
             "property ushort globalId", "property uchar NYU40",
             "property uchar Eigen13", "property uchar RIO27",
             f"element face {len(F)}",
             "property list uchar uint vertex_indices", "end_header"]
    lines += [f"{x} {y} {z} 1 2 3 {l} 0 {n} 0 0" for (x, y, z), l, n
              in zip(V, L, N)]
    lines += [f"3 {a} {b} {c}" for a, b, c in F]
    p = tmp_path / "labels.instances.annotated.v2.ply"
    p.write_text("\n".join(lines) + "\n")
    ply = rscan.read_label_ply(str(p))
    assert ply["xyz"].shape == (16, 3) and ply["faces"].shape == (24, 3)
    assert set(ply["objectId"].tolist()) == {2, 5}
    assert rscan.structure_mask(ply) == {2: False, 5: True}
    objs = rscan.object_samples(ply)
    assert set(objs) == {5}
    W = rscan.floor_shift(ply)
    assert np.isclose(W[2, 3], 0.1, atol=0.02)   # floor cube bottom ~ -0.1


def test_read_info(tmp_path):
    p = tmp_path / "_info.txt"
    p.write_text("m_colorWidth = 960\nm_colorHeight = 540\n"
                 "m_depthWidth = 224\nm_depthHeight = 172\n"
                 "m_depthShift = 1000\n"
                 "m_calibrationColorIntrinsic = 756.8 0 492.9 0 0 756.0 "
                 "270.4 0 0 0 1 0 0 0 0 1\n"
                 "m_calibrationDepthIntrinsic = 176.6 0 114.6 0 0 240.8 "
                 "85.8 0 0 0 1 0 0 0 0 1\n")
    info = rscan.read_info(str(p))
    assert info["color_size"] == (960, 540)
    assert info["depth_size"] == (224, 172)
    assert np.isclose(info["color_K"][1, 2], 270.4)
    assert np.isclose(info["depth_K"][0, 0], 176.6)


def _sess(k, objs):
    return SessionData(scene="s", session_index=k, time=k * 86400.0,
                       tracks=[], T_wc_est=np.zeros((0, 4, 4)), K=np.eye(3),
                       depth_stack=np.zeros((0, 2, 2), np.float16),
                       pose_cov=np.eye(3), gt_objects=objs)


def test_unknown_state_gives_no_label():
    def g(present, moved=False):
        return {"gt_id": "x", "type": "box", "present": present,
                "moved_since_prev": moved}
    gt = gt_tables([_sess(0, [g(True)]), _sess(1, [g(None)]),
                    _sess(2, [g(True)]), _sess(3, [g(True)])])
    assert gt[1]["x"]["present"] is None
    assert gt_labels(gt, "x", 0, 0) == (1, None)     # next state unknown
    assert gt_labels(gt, "x", 0, 2) == (None, None)  # span crosses unknown
    assert gt_labels(gt, "x", 2, 2) == (1, 1)
    assert gt_labels(gt, "x", 3, 3) == (1, None)     # last session
    # priors: an unknown next state is neither a survival nor a vanish
    S, gaps = exposure_counts([[(0.0, [g(True)]), (86400.0, [g(None)]),
                                (172800.0, [g(True)])]])
    assert S == {} and gaps == {}


def test_maintenance_ignores_unknown_sessions():
    rows = [{"model": "m", "scene": "s", "oid": 1, "k": k,
             "gt_present_now": p, "pruned": pr, "committed": False,
             "label_next": None}
            for k, p, pr in ((0, 1, False), (1, None, True), (2, 0, False))]
    c = pm.maintenance_metrics(rows)
    assert c["n_pruned"] == 0            # first prune at an unknown session
    assert c["n_departed"] == 0          # unknown breaks the absent run
