"""Unit tests for hiper.evaluation.thor_world (pure parts, no simulator)."""

import math

import numpy as np
import pytest

from hiper.evaluation.thor_world import (
    ChangeParams,
    ChangeProcess,
    GAP_CHOICES_DAYS,
    PoseErrorParams,
    apply_pose_error,
    backproject,
    camera_T_wc,
    choose_duplicate_types,
    detect_attachments,
    diff_flags,
    draw_gaps,
    index_image,
    intrinsics,
    make_rng,
    object_group,
    plan_exploration,
    pose_cov_from_model,
    rotation_angle_deg,
    sample_pose_error,
    topdown_matrix,
    unity_to_world,
)


# ── seeding / process ────────────────────────────────────────────────────────

def test_make_rng_is_reproducible_and_tag_dependent():
    a = make_rng("val", "FloorPlan21", 0, "change", 1).random(5)
    b = make_rng("val", "FloorPlan21", 0, "change", 1).random(5)
    c = make_rng("val", "FloorPlan21", 0, "change", 2).random(5)
    d = make_rng("test", "FloorPlan21", 0, "change", 1).random(5)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c) and not np.allclose(a, d)


def test_object_group_from_flags():
    assert object_group({"pickupable": True, "moveable": False}) == "pickupable"
    assert object_group({"pickupable": True, "moveable": True}) == "pickupable"
    assert object_group({"pickupable": False, "moveable": True}) == "moveable"
    assert object_group({"pickupable": False, "moveable": False}) == "fixed"


def test_draw_gaps():
    g = draw_gaps(8, make_rng("s", "x", 0, "gaps"))
    assert len(g) == 7 and all(v in GAP_CHOICES_DAYS for v in g)
    assert draw_gaps(1, make_rng("s", "x", 0)) == []


def test_choose_duplicate_types():
    t = choose_duplicate_types(["Apple", "Mug", "Cup", "Mug", "Egg"],
                               make_rng("s", "x", 0, "dup"))
    assert len(t) == 3 and len(set(t)) == 3
    assert set(t) <= {"Apple", "Mug", "Cup", "Egg"}
    assert choose_duplicate_types(["Apple"], make_rng("s", "x", 0)) == \
        ["Apple"]


def _groups(n_pick=4000, n_move=4000, n_fixed=50):
    g = {f"p{i:05d}": "pickupable" for i in range(n_pick)}
    g.update({f"m{i:05d}": "moveable" for i in range(n_move)})
    g.update({f"f{i:05d}": "fixed" for i in range(n_fixed)})
    return g


def test_hazards_lognormal_by_group_only():
    groups = _groups()
    proc = ChangeProcess(groups, make_rng("s", "x", 0, "hazard"),
                         attached=["m00000"])
    p = ChangeParams()
    lp = np.log([proc.hazard[n] / p.hazard_pickup for n in groups
                 if groups[n] == "pickupable"])
    lm = np.log([proc.hazard[n] / p.hazard_move for n in groups
                 if groups[n] == "moveable" and n != "m00000"])
    for v in (lp, lm):
        assert abs(v.mean()) < 0.06 and abs(v.std() - 1.0) < 0.05
    assert all(proc.hazard[n] == 0.0 for n in groups if n[0] == "f")
    assert proc.hazard["m00000"] == 0.0          # attached part


def test_step_statistics_match_ctmc():
    groups = _groups(n_fixed=0)
    proc = ChangeProcess(groups, make_rng("s", "x", 1, "hazard"))
    gap = 2.0
    present = {n: True for n in groups}
    ch = proc.step(present, gap, make_rng("s", "x", 1, "change", 1))
    for grp in ("pickupable", "moveable"):
        names = [n for n in groups if groups[n] == grp]
        # P(any event in the gap) = 1 - exp(-h gap), averaged over objects
        expect = np.mean([1 - math.exp(-proc.hazard[n] * gap) for n in names])
        got = np.mean([n in ch for n in names])
        assert abs(got - expect) < 0.025, (grp, got, expect)
    nets = [ch[n]["net"] for n in ch if groups[n] == "moveable"]
    assert set(nets) == {"relocate"}                 # moveables never vanish
    nets = [ch[n]["net"] for n in ch if groups[n] == "pickupable"]
    frac_dis = np.mean([v == "disappear" for v in nets])
    # first event: disappear w.p. 0.4; a later reappearance (0.2/day) or a
    # second event can turn it into 'relocate'/'none' within 2 days
    assert 0.30 < frac_dis < 0.42
    assert set(nets) <= {"relocate", "disappear", "none"}


def test_reappearance_rate_and_determinism():
    groups = {f"p{i:05d}": "pickupable" for i in range(6000)}
    proc = ChangeProcess(groups, make_rng("s", "x", 2, "hazard"),
                         params=ChangeParams(hazard_pickup=1e-9))
    absent = {n: False for n in groups}
    gap = 4.0
    ch = proc.step(absent, gap, make_rng("s", "x", 2, "change", 1))
    frac = np.mean([n in ch and ch[n]["net"] == "reappear" for n in groups])
    assert abs(frac - (1 - math.exp(-0.2 * gap))) < 0.02
    ch2 = proc.step(absent, gap, make_rng("s", "x", 2, "change", 1))
    assert ch == ch2
    ch3 = proc.step(absent, gap, make_rng("s", "x", 3, "change", 1))
    assert ch != ch3


def test_events_inside_gap_and_ordered():
    groups = {f"p{i:03d}": "pickupable" for i in range(300)}
    proc = ChangeProcess(groups, make_rng("s", "y", 0, "hazard"),
                         params=ChangeParams(hazard_pickup=2.0))
    ch = proc.step({n: True for n in groups}, 1.0,
                   make_rng("s", "y", 0, "c"))
    for v in ch.values():
        ts = [t for t, _ in v["events"]]
        assert ts == sorted(ts) and all(0 < t <= 1.0 for t in ts)


# ── geometry ─────────────────────────────────────────────────────────────────

def test_intrinsics_vertical_fov():
    K = intrinsics(640, 480, 90.0)
    assert np.allclose(K, [[240, 0, 320], [0, 240, 240], [0, 0, 1]])


def test_unity_to_world_swaps_y_z():
    assert np.allclose(unity_to_world([1, 2, 3]), [1, 3, 2])
    assert np.allclose(unity_to_world({"x": 1, "y": 2, "z": 3}), [1, 3, 2])
    assert np.allclose(unity_to_world(np.array([[1, 2, 3], [4, 5, 6]])),
                       [[1, 3, 2], [4, 6, 5]])


def test_camera_T_wc_axes():
    T = camera_T_wc({"x": 1.0, "y": 1.5, "z": 2.0}, 0.0, 0.0)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 1.5])
    assert np.allclose(R[:, 2], [0, 1, 0])    # optical axis = Unity +z = +Y
    assert np.allclose(R[:, 0], [1, 0, 0])    # image right = +X
    assert np.allclose(R[:, 1], [0, 0, -1])   # image down = -Z
    T = camera_T_wc({"x": 0, "y": 0, "z": 0}, 90.0, 0.0)
    assert np.allclose(T[:3, 2], [1, 0, 0])   # Unity yaw 90 faces +x
    assert np.allclose(T[:3, 0], [0, -1, 0])  # right hand side = -Y
    T = camera_T_wc({"x": 0, "y": 0, "z": 0}, 0.0, 30.0)
    assert np.allclose(T[:3, 2], [0, math.cos(math.radians(30)),
                                  -math.sin(math.radians(30))])


def test_rotation_angle_deg():
    assert rotation_angle_deg([0, 0, 0], [0, 360, 0]) == pytest.approx(0,
                                                                       abs=1e-6)
    assert rotation_angle_deg([0, 10, 0], [0, 40, 0]) == pytest.approx(30)
    assert rotation_angle_deg([0, 0, 0], [90, 0, 0]) == pytest.approx(90)


def test_backproject_plane_and_pixel_centres():
    K = intrinsics(8, 6, 90.0)                 # f = 3, c = (4, 3)
    depth = np.full((6, 8), 2.0)
    depth[0, 0] = 0.0                          # invalid
    P, pix = backproject(depth, K, np.eye(4))
    assert len(P) == 47 and np.allclose(P[:, 2], 2.0)
    i = int(np.nonzero((pix[:, 0] == 3) & (pix[:, 1] == 4))[0][0])
    assert np.allclose(P[i], [0.5 / 3 * 2, 0.5 / 3 * 2, 2.0])
    mask = np.zeros((6, 8), bool)
    mask[2:4, 2:6] = True
    P2, pix2 = backproject(depth, K, np.eye(4), mask=mask, stride=2)
    assert len(P2) == 2 and set(map(tuple, pix2)) == {(2, 2), (2, 4)}
    T = camera_T_wc({"x": 0, "y": 1.0, "z": 0}, 0.0, 0.0)
    P3, _ = backproject(np.full((6, 8), 2.0), K, T)
    assert np.allclose(P3[:, 1], 2.0)          # 2 m ahead along +Y


def test_diff_flags():
    a = {"present": True, "position": [0, 0, 0], "rotation_unity": [0, 0, 0]}
    b = {"present": True, "position": [0.04, 0, 0],
         "rotation_unity": [0, 5, 0]}
    assert not diff_flags(a, b)["moved_since_prev"]
    b2 = dict(b, position=[0.06, 0, 0])
    assert diff_flags(a, b2)["moved_since_prev"]
    b3 = dict(b, rotation_unity=[0, 12, 0])
    assert diff_flags(a, b3)["moved_since_prev"]
    gone = dict(a, present=False)
    assert diff_flags(a, gone)["disappeared_since_prev"]
    assert diff_flags(gone, a)["appeared_since_prev"]
    assert not any(v for k, v in diff_flags(None, a).items()
                   if k.endswith("prev"))


def test_index_image():
    seg = np.zeros((2, 3, 3), np.uint8)
    seg[0, 0] = (10, 20, 30)
    seg[0, 1] = (10, 20, 31)
    seg[1, 2] = (1, 2, 3)
    c2id = {(10, 20, 30): "A|1", (10, 20, 31): "Floor|0", (1, 2, 3): "B|2",
            (9, 9, 9): "Wall"}
    n2i = {}
    idx = index_image(seg, c2id, {"A|1": "A_x", "B|2": "B_y"}, n2i)
    assert idx.dtype == np.uint16
    assert idx[0, 0] == n2i["A_x"] and idx[1, 2] == n2i["B_y"]
    assert idx[0, 1] == 0 and idx[1, 1] == 0
    assert sorted(n2i.values()) == [1, 2]


def test_topdown_matrix():
    A = topdown_matrix({"x": 1.0, "y": 3.0, "z": 2.0},
                       {"x": 90.0, "y": 0.0, "z": 0.0}, 2.0, 400, 400)
    u, v, _ = A @ [1.0, 2.0, 1.0]
    assert (u, v) == pytest.approx((200, 200))
    u, v, _ = A @ [2.0, 2.0, 1.0]                 # +x -> right
    assert (u, v) == pytest.approx((300, 200))
    u, v, _ = A @ [1.0, 3.0, 1.0]                 # +y (Unity +z) -> up
    assert (u, v) == pytest.approx((200, 100))


def test_detect_attachments():
    def box(t, c, s, pick=False, move=False):
        return {"objectType": t, "pickupable": pick, "moveable": move,
                "axisAlignedBoundingBox": {
                    "center": dict(zip("xyz", c)),
                    "size": dict(zip("xyz", s))}}
    objs = {"dresser": box("Dresser", (0, 0.5, 0), (1, 1, 0.5), move=True),
            "drawer": box("Drawer", (0, 0.3, 0.05), (0.8, 0.2, 0.45)),
            "shelf_far": box("Shelf", (3, 0.5, 0), (0.5, 0.1, 0.3)),
            "floor": box("Floor", (0, 0, 0), (10, 0.01, 10)),
            "cup": box("Cup", (0, 1.05, 0), (0.1, 0.1, 0.1), pick=True)}
    assert detect_attachments(objs) == {"drawer": "dresser"}


# ── exploration / pose error ─────────────────────────────────────────────────

def test_plan_exploration_room():
    xs, zs = np.meshgrid(np.arange(0, 3.01, 0.25), np.arange(0, 2.01, 0.25))
    xz = np.stack([xs.ravel(), zs.ravel()], 1)
    island = np.array([[10.0, 10.0]])                # unreachable component
    plan = plan_exploration(np.vstack([xz, island]),
                            make_rng("s", "x", 0, "explore", 0), 12)
    wps = plan["waypoints"]
    assert len(wps) == 12 and len(set(wps)) == 12
    assert len(xz) not in wps or plan["start"] == len(xz)
    s = np.asarray(plan["path_s"])
    assert s[0] == 0 and np.all(np.diff(s) > 0)
    assert all(len(y) == 8 and np.allclose(np.diff(sorted(y)), 45)
               for y in plan["yaws"])
    again = plan_exploration(np.vstack([xz, island]),
                             make_rng("s", "x", 0, "explore", 0), 12)
    assert again == plan


def test_pose_error_rigid_part_and_drift_scale():
    p = PoseErrorParams()
    pivot = np.array([1.0, 2.0, 0.9])
    err = sample_pose_error(make_rng("s", "x", 0, "pose", 1), pivot,
                            [0.0, 1.0, 3.0], p)
    assert np.allclose(err.drift[0], 0) and np.allclose(err.drift[:, 2], 0)
    T = camera_T_wc({"x": 1.0, "y": 0.9, "z": 2.0}, 30.0, 30.0)
    Te = apply_pose_error(T, err, 0)
    # the pivot is a fixed point of the rotation part
    assert np.allclose(Te[:3, 3], pivot + err.offset)
    R = Te[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    # drift std after path length s is drift_frac * s per axis
    ends = np.array([sample_pose_error(make_rng("s", "x", i, "pose"), pivot,
                                       [0.0, 4.0, 10.0, 20.0], p).drift[-1, 0]
                     for i in range(3000)])
    assert abs(ends.std() - 0.01 * 20.0) < 0.015


def test_pose_cov_from_model():
    p = PoseErrorParams(sigma_yaw_deg=0.0, drift_frac=0.0)
    C = pose_cov_from_model(p, [0, 0, 0], np.zeros((5, 3)), [0, 1])
    assert np.allclose(C, np.diag([0.05 ** 2, 0.05 ** 2, 0.01 ** 2]))
    p = PoseErrorParams(sigma_xy=0, sigma_z=0, sigma_yaw_deg=1.0,
                        drift_frac=0.0)
    C = pose_cov_from_model(p, [0, 0, 0], np.array([[2.0, 0, 0]]), [0])
    assert np.allclose(C, np.diag([0, (2 * math.radians(1)) ** 2, 0]))
    p = PoseErrorParams(sigma_xy=0, sigma_z=0, sigma_yaw_deg=0,
                        drift_frac=0.01)
    C = pose_cov_from_model(p, [0, 0, 0], np.zeros((0, 3)), [0.0, 10.0])
    assert np.allclose(C, np.diag([0.005, 0.005, 0.0]))


# ── top-down render logic with a fake simulator (no Unity) ───────────────────

class _FakeEvent:
    def __init__(self, md, frames=None, segs=None, c2id=None, frame=None):
        self.metadata = md
        self.third_party_camera_frames = frames or []
        self.third_party_instance_segmentation_frames = segs or []
        self.color_to_object_id = c2id or {}
        self.frame = frame


class _FakeSim:
    """Ceiling blocks the top-down view when the camera is above 2.5 m."""
    width, height = 64, 48

    def __init__(self, objs):
        self.objs = objs
        self.res = (64, 48)
        self.cam = None
        self.calls = []
        self.event = self._ev()

    def objects(self):
        return self.objs

    def _ev(self, md_extra=None, third=False):
        md = {"lastActionSuccess": True, "errorMessage": "",
              "actionReturn": {"position": {"x": 0.0, "y": 3.2, "z": 0.0},
                               "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                               "orthographic": True,
                               "orthographicSize": 2.0}}
        W, H = self.res
        frame = np.zeros((H, W, 3), np.uint8)
        if not third:
            return _FakeEvent(md, frame=frame)
        seg = np.zeros((H, W, 3), np.uint8)
        c2id = {}
        if self.cam["position"]["y"] <= 2.5:
            A = topdown_matrix(self.cam["position"], self.cam["rotation"],
                               self.cam["orthographicSize"], W, H)
            for j, (n, o) in enumerate(self.objs.items()):
                x, z = o["axisAlignedBoundingBox"]["center"]["x"], \
                    o["axisAlignedBoundingBox"]["center"]["z"]
                u, v, _ = A @ [x, z, 1.0]
                col = (10 + j, 20, 30)
                seg[int(v) - 15:int(v) + 15, int(u) - 15:int(u) + 15] = col
                c2id[col] = o["objectId"]
        return _FakeEvent(md, frames=[np.dstack([frame, frame[..., :1]])],
                          segs=[seg], c2id=c2id, frame=frame)

    def step(self, action, **kw):
        self.calls.append((action, kw.get("position", {}).get("y")))
        if action == "ChangeResolution":
            self.res = (kw["x"], kw["y"])
            self.event = self._ev()
        elif action in ("AddThirdPartyCamera", "UpdateThirdPartyCamera"):
            self.cam = {**(self.cam or {}), **kw}
            self.event = self._ev(third=True)
        else:
            self.event = self._ev()
        return self.event


def test_topdown_lowers_camera_below_ceiling(tmp_path):
    from hiper.evaluation.thor_world import SequenceGenerator

    def obj(n, x, z):
        return {"name": n, "objectId": f"{n}|id", "objectType": n,
                "axisAlignedBoundingBox": {"center": {"x": x, "y": 0.5,
                                                      "z": z},
                                           "size": {"x": .2, "y": .2,
                                                    "z": .2}}}
    objs = {"A": obj("A", -1.0, 0.5), "B": obj("B", 0.5, -0.5),
            "C": obj("C", 1.2, 1.0)}
    sim = _FakeSim(objs)
    gen = SequenceGenerator.__new__(SequenceGenerator)
    gen.sim, gen.td_props, gen.td_added, gen.floor_y = sim, None, False, 0.0
    snap = {n: {"present": True, "type": n,
                "position": [o["axisAlignedBoundingBox"]["center"]["x"],
                             o["axisAlignedBoundingBox"]["center"]["z"],
                             0.5]} for n, o in objs.items()}
    out = gen.topdown(0, str(tmp_path), snap)
    assert out["lowered"] and out["camera_position_unity"]["y"] == 2.3
    assert out["check"]["n_objects"] == 3
    assert out["check"]["median_err_px"] < 1.0
    assert (tmp_path / "topdown.png").exists()
    # the camera ends parked below the floor; resolution restored
    assert sim.calls[-1][0] == "UpdateThirdPartyCamera" and \
        sim.calls[-1][1] == -50.0
    assert sim.res == (64, 48)


class _NudgeSim:
    """TeleportObject on 'Dresser' moves its child 'Drawer' rigidly."""

    def __init__(self):
        def o(n, x, y, z, s, pick=False, move=False):
            return {"name": n, "objectId": f"{n}|0", "objectType": n,
                    "pickupable": pick, "moveable": move,
                    "position": {"x": x, "y": y, "z": z},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "axisAlignedBoundingBox": {
                        "center": {"x": x, "y": y + s / 2, "z": z},
                        "size": {"x": s, "y": s, "z": s}}}
        self.objs = {"Dresser": o("Dresser", 0, 0.0, 0, 1.0, move=True),
                     "Drawer": o("Drawer", 0, 0.3, 3.0, 0.2),
                     "Lamp": o("Lamp", 2, 1.0, 2, 0.3, move=True),
                     "Shelf": o("Shelf", 5, 0.5, 5, 0.3)}
        self.children = {"Dresser": ["Drawer"]}

    def objects(self):
        import copy
        return copy.deepcopy(self.objs)

    def step(self, action, **kw):
        if action == "TeleportObject":
            n = kw["objectId"].split("|")[0]
            old = self.objs[n]["position"]
            d = {q: kw["position"][q] - old[q] for q in "xyz"}
            for m in [n] + self.children.get(n, []):
                for q in "xyz":
                    self.objs[m]["position"][q] += d[q]
                    self.objs[m]["axisAlignedBoundingBox"]["center"][q] += \
                        d[q]
        return _FakeEvent({"lastActionSuccess": True, "errorMessage": "",
                           "isSceneAtRest": True})


def test_nudge_attachment_detection():
    from hiper.evaluation.thor_world import SequenceGenerator
    sim = _NudgeSim()
    gen = SequenceGenerator.__new__(SequenceGenerator)
    gen.sim, gen.floor_y, gen.kinematic = sim, 0.0, set()
    gen.names = sorted(sim.objs)
    gen.flag_group = {n: object_group(o) for n, o in sim.objs.items()}
    found = gen._nudge_attachments()
    # Drawer lies outside the Dresser AABB (z = 3) but moves with it
    assert found == {"Drawer": "Dresser"}
    assert gen.kinematic == {"Dresser"}      # Lamp is not floor-standing
    assert sim.objs["Dresser"]["position"]["y"] == pytest.approx(0.0)
    assert sim.objs["Drawer"]["position"]["y"] == pytest.approx(0.3)
    assert detect_attachments(sim.objs) == {}
