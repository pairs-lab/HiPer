"""
tests/test_lifelong_objects.py
==============================
Unit tests for the offline lifelong object mapper
(hiper.evaluation.lifelong_objects) on hand-built SessionData (no
simulator): HIT / RELOCATE / MISS / NONE / CREATE semantics, tentative
matches, one-to-one association of duplicates, pose inflation, the
label_next / gt_present_now definitions, open-loop replay, the closed loop
and the CLI (scripts/run_lifelong_objects.py).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hiper.ephemerality.persistence_v2 import HierarchicalPersistenceFilter
from hiper.evaluation.lifelong_objects import (
    MapObject,
    ObjectMapper,
    gt_labels,
    gt_tables,
    load_class_lambdas,
    load_stream,
    replay,
    run_closed_loop,
    run_mapping,
    save_stream,
)
from hiper.evaluation.object_session import SessionData, Track, save_session
from hiper.object_encoder.association_v2 import AssocConfig

DAY = 86400.0
K_CAM = np.array([[30.0, 0.0, 32.0], [0.0, 30.0, 24.0], [0.0, 0.0, 1.0]])
PRED_KEYS = {"scene", "split", "model", "oid", "k", "t_k", "t_next",
             "class_label", "gt_id", "gt_group", "pred_presence",
             "rate_mean", "rate_var", "label_next", "gt_present_now",
             "pruned", "committed", "last_evidence"}
ASSOC_KEYS = {"scene", "split", "config", "k", "track_id", "gt_id",
              "matched_oid", "matched_gt_id", "score", "d2", "tentative",
              "relocated_pred", "gt_mapped_before", "gt_moved",
              "gt_reappeared"}
P_A = (0.0, 0.0, 2.0)          # in view of the default camera


def _emb(seed: int, dim: int = 16) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


EA = _emb(1)


def _track(tid, pos, gt_id, emb=EA, label="mug", disp=0.05, cov=1e-4):
    return Track(track_id=tid, centroid=np.asarray(pos, dtype=np.float64),
                 centroid_cov=cov * np.eye(3), n_obs=10, clip_mean=emb,
                 clip_disp=disp, clip_label=label, clip_label_conf=0.9,
                 gt_id=gt_id)


def _gt(gt_id, present=True, moved=False, appeared=False):
    return {"gt_id": gt_id, "type": "Mug", "position": [0.0, 0.0, 0.0],
            "present": present, "group": "pickupable",
            "moved_since_prev": moved, "appeared_since_prev": appeared}


def _session(k, tracks, gt, look_away=False, pose_std=0.01, scene="SceneA"):
    """Three identical frames at the origin looking along +z (or -z)."""
    T = np.eye(4)
    if look_away:
        T[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    return SessionData(
        scene=scene, session_index=k, time=k * DAY, tracks=tracks,
        T_wc_est=np.repeat(T[None], 3, axis=0), K=K_CAM.copy(),
        depth_stack=np.zeros((3, 48, 64), dtype=np.float16),
        pose_cov=pose_std ** 2 * np.eye(3), gt_objects=gt)


def _events(stream, oid):
    return [e["ev"] for e in stream["objects"][oid]["events"]]


def _by_k(rows):
    return {r["k"]: r for r in rows}


# ── Event semantics ──────────────────────────────────────────────────────────

class TestEvents:
    def test_stationary_object_is_hit(self):
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
             _session(1, [_track(1, (0.01, 0.0, 2.0), "a")], [_gt("a")])]
        stream, assoc = run_mapping(s, AssocConfig(), split="val")
        assert list(stream["objects"]) == ["s0_t1"]
        assert _events(stream, "s0_t1") == ["create", "hit"]
        (row,) = assoc
        assert set(row) == ASSOC_KEYS
        assert row["matched_oid"] == "s0_t1" and row["matched_gt_id"] == "a"
        assert row["gt_mapped_before"] and row["gt_moved"] is False
        assert not row["tentative"] and not row["relocated_pred"]

        rows = _by_k(replay(stream, HierarchicalPersistenceFilter()))
        assert (rows[0]["label_next"], rows[0]["gt_present_now"]) == (1, 1)
        assert rows[1]["label_next"] is None and rows[1]["t_next"] is None
        assert rows[1]["pred_presence"] is None
        assert rows[0]["pred_presence"] == pytest.approx(
            HierarchicalPersistenceFilter().prior_mean("mug"), rel=1e-9)

    def test_hit_fuses_position_and_covariance(self):
        mapper = ObjectMapper(AssocConfig(pose_inflation=False,
                                          centroid_std=0.0))
        mapper.step(_session(0, [_track(1, P_A, "a")], [_gt("a")]), 0)
        mapper.step(_session(1, [_track(1, (0.01, 0, 2), "a")], [_gt("a")]),
                    1)
        o = mapper.objects["s0_t1"]
        assert o.position == pytest.approx([0.005, 0.0, 2.0], abs=1e-6)
        assert np.diag(o.cov) == pytest.approx([5e-5] * 3, rel=1e-4)
        assert o.n_fused == 2 and o.pos_session == 1

    def test_moved_object_is_relocated_with_labels(self):
        P_B = (1.0, 0.0, 2.5)
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
             _session(1, [_track(4, P_B, "a")], [_gt("a", moved=True)]),
             _session(2, [_track(2, P_B, "a")], [_gt("a")])]
        stream, assoc = run_mapping(s, AssocConfig())
        assert _events(stream, "s0_t1") == ["create", "relocate", "hit"]
        assert len(stream["objects"]) == 1
        r1 = assoc[0]
        assert r1["k"] == 1 and r1["relocated_pred"] and r1["gt_moved"]
        assert r1["matched_gt_id"] == r1["gt_id"] == "a"
        rows = _by_k(replay(stream, HierarchicalPersistenceFilter()))
        # Before the move the map held the old position: not there at k=1
        assert rows[0]["label_next"] == 0
        # The relocation reset the position: present and unmoved at 1 and 2
        assert (rows[1]["gt_present_now"], rows[1]["label_next"]) == (1, 1)
        assert rows[1]["last_evidence"] == "relocate"
        assert stream["objects"]["s0_t1"]["events"][1]["pos_session"] == 1

    def test_relocation_resets_position(self):
        mapper = ObjectMapper(AssocConfig())
        mapper.step(_session(0, [_track(1, P_A, "a")], [_gt("a")]), 0)
        mapper.step(_session(1, [_track(1, (1.0, 0, 2.5), "a")],
                             [_gt("a", moved=True)]), 1)
        assert mapper.objects["s0_t1"].position == \
            pytest.approx([1.0, 0.0, 2.5])

    def test_removed_but_visible_is_miss(self):
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
             _session(1, [], [_gt("a", present=False)])]
        stream, assoc = run_mapping(s, AssocConfig())
        ev = stream["objects"]["s0_t1"]["events"][1]
        assert ev["ev"] == "miss" and ev["visible"] is True
        assert assoc == []
        rows = _by_k(replay(stream, HierarchicalPersistenceFilter()))
        assert rows[0]["label_next"] == 0
        assert rows[1]["gt_present_now"] == 0

    def test_out_of_view_is_none(self):
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
             _session(1, [], [_gt("a")], look_away=True)]
        stream, _ = run_mapping(s, AssocConfig())
        ev = stream["objects"]["s0_t1"]["events"][1]
        assert ev["ev"] == "none" and ev["visible"] is False
        assert not ev["tentative"]
        rows = _by_k(replay(stream, HierarchicalPersistenceFilter()))
        assert rows[0]["label_next"] == 1          # unseen but still there

    def test_tentative_match_is_none_without_update(self):
        mapper = ObjectMapper(AssocConfig(tau_u=0.5))
        mapper.step(_session(0, [_track(1, P_A, "a")], [_gt("a")]), 0)
        o = mapper.objects["s0_t1"]
        pos, cov, u = o.position.copy(), o.cov.copy(), o.u
        events, matches, _ = mapper.step(
            _session(1, [_track(1, (0.02, 0, 2), "a", disp=0.6)],
                     [_gt("a")]), 1)
        (ev,) = events                  # matched, so no new object either
        assert ev["ev"] == "none" and ev["tentative"]
        assert matches[0].obj == 0 and matches[0].tentative
        assert np.array_equal(o.position, pos) and np.array_equal(o.cov, cov)
        assert o.u == u and o.n_fused == 1 and o.pos_session == 0
        assert list(mapper.objects) == ["s0_t1"]

    def test_tentative_gate_off_gives_hit(self):
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
             _session(1, [_track(1, P_A, "a", disp=0.6)], [_gt("a")])]
        stream, _ = run_mapping(s, AssocConfig(use_tau_u=False))
        assert _events(stream, "s0_t1") == ["create", "hit"]


class TestAssociationInMap:
    def test_duplicates_one_to_one(self):
        """Look-alike duplicates keep their identities through geometry;
        a third look-alike becomes a new object."""
        s = [_session(0, [_track(1, (-0.5, 0, 2), "d1"),
                          _track(2, (0.5, 0, 2), "d2")],
                      [_gt("d1"), _gt("d2")]),
             _session(1, [_track(7, (0.5, 0, 2), "d2"),
                          _track(8, (-0.5, 0, 2), "d1"),
                          _track(9, (0.0, 0, 3.5), "d3")],
                      [_gt("d1"), _gt("d2"), _gt("d3", appeared=True)])]
        stream, assoc = run_mapping(s, AssocConfig())
        assert _events(stream, "s0_t1") == ["create", "hit"]
        assert _events(stream, "s0_t2") == ["create", "hit"]
        assert _events(stream, "s1_t9") == ["create"]
        rows = {r["track_id"]: r for r in assoc}
        assert rows[7]["matched_gt_id"] == "d2"
        assert rows[8]["matched_gt_id"] == "d1"
        assert rows[9]["matched_oid"] is None
        assert not rows[9]["gt_mapped_before"] and rows[9]["gt_moved"] is None

    def test_bye_separates_relocated_duplicates(self):
        """Both duplicates move far (geometry useless) and CLIP is identical:
        the optional BYE embeddings decide."""
        b1, b2 = _emb(11), _emb(12)
        b2 = b2 - (b2 @ b1) * b1
        s = [_session(0, [_track(1, (-0.5, 0, 2), "d1"),
                          _track(2, (0.5, 0, 2), "d2")],
                      [_gt("d1"), _gt("d2")]),
             _session(1, [_track(1, (0.5, 0, 3), "d1"),
                          _track(2, (-0.5, 0, 3), "d2")],
                      [_gt("d1", moved=True), _gt("d2", moved=True)])]
        bye = {0: {1: b1, 2: b2}, 1: {1: b1, 2: b2}}
        stream, assoc = run_mapping(s, AssocConfig(w=0.5), bye=bye)
        assert [r["matched_gt_id"] for r in assoc] == ["d1", "d2"]
        assert all(r["relocated_pred"] and r["gt_moved"] for r in assoc)

    def test_pose_inflation_turns_drift_relocation_into_hit(self):
        """A 0.3 m drift with 0.2 m pose std: a false relocation without
        pose inflation, a HIT with it."""
        s = [_session(0, [_track(1, P_A, "a")], [_gt("a")], pose_std=0.2),
             _session(1, [_track(1, (0.3, 0, 2), "a")], [_gt("a")],
                      pose_std=0.2)]
        on, assoc_on = run_mapping(s, AssocConfig(pose_inflation=True))
        off, assoc_off = run_mapping(s, AssocConfig(pose_inflation=False))
        assert _events(on, "s0_t1") == ["create", "hit"]
        assert _events(off, "s0_t1") == ["create", "relocate"]
        assert not assoc_on[0]["relocated_pred"]
        assert assoc_off[0]["relocated_pred"] and not assoc_off[0]["gt_moved"]
        assert assoc_on[0]["d2"] < 11.34 < assoc_off[0]["d2"]


# ── Map state and labels ─────────────────────────────────────────────────────

class TestMapState:
    def test_u_is_pooled_unit_embedding_dispersion(self):
        """u_j = mean within + between dispersion = 1 - |mean raw mean|^2."""
        d1, d2 = np.eye(4)[0], np.eye(4)[1]
        o = MapObject("o", np.zeros(3), np.eye(3), "mug", 0, 0)
        o.absorb(_track(1, P_A, "a", emb=d1, disp=0.1), None, 0)
        o.absorb(_track(1, P_A, "a", emb=d2, disp=0.3), None, 1)
        mu = (np.sqrt(0.9) * d1 + np.sqrt(0.7) * d2) / 2.0
        assert o.clip_within == pytest.approx(0.2)
        assert o.u == pytest.approx(1.0 - mu @ mu)          # = 0.6
        assert o.clip == pytest.approx(mu / np.linalg.norm(mu))
        # Antipodal, very dispersed tracks: still inside [0, 1]
        o2 = MapObject("o2", np.zeros(3), np.eye(3), "mug", 0, 0)
        o2.absorb(_track(1, P_A, "a", emb=d1, disp=0.9), None, 0)
        o2.absorb(_track(1, P_A, "a", emb=-d1, disp=0.9), None, 1)
        assert o2.u == pytest.approx(1.0)

    def test_majority_label_keeps_creation_label(self):
        o = MapObject("o", np.zeros(3), np.eye(3), "cup", 0, 0)
        for lab in ("cup", "mug", "mug"):
            o.absorb(_track(1, P_A, "a", label=lab), None, 0)
        assert o.class_label == "cup" and o.majority_label == "mug"

    def test_gt_labels(self):
        seq = [dict(present=True), dict(present=True),
               dict(present=True, moved=True), dict(present=False),
               dict(present=True, appeared=True)]
        gt = gt_tables([_session(k, [], [_gt("x", **f)])
                        for k, f in enumerate(seq)])
        assert gt_labels(gt, "x", 0, 0) == (1, 1)
        assert gt_labels(gt, "x", 0, 1) == (1, 0)       # moves at 2
        assert gt_labels(gt, "x", 2, 2) == (1, 0)       # position set at 2
        assert gt_labels(gt, "x", 2, 3) == (0, 0)       # gone, re-appears
        assert gt_labels(gt, "x", 4, 4) == (1, None)    # last session
        assert gt_labels(gt, None, 0, 0) == (0, 0)
        assert gt_labels(None, "x", 0, 0) == (None, None)


# ── Replay and closed loop ───────────────────────────────────────────────────

def _fast_hpf():
    return HierarchicalPersistenceFilter(lambda_table={"mug": 1.0 / DAY})


def _removed_then_back():
    """Visible removal at k=1 (MISS), back at the same place at k=2."""
    return [_session(0, [_track(1, P_A, "a")], [_gt("a")]),
            _session(1, [], [_gt("a", present=False)]),
            _session(2, [_track(3, P_A, "a")], [_gt("a", appeared=True)])]


class TestReplay:
    def test_rows_schema_and_open_loop_keeps_pruned(self):
        stream, _ = run_mapping(_removed_then_back(), AssocConfig(),
                                split="val")
        rows = replay(stream, _fast_hpf(), model_name="hpf_fast")
        assert all(set(r) == PRED_KEYS for r in rows)
        r = _by_k(rows)
        assert r[1]["last_evidence"] == "miss" and r[1]["pruned"]
        assert r[2]["oid"] == "s0_t1" and r[2]["last_evidence"] == "hit"
        assert r[0]["model"] == "hpf_fast" and r[0]["split"] == "val"
        assert r[0]["gt_group"] == "pickupable"
        assert r[0]["pred_presence"] > r[1]["pred_presence"]

    def test_stream_json_roundtrip(self, tmp_path):
        stream, _ = run_mapping(_removed_then_back(), AssocConfig())
        path = tmp_path / "streams" / "SceneA.json"
        save_stream(stream, str(path))
        rows_a = replay(stream, _fast_hpf())
        rows_b = replay(load_stream(str(path)), _fast_hpf())
        assert json.dumps(rows_a) == json.dumps(rows_b)


class TestClosedLoop:
    def test_pruned_object_cannot_be_matched_again(self):
        out = run_closed_loop(_removed_then_back(), AssocConfig(),
                              _fast_hpf(), split="val", model_name="hpf")
        sizes = [(m["n_objects"], m["n_pruned"], m["n_created"])
                 for m in out["map_size"]]
        assert sizes == [(1, 0, 1), (0, 1, 0), (1, 0, 1)]
        assert [r["oid"] for r in out["pred_rows"]] == \
            ["s0_t1", "s0_t1", "s2_t3"]
        (row,) = out["assoc_rows"]
        assert row["matched_oid"] is None and not row["gt_mapped_before"]
        assert row["model"] == "hpf"

    def test_model_that_never_prunes_matches_open_loop(self):
        s = _removed_then_back()
        stream, assoc = run_mapping(s, AssocConfig())
        keep = HierarchicalPersistenceFilter(tau_pi=0.0)
        out = run_closed_loop(s, AssocConfig(), keep)
        assert out["pred_rows"] == replay(stream, HierarchicalPersistenceFilter(
            tau_pi=0.0))
        assert [{k: v for k, v in r.items() if k != "model"}
                for r in out["assoc_rows"]] == assoc


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / \
        "run_lifelong_objects.py"
    spec = importlib.util.spec_from_file_location("run_lifelong_objects",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strict_jsonl(path):
    def bad(c):
        raise ValueError(f"non-finite constant {c}")
    return [json.loads(line, parse_constant=bad)
            for line in path.read_text().splitlines()]


class TestCLI:
    def test_end_to_end(self, tmp_path):
        cli = _load_cli()
        root = tmp_path / "sessions"
        for scene in ("SceneA", "SceneB"):
            for s in _removed_then_back():
                s.scene = scene
                save_session(s, str(root / scene /
                                    f"session_{s.session_index}.npz"))
        # BYE for SceneA only, mirrored layout
        bye_dir = tmp_path / "bye" / "SceneA"
        bye_dir.mkdir(parents=True)
        for k in range(3):
            np.savez(bye_dir / f"session_{k}.npz",
                     track_ids=np.array([1, 3]),
                     embeddings=np.stack([_emb(5), _emb(5)]))
        common = ["--sessions-root", str(root), "--split", "val",
                  "--scenes", "SceneA", "SceneB",
                  "--out-root", str(tmp_path / "results"),
                  "--models", "hpf", "pf_class",
                  "lazy=persistence_v2:HierarchicalPersistenceFilter",
                  "--model-kwargs", "lazy", '{"learn_rate": false}']
        assert cli.main(common + ["--run-name", "r1", "--closed-loop",
                                  "--bye-dir", str(tmp_path / "bye"),
                                  "--assoc", "name=test", "w_geo=0.4"]) == 0
        out = tmp_path / "results" / "r1"
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["assoc"]["name"] == "test" and cfg["assoc"]["w_geo"] == 0.4
        assert cfg["per_scene"]["SceneA"]["bye_sessions"] == 3
        assert cfg["per_scene"]["SceneB"]["bye_sessions"] == 0
        assert (out / "streams" / "SceneB.json").exists()
        pred = _strict_jsonl(out / "pred_hpf.jsonl")
        assert len(pred) == 6 and all(set(r) == PRED_KEYS for r in pred)
        assert {r["scene"] for r in pred} == {"SceneA", "SceneB"}
        # The lazily imported class with learn_rate=False is PF(lambda_c)
        lazy = _strict_jsonl(out / "pred_lazy.jsonl")
        pfc = _strict_jsonl(out / "pred_pf_class.jsonl")
        assert [r["pred_presence"] for r in lazy] == \
            [r["pred_presence"] for r in pfc]
        assoc = _strict_jsonl(out / "assoc.jsonl")
        assert all(set(r) == ASSOC_KEYS for r in assoc)
        assert all(r["config"] == "test" for r in assoc)
        sizes = _strict_jsonl(out / "closed_mapsize_hpf.jsonl")
        assert len(sizes) == 6
        # Replaying the saved streams gives the same prediction rows
        assert cli.main(common + ["--run-name", "r1", "--reuse-streams"]) == 0
        assert _strict_jsonl(out / "pred_hpf.jsonl") == pred

    def test_missing_scene_fails_early(self, tmp_path):
        cli = _load_cli()
        with pytest.raises(SystemExit):
            cli.main(["--sessions-root", str(tmp_path), "--scenes", "Nope",
                      "--run-name", "x", "--out-root", str(tmp_path)])
        assert not (tmp_path / "x").exists()

    def test_class_lambdas_from_yaml(self):
        path = Path(__file__).resolve().parents[1] / "config" / \
            "class_priors_indoor.yaml"
        table, default = load_class_lambdas(str(path))
        assert table["AlarmClock"] == table["alarm clock"] == \
            table["alarmclock"] == pytest.approx(1.0 / (90 * DAY))
        assert table["tv stand"] == pytest.approx(1.0 / (730 * DAY))
        assert default == pytest.approx(1.0 / (30 * DAY))
