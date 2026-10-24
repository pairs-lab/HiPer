#!/usr/bin/env python3
"""
scripts/explore_common.py
=========================
Shared, VALIDATION-ONLY exploration harness (docs/reports/2026-10-20-widen-margin).

  python scripts/explore_common.py build     # write the cache (once)
  python scripts/explore_common.py check     # reproduce the reference val NLL

Cache (RESULTS/explore_cache/, pickles), built with the frozen association of
run v2 (final.json):
  streams_<split>.pkl   {scene: stream} for val, train, train2 (open loop;
                        CREATE events carry "emb" = DINOv2 of the creating track)
  assoc_<split>.pkl     association rows (gt_mapped_before, matched_gt_id, ...)
  feats_<split>.pkl     {(scene, oid): creation-track features} : clip, dino
                        (unit vectors), clip_label, clip_label_conf, clip_disp,
                        extent (3,), height (centroid z), pixel_area, n_obs
  records_<split>.pkl   training-object histories (GT objects that were
                        tracked): emb (DINOv2 mean), clip (CLIP mean), extent,
                        height, pixel_area, n_tracks, major (CLIP majority
                        label), type, group, S [s], gaps [s], true_hazard
  truth_<split>.pkl     {(scene, gt_id): true hazard per day}  DIAGNOSTIC ONLY

Rules for every experiment built on this harness:
  * tune and select on val only; never read test / test2 / test3 / obchange;
  * methods may use records_train*.pkl and the val streams/feats; they must
    NOT use true_hazard or truth_*.pkl except in a clearly labelled oracle
    diagnostic, and must not fit anything on val labels (label_next,
    gt_present_now, gt_*) except the declared hyper-parameter selection.
"""

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HierarchicalPersistenceFilter, _HPFState)
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import replay  # noqa: E402

CACHE = te.RESULTS / "explore_cache"
RAW = te.THOR.parent / "raw"
DAY = 86400.0
RUN = "v2"


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


# ── Build ────────────────────────────────────────────────────────────────────

def true_hazards(split: str) -> Dict[tuple, float]:
    out = {}
    for d in sorted((RAW / split).iterdir()):
        f = d / "sequence.json"
        if not f.exists():
            continue
        seq = json.load(open(f))
        for g, o in seq["objects"].items():
            out[(d.name, g)] = float(o.get("hazard_per_day", 0.0))
    return out


def creation_feats(split, streams):
    out = {}
    files = te.scene_files(split)
    for scene, st in streams.items():
        ss = {s.session_index: s for s in (te.load_session(f)
                                           for f in files[scene])}
        order = st["session_index"]
        for oid, ob in st["objects"].items():
            ev = ob["events"][0]
            s = ss[order[ev["k"]]]
            tr = next(t for t in s.tracks
                      if int(t.track_id) == int(ev["track_id"]))
            out[(scene, oid)] = {
                "clip": _unit(tr.clip_mean), "dino": _unit(tr.dino_mean),
                "clip_label": tr.clip_label,
                "clip_label_conf": float(tr.clip_label_conf),
                "clip_disp": float(tr.clip_disp),
                "extent": (None if tr.extent is None
                           else np.asarray(tr.extent, float)),
                "height": float(np.asarray(tr.centroid)[2]),
                "pixel_area": float(tr.pixel_area), "n_obs": int(tr.n_obs)}
    return out


def object_records(split, hz):
    recs = []
    for files in te.scene_files(split).values():
        ss = sorted((te.load_session(f) for f in files),
                    key=lambda s: s.session_index)
        scene = ss[0].scene
        acc = defaultdict(lambda: defaultdict(list))
        votes = defaultdict(Counter)
        for s in ss:
            for t in s.tracks:
                if t.gt_id is None:
                    continue
                a = acc[t.gt_id]
                votes[t.gt_id][t.clip_label] += 1
                if t.dino_mean is not None:
                    a["dino"].append(_unit(t.dino_mean))
                if t.clip_mean is not None:
                    a["clip"].append(_unit(t.clip_mean))
                if t.extent is not None:
                    a["extent"].append(np.asarray(t.extent, float))
                a["height"].append(float(np.asarray(t.centroid)[2]))
                a["pixel_area"].append(float(t.pixel_area))
        surv, gaps, meta = defaultdict(float), defaultdict(list), {}
        for s0, s1 in zip(ss, ss[1:]):
            d = float(s1.time - s0.time)
            nxt = {o["gt_id"]: o for o in s1.gt_objects}
            for o in s0.gt_objects:
                meta[o["gt_id"]] = (o["type"], o.get("group"))
                if not o["present"]:
                    continue
                n = nxt.get(o["gt_id"])
                if (n is None or not n["present"] or n.get("moved_since_prev")
                        or n.get("appeared_since_prev")):
                    gaps[o["gt_id"]].append(d)
                else:
                    surv[o["gt_id"]] += d
        for g, a in acc.items():
            if not a["dino"]:
                continue
            typ, grp = meta.get(g, (votes[g].most_common(1)[0][0], None))
            recs.append({
                "scene": scene, "gt_id": g,
                "emb": _unit(np.mean(a["dino"], 0)),
                "clip": _unit(np.mean(a["clip"], 0)) if a["clip"] else None,
                "extent": (np.mean(a["extent"], 0) if a["extent"] else None),
                "height": float(np.mean(a["height"])),
                "pixel_area": float(np.mean(a["pixel_area"])),
                "n_tracks": len(a["height"]),
                "major": votes[g].most_common(1)[0][0], "type": typ,
                "group": grp, "S": surv.get(g, 0.0),
                "gaps": list(gaps.get(g, [])),
                "true_hazard": hz.get((scene, g))})   # DIAGNOSTIC ONLY
    return recs


def build(workers):
    CACHE.mkdir(parents=True, exist_ok=True)
    fin = json.load(open(te.RESULTS / RUN / "final.json"))
    a = {**fin["assoc"], "name": "ours"}
    for split in ("val", "train", "train2"):
        hz = true_hazards(split)
        streams, rows, _ = te.map_split_serial(split, a, workers)
        pickle.dump(streams, open(CACHE / f"streams_{split}.pkl", "wb"))
        pickle.dump(rows, open(CACHE / f"assoc_{split}.pkl", "wb"))
        pickle.dump(creation_feats(split, streams),
                    open(CACHE / f"feats_{split}.pkl", "wb"))
        pickle.dump(object_records(split, hz),
                    open(CACHE / f"records_{split}.pkl", "wb"))
        pickle.dump(hz, open(CACHE / f"truth_{split}.pkl", "wb"))
        print("cached", split, len(streams), "scenes", flush=True)
    json.dump({"run": RUN, "assoc": a, "ours_params":
               fin["families"]["ours"]["params"],
               "pf_params": fin["families"]["pf"]["params"],
               "pf_nbr_params": fin["families"]["pf_nbr"]["params"]},
              open(CACHE / "meta.json", "w"), indent=1, default=float)


# ── Use ──────────────────────────────────────────────────────────────────────

def load(name: str, split: str = "val"):
    return pickle.load(open(CACHE / f"{name}_{split}.pkl", "rb"))


def meta() -> dict:
    return json.load(open(CACHE / "meta.json"))


def keyed(streams):
    """Copy of the streams whose class_label is the unique key 'scene/oid'
    (so a per-object prior can be looked up in create())."""
    out = {}
    for sc, st in streams.items():
        s2 = dict(st)
        s2["objects"] = {oid: {**ob, "class_label": f"{sc}/{oid}"}
                         for oid, ob in st["objects"].items()}
        out[sc] = s2
    return out


class PerObjectPriorHPF(HierarchicalPersistenceFilter):
    """HiPer whose Beta prior (mean m, concentration kappa) is given per
    object, keyed by 'scene/oid' (use with keyed()). Objects without an
    entry fall back to (default mean, kappa0)."""

    def __init__(self, prior: Dict[str, tuple], **kw):
        super().__init__(**kw)
        self.prior = prior

    def create(self, oid, class_label, t, emb=None):
        m, kappa = self.prior.get(class_label,
                                  (self.prior_mean(class_label), self.kappa0))
        m = min(max(float(m), self.m_clip), 1.0 - self.m_clip)
        kappa = float(kappa)
        self._s[oid] = _HPFState(class_label=class_label, pi=self.pi_new,
                                 a=kappa * m, b=kappa * (1.0 - m),
                                 t=float(t), m=m)


def scores(rows) -> dict:
    """Val criterion (NLL, mean over scenes) and the other metrics."""
    name = rows[0]["model"]
    s = pm.summarize(rows)[name]
    return {k: float(s[k][0]) for k in ("nll", "brier", "ece", "auroc")}


def run_model(streams, factory: Callable, name: str = "x") -> dict:
    """Replay every scene through a fresh model from factory(scene)."""
    rows = []
    for sc in sorted(streams):
        rows += replay(streams[sc], factory(sc), name)
    return scores(rows)


def reference(streams=None) -> dict:
    """HiPer v2 (retrieved prior) and PF(lambda) on the cached val streams.
    Expected: HiPer NLL 0.2909, PF(lambda) 0.3591."""
    streams = streams or load("streams")
    m = meta()
    te.NBR["prior"] = te.NeighbourSurvivalPrior(
        [{"emb": r["emb"], "S": r["S"], "gaps": r["gaps"]}
         for r in load("records", "train")])
    te._G.update(lam={}, lam0=1e-6, learned=({}, 1e-6))
    out = {}
    for name, fam, p in (("ours", "hpf_nbr", m["ours_params"]),
                         ("pf", "pf", m["pf_params"])):
        out[name] = run_model(streams, lambda sc: te.make_model(fam, p), name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.workers)
    else:
        print(json.dumps(reference(), indent=1))


if __name__ == "__main__":
    main()
