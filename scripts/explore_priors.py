#!/usr/bin/env python3
"""
scripts/explore_priors.py
=========================
VALIDATION-ONLY exploration of the survival prior of HiPer
(docs/reports/2026-10-19-improve-results). Never touches test splits.

The map object's prior rate lambda(x) comes from the track that created it.
Variants (each re-tuned on val with the same detection x kappa0 grid):
  hard       learned class rate of the creation track's CLIP label (paper)
  hard_k     same, class rates shrunk to the pooled rate with strength k
  soft       mixture over the CLIP label distribution p(c | x):
             m(x) = sum_c p(c|x) exp(-lambda_c dt_ref)
  knn        kernel-weighted rate of the K nearest TRAIN objects in an
             embedding space (CLIP or DINOv2), MAP with Gamma shrinkage
             to the pooled rate (the class-rate estimator with soft keys)
  oracle_gt  DIAGNOSTIC: rates keyed by ground-truth type, GT labels

  python scripts/explore_priors.py --run main --out improve
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_and_eval as te  # noqa: E402  (registers the hiper package)

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HierarchicalPersistenceFilter)
from hiper.ephemerality.priors import learn_class_rates  # noqa: E402
from hiper.evaluation.lifelong_objects import replay, write_jsonl  # noqa: E402
from hiper.evaluation.object_session import load_session  # noqa: E402
from hiper.evaluation import persistence_metrics as pm  # noqa: E402

DAY = 86400.0


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


# ── Train objects: embeddings + per-object exposure ─────────────────────────

def train_objects(splits):
    """One record per tracked GT object of the train scenes: mean CLIP and
    DINOv2 embedding of its tracks, majority CLIP label, GT type, survived
    exposure S [s] and the gaps [s] that ended in a vanish (same rules as
    priors.exposure_counts)."""
    recs = []
    for split in splits:
        for files in te.scene_files(split).values():
            ss = sorted((load_session(f) for f in files),
                        key=lambda s: s.session_index)
            clip, dino, votes = (defaultdict(list), defaultdict(list),
                                 defaultdict(Counter))
            for s in ss:
                for t in s.tracks:
                    if t.gt_id is None:
                        continue
                    votes[t.gt_id][t.clip_label] += 1
                    if t.clip_mean is not None:
                        clip[t.gt_id].append(_unit(t.clip_mean))
                    if t.dino_mean is not None:
                        dino[t.gt_id].append(_unit(t.dino_mean))
            surv, gaps, gtype = defaultdict(float), defaultdict(list), {}
            for s0, s1 in zip(ss, ss[1:]):
                d = float(s1.time - s0.time)
                nxt = {o["gt_id"]: o for o in s1.gt_objects}
                for o in s0.gt_objects:
                    gtype[o["gt_id"]] = o["type"]
                    if not o["present"]:
                        continue
                    n = nxt.get(o["gt_id"])
                    if (n is None or not n["present"]
                            or n.get("moved_since_prev")
                            or n.get("appeared_since_prev")):
                        gaps[o["gt_id"]].append(d)
                    else:
                        surv[o["gt_id"]] += d
            for g in votes:
                if not clip[g] or not dino[g]:
                    continue
                recs.append({
                    "clip": _unit(np.mean(clip[g], 0)),
                    "dino": _unit(np.mean(dino[g], 0)),
                    "major": votes[g].most_common(1)[0][0],
                    "type": gtype.get(g, votes[g].most_common(1)[0][0]),
                    "S": surv.get(g, 0.0), "gaps": gaps.get(g, [])})
    return recs


def weighted_map_rate(w, S, gaps, gap_w, k, lam0):
    """MAP rate with weighted objects: root of
    k/l - k/lam0 - sum w S + sum_j gw_j d_j / (e^{l d_j} - 1)."""
    Sw = float(np.dot(w, S))
    n = float(gap_w.sum())
    beta = k / lam0
    if n + k <= 0:
        return lam0
    if n == 0:
        return k / (beta + Sw)

    def score(lam):
        x = lam * gaps
        return k / lam - beta - Sw + float(
            np.sum(gap_w * gaps * np.exp(-x) / -np.expm1(-x)))

    lo = (n + k) / (beta + Sw + float(np.dot(gap_w, gaps)))
    hi = (n + k) / (beta + Sw)
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


class KnnPrior:
    def __init__(self, recs, space, lam0):
        self.E = np.stack([r[space] for r in recs])
        self.S = np.array([r["S"] for r in recs])
        self.gaps = [np.asarray(r["gaps"], float) for r in recs]
        self.lam0 = lam0

    def rate(self, e, K, tau, k):
        sims = self.E @ e
        idx = np.argpartition(-sims, min(K, len(sims) - 1))[:K]
        w = np.exp((sims[idx] - sims[idx].max()) / tau) if tau else \
            np.ones(len(idx))
        w = w / w.max()
        g = np.concatenate([self.gaps[i] for i in idx]) if len(idx) else \
            np.zeros(0)
        gw = np.concatenate([np.full(len(self.gaps[i]), wi)
                             for i, wi in zip(idx, w)]) if len(idx) else \
            np.zeros(0)
        return weighted_map_rate(w, self.S[idx], g, gw, k, self.lam0)


# ── Val streams + creation-track features ───────────────────────────────────

def creation_features(split, streams):
    """{(scene, oid): (clip, dino, clip_label, gt_type)} of the track that
    created each map object."""
    out = {}
    files = te.scene_files(split)
    for scene, st in streams.items():
        ss = {s.session_index: s for s in (load_session(f)
                                           for f in files[scene])}
        order = st["session_index"]
        for oid, ob in st["objects"].items():
            ev = ob["events"][0]
            s = ss[order[ev["k"]]]
            tr = next(t for t in s.tracks if int(t.track_id) ==
                      int(ev["track_id"]))
            g = (st["gt"][ev["k"]].get(ev["gt_id"]) or {}) \
                if ev.get("gt_id") else {}
            out[(scene, oid)] = (_unit(tr.clip_mean), _unit(tr.dino_mean),
                                 tr.clip_label, g.get("type"))
    return out


_G = {}


def _init(streams):
    _G["streams"] = streams


def _replay(args):
    name, lam_of, params = args          # lam_of: {(scene, oid): lambda}
    rows = []
    for scene, st in sorted(_G["streams"].items()):
        st2 = dict(st)
        st2["objects"] = {oid: {**ob, "class_label": f"{scene}/{oid}"}
                          for oid, ob in st["objects"].items()}
        tab = {f"{scene}/{oid}": lam_of[(scene, oid)]
               for oid in st["objects"]}
        m = HierarchicalPersistenceFilter(lambda_table=tab, **params)
        rows += replay(st2, m, name)
    return name, params, rows


def tune(pool, name, lam_of, det_kappa, log):
    best = (math.inf, None, None)
    jobs = [(name, lam_of, {**p, "variance_gate": False, "tau_pi": 0.3})
            for p in det_kappa]
    for nm, params, rows in pool.imap_unordered(_replay, jobs):
        s = te._scores(rows)
        log.append({"variant": nm, "params": params, "nll": s["nll"]})
        if s["nll"] < best[0]:
            best = (s["nll"], params, rows)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--out", default="improve")
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="final.json detection params only (no re-tuning)")
    args = ap.parse_args()
    out = te.RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    fin = json.load(open(te.RESULTS / args.run / "final.json"))
    a = {**fin["assoc"], "name": "ours"}
    op = fin["families"]["ours"]["params"]

    recs = train_objects(args.train_splits)
    hist_major, hist_type = [], []
    for files in [f for sp in args.train_splits
                  for f in te.scene_files(sp).values()]:
        ss = [load_session(f) for f in files]
        votes = defaultdict(Counter)
        for s in ss:
            for t in s.tracks:
                votes[t.gt_id][t.clip_label] += 1
        major = {g: c.most_common(1)[0][0] for g, c in votes.items()}
        hist_major.append([(s.time, [{**o, "type": major[o["gt_id"]]}
                                     for o in s.gt_objects
                                     if o["gt_id"] in major]) for s in ss])
        hist_type.append([(s.time, [o for o in s.gt_objects
                                    if o["gt_id"] in major]) for s in ss])
    print("train objects", len(recs), flush=True)

    streams, _, _ = te.map_split_serial("val", a, args.workers)
    feats = creation_features("val", streams)
    keys = list(feats)
    print("val objects", len(keys), flush=True)

    det = te.grid(te.DET_GRID)
    det_kappa = ([{k: op[k] for k in ("p_miss", "p_false", "kappa0")}]
                 if args.quick else
                 [{**d, "kappa0": k} for d in det for k in te.KAPPA_FINAL])
    variants = {}
    for k in (0.25, 1.0, 4.0, 16.0, 64.0):
        tab, lam0 = learn_class_rates(hist_major, prior_strength=k)
        variants[f"hard_k{k:g}"] = {
            key: tab.get(feats[key][2], lam0) for key in keys}
        if k == 1.0:
            # soft labels over the vocabulary
            text = text_embeddings(out)
            types, T = text["types"], text["emb"]
            lam_c = np.array([tab.get(t, lam0) for t in types])
            m_c = np.exp(-lam_c * DAY)
            for scale in (25.0, 50.0, 100.0):
                lam = {}
                for key in keys:
                    e = feats[key][0]
                    if e is None:
                        lam[key] = tab.get(feats[key][2], lam0)
                        continue
                    z = scale * (T @ e)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    m = float(np.clip(p @ m_c, 1e-6, 1 - 1e-9))
                    lam[key] = -math.log(m) / DAY
                variants[f"soft_s{scale:g}"] = lam
    _, lam0 = learn_class_rates(hist_major, prior_strength=1.0)
    for space in ("clip", "dino"):
        kp = KnnPrior(recs, space, lam0)
        for K in (10, 30, 100, 300):
            for tau in (0.02, 0.05, None):
                for ks in (1.0, 4.0):
                    lam = {}
                    for key in keys:
                        e = feats[key][0 if space == "clip" else 1]
                        lam[key] = (kp.rate(e, K, tau, ks) if e is not None
                                    else lam0)
                    variants[f"knn_{space}_K{K}_t{tau}_k{ks:g}"] = lam
    tab_t, lam0_t = learn_class_rates(hist_type, prior_strength=1.0)
    variants["oracle_gt"] = {key: tab_t.get(feats[key][3] or "", lam0_t)
                             for key in keys}

    log, res = [], {}
    with Pool(args.workers, _init, (streams,)) as pool:
        for name, lam_of in variants.items():
            nll, params, rows = tune(pool, name, lam_of, det_kappa, log)
            summ = pm.summarize(rows, split="val")[name]
            res[name] = {"val_nll": nll, "params": params,
                         **{m: summ[m][0] for m in ("brier", "ece",
                                                         "auroc")}}
            print(f"{name:32s} nll {nll:.4f} brier {res[name]['brier']:.4f}"
                  f" ece {res[name]['ece']:.4f} auroc "
                  f"{res[name]['auroc']:.4f} {params}", flush=True)
    write_jsonl(log, str(out / "prior_grid_val.jsonl"))
    json.dump(res, open(out / "prior_variants_val.json", "w"), indent=1,
              default=float)


def text_embeddings(out):
    f = out / "clip_text_emb.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {"types": list(d["types"]), "emb": d["emb"]}
    from hiper.evaluation.clip_vocab import ClipVocab
    v = ClipVocab(device="cpu")
    np.savez(f, types=np.array(v.types), emb=v.text_emb)
    return {"types": v.types, "emb": v.text_emb}


if __name__ == "__main__":
    main()
