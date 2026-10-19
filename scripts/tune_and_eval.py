#!/usr/bin/env python3
"""
scripts/tune_and_eval.py
========================
Validation tuning with declared criteria, then ONE run on the test split and
one zero-shot run on ObChange (design.md 'Evaluation').

  tune   association grid on THOR val (criterion: association accuracy),
         tau_u by the full model's val NLL, then every persistence family:
         prediction parameters by val NLL, decision thresholds by
         0.5 (premature_prune_rate + stale_rate) on val (pooled counts).
  test   selected configs on THOR test + ObChange: E2 rows, E3 rows,
         E4 ablations, closed-loop map size, summaries with bootstrap CIs.

Usage
  python scripts/tune_and_eval.py tune --run main
  python scripts/tune_and_eval.py test --run main
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import conftest  # noqa: E402,F401  (registers the hiper package)

from hiper.ephemerality.persistence_baselines import (  # noqa: E402
    AlwaysPersistModel, ClassRateModel, DraftV1Model, ELiteScalarModel)
from hiper.ephemerality.persistence_pocd import POCDBetaModel  # noqa: E402
from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HierarchicalPersistenceFilter)
from hiper.ephemerality.priors import learn_class_rates  # noqa: E402
from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.lifelong_objects import (  # noqa: E402
    load_class_lambdas, replay, run_closed_loop, run_mapping, write_jsonl)
from hiper.evaluation.object_session import load_session  # noqa: E402
from hiper.object_encoder.association_v2 import AssocConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
from hiper.evaluation.paths import (  # noqa: E402
    OBCHANGE_SESSIONS as OBC, RESULTS_ROOT as RESULTS, THOR_SESSIONS as THOR)
PRIORS = REPO / "config/class_priors_indoor.yaml"
DAY = 86400.0
INF = float("inf")

# ── Declared grids (fixed before any result was seen) ────────────────────────
ASSOC_GRID = dict(appearance=["clip", "clip+dino", "dino", "none"],
                  w_geo=[0.0, 0.25, 0.5, 1.0],
                  tau_match=[0.5, 0.6, 0.7, 0.8, 0.9],
                  centroid_std=[0.02, 0.05, 0.1],
                  sigma_u=[None, 0.1, 0.2, 0.4])          # None = no gamma
TAU_U_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, INF]
DET_GRID = dict(p_miss=[0.05, 0.1, 0.2, 0.3], p_false=[0.01, 0.05, 0.1, 0.2])
KAPPA_GRID = [0.5, 1, 2, 4, 8, 16, 32]
LIFETIME_GRID_DAYS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
TAU_PI_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
TAU_V_GRID = [0.001, 0.003, 0.01, 0.03, INF]
ELITE_GRID = dict(eps_hit=[0.05, 0.1, 0.2, 0.3, 0.4],
                  eps_miss=[0.6, 0.7, 0.8, 0.9, 0.95], eps_init=[0.3, 0.5, 0.7])
ELITE_TAU = [0.3, 0.5, 0.7, 0.9]
POCD_GRID = dict(steps_per_session=[1, 3, 5], idle_decay=[True, False])
POCD_TAU = [0.3, 0.4, 0.5, 0.6, 0.7]


# ── Data ─────────────────────────────────────────────────────────────────────

SMOKE = {"on": False}


def scene_files(split: str) -> Dict[str, List[str]]:
    root = OBC if split == "obchange" else THOR / split
    out = {}
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in d.glob("session_*.npz"))
        if files:
            out[d.name] = files
    if SMOKE["on"]:
        out = dict(list(out.items())[:2])
    return out


def _smoke_grids():
    """Tiny grids for a bug-finding run (never for reported numbers)."""
    g = globals()
    g["ASSOC_GRID"] = dict(appearance=["clip", "dino", "none"], w_geo=[0.5],
                           tau_match=[0.6], centroid_std=[0.05],
                           sigma_u=[None, 0.2])
    g["TAU_U_GRID"] = [0.3, INF]
    g["DET_GRID"] = dict(p_miss=[0.1], p_false=[0.05])
    g["KAPPA_GRID"] = [4]
    g["LIFETIME_GRID_DAYS"] = [4]
    g["TAU_PI_GRID"] = [0.1]
    g["TAU_V_GRID"] = [0.01, INF]
    g["ELITE_GRID"] = dict(eps_hit=[0.1], eps_miss=[0.9], eps_init=[0.5])
    g["ELITE_TAU"] = [0.5]
    g["POCD_GRID"] = dict(steps_per_session=[1], idle_decay=[True])
    g["POCD_TAU"] = [0.5]


def _appearance(cfg: dict):
    """(w, use_dino): CLIP-only, DINOv2-only (in the 'bye' slot) or both."""
    mode = cfg.get("appearance", "clip")
    return {"clip": (1.0, False), "dino": (0.0, True),
            "clip+dino": (0.5, True), "none": (1.0, False)}[mode]


def assoc_config(cfg: dict) -> AssocConfig:
    w, _ = _appearance(cfg)
    sig = cfg.get("sigma_u")
    return AssocConfig(
        name=cfg.get("name", "assoc"), w=w, w_geo=cfg["w_geo"],
        sigma_u=sig if sig is not None else 1.0,
        use_gamma=sig is not None, tau_u=cfg.get("tau_u", INF),
        use_tau_u=cfg.get("tau_u", INF) < INF,
        tau_match=cfg["tau_match"], centroid_std=cfg["centroid_std"],
        pose_inflation=cfg.get("pose_inflation", True))


def appearance_inputs(sessions, cfg: dict):
    """The appearance a mapping run needs for ``cfg``: strips CLIP for
    geometry-only runs (in place) and returns the per-session DINOv2
    embeddings for the 'bye' slot (None if unused). EVERY mapping call
    (open loop, closed loop, leave-one-room-out, figures) must use it."""
    if cfg.get("appearance") == "none":
        for s in sessions:
            for t in s.tracks:
                t.clip_mean = None
    _, use_dino = _appearance(cfg)
    if not use_dino:
        return None
    return {s.session_index: {int(t.track_id): t.dino_mean
                              for t in s.tracks if t.dino_mean is not None}
            for s in sessions}


def _map_job(args):
    scene, files, cfg, split = args
    sessions = [load_session(f) for f in files]
    dino = appearance_inputs(sessions, cfg)
    t0 = time.perf_counter()
    stream, rows = run_mapping(sessions, assoc_config(cfg), bye=dino,
                               split=split)
    dt = time.perf_counter() - t0
    return scene, stream, rows, dt, len(sessions)


def map_split(split: str, cfg: dict, pool: Pool):
    jobs = [(sc, f, cfg, split) for sc, f in scene_files(split).items()]
    streams, rows, secs, n_sess = {}, [], 0.0, 0
    for scene, stream, r, dt, n in pool.imap_unordered(_map_job, jobs):
        streams[scene] = stream
        rows += r
        secs += dt
        n_sess += n
    return streams, rows, secs / max(n_sess, 1)


def assoc_accuracy(rows: List[dict]) -> float:
    """Mean over scenes of the share of tracks (k >= 1) handled correctly:
    a previously mapped object matched to its own map object, or a novel
    object left unmatched."""
    per = {}
    for r in rows:
        ok = (r["matched_gt_id"] == r["gt_id"] if r["gt_mapped_before"]
              else r["matched_oid"] is None)
        per.setdefault(r["scene"], []).append(bool(ok))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else 0.0


# ── Persistence models ───────────────────────────────────────────────────────

_G: Dict = {}


def _init_worker(streams, lam, lam0, learned):
    _G.update(streams=streams, lam=lam, lam0=lam0, learned=learned)


def make_model(family: str, params: dict):
    lam, lam0 = _G["lam"], _G["lam0"]
    p = dict(params)
    if family == "hpf":
        return HierarchicalPersistenceFilter(lambda_table=lam,
                                             default_lambda=lam0, **p)
    if family == "hpf_noprior":
        days = p.pop("lifetime_days")
        return HierarchicalPersistenceFilter(
            default_lambda=1.0 / (days * DAY), use_class_prior=False, **p)
    if family == "pf_class":
        return HierarchicalPersistenceFilter(
            lambda_table=lam, default_lambda=lam0, learn_rate=False, **p)
    if family in ("hpf_scaled", "pf_class_scaled"):
        # hand table with every lifetime multiplied by one scalar (val-tuned)
        sc = p.pop("scale")
        tab = {k: v / sc for k, v in lam.items()}
        return HierarchicalPersistenceFilter(
            lambda_table=tab, default_lambda=lam0 / sc,
            learn_rate=(family == "hpf_scaled"), **p)
    if family == "pf":
        days = p.pop("lifetime_days")
        return HierarchicalPersistenceFilter(
            default_lambda=1.0 / (days * DAY), learn_rate=False,
            use_class_prior=False, **p)
    if family == "class_rate":
        tab, l0 = _G["learned"]
        return ClassRateModel(tab, l0, **p)
    if family == "hpf_learned":       # ours with the learned class prior
        tab, l0 = _G["learned"]
        return HierarchicalPersistenceFilter(lambda_table=tab,
                                             default_lambda=l0, **p)
    if family == "elite":
        return ELiteScalarModel(**p)
    if family == "pocd":
        return POCDBetaModel(lambda_table=lam, default_lambda=lam0, **p)
    if family == "draft_v1":
        return DraftV1Model(lam, lam0, **p)
    if family == "always":
        return AlwaysPersistModel()
    raise ValueError(family)


def _replay_job(args):
    family, params, name = args
    rows = []
    for scene in sorted(_G["streams"]):
        rows += replay(_G["streams"][scene], make_model(family, params),
                       name or family)
    return family, params, rows


def _scores(rows: List[dict]) -> dict:
    per = pm.per_scene_metrics(rows)
    (model, by_scene), = per.items()
    nll = float(np.nanmean([m["nll"] for m in by_scene.values()]))
    mt = pm.maintenance_metrics(rows)
    prem = mt["premature_prune_rate"]
    prem = 0.0 if not math.isfinite(prem) else prem
    stale = mt["stale_rate"]
    stale = 0.0 if not math.isfinite(stale) else stale
    return {"nll": nll, "decision": 0.5 * (prem + stale),
            "premature_prune_rate": prem, "stale_rate": stale}


def grid(d: dict) -> List[dict]:
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*d.values())]


def tune_family(pool, family, pred_grid, dec_grid, log):
    """Prediction params by val NLL, then thresholds by the decision score."""
    best_p, best_nll = None, INF
    for fam, params, rows in pool.imap_unordered(
            _replay_job, [(family, g, None) for g in pred_grid]):
        s = _scores(rows)
        log.append({"family": fam, "stage": "pred", "params": params, **s})
        if s["nll"] < best_nll:
            best_p, best_nll = params, s["nll"]
    best, best_dec = dict(best_p), INF
    for fam, params, rows in pool.imap_unordered(
            _replay_job, [(family, {**best_p, **g}, None) for g in dec_grid]):
        s = _scores(rows)
        log.append({"family": fam, "stage": "decision", "params": params, **s})
        if s["decision"] < best_dec:
            best, best_dec = params, s["decision"]
    return best, {"val_nll": best_nll, "val_decision": best_dec}


# ── Commands ─────────────────────────────────────────────────────────────────

def learned_rates() -> tuple:
    """Class rates learned on the TRAIN split, keyed by the label the map
    sees at run time: each GT object's majority CLIP label over its tracks
    (objects never tracked never enter a map and are left out). Absorbs
    the zero-shot label noise into the rates (empirical Bayes)."""
    from collections import Counter
    hist = []
    files_by_scene = scene_files("train")
    if not files_by_scene:
        print("WARNING: no train sessions; class_rate uses the hand table")
        return load_class_lambdas(str(PRIORS))
    for files in files_by_scene.values():
        ss = [load_session(f) for f in files]
        votes: Dict[str, Counter] = {}
        for s in ss:
            for t in s.tracks:
                votes.setdefault(t.gt_id, Counter())[t.clip_label] += 1
        major = {g: c.most_common(1)[0][0] for g, c in votes.items()}
        hist.append([(s.time, [{**o, "type": major[o["gt_id"]]}
                               for o in s.gt_objects if o["gt_id"] in major])
                     for s in ss])
    return learn_class_rates(hist)


def cmd_tune(args):
    out = RESULTS / args.run
    (out / "tuning").mkdir(parents=True, exist_ok=True)
    lam, lam0 = load_class_lambdas(str(PRIORS))
    have_train = bool(scene_files("train"))
    learned = learned_rates() if have_train else ({}, lam0)
    t_start = time.time()
    with Pool(args.workers) as pool:
        # 1. Association on val, two stages: (a) appearance x geometry
        # with gamma off; (b) gamma scales for the best CLIP-based configs
        # (gamma scales appearance only, so it is moot for 'none').
        log = []

        def run(cfg):
            _, rows, _ = map_split("val", cfg, pool)
            log.append({**cfg, "val_assoc_accuracy": assoc_accuracy(rows)})
            print("assoc", cfg, round(log[-1]["val_assoc_accuracy"], 4),
                  flush=True)

        stage_a = {**ASSOC_GRID, "sigma_u": [None]}
        for cfg in grid(stage_a):
            run(cfg)
        for mode in ("clip", "clip+dino"):
            cand = [r for r in log if r["appearance"] == mode]
            if not cand:
                continue
            b = max(cand, key=lambda r: r["val_assoc_accuracy"])
            for su in ASSOC_GRID["sigma_u"]:
                if su is not None:
                    run({k: b[k] for k in ASSOC_GRID if k != "sigma_u"}
                        | {"sigma_u": su})
        best = max(log, key=lambda r: r["val_assoc_accuracy"])
        best_a = {k: best[k] for k in ASSOC_GRID}
        best_acc = best["val_assoc_accuracy"]
        write_jsonl(log, str(out / "tuning/assoc_grid.jsonl"))
        print("assoc", best_a, round(best_acc, 4), flush=True)
    # E3 baselines: best plain config (no gamma) per appearance mode
    assoc_e3 = {}
    for mode in ASSOC_GRID["appearance"]:
        cand = [r for r in log if r["appearance"] == mode
                and r["sigma_u"] is None]
        b = max(cand, key=lambda r: r["val_assoc_accuracy"])
        assoc_e3[f"e3_{mode}"] = {k: b[k] for k in ASSOC_GRID}
    # 2. HPF prediction params on the selected association, tau_u = inf
    plog = []
    streams, _, _ = map_split_serial("val", best_a, args.workers)
    with Pool(args.workers, _init_worker,
              (streams, lam, lam0, learned)) as pool:
        hpf_pred = [{**d, "kappa0": k} for d in grid(DET_GRID)
                    for k in KAPPA_GRID]
        hpf0, _ = tune_family(pool, "hpf", hpf_pred, [{}], plog)
    # 3. tau_u by the full model's val NLL
    best_tu, best_nll = INF, INF
    for tu in TAU_U_GRID:
        cfg = {**best_a, "tau_u": tu}
        st, _, _ = map_split_serial("val", cfg, args.workers)
        with Pool(args.workers, _init_worker, (st, lam, lam0, learned)) as p:
            _, _, rows = _replay_job_in(p, "hpf", hpf0)
        s = _scores(rows)
        plog.append({"family": "assoc_tau_u", "tau_u": tu, **s})
        if s["nll"] < best_nll:
            best_tu, best_nll = tu, s["nll"]
    assoc_sel = {**best_a, "tau_u": best_tu}
    # 4. Every persistence family on the final val streams
    streams, _, _ = map_split_serial("val", assoc_sel, args.workers)
    sel = {"assoc": assoc_sel, "val_assoc_accuracy": best_acc,
           "assoc_e3": assoc_e3}
    det = grid(DET_GRID)
    fams = {
        "hpf": ([{**d, "kappa0": k} for d in det for k in KAPPA_GRID],
                [{"tau_pi": a, "tau_v": b} for a in TAU_PI_GRID
                 for b in TAU_V_GRID]),
        "pf_class": (det, [{"tau_pi": a} for a in TAU_PI_GRID]),
        "pf": ([{**d, "lifetime_days": L} for d in det
                for L in LIFETIME_GRID_DAYS],
               [{"tau_pi": a} for a in TAU_PI_GRID]),
        "elite": (grid(ELITE_GRID), [{"tau_e": a} for a in ELITE_TAU]),
        "pocd": (grid(POCD_GRID), [{"tau_remove": a} for a in POCD_TAU]),
    }
    with Pool(args.workers, _init_worker,
              (streams, lam, lam0, learned)) as pool:
        for fam, (pg, dg) in fams.items():
            best, sc = tune_family(pool, fam, pg, dg, plog)
            sel[fam] = {"params": best, **sc}
            print(fam, best, sc, flush=True)
        # ablation: no class prior, detection params of the full model
        hp = sel["hpf"]["params"]
        keep = {k: hp[k] for k in ("p_miss", "p_false", "kappa0", "tau_pi",
                                   "tau_v")}
        best, sc = tune_family(
            pool, "hpf_noprior",
            [{**keep, "lifetime_days": L} for L in LIFETIME_GRID_DAYS],
            [{}], plog)
        sel["hpf_noprior"] = {"params": best, **sc}
    write_jsonl(plog, str(out / "tuning/persistence_grid.jsonl"))
    sel["seconds"] = round(time.time() - t_start, 1)
    with open(out / "selected.json", "w") as f:
        json.dump(sel, f, indent=1)
    if have_train:
        cmd_tune_classrate(args)
    else:
        print("class_rate pending: run 'tune_classrate' once train exists")


def cmd_tune_classrate(args):
    """Tune the learned-class-rate baseline (needs the train split) on the
    val streams of the selected association; merge into selected.json."""
    out = RESULTS / args.run
    sel = json.load(open(out / "selected.json"))
    lam, lam0 = load_class_lambdas(str(PRIORS))
    learned = learned_rates()
    streams, _, _ = map_split_serial("val", sel["assoc"], args.workers)
    plog = []
    with Pool(args.workers, _init_worker,
              (streams, lam, lam0, learned)) as pool:
        best, sc = tune_family(pool, "class_rate", grid(DET_GRID),
                               [{"tau_pi": a} for a in TAU_PI_GRID], plog)
    sel["class_rate"] = {"params": best, **sc}
    sel["learned_rates"] = {"lambda0": learned[1],
                            "n_keys": len(learned[0]),
                            "train_scenes": len(scene_files("train"))}
    write_jsonl(plog, str(out / "tuning/class_rate_grid.jsonl"))
    with open(out / "selected.json", "w") as f:
        json.dump(sel, f, indent=1)
    print("class_rate", best, sc)


KAPPA_FINAL = [1, 2, 4, 8, 16, 32, 64]


def final_families(det, taus):
    """Every family re-tuned on the final val streams (after attempts 02-07:
    the method is the learned CLIP-keyed class prior + per-object rate
    learning, no variance gate; association without gamma / tau_u)."""
    ours = [{**d, "kappa0": k, "variance_gate": False}
            for d in det for k in KAPPA_FINAL]
    return {
        "ours": ("hpf_learned", ours, taus),
        "pf_learned": ("class_rate", det, taus),
        "pf": ("pf", [{**d, "lifetime_days": L} for d in det
                      for L in LIFETIME_GRID_DAYS], taus),
        "pf_class": ("pf_class", det, taus),
        "ours_hand_prior": ("hpf", ours, taus),
        "elite": ("elite", grid(ELITE_GRID),
                  [{"tau_e": a} for a in ELITE_TAU]),
        "pocd": ("pocd", grid(POCD_GRID),
                 [{"tau_remove": a} for a in POCD_TAU]),
    }


def cmd_tune_final(args):
    out = RESULTS / args.run
    sel = json.load(open(out / "selected.json"))
    a = {**sel["assoc"], "sigma_u": None, "tau_u": INF}
    lam, lam0 = load_class_lambdas(str(PRIORS))
    learned = learned_rates()
    streams, _, _ = map_split_serial("val", a, args.workers)
    taus = [{"tau_pi": t} for t in TAU_PI_GRID]
    final = {"assoc": a, "assoc_e3": sel["assoc_e3"], "families": {},
             "learned_rates": {"lambda0_days": 1.0 / learned[1] / DAY,
                               "n_keys": len(learned[0])}}
    plog = []
    with Pool(args.workers, _init_worker,
              (streams, lam, lam0, learned)) as pool:
        for name, (fam, pg, dg) in final_families(grid(DET_GRID),
                                                  taus).items():
            best, sc = tune_family(pool, fam, pg, dg, plog)
            final["families"][name] = {"family": fam, "params": best, **sc}
            print(name, best, sc, flush=True)
    write_jsonl(plog, str(out / "tuning/final_grid.jsonl"))
    with open(out / "final.json", "w") as f:
        json.dump(final, f, indent=1)


def map_split_serial(split, cfg, workers):
    with Pool(workers) as pool:
        return map_split(split, cfg, pool)


def _replay_job_in(pool, family, params):
    return pool.apply(_replay_job, ((family, params, None),))


def gt_labelled(streams: dict) -> dict:
    """Diagnostic copy of the streams in which every object's class label is
    the ground-truth type of the object that created it (EVAL ONLY)."""
    import copy
    out = {}
    for sc, st in streams.items():
        st2 = copy.deepcopy(st)
        for ob in st2["objects"].values():
            ev = ob["events"][0]
            g = (st2["gt"][ev["k"]].get(ev["gt_id"]) or {}) \
                if ev.get("gt_id") else {}
            ob["class_label"] = g.get("type") or ob["class_label"]
        out[sc] = st2
    return out


def sign_flip_p(a: Dict[str, float], b: Dict[str, float]) -> dict:
    """Exact two-sided paired sign-flip test over scenes (all 2^n sign
    patterns); for n = 5 the smallest attainable p is 2/32 = 0.0625."""
    d = np.array([a[s] - b[s] for s in sorted(a)
                  if np.isfinite(a[s]) and np.isfinite(b[s])])
    n = len(d)
    if n == 0:
        return {"diff": math.nan, "p_value": math.nan, "n_scenes": 0}
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    null = np.abs((signs * d).mean(axis=1))
    return {"diff": float(d.mean()),
            "p_value": float(np.mean(null >= obs - 1e-12)), "n_scenes": n}


def eval_split(split: str, fin: dict, out: Path, workers: int,
               lam, lam0, learned) -> None:
    """E2/E4 rows, E3 rows, closed loop and summaries for one split, with the
    method frozen in final.json (val only)."""
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    a = fin["assoc"]
    fam = fin["families"]
    streams, arows, sec = map_split_serial(split, {**a, "name": "ours"},
                                           workers)
    timing["mapping_s_per_session"] = sec
    op = fam["ours"]["params"]
    det = {k: op[k] for k in ("p_miss", "p_false", "tau_pi")}
    models = {n: (v["family"], v["params"]) for n, v in fam.items()}
    models.update({
        "draft_v1": ("draft_v1", {}),
        "always": ("always", {}),
        # pure ablations of the final model (no re-tuning)
        "abl_no_rate_learning": ("class_rate", det),
        "abl_no_class_prior": ("hpf_noprior", {
            **op, "lifetime_days": fin["learned_rates"]["lambda0_days"]}),
        "abl_hand_prior": ("hpf", op),
        "abl_reset_rate_on_relocation": ("hpf_learned", {
            **op, "relocation_resets_rate": True}),
    })
    rows = []
    with Pool(workers, _init_worker, (streams, lam, lam0, learned)) as pool:
        jobs = [(f, p, name) for name, (f, p) in models.items()]
        t0 = time.perf_counter()
        for _, _, r in pool.imap_unordered(_replay_job, jobs):
            rows += r
        timing["replay_all_models_s"] = time.perf_counter() - t0
    # diagnostic: hand-set table with ground-truth labels (separates label
    # noise from table error)
    with Pool(workers, _init_worker,
              (gt_labelled(streams), lam, lam0, learned)) as pool:
        for fam_, p_, name in (("hpf", op, "diag_hand_prior_gt_labels"),
                               ("pf_class", det, "diag_pf_hand_gt_labels")):
            _, _, r = pool.apply(_replay_job, ((fam_, p_, name),))
            rows += r
    write_jsonl(rows, str(out / "pred.jsonl"))
    summary = pm.summarize(rows, split=split)
    per = pm.per_scene_metrics(rows)
    paired = {}
    for m in models:
        if m == "ours":
            continue
        paired[m] = {met: pm.paired_bootstrap(
            {s_: v[met] for s_, v in per["ours"].items()},
            {s_: v[met] for s_, v in per[m].items()})
            for met in ("nll", "brier", "ece", "auroc", "ap_absent",
                        "premature_prune_rate", "stale_rate")}
    for m in ("ours", "pf_learned", "pf", "pf_class", "elite", "pocd",
              "draft_v1"):
        p_, y_ = pm.labeled_arrays([r for r in rows if r["model"] == m])
        pm.write_reliability_csv(pm.reliability_bins(p_, y_),
                                 str(out / f"reliability_{m}.csv"))
    # E3 (appearance baselines) + association ablations of the final model
    assoc_rows = list(arows)
    abl = {"abl_no_pose_inflation": {**a, "pose_inflation": False},
           "abl_clip_appearance": {**a, "appearance": "clip"},
           **{k: {**v, "tau_u": INF} for k, v in fin["assoc_e3"].items()}}
    abl_rows = []
    for name, cfg in abl.items():
        st, ar, _ = map_split_serial(split, {**cfg, "name": name}, workers)
        assoc_rows += ar
        with Pool(workers, _init_worker, (st, lam, lam0, learned)) as pool:
            _, _, r = _replay_job_in(pool, "hpf_learned", op)
        for x in r:
            x["model"] = name
        abl_rows += r
    write_jsonl(assoc_rows, str(out / "assoc.jsonl"))
    write_jsonl(abl_rows, str(out / "pred_assoc_ablations.jsonl"))
    summ_assoc = pm.summarize_association(assoc_rows, split=split)
    summ_abl = pm.summarize(abl_rows + [r for r in rows
                                        if r["model"] == "ours"], split=split)
    # Closed loop: map size with the model in the loop
    mapsize = []
    _init_worker({}, lam, lam0, learned)
    for name in ("ours", "pf_learned", "pf", "elite", "pocd", "draft_v1",
                 "always"):
        f_, p_ = models[name]
        for scene, files in scene_files(split).items():
            sessions = [load_session(f) for f in files]
            res = run_closed_loop(sessions, assoc_config(a),
                                  make_model(f_, p_),
                                  bye=appearance_inputs(sessions, a),
                                  split=split, model_name=name)
            mapsize += res["map_size"]
    write_jsonl(mapsize, str(out / "closed_mapsize.jsonl"))
    with open(out / "summary.json", "w") as f:
        json.dump({"split": split, "persistence": summary,
                   "paired_vs_ours": paired, "association": summ_assoc,
                   "assoc_ablations_persistence": summ_abl,
                   "timing": timing}, f, indent=1, default=float)
    print(split, "done", flush=True)


def rates_from_sessions(session_lists) -> tuple:
    """learned_rates() for arbitrary scenes: CLIP-majority-keyed class rates
    from the GT histories of the given session lists."""
    from collections import Counter
    hist = []
    for ss in session_lists:
        votes: Dict[str, Counter] = {}
        for s in ss:
            for t in s.tracks:
                votes.setdefault(t.gt_id, Counter())[t.clip_label] += 1
        major = {g: c.most_common(1)[0][0] for g, c in votes.items()}
        hist.append([(s.time, [{**o, "type": major[o["gt_id"]]}
                               for o in s.gt_objects if o["gt_id"] in major])
                     for s in ss])
    return learn_class_rates(hist)


def cmd_loro(args):
    """ObChange leave-one-room-out (design.md change log, 2026-10-19)."""
    res = RESULTS / args.run
    out = res / "obchange_loro"
    out.mkdir(parents=True, exist_ok=True)
    fin = json.load(open(res / "final.json"))
    a = {**fin["assoc"], "name": "ours"}
    lam, lam0 = load_class_lambdas(str(PRIORS))
    rooms = scene_files("obchange")
    sess = {r: [load_session(f) for f in fl] for r, fl in rooms.items()}
    streams = {r: run_mapping(ss, assoc_config(a),
                              bye=appearance_inputs(ss, a),
                              split="obchange")[0]
               for r, ss in sess.items()}
    det = grid(DET_GRID)
    taus = [{"tau_pi": t} for t in TAU_PI_GRID]
    rows, chosen, plog = [], {}, []
    for r in sorted(rooms):
        train = [x for x in rooms if x != r]
        learned = rates_from_sessions([sess[x] for x in train])
        chosen[r] = {}
        with Pool(args.workers, _init_worker,
                  ({x: streams[x] for x in train}, lam, lam0,
                   learned)) as pool:
            for name, (fam, pg, dg) in final_families(det, taus).items():
                best, sc = tune_family(pool, fam, pg, dg, plog)
                chosen[r][name] = {"family": fam, "params": best, **sc}
        _init_worker({r: streams[r]}, lam, lam0, learned)
        for name, v in list(chosen[r].items()) + [
                ("draft_v1", {"family": "draft_v1", "params": {}}),
                ("always", {"family": "always", "params": {}})]:
            rows += replay(streams[r], make_model(v["family"], v["params"]),
                           name)
        print("held-out", r, {k: round(v["val_nll"], 3)
                              for k, v in chosen[r].items()}, flush=True)
    write_jsonl(rows, str(out / "pred.jsonl"))
    write_jsonl(plog, str(out / "tuning_grid.jsonl"))
    summary = pm.summarize(rows, split="obchange")
    per = pm.per_scene_metrics(rows)
    paired = {m: {met: pm.paired_bootstrap(
        {s_: v[met] for s_, v in per["ours"].items()},
        {s_: v[met] for s_, v in per[m].items()})
        for met in ("nll", "brier", "ece", "auroc")}
        for m in per if m != "ours"}
    flips = {m: {met: sign_flip_p(
        {s_: v[met] for s_, v in per["ours"].items()},
        {s_: v[met] for s_, v in per[m].items()})
        for met in ("nll", "brier", "ece", "auroc")}
        for m in per if m != "ours"}
    with open(out / "summary.json", "w") as f:
        json.dump({"split": "obchange_loro", "persistence": summary,
                   "paired_vs_ours": paired, "signflip_vs_ours": flips,
                   "chosen": chosen}, f, indent=1, default=float)
    print("loro done", flush=True)


def cmd_test(args):
    out = RESULTS / args.run
    sel = json.load(open(out / "final.json"))
    lam, lam0 = load_class_lambdas(str(PRIORS))
    learned = learned_rates()
    for split in args.splits:
        eval_split(split, sel, out / split, args.workers, lam, lam0, learned)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["tune", "tune_classrate", "tune_final",
                                    "test", "loro"])
    ap.add_argument("--run", default="main")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=["test", "obchange"])
    ap.add_argument("--smoke", action="store_true",
                    help="2 scenes, tiny grids: bug finding only")
    args = ap.parse_args()
    if args.smoke:
        SMOKE["on"] = True
        _smoke_grids()
        args.run = "smoke_" + args.run
    {"tune": cmd_tune, "tune_classrate": cmd_tune_classrate,
     "tune_final": cmd_tune_final, "test": cmd_test,
     "loro": cmd_loro}[args.cmd](args)


if __name__ == "__main__":
    main()
