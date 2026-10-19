#!/usr/bin/env python3
"""
scripts/run_lifelong_objects.py
===============================
THOR-Lifelong object-map runs (design.md 'Evaluation'). Per scene:

  1. open-loop mapping with association v2 -> event stream + association rows,
  2. (--closed-loop) mapping with each model in the loop, prunes applied;
then every model replays the SAME open-loop streams -> prediction rows.

Output under <out-root>/<run-name>/:
  config.json                     arguments, association config, models, git
  streams/<scene>.json            open-loop event streams
  assoc.jsonl                     association rows (open loop)
  pred_<model>.jsonl              prediction rows (open-loop replay)
  closed_pred_<model>.jsonl, closed_assoc_<model>.jsonl,
  closed_mapsize_<model>.jsonl    closed loop

Sessions: <sessions-root>/<scene>/*.npz or <sessions-root>/<split>/<scene>/
*.npz (object_session.save_session). BYE embeddings (optional): one .npz per
session under --bye-dir at the same relative path as the session file, with
arrays 'track_ids' (N,) and 'embeddings' (N, d).

Models: built-in HPF variants (hpf, hpf_novar, hpf_noprior, pf_class, pf),
or 'module:Class' imported lazily (a bare module name is looked up in
hiper.ephemerality, e.g. persistence_baselines:SomeModel). Prefix
'label=' to name a variant; constructor kwargs via --model-kwargs LABEL JSON.
The class priors (lambda_table, default_lambda) go to every constructor that
accepts them. A model with a fit(streams) method is fitted once on the
streams of --fit-run (e.g. the train split) before any replay.

Usage
  python scripts/run_lifelong_objects.py --split val --run-name val_full \
      --models hpf pf_class pf --closed-loop
  python scripts/run_lifelong_objects.py --split val --run-name val_clip \
      --assoc name=clip_only w=1.0
"""

import argparse
import copy
import importlib
import inspect
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allow running without catkin: register src/ as the hiper package
_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402  (registers hiper pkg + rospy stub)

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HierarchicalPersistenceFilter,
)
from hiper.evaluation.lifelong_objects import (  # noqa: E402
    load_class_lambdas,
    load_stream,
    replay,
    run_closed_loop,
    run_mapping,
    save_stream,
    write_jsonl,
)
from hiper.evaluation.object_session import load_session  # noqa: E402
from hiper.object_encoder.association_v2 import AssocConfig  # noqa: E402

from hiper.evaluation.paths import THOR_ROOT as DATA_ROOT  # noqa: E402


def _fp(*ranges) -> List[str]:
    return [f"FloorPlan{i}" for a, b in ranges for i in range(a, b + 1)]


# RoomR splits (design.md 'Benchmark'); bye_* = BYE-protocol scene set
_BYE = _fp((1, 2), (4, 6), (301, 305))
SPLITS = {
    "train": _fp((1, 20), (201, 220), (301, 320), (401, 420)),
    "val": _fp((21, 25), (221, 225), (321, 325), (421, 425)),
    "test": _fp((26, 30), (226, 230), (326, 330), (426, 430)),
    "bye_easy": _BYE,
    "bye_hard": _BYE,
}

HPF_VARIANTS = {
    "hpf": {},
    "hpf_novar": {"variance_gate": False},
    "hpf_noprior": {"use_class_prior": False},
    "pf_class": {"learn_rate": False},
    "pf": {"learn_rate": False, "use_class_prior": False},
}


def parse_assoc(items: Optional[List[str]]) -> AssocConfig:
    """KEY=VALUE overrides of AssocConfig."""
    cfg = AssocConfig()
    for it in items or []:
        key, sep, val = it.partition("=")
        if not sep or not hasattr(cfg, key):
            raise SystemExit(f"bad --assoc item {it!r}; keys: "
                             f"{sorted(asdict(cfg))}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            if val.lower() not in ("1", "0", "true", "false"):
                raise SystemExit(f"--assoc {key} needs 1/0/true/false")
            setattr(cfg, key, val.lower() in ("1", "true"))
        elif isinstance(cur, float):
            setattr(cfg, key, float(val))
        else:
            setattr(cfg, key, val)
    return cfg


def build_models(specs: List[str], kwargs_items: Optional[List[List[str]]],
                 lam_table: Dict[str, float], lam_default: float) -> dict:
    """label -> (prototype model, target, kwargs). Imports lazily."""
    kw_by_label = {lab: json.loads(js) for lab, js in (kwargs_items or [])}
    models = {}
    for spec in specs:
        label, sep, target = spec.partition("=")
        if not sep:
            target, label = spec, spec.split(":")[-1]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise SystemExit(f"model label {label!r}: use [A-Za-z0-9_.-]")
        if label in models:
            raise SystemExit(f"duplicate model label {label!r}")
        kw = dict(kw_by_label.get(label, {}))
        if target in HPF_VARIANTS:
            cls = HierarchicalPersistenceFilter
            kw = {**HPF_VARIANTS[target], **kw}
        else:
            mod, _, cname = target.partition(":")
            if not cname:
                raise SystemExit(f"unknown model {target!r}: use one of "
                                 f"{sorted(HPF_VARIANTS)} or module:Class")
            if "." not in mod:
                mod = "hiper.ephemerality." + mod
            cls = getattr(importlib.import_module(mod), cname)
        params = inspect.signature(cls).parameters
        if "lambda_table" in params:
            kw.setdefault("lambda_table", lam_table)
        if "default_lambda" in params:
            kw.setdefault("default_lambda", lam_default)
        models[label] = (cls(**kw), target, kw)
    return models


def find_sessions(root: Path, split: str, scene: str) -> List[Path]:
    for d in (root / scene, root / split / scene):
        files = sorted(d.glob("*.npz")) if d.is_dir() else []
        if files:
            return files
    return []


def load_bye(bye_dir: Path, root: Path, files: List[Path],
             sessions: list) -> Dict[int, Dict[int, np.ndarray]]:
    """session_index -> {track_id: BYE embedding} for sessions with a file."""
    out = {}
    for f, s in zip(files, sessions):
        p = bye_dir / f.relative_to(root)
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            out[int(s.session_index)] = {
                int(t): np.asarray(e, dtype=np.float64)
                for t, e in zip(z["track_ids"], z["embeddings"])}
    return out


def git_state() -> str:
    try:
        def git(*a):
            return subprocess.run(["git", "-C", str(_PKG_ROOT), *a],
                                  capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        dirty = git("status", "--porcelain", "--", "src", "scripts")
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if dirty
                                                      else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sessions-root", default=str(DATA_ROOT / "sessions"))
    ap.add_argument("--split", default="val",
                    help=f"row label; scene list if --scenes is not given: "
                         f"{sorted(SPLITS)}")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default=str(DATA_ROOT / "results"))
    ap.add_argument("--assoc", nargs="*", default=[], metavar="KEY=VALUE",
                    help="AssocConfig overrides, e.g. name=no_gamma "
                         "use_gamma=0")
    ap.add_argument("--bye-dir", default=None)
    ap.add_argument("--models", nargs="*", default=["hpf", "pf_class", "pf"])
    ap.add_argument("--model-kwargs", nargs=2, action="append",
                    metavar=("LABEL", "JSON"), default=[])
    ap.add_argument("--priors", default=str(
        _PKG_ROOT / "config" / "class_priors_indoor.yaml"))
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--reuse-streams", action="store_true",
                    help="replay the streams already saved in this run")
    ap.add_argument("--fit-run", default=None,
                    help="run name whose streams fit models with fit()")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip scenes without sessions instead of failing")
    args = ap.parse_args(argv)
    if args.reuse_streams and args.closed_loop:
        ap.error("--closed-loop needs sessions; drop --reuse-streams")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    cfg = parse_assoc(args.assoc)
    scenes = args.scenes or SPLITS.get(args.split)
    if not scenes:
        raise SystemExit(f"unknown split {args.split!r}; pass --scenes")
    out = Path(args.out_root) / args.run_name
    root = Path(args.sessions_root)
    lam_table, lam_default = load_class_lambdas(args.priors)
    models = build_models(args.models, args.model_kwargs, lam_table,
                          lam_default)
    if args.fit_run:
        fit_files = sorted((Path(args.out_root) / args.fit_run / "streams")
                           .glob("*.json"))
        fit_streams = [load_stream(str(p)) for p in fit_files]
        for proto, _, _ in models.values():
            if hasattr(proto, "fit"):
                proto.fit(fit_streams)

    # Resolve inputs first so a missing scene fails before any work
    if args.reuse_streams:
        inputs = {s: out / "streams" / f"{s}.json" for s in scenes}
        inputs = {s: p for s, p in inputs.items() if p.exists()}
    else:
        inputs = {s: find_sessions(root, args.split, s) for s in scenes}
        inputs = {s: f for s, f in inputs.items() if f}
    missing = [s for s in scenes if s not in inputs]
    if missing and not args.allow_missing:
        raise SystemExit(f"no input for {len(missing)} scene(s): {missing} "
                         f"(use --allow-missing to skip)")

    streams, assoc_rows, per_scene = [], [], {}
    closed = {lab: {"pred_rows": [], "assoc_rows": [], "map_size": []}
              for lab in models} if args.closed_loop else {}
    for scene, src in inputs.items():
        t0 = time.time()
        if args.reuse_streams:
            streams.append(load_stream(str(src)))
            continue
        sessions = [load_session(str(f)) for f in src]
        bye = (load_bye(Path(args.bye_dir), root, src, sessions)
               if args.bye_dir else None)
        stream, rows = run_mapping(sessions, cfg, bye=bye, split=args.split)
        save_stream(stream, str(out / "streams" / f"{scene}.json"))
        streams.append(stream)
        assoc_rows += rows
        for lab, (proto, _, _) in models.items() if args.closed_loop else ():
            res = run_closed_loop(sessions, cfg, copy.deepcopy(proto),
                                  bye=bye, split=args.split, model_name=lab)
            for key in closed[lab]:
                closed[lab][key] += res[key]
        events = Counter(ev["ev"] for ob in stream["objects"].values()
                         for ev in ob["events"])
        per_scene[scene] = {"n_sessions": len(sessions),
                            "n_objects": len(stream["objects"]),
                            "events": dict(events),
                            "bye_sessions": len(bye or {}),
                            "seconds": round(time.time() - t0, 2)}
        print(f"{scene}: {per_scene[scene]}", flush=True)

    n_rows = {}
    if not args.reuse_streams:
        write_jsonl(assoc_rows, str(out / "assoc.jsonl"))
        n_rows["assoc"] = len(assoc_rows)
    for lab, (proto, _, _) in models.items():
        rows = []
        for st in streams:
            rows += replay(st, copy.deepcopy(proto), model_name=lab)
        write_jsonl(rows, str(out / f"pred_{lab}.jsonl"))
        n_rows[f"pred_{lab}"] = len(rows)
        for key, prefix in (("pred_rows", "closed_pred"),
                            ("assoc_rows", "closed_assoc"),
                            ("map_size", "closed_mapsize")):
            if lab in closed:
                write_jsonl(closed[lab][key], str(out / f"{prefix}_{lab}.jsonl"))
                n_rows[f"{prefix}_{lab}"] = len(closed[lab][key])

    config = {
        "argv": sys.argv if argv is None else list(argv),
        "args": vars(args), "assoc": asdict(cfg),
        "models": {lab: {"target": tgt, "kwargs": {
            k: (f"<from {args.priors}>" if k == "lambda_table" else v)
            for k, v in kw.items()}} for lab, (_, tgt, kw) in models.items()},
        "scenes": list(inputs), "missing": missing, "per_scene": per_scene,
        "n_rows": n_rows, "git": git_state(), "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t_start, 2),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"wrote {out} rows={n_rows} in {config['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
