"""
hiper.evaluation.lifelong_objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offline lifelong object mapper and persistence replay (design.md,
'Association v2', 'Evaluation', 'Row schemas').

run_mapping      open loop: associate each session with association v2 and
                 turn every mapped object's history into an event stream.
                 Nothing is pruned, so every persistence model replays the
                 SAME stream.
replay           feed one stream to a PersistenceModel; one prediction row
                 per map object per session.
run_closed_loop  the model's prune decisions are applied: pruned objects
                 leave the map and cannot be matched later.

Events of session k, for every map object j:
  matched, not tentative, d2 <= chi2  -> HIT       fuse position/cov/embeddings
  matched, not tentative, d2 >  chi2  -> RELOCATE  reset position to the track
  matched, tentative                  -> NONE      no belief/embedding update
  unmatched, mapped position in view  -> MISS
  unmatched, out of view              -> NONE
and every unmatched track -> CREATE.

The mapper never reads ground truth for its decisions. A map object only
carries the gt_id of the track that last set its position (CREATE, HIT or
RELOCATE), which the labels need.
"""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hiper.ephemerality.persistence_v2 import (
    HIT,
    MISS,
    NONE,
    PersistenceModel,
)
from hiper.ephemerality.priors import geo_features
from hiper.evaluation.object_session import SessionData, Track
from hiper.object_encoder.association_v2 import (
    AssocConfig,
    AssocSet,
    Match,
    associate,
)

CREATE, RELOCATE = "create", "relocate"

# Per-session BYE embeddings: session_index -> {track_id: vector}
ByeInput = Optional[Dict[int, Dict[int, np.ndarray]]]


def _unit(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    return None if n < 1e-8 else np.asarray(v, dtype=np.float64) / n


def _stack(rows: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Stack optional vectors; a missing one becomes a zero row."""
    dim = next((np.size(r) for r in rows if r is not None), None)
    if dim is None:
        return None
    out = np.zeros((len(rows), dim))
    for i, r in enumerate(rows):
        if r is not None:
            out[i] = np.asarray(r, dtype=np.float64).ravel()
    return out


def _pose_cov(sess: SessionData) -> np.ndarray:
    P = np.asarray(sess.pose_cov, dtype=np.float64)
    if P.shape == (6, 6):
        P = P[:3, :3]
    return P if P.shape == (3, 3) else np.zeros((3, 3))


# ── Map state ────────────────────────────────────────────────────────────────

@dataclass
class MapObject:
    oid: str
    position: np.ndarray            # (3,) estimated world frame
    cov: np.ndarray                 # (3,3) Sigma_j, fused track covariances
    class_label: str                # CLIP label at creation (model prior)
    created: int                    # session of creation
    pos_session: int                # session that last set the position
    gt_id: Optional[str] = None     # EVAL ONLY, see module docstring
    labels: Dict[str, int] = field(default_factory=dict)  # CLIP label votes
    clip_mu: Optional[np.ndarray] = None  # mean of track means m_s
    clip_within: float = 0.0        # mean within-session dispersion
    clip_m2: float = 0.0            # Welford sum of |m_s - mean|^2
    clip_n: int = 0
    bye_sum: Optional[np.ndarray] = None
    n_fused: int = 0                # sessions with a confident match

    @property
    def u(self) -> float:
        """u_j = mean within-session + between-session dispersion of the
        track means. With m_s = sqrt(1 - u_s) * clip_mean_s (the raw mean of
        unit embeddings) this equals tr(C) of all unit embeddings pooled
        with equal session weights, so it stays in [0, 1]."""
        if self.clip_n == 0:
            return 0.0
        u = self.clip_within + self.clip_m2 / self.clip_n
        return float(min(max(u, 0.0), 1.0))

    @property
    def clip(self) -> Optional[np.ndarray]:
        return _unit(self.clip_mu)

    @property
    def bye(self) -> Optional[np.ndarray]:
        return _unit(self.bye_sum)

    @property
    def majority_label(self) -> str:
        if not self.labels:
            return self.class_label
        return max(self.labels, key=self.labels.get)   # ties: first seen

    def absorb(self, tr: Track, bye: Optional[np.ndarray], k: int) -> None:
        """Fuse the appearance of a confidently matched track and take its
        identity as the one that set the position at session k."""
        self.labels[tr.clip_label] = self.labels.get(tr.clip_label, 0) + 1
        d = _unit(tr.clip_mean)
        if d is not None:
            u_s = min(max(float(tr.clip_disp), 0.0), 1.0)
            m = math.sqrt(1.0 - u_s) * d
            self.clip_n += 1
            if self.clip_mu is None:
                self.clip_mu = m.copy()
                self.clip_within, self.clip_m2 = u_s, 0.0
            else:
                delta = m - self.clip_mu
                self.clip_mu = self.clip_mu + delta / self.clip_n
                self.clip_m2 += float(delta @ (m - self.clip_mu))
                self.clip_within += (u_s - self.clip_within) / self.clip_n
        b = _unit(bye)
        if b is not None:
            self.bye_sum = b if self.bye_sum is None else self.bye_sum + b
        self.n_fused += 1
        self.gt_id = tr.gt_id
        self.pos_session = k


class ObjectMapper:
    """Map objects + association v2. Never prunes; callers use remove()."""

    def __init__(self, cfg: AssocConfig):
        self.cfg = cfg
        self.objects: Dict[str, MapObject] = {}   # creation order

    def remove(self, oid: str) -> None:
        self.objects.pop(oid, None)

    def _track_cov(self, tr: Track) -> np.ndarray:
        R = np.asarray(tr.centroid_cov, dtype=np.float64).reshape(3, 3)
        return R + (self.cfg.centroid_std ** 2 + 1e-9) * np.eye(3)

    def step(self, sess: SessionData, k: int,
             bye: Optional[Dict[int, np.ndarray]] = None
             ) -> Tuple[List[dict], List[Match], List[str]]:
        """Process session k. Returns (events, one Match per track, oids
        of the map objects that took part in the association)."""
        cfg, bye = self.cfg, bye or {}
        P = _pose_cov(sess)
        tracks = list(sess.tracks)
        oids = list(self.objects)
        objs = [self.objects[o] for o in oids]
        R = [self._track_cov(tr) for tr in tracks]
        tset = AssocSet(
            pos=np.array([tr.centroid for tr in tracks],
                         dtype=np.float64).reshape(-1, 3),
            cov=np.array(R).reshape(-1, 3, 3),
            u=np.array([min(max(float(tr.clip_disp), 0.0), 1.0)
                        for tr in tracks]),
            clip=_stack([tr.clip_mean for tr in tracks]),
            bye=_stack([bye.get(int(tr.track_id)) for tr in tracks]))
        oset = AssocSet(
            pos=np.array([o.position for o in objs]).reshape(-1, 3),
            cov=np.array([o.cov for o in objs]).reshape(-1, 3, 3),
            u=np.array([o.u for o in objs]),
            clip=_stack([o.clip for o in objs]),
            bye=_stack([o.bye for o in objs]))
        matches = associate(tset, oset, cfg, P)
        vis = (sess.visible(oset.pos) if objs
               else np.zeros(0, dtype=bool))
        # Measurement covariance of a track in the map frame
        Rm = [r + P if cfg.pose_inflation else r for r in R]

        t = float(sess.time)
        by_obj = {m.obj: m for m in matches if m.obj >= 0}
        events = []
        for j, o in enumerate(objs):
            m = by_obj.get(j)
            tr = tracks[m.track] if m is not None else None
            if m is None:
                ev = MISS if vis[j] else NONE
            elif m.tentative:
                ev = NONE
            else:
                z = np.asarray(tr.centroid, dtype=np.float64)
                if m.relocated:
                    ev = RELOCATE
                    o.position, o.cov = z.copy(), Rm[m.track].copy()
                else:
                    ev = HIT
                    Si = np.linalg.inv(o.cov)
                    Ri = np.linalg.inv(Rm[m.track])
                    C = np.linalg.inv(Si + Ri)
                    o.position = C @ (Si @ o.position + Ri @ z)
                    o.cov = 0.5 * (C + C.T)
                o.absorb(tr, bye.get(int(tr.track_id)), k)
            events.append(_event(o, k, t, ev, tr, m, bool(vis[j])))
        for m in matches:
            if m.obj >= 0:
                continue
            tr = tracks[m.track]
            oid = base = f"s{k}_t{tr.track_id}"
            n = 1
            while oid in self.objects:
                oid, n = f"{base}_{n}", n + 1
            o = MapObject(
                oid=oid, position=np.asarray(tr.centroid, np.float64).copy(),
                cov=Rm[m.track].copy(), class_label=tr.clip_label,
                created=k, pos_session=k)
            o.absorb(tr, bye.get(int(tr.track_id)), k)
            self.objects[oid] = o
            events.append(_event(o, k, t, CREATE, tr, None, None))
        return events, matches, oids


def _event(o: MapObject, k: int, t: float, ev: str, tr: Optional[Track],
           m: Optional[Match], visible: Optional[bool]) -> dict:
    matched = m is not None and m.obj >= 0
    return {
        "oid": o.oid, "k": k, "t": t, "ev": ev,
        "gt_id": o.gt_id, "pos_session": o.pos_session,
        "track_id": None if tr is None else int(tr.track_id),
        "label": None if tr is None else tr.clip_label,
        "score": m.score if matched else None,
        "d2": m.d2 if matched else None,
        "tentative": bool(matched and m.tentative),
        "visible": visible,
        "n_obs": None if tr is None else int(tr.n_obs),
        "eps_l": 0.0 if tr is None else float(tr.eps_l),
        # appearance and geometry of the creating track (retrieved prior)
        "emb": (np.asarray(tr.dino_mean, np.float32)
                if ev == CREATE and tr is not None
                and tr.dino_mean is not None else None),
        "emb_clip": (np.asarray(tr.clip_mean, np.float32)
                     if ev == CREATE and tr is not None
                     and tr.clip_mean is not None else None),
        "geo": (geo_features(tr.extent, float(np.asarray(tr.centroid)[2]),
                             tr.pixel_area)
                if ev == CREATE and tr is not None else None),
    }


# ── Ground truth (EVAL ONLY) ─────────────────────────────────────────────────

GTTables = Optional[List[Dict[str, dict]]]


def gt_tables(sessions: Sequence[SessionData]) -> GTTables:
    """Per session {gt_id: {present, moved, appeared, group}}; None when a
    session has no ground truth. ``present`` None marks an unknown state
    (3RScan: not annotated in that scan); labels spanning it are None."""
    out = []
    for s in sessions:
        if s.gt_objects is None:
            return None
        out.append({str(g["gt_id"]): {
            "present": (None if "present" in g and g["present"] is None
                        else bool(g.get("present", True))),
            "moved": bool(g.get("moved_since_prev", False)),
            "appeared": bool(g.get("appeared_since_prev", False)),
            "group": g.get("group"), "type": g.get("type")}
            for g in s.gt_objects})
    return out


def gt_labels(gt: GTTables, gt_id: Optional[str], pos_session: int,
              k: int) -> Tuple[Optional[int], Optional[int]]:
    """(gt_present_now, label_next) of a map object whose position was last
    set at ``pos_session`` from GT object ``gt_id`` (design.md row schema):
    1 iff the GT object is present at k (k+1) and has neither moved nor
    re-appeared since pos_session. label_next is None at the last session,
    and either label is None when the object's state is unknown at a
    session of the span."""
    if gt is None:
        return None, None

    def stays(s_end: int) -> Optional[int]:
        e = gt[pos_session].get(gt_id) if gt_id is not None else None
        if e and e["present"] is None:
            return None
        if not (e and e["present"]):
            return 0
        for s in range(pos_session + 1, s_end + 1):
            e = gt[s].get(gt_id)
            if e and e["present"] is None:
                return None
            if not (e and e["present"]) or e["moved"] or e["appeared"]:
                return 0
        return 1

    return stays(k), (stays(k + 1) if k + 1 < len(gt) else None)


def _gt_group(gt: GTTables, gt_id: Optional[str], k: int) -> Optional[str]:
    if gt is None or gt_id is None:
        return None
    for s in range(min(k, len(gt) - 1), -1, -1):
        if gt_id in gt[s]:
            return gt[s][gt_id]["group"]
    return None


def _gt_changes(gt: List[Dict[str, dict]], gt_id: str, since: int,
                k: int) -> Tuple[bool, bool]:
    """(moved or re-appeared, re-appeared) in sessions (since, k]."""
    moved = reapp = False
    for s in range(since + 1, k + 1):
        e = gt[s].get(gt_id) or {}
        moved |= bool(e.get("moved") or e.get("appeared"))
        reapp |= bool(e.get("appeared"))
    return moved, reapp


# ── One session of the mapping run ───────────────────────────────────────────

def _map_session(mapper: ObjectMapper, sess: SessionData, k: int,
                 gt: GTTables, bye_k: Optional[Dict[int, np.ndarray]],
                 base: dict) -> Tuple[List[dict], List[dict]]:
    """Step the mapper; return (events, association rows) of session k."""
    pre_gt = {oid: o.gt_id for oid, o in mapper.objects.items()}
    holder = {}   # gt_id -> latest pos_session among mapped objects
    for o in mapper.objects.values():
        if o.gt_id is not None:
            holder[o.gt_id] = max(holder.get(o.gt_id, -1), o.pos_session)
    events, matches, oids = mapper.step(sess, k, bye_k)
    rows = []
    if k == 0:
        return events, rows
    for m in matches:
        tr = sess.tracks[m.track]
        g = tr.gt_id
        mapped = g is not None and g in holder
        moved = reapp = None
        if mapped and gt is not None:
            moved, reapp = _gt_changes(gt, g, holder[g], k)
        oid = oids[m.obj] if m.obj >= 0 else None
        rows.append({
            **base, "k": k, "track_id": int(tr.track_id), "gt_id": g,
            "matched_oid": oid,
            "matched_gt_id": pre_gt[oid] if oid is not None else None,
            "score": m.score if oid is not None else None,
            "d2": m.d2 if oid is not None else None,
            "tentative": m.tentative, "relocated_pred": m.relocated,
            "gt_mapped_before": mapped, "gt_moved": moved,
            "gt_reappeared": reapp})
    return events, rows


def _sorted(sessions: Sequence[SessionData]) -> List[SessionData]:
    return sorted(sessions, key=lambda s: s.session_index)


# ── Open loop ────────────────────────────────────────────────────────────────

def run_mapping(sessions: Sequence[SessionData], cfg: AssocConfig,
                bye: ByeInput = None, split: str = ""
                ) -> Tuple[dict, List[dict]]:
    """Open-loop mapping of one scene. Returns (stream, association rows).
    ``bye`` maps session_index -> {track_id: BYE embedding} (optional)."""
    sessions = _sorted(sessions)
    gt = gt_tables(sessions)
    mapper = ObjectMapper(cfg)
    scene = sessions[0].scene if sessions else ""
    base = {"scene": scene, "split": split, "config": cfg.name}
    objects: Dict[str, dict] = {}
    assoc_rows: List[dict] = []
    for k, sess in enumerate(sessions):
        events, rows = _map_session(mapper, sess, k, gt,
                                    (bye or {}).get(sess.session_index),
                                    base)
        assoc_rows += rows
        for ev in events:
            oid = ev.pop("oid")
            if oid not in objects:
                objects[oid] = {
                    "class_label": mapper.objects[oid].class_label,
                    "created": k, "events": []}
            objects[oid]["events"].append(ev)
    for oid, ob in objects.items():
        ob["class_major"] = mapper.objects[oid].majority_label
    stream = {
        "scene": scene, "split": split, "config": cfg.name,
        "assoc_cfg": asdict(cfg),
        "session_index": [int(s.session_index) for s in sessions],
        "times": [float(s.time) for s in sessions],
        "gt": gt, "objects": objects,
    }
    return stream, assoc_rows


def creation_query(ev: dict):
    """Query of the retrieved prior from a CREATE event: the DINOv2
    embedding alone (older streams) or {"dino", "clip", "geo"}."""
    if ev.get("emb_clip") is None and ev.get("geo") is None:
        return ev.get("emb")
    return {"dino": ev.get("emb"), "clip": ev.get("emb_clip"),
            "geo": ev.get("geo")}


def apply_event(model: PersistenceModel, oid: str, class_label: str,
                ev: dict) -> None:
    """Feed one stream event to a persistence model."""
    e, t = ev["ev"], ev["t"]
    if e == CREATE:
        # The creating detection is itself a HIT (as in the draft runner and
        # ELite, which fuse the creation session's evidence); a no-op for the
        # v2 filter, whose presence starts at pi_new = 1 with r = 0.
        if getattr(model, "wants_embedding", False):
            model.create(oid, class_label, t, emb=creation_query(ev))
        else:
            model.create(oid, class_label, t)
        model.observe(oid, t, HIT, eps_l=ev["eps_l"], n_obs=ev["n_obs"],
                      score=None)
    elif e == RELOCATE:
        model.relocate(oid, t)
    elif e == HIT:
        model.observe(oid, t, HIT, eps_l=ev["eps_l"], n_obs=ev["n_obs"],
                      score=ev["score"])
    elif e == MISS:
        model.observe(oid, t, MISS)
    else:
        model.observe(oid, t, NONE, tentative=ev["tentative"])


def _num(x) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def _prediction_row(base: dict, model: PersistenceModel, oid: str,
                    class_label: str, ev: dict, k: int,
                    times: Sequence[float], gt: GTTables) -> dict:
    t_k = times[k]
    t_next = times[k + 1] if k + 1 < len(times) else None
    now, nxt = gt_labels(gt, ev["gt_id"], ev["pos_session"], k)
    return {
        **base, "oid": oid, "k": k, "t_k": t_k, "t_next": t_next,
        "class_label": class_label, "gt_id": ev["gt_id"],
        "gt_group": _gt_group(gt, ev["gt_id"], k),
        "pred_presence": (None if t_next is None
                          else _num(model.presence(oid, t_next))),
        "rate_mean": _num(model.rate_mean(oid)),
        "rate_var": _num(model.rate_var(oid)),
        "label_next": nxt, "gt_present_now": now,
        "pruned": bool(model.should_prune(oid, t_k)),
        "committed": bool(model.is_committed(oid, t_k)),
        "last_evidence": ev["ev"],
    }


def replay(stream: dict, model: PersistenceModel,
           model_name: Optional[str] = None) -> List[dict]:
    """Replay an open-loop stream through a FRESH model. Prune and commit
    decisions at the end of each session are recorded, not applied.
    ``last_evidence`` is the event applied at session k."""
    times = stream["times"]
    gt = stream.get("gt")
    per_k: List[list] = [[] for _ in times]
    for oid, ob in stream["objects"].items():
        for ev in ob["events"]:
            per_k[ev["k"]].append((oid, ob["class_label"], ev))
    base = {"scene": stream["scene"], "split": stream["split"],
            "model": model_name or model.name}
    rows = []
    for k, items in enumerate(per_k):
        for oid, label, ev in items:
            apply_event(model, oid, label, ev)
        for oid, label, ev in items:
            rows.append(_prediction_row(base, model, oid, label, ev, k,
                                        times, gt))
    return rows


# ── Closed loop ──────────────────────────────────────────────────────────────

def run_closed_loop(sessions: Sequence[SessionData], cfg: AssocConfig,
                    model: PersistenceModel, bye: ByeInput = None,
                    split: str = "", model_name: Optional[str] = None
                    ) -> Dict[str, List[dict]]:
    """Mapping with the model in the loop: objects the model prunes at the
    end of a session leave the map (and the model) for good. Returns
    {"pred_rows", "assoc_rows", "map_size"}."""
    sessions = _sorted(sessions)
    gt = gt_tables(sessions)
    times = [float(s.time) for s in sessions]
    mapper = ObjectMapper(cfg)
    name = model_name or model.name
    scene = sessions[0].scene if sessions else ""
    base_p = {"scene": scene, "split": split, "model": name}
    base_a = {"scene": scene, "split": split, "config": cfg.name,
              "model": name}
    out: Dict[str, List[dict]] = {"pred_rows": [], "assoc_rows": [],
                                  "map_size": []}
    for k, sess in enumerate(sessions):
        events, rows = _map_session(mapper, sess, k, gt,
                                    (bye or {}).get(sess.session_index),
                                    base_a)
        out["assoc_rows"] += rows
        for ev in events:
            apply_event(model, ev["oid"],
                        mapper.objects[ev["oid"]].class_label, ev)
        pruned = []
        for ev in events:
            oid = ev["oid"]
            row = _prediction_row(base_p, model, oid,
                                  mapper.objects[oid].class_label, ev, k,
                                  times, gt)
            out["pred_rows"].append(row)
            if row["pruned"]:
                pruned.append(oid)
        for oid in pruned:
            mapper.remove(oid)
            model.remove(oid)
        out["map_size"].append({
            **base_p, "config": cfg.name, "k": k, "t_k": times[k],
            "n_objects": len(mapper.objects), "n_pruned": len(pruned),
            "n_created": sum(ev["ev"] == CREATE for ev in events)})
    return out


# ── I/O ──────────────────────────────────────────────────────────────────────

def _jsonable(x):
    """numpy -> python, non-finite floats -> None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def save_stream(stream: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(stream), f, allow_nan=False)


def load_stream(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_jsonl(rows: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(_jsonable(r), allow_nan=False) + "\n")


def load_class_lambdas(path: str) -> Tuple[Dict[str, float], float]:
    """Read a class-prior YAML (lifetimes in days, see
    config/class_priors_indoor.yaml) into ({label: lambda_c in 1/s},
    default lambda). Each THOR type is also keyed by its CLIP vocabulary
    form ("AlarmClock" -> "alarm clock") and a compact form ("alarmclock")."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    table: Dict[str, float] = {}
    for key, days in (cfg.get("lifetimes_days") or {}).items():
        lam = 1.0 / (float(days) * 86400.0)
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                       " ", key).lower()
        for k in (key, words, words.replace(" ", "")):
            table.setdefault(k, lam)
    return table, 1.0 / (float(cfg.get("default_days", 30.0)) * 86400.0)
