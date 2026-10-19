"""
hiper.evaluation.object_session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data contract between perception adapters (AI2THOR, ObChange, OpenLORIS)
and the offline lifelong object-map runner.

One SessionData holds everything the mapper may use from one visit:
  - per-instance tracks aggregated over the session (within-session
    instance tracking is assumed, as in BYE [Huang et al., RA-L 2025]),
  - the estimated camera trajectory (possibly with injected drift) and a
    compact depth stack, used by the in-view test for mapped objects,
  - the translation covariance of the session's poses (drift model).

Fields prefixed ``gt_`` / ``cam_T_world_gt`` are for evaluation only; the
mapper must never read them.

World frame: right-handed, z-up, metres. Camera frame: OpenCV (x right,
y down, z forward). ``T_wc`` maps camera points to world: X_w = T_wc @ X_c.
For AI2THOR (Unity, left-handed, y-up) adapters map (x_u, y_u, z_u) to
(x_u, z_u, y_u), which is right-handed with z up.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Track:
    """One object instance aggregated over a session."""
    track_id: int
    centroid: np.ndarray                 # (3,) world frame (estimated poses)
    centroid_cov: np.ndarray             # (3,3) measurement covariance
    n_obs: int                           # frames in which it was segmented
    clip_mean: Optional[np.ndarray]      # (512,) L2-normalised mean CLIP crop emb.
    clip_disp: float                     # u_i = tr(C) = 1 - ||mean of unit embs||^2
    clip_label: str                      # zero-shot label (vocabulary entry)
    clip_label_conf: float               # softmax probability of clip_label
    dino_mean: Optional[np.ndarray] = None   # (d,) L2-normalised mean DINOv2 emb.
    points: Optional[np.ndarray] = None      # (N,6) xyz world + rgb in [0,1]
    extent: Optional[np.ndarray] = None      # (3,) axis-aligned extent (m)
    pixel_area: float = 0.0                  # mean mask area fraction
    eps_l: float = 0.0                       # optional local dynamic cue [0,1)
    gt_id: Optional[str] = None              # EVAL ONLY


@dataclass
class SessionData:
    scene: str
    session_index: int
    time: float                              # seconds, global clock
    tracks: List[Track]
    T_wc_est: np.ndarray                     # (F,4,4) estimated camera poses
    K: np.ndarray                            # (3,3) intrinsics of depth_stack
    depth_stack: np.ndarray                  # (F,h,w) float16 metres
    pose_cov: np.ndarray                     # (3,3) session translation cov.
    T_wc_gt: Optional[np.ndarray] = None     # EVAL ONLY
    gt_objects: Optional[List[dict]] = None  # EVAL ONLY, see below
    meta: Dict = field(default_factory=dict)
    # gt_objects entries: {"gt_id": str, "type": str, "position": [x,y,z],
    #   "present": bool, "group": "fixed"|"moveable"|"pickupable",
    #   "moved_since_prev": bool, "appeared_since_prev": bool}

    # ── In-view test for mapped objects ──────────────────────────────────────

    def visible(self, positions: np.ndarray, min_frames: int = 3,
                max_range: float = 4.0, occl_margin: float = 0.15,
                border: int = 2) -> np.ndarray:
        """
        Return a bool (M,) mask: True if a mapped position projects inside
        the image of at least ``min_frames`` frames, lies within
        ``max_range`` metres, and is not occluded (the depth at its pixel is
        no closer than the point minus ``occl_margin``). Uses the ESTIMATED
        poses, as a deployed system would.
        """
        P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        M = len(P)
        if M == 0 or len(self.T_wc_est) == 0:
            return np.zeros(M, dtype=bool)
        F, h, w = self.depth_stack.shape
        counts = np.zeros(M, dtype=np.int32)
        Ph = np.hstack([P, np.ones((M, 1))])
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        for f in range(F):
            T_cw = np.linalg.inv(self.T_wc_est[f])
            Xc = (T_cw @ Ph.T).T[:, :3]
            z = Xc[:, 2]
            ok = (z > 0.05) & (z < max_range)
            u = np.full(M, -1.0)
            v = np.full(M, -1.0)
            u[ok] = fx * Xc[ok, 0] / z[ok] + cx
            v[ok] = fy * Xc[ok, 1] / z[ok] + cy
            ok &= (u >= border) & (u < w - border) & (v >= border) & \
                (v < h - border)
            if not ok.any():
                continue
            ui = u[ok].astype(np.int64)
            vi = v[ok].astype(np.int64)
            d = self.depth_stack[f][vi, ui].astype(np.float64)
            not_occ = (d <= 0) | (d >= z[ok] - occl_margin)
            idx = np.nonzero(ok)[0][not_occ]
            counts[idx] += 1
        return counts >= min_frames


# ── Serialisation (one .npz per session + JSON metadata) ─────────────────────

_TRACK_ARRAYS = ("centroid", "centroid_cov", "clip_mean", "dino_mean",
                 "points", "extent")


def save_session(sess: SessionData, path: str) -> None:
    """Write ``path`` (.npz) holding arrays and a JSON header."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        "T_wc_est": np.asarray(sess.T_wc_est, dtype=np.float64),
        "K": np.asarray(sess.K, dtype=np.float64),
        "depth_stack": np.asarray(sess.depth_stack, dtype=np.float16),
        "pose_cov": np.asarray(sess.pose_cov, dtype=np.float64),
    }
    if sess.T_wc_gt is not None:
        arrays["T_wc_gt"] = np.asarray(sess.T_wc_gt, dtype=np.float64)
    tracks_meta = []
    for i, tr in enumerate(sess.tracks):
        d = {k: v for k, v in asdict(tr).items() if k not in _TRACK_ARRAYS}
        tracks_meta.append(d)
        for k in _TRACK_ARRAYS:
            v = getattr(tr, k)
            if v is not None:
                arrays[f"t{i}_{k}"] = np.asarray(
                    v, dtype=np.float16 if k == "points" else np.float32)
    header = {
        "scene": sess.scene, "session_index": sess.session_index,
        "time": sess.time, "tracks": tracks_meta,
        "gt_objects": sess.gt_objects, "meta": sess.meta,
    }
    arrays["header_json"] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_session(path: str) -> SessionData:
    z = np.load(path, allow_pickle=False)
    header = json.loads(bytes(z["header_json"]).decode("utf-8"))
    tracks = []
    for i, d in enumerate(header["tracks"]):
        kw = dict(d)
        for k in _TRACK_ARRAYS:
            key = f"t{i}_{k}"
            kw[k] = z[key].astype(np.float32) if key in z.files else None
        tracks.append(Track(**kw))
    return SessionData(
        scene=header["scene"], session_index=header["session_index"],
        time=header["time"], tracks=tracks,
        T_wc_est=z["T_wc_est"], K=z["K"],
        depth_stack=z["depth_stack"], pose_cov=z["pose_cov"],
        T_wc_gt=z["T_wc_gt"] if "T_wc_gt" in z.files else None,
        gt_objects=header.get("gt_objects"), meta=header.get("meta", {}),
    )
