"""
hiper.evaluation.openloris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reader for the OpenLORIS-Scene lifelong SLAM dataset
(https://lifelong-robotic-vision.github.io/dataset/scene).

OpenLORIS-Scene distributes each sequence either as a ROS bag or as an
extracted "package" directory in TUM RGB-D style:

    office1-1/
        color.txt            # "timestamp filename" index
        color/               #   848x480 RGB images (D435i)
        aligned_depth.txt
        aligned_depth/       #   depth aligned to color, uint16 mm
        depth.txt
        depth/
        groundtruth.txt      # "t tx ty tz qx qy qz qw"  (TUM format)
        odom.txt             # wheel odometry: pose (+ optional twist)
        sensors.yaml         # intrinsics / extrinsics (optional here)

Scenes: office (7 seqs), corridor (5), home (5), cafe (2), market (3).
Within one scene, the numbered sequences were recorded days-to-months
apart — they are the lifelong *sessions* this module exposes.

The reader is dependency-light: pure numpy for the index/trajectory
files; images are loaded lazily through cv2 only when requested.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Low-level file parsing ────────────────────────────────────────────────────

def read_file_index(path: str) -> List[Tuple[float, str]]:
    """Parse a 'timestamp filename' index file (e.g. color.txt)."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1]))
    entries.sort(key=lambda e: e[0])
    return entries


def read_trajectory(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a TUM-format trajectory file: t tx ty tz qx qy qz qw [...].
    Returns (t (N,), xyz (N,3), quat_xyzw (N,4)). Extra columns (e.g. the
    twist in odom.txt) are ignored here — see read_odometry for those.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts[:8]])
    if not rows:
        return (np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4)))
    arr = np.array(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def read_odometry(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse odom.txt. Returns (t (N,), xyz (N,3), speed (N,)).
    If the file carries a twist (>= 11 columns: pose + linear velocity),
    speed is the linear-velocity norm; otherwise it is computed by finite
    differences of the positions.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts])
    if not rows:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0)

    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    t = arr[:, 0]
    xyz = arr[:, 1:4]

    if width >= 11:
        speed = np.linalg.norm(arr[:, 8:11], axis=1)
    else:
        speed = np.zeros(len(t))
        if len(t) > 1:
            dt = np.diff(t)
            dt[dt <= 0] = 1e-6
            speed[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
            speed[0] = speed[1]
    return t, xyz, speed


def read_intrinsics(seq_dir: str) -> Optional[Dict[str, float]]:
    """
    Best-effort extraction of the color-camera intrinsics from
    sensors.yaml. Returns {"fx", "fy", "cx", "cy"} or None.
    Tolerant of format variations: searches any mapping whose key
    mentions 'color' for a 3x3 camera/intrinsic matrix.
    """
    path = os.path.join(seq_dir, "sensors.yaml")
    if not os.path.exists(path):
        return None
    try:
        import yaml
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
    except Exception:
        return None

    def _find(node, in_color: bool):
        if isinstance(node, dict):
            for k, v in node.items():
                key_color = in_color or ("color" in str(k).lower())
                if str(k).lower() in ("camera_matrix", "intrinsics", "k") \
                        and key_color:
                    data = v.get("data", v) if isinstance(v, dict) else v
                    if isinstance(data, (list, tuple)) and len(data) >= 9:
                        return {"fx": float(data[0]), "fy": float(data[4]),
                                "cx": float(data[2]), "cy": float(data[5])}
                found = _find(v, key_color)
                if found:
                    return found
        elif isinstance(node, (list, tuple)):
            for item in node:
                found = _find(item, in_color)
                if found:
                    return found
        return None

    return _find(doc, False)


# ── Sequence / session model ──────────────────────────────────────────────────

@dataclass
class OpenLorisSequence:
    """One OpenLORIS-Scene sequence (= one lifelong session)."""
    seq_dir: str
    name: str = ""
    frames: List[dict] = field(default_factory=list)   # per-keyframe dicts
    gt_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gt_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    gt_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    odom_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    odom_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    intrinsics: Optional[Dict[str, float]] = None


def load_sequence(
    seq_dir: str,
    depth_index: str = "aligned_depth.txt",
    max_dt: float = 0.05,
    stride: int = 1,
) -> OpenLorisSequence:
    """
    Index one sequence directory. Builds a frame list of dicts:
        {"stamp", "color_path", "depth_path", "velocity"}
    Depth frames are matched to color by nearest timestamp (<= max_dt);
    velocity comes from the wheel odometry twist (or finite differences).
    Images are NOT loaded here — see load_frame_images().
    """
    from hiper.evaluation.trajectory_metrics import associate_timestamps

    seq = OpenLorisSequence(seq_dir=seq_dir,
                            name=os.path.basename(os.path.normpath(seq_dir)))

    color_index_path = os.path.join(seq_dir, "color.txt")
    if not os.path.exists(color_index_path):
        raise FileNotFoundError(
            f"{color_index_path} not found — is this an extracted "
            f"OpenLORIS-Scene 'package' directory? (See "
            f"scripts/openloris_to_sessions.py for bag extraction.)")
    color = read_file_index(color_index_path)

    depth_index_path = os.path.join(seq_dir, depth_index)
    if not os.path.exists(depth_index_path):
        depth_index_path = os.path.join(seq_dir, "depth.txt")
    depth = (read_file_index(depth_index_path)
             if os.path.exists(depth_index_path) else [])

    gt_path = os.path.join(seq_dir, "groundtruth.txt")
    if os.path.exists(gt_path):
        seq.gt_t, seq.gt_xyz, seq.gt_quat = read_trajectory(gt_path)

    odom_path = os.path.join(seq_dir, "odom.txt")
    odom_speed = np.zeros(0)
    if os.path.exists(odom_path):
        seq.odom_t, seq.odom_xyz, odom_speed = read_odometry(odom_path)
        _, _, seq.odom_quat = read_trajectory(odom_path)

    # Match depth to color by timestamp
    c_t = np.array([t for t, _ in color])
    d_t = np.array([t for t, _ in depth])
    depth_for_color = {}
    if len(d_t) > 0:
        idx_c, idx_d = associate_timestamps(c_t, d_t, max_dt=max_dt)
        for ic, idd in zip(idx_c, idx_d):
            depth_for_color[ic] = depth[idd][1]

    for i in range(0, len(color), max(stride, 1)):
        t, rel = color[i]
        vel = 0.0
        if len(seq.odom_t) > 0:
            j = int(np.argmin(np.abs(seq.odom_t - t)))
            if abs(seq.odom_t[j] - t) < 0.5:
                vel = float(odom_speed[j])
        dp = depth_for_color.get(i)
        seq.frames.append({
            "stamp": t,
            "color_path": os.path.join(seq_dir, rel),
            "depth_path": os.path.join(seq_dir, dp) if dp else None,
            "velocity": vel,
        })

    seq.intrinsics = read_intrinsics(seq_dir)
    return seq


def load_frame_images(frame: dict, size: Tuple[int, int] = (640, 192)) -> dict:
    """
    Load and normalise the images for one frame entry (requires cv2).
    Adds "img_np" (H,W,3 float32 in [0,1]) and "depth_np" (H,W float32 m
    or None) to a copy of the frame dict.
    """
    import cv2

    out = dict(frame)
    img = cv2.imread(frame["color_path"], cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {frame['color_path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    out["img_np"] = img.astype(np.float32) / 255.0

    out["depth_np"] = None
    if frame.get("depth_path"):
        d = cv2.imread(frame["depth_path"], cv2.IMREAD_UNCHANGED)
        if d is not None:
            d = cv2.resize(d.astype(np.float32), size,
                           interpolation=cv2.INTER_NEAREST)
            out["depth_np"] = d / 1000.0   # uint16 mm -> m
    return out


# ── Scene/session discovery ───────────────────────────────────────────────────

_SEQ_RE = re.compile(r"^(?P<scene>[a-zA-Z]+)(?P<site>\d*)-(?P<idx>\d+)$")


def discover_scene_sessions(root: str, scene: str) -> List[str]:
    """
    Find the session sequence directories of one scene under root,
    ordered by recording index (office1-1, office1-2, ... office1-7).
    `scene` may be a bare scene name ("office") or a scene+site prefix
    ("office1").
    """
    sessions = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        m = _SEQ_RE.match(entry)
        if not m:
            continue
        prefix = m.group("scene") + m.group("site")
        if not (m.group("scene") == scene or prefix == scene):
            continue
        sessions.append((prefix, int(m.group("idx")), full))
    sessions.sort(key=lambda s: (s[0], s[1]))
    return [s[2] for s in sessions]
