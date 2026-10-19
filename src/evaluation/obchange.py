"""
hiper.evaluation.obchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adapter for the ObChange dataset (Langer, Patten, Vincze, TU Wien,
CC BY 4.0, DOI 10.48436/y3ggy-hxp10): real robot visits of five rooms with
YCB objects placed on horizontal planes, annotated per plane and visit.

Layout (one ``<Room>/<sceneN>`` per robot visit, all visits of a room in
one consistent room frame, z up, metres):

    <Room>/<sceneN>/table.txt                 plane centre/height/hull points
    <Room>/<sceneN>/planes/<p>/merged_plane_clouds_ds002.pcd
    <Room>/<sceneN>/planes/<p>/merged_plane_clouds_ds002_GT.anno
    <Room>/<sceneN>/planes/<p>/merged_plane_clouds_ds002.anno

This module turns every visit into one ``SessionData`` (see
``object_session.py``):

  * segmentation is ORACLE: one Track per annotated YCB instance, built
    from the GT point indices (mirrors the simulated protocol, where GT
    instance masks are used). Objects flagged as lying on the floor
    ("reconstructed by chance", dataset README) are excluded, as in the
    authors' evaluation; they are also the only source of repeated names
    within one visit (the same floor object inside two plane clouds);
  * appearance comes from 8 virtual renders of each object's coloured
    points (CLIP ViT-B-32 'openai' + DINOv2 ViT-B/14), with zero-shot CLIP
    labels over the AI2-THOR type vocabulary used by the simulated
    benchmark;
  * visibility uses one virtual top-down camera per reconstructed plane
    and a depth image rendered from that plane's cloud. Pixels outside the
    plane footprint hold NaN (not observed), so ``SessionData.visible``
    rejects positions off every plane. Each plane view is stored
    ``FRAMES_PER_PLANE`` times so that ``visible()`` with its default
    ``min_frames=3`` accepts a position seen in one plane view;
  * visits carry no timestamps: they are assumed one day apart;
  * change labels follow the dataset authors: same name in both visits and
    centroid distance < 0.20 m -> static, otherwise moved
    (``max_dist_for_being_static = 0.2`` in tidyupvisionpipeline).

Parsing and geometry are pure numpy/scipy; the CLIP/DINO encoders are
imported lazily (torch, open_clip, timm) so everything else can be
unit-tested without them.
"""

import json
import math
import os
import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOMS = ("BigRoom", "Kitchen", "LivingArea", "Office", "SmallRoom")
PCD_NAME = "merged_plane_clouds_ds002.pcd"
GT_ANNO_NAME = "merged_plane_clouds_ds002_GT.anno"
POSE_ANNO_NAME = "merged_plane_clouds_ds002.anno"
DAY_S = 86400.0

MOVED_THRESHOLD_M = 0.20      # ObChange: static iff centroid distance < 0.2 m
MIN_TRACK_POINTS = 50         # fewer GT points: present in GT, but no Track
MERGE_RADIUS_M = 0.10         # same name twice, centroids closer: one object
TRACK_POINTS = 2048           # Track.points subsample
VIEW_POINTS = 1024            # BYE partial-view size
MIN_VIEW_POINTS = 32          # partial views with fewer points are skipped
N_VIEWS = 8                   # virtual object views (azimuth every 45 deg)
VIEW_ELEV_DEG = 35.0
VIEW_DIST_FACTOR = 3.0        # camera distance = 3 x object radius
RENDER_SIZE = 224
DEPTH_W, DEPTH_H = 160, 120   # virtual plane-camera depth images
CAM_HEIGHT_M = 1.2            # plane camera height above the plane centre
FRAMES_PER_PLANE = 3          # = SessionData.visible() default min_frames
FOOTPRINT_MARGIN_M = 0.20     # hull dilation: objects lie up to ~15 cm past
                              # the table.txt hull (measured, SmallRoom)
FOOTPRINT_BELOW_M = 0.10      # footprint prism: plane - 0.10 m ...
FOOTPRINT_ABOVE_M = 0.20      # ... to plane + 0.20 m (object centroids)
POSE_SIGMA_M = 0.02
CENTROID_SIGMA_M = 0.01
CLIP_MODEL = "ViT-B-32-quickgelu"   # OpenAI CLIP ViT-B/32 architecture
CLIP_PRETRAINED = "openai"
DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
CLIP_PROMPTS = ("a photo of a {}.", "a {} in a room.")
CLIP_LOGIT_SCALE = 100.0
CITATION = ("E. Langer, T. Patten, M. Vincze, ObChange dataset, TU Wien, "
            "2022, DOI 10.48436/y3ggy-hxp10 (CC BY 4.0); paper: 'Where Does "
            "It Belong? Autonomous Object Mapping in Open-World Settings', "
            "Front. Robot. AI 2022, DOI 10.3389/frobt.2022.828732")


# ── Low-level file parsing ────────────────────────────────────────────────────

_PCD_NP_TYPES = {("F", 4): "<f4", ("F", 8): "<f8",
                 ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
                 ("U", 8): "<u8",
                 ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
                 ("I", 8): "<i8"}


def read_pcd_header(f) -> Tuple[Dict[str, List[str]], int]:
    """Read PCD header lines from an open binary file.
    Returns (header dict: key -> list of tokens, byte offset of data)."""
    header: Dict[str, List[str]] = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PCD header ended before DATA line")
        text = line.decode("ascii", errors="replace").strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        header[parts[0].upper()] = parts[1:]
        if parts[0].upper() == "DATA":
            return header, f.tell()


def read_pcd(path: str) -> Dict[str, np.ndarray]:
    """
    Parse a PCD v0.7 file (DATA ascii or binary; binary_compressed is not
    used by ObChange and raises NotImplementedError).

    Returns a dict with 'xyz' (N,3) float64, 'rgb' (N,3) float32 in [0,1]
    (when an rgb/rgba field exists) and every other field by name, e.g.
    'normal_x', 'curvature'.
    """
    with open(path, "rb") as f:
        header, offset = read_pcd_header(f)
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = [t.upper() for t in header["TYPE"]]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n_points = int(header["POINTS"][0])
        data_kind = header["DATA"][0].lower()
        dtype_list = []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            np_t = _PCD_NP_TYPES[(typ, size)]
            if name == "_":
                name = f"_pad{len(dtype_list)}"
            dtype_list.append((name, np_t) if cnt == 1 else
                              (name, np_t, (cnt,)))
        dtype = np.dtype(dtype_list)
        if data_kind == "binary":
            raw = f.read(n_points * dtype.itemsize)
            if len(raw) != n_points * dtype.itemsize:
                raise ValueError(
                    f"{path}: truncated binary PCD ({len(raw)} bytes, "
                    f"expected {n_points * dtype.itemsize})")
            arr = np.frombuffer(raw, dtype=dtype, count=n_points)
        elif data_kind == "ascii":
            rows = np.loadtxt(f, dtype=np.float64, ndmin=2)
            if len(rows) != n_points:
                raise ValueError(f"{path}: {len(rows)} rows, header says "
                                 f"{n_points}")
            arr = np.zeros(n_points, dtype=dtype)
            col = 0
            for name, cnt in zip(dtype.names, counts):
                val = rows[:, col:col + cnt]
                if name in ("rgb", "rgba") and \
                        dtype[name].kind == "f":
                    # ascii PCL stores packed colour as a float whose bit
                    # pattern is the uint32; keep the bits.
                    val = val.astype(np.float32)
                arr[name] = val[:, 0] if cnt == 1 else val
                col += cnt
        else:
            raise NotImplementedError(f"PCD DATA {data_kind} not supported")

    out: Dict[str, np.ndarray] = {}
    out["xyz"] = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(
        np.float64)
    for name in dtype.names:
        if name in ("x", "y", "z") or name.startswith("_pad"):
            continue
        if name in ("rgb", "rgba"):
            packed = arr[name]
            if packed.dtype.kind == "f":
                packed = packed.astype(np.float32).view(np.uint32)
            out["rgb"] = unpack_rgb(packed)
        else:
            out[name] = np.array(arr[name])
    return out


def unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """PCL packed colour (uint32 0x00RRGGBB) -> (N,3) float32 in [0,1]."""
    p = np.asarray(packed).astype(np.uint32)
    r = (p >> 16) & 0xFF
    g = (p >> 8) & 0xFF
    b = p & 0xFF
    return np.stack([r, g, b], axis=1).astype(np.float32) / 255.0


def write_pcd_binary(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a minimal binary PCD v0.7 (x y z rgb) - used by unit tests."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb8 = np.clip(np.round(np.asarray(rgb, dtype=np.float64) * 255), 0,
                   255).astype(np.uint32).reshape(-1, 3)
    packed = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    n = len(xyz)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("rgb", "<u4")])
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["rgb"] = packed
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
            "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
            f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\n"
            "DATA binary\n")
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(arr.tobytes())


def read_gt_anno(path: str) -> List[Dict]:
    """
    Parse ``*_GT.anno``: one line per YCB object instance,
    '<ycb_name> <point index> ... <on_floor bool>' (indices into the PCD).
    The trailing 'true'/'false' marks an object lying on the floor that was
    reconstructed by chance (dataset README); the authors' evaluation
    (tidyupvisionpipeline, evaluation.cpp) keeps only 'false' objects.

    Returns one dict per line, in file order:
      {'name', 'indices' (int64), 'on_floor' (bool or None if no flag),
       'flag_position': 'last' | 'inside' | 'missing'}
    A few lines carry the flag before their last indices ('inside'); all
    integer tokens are kept as indices. A missing file -> [].
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            flags = [(i, p.lower()) for i, p in enumerate(parts[1:], 1)
                     if p.lower() in ("true", "false")]
            idx = np.array([int(p) for p in parts[1:]
                            if p.lower() not in ("true", "false")],
                           dtype=np.int64)
            if not flags:
                on_floor, pos = None, "missing"
            else:
                on_floor = flags[-1][1] == "true"
                pos = "last" if flags[-1][0] == len(parts) - 1 else "inside"
            out.append({"name": parts[0], "indices": idx,
                        "on_floor": on_floor, "flag_position": pos})
    return out


_POSE_RE = re.compile(r"^\s*(\S+)\s*\((\d+)\)\s*:\s*(.*)$")


def read_pose_anno(path: str) -> List[Tuple[str, int, np.ndarray]]:
    """
    Parse ``*.anno`` (object poses): one line per object,
    '<ycb_name> (<n>): <16 numbers, row-major 4x4 pose>'.
    Returns [(name, n, T (4,4))]. Missing/empty file -> [].
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            m = _POSE_RE.match(line)
            if m is None:
                raise ValueError(f"{path}: cannot parse line {line!r}")
            vals = [float(v) for v in m.group(3).split()]
            if len(vals) != 16:
                raise ValueError(f"{path}: expected 16 numbers, got "
                                 f"{len(vals)} in {line!r}")
            out.append((m.group(1), int(m.group(2)),
                        np.array(vals, dtype=np.float64).reshape(4, 4)))
    return out


def read_table_txt(path: str) -> List[Dict[str, np.ndarray]]:
    """
    Parse ``table.txt``: a sequence of YAML-like plane blocks
        center: {x,y,z}  height: h  points: [- {x,y,z} ...]
        plane: {x,y,z,d}  (unit normal and offset, n.X + d = 0)
    Block i describes ``planes/<i>``. Returns one dict per block with
    'center' (3,), 'height' float, 'points' (P,3) and 'plane' (4,).
    """
    blocks: List[Dict] = []
    cur: Optional[Dict] = None
    section = None          # 'center' | 'points' | 'plane'
    pt: Optional[Dict] = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 0:
                key, _, val = s.partition(":")
                key = key.strip()
                if key == "center":
                    if cur is not None and "plane" in cur:
                        blocks.append(cur)
                        cur = None
                    if cur is None:
                        cur = {"points": []}
                    cur["center"] = {}
                    section = "center"
                elif key == "height":
                    cur["height"] = float(val)
                    section = None
                elif key == "points":
                    section = "points"
                elif key == "plane":
                    cur["plane"] = {}
                    section = "plane"
                else:
                    section = None
                continue
            if s == "-":
                pt = {}
                cur["points"].append(pt)
                continue
            key, _, val = s.lstrip("- ").partition(":")
            key = key.strip()
            if section == "center":
                cur["center"][key] = float(val)
            elif section == "plane":
                cur["plane"][key] = float(val)
            elif section == "points":
                if s.startswith("-"):           # inline '- x: ..' form
                    pt = {}
                    cur["points"].append(pt)
                pt[key] = float(val)
    if cur is not None:
        blocks.append(cur)
    out = []
    for b in blocks:
        c = b.get("center", {})
        pl = b.get("plane", {})
        pts = [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
               for p in b["points"]]
        out.append({
            "center": np.array([c.get("x", 0.0), c.get("y", 0.0),
                                c.get("z", 0.0)]),
            "height": float(b.get("height", c.get("z", 0.0))),
            "points": np.array(pts, dtype=np.float64).reshape(-1, 3),
            "plane": np.array([pl.get("x", 0.0), pl.get("y", 0.0),
                               pl.get("z", 1.0), pl.get("d", 0.0)]),
        })
    return out


def list_visits(root: str, room: str) -> List[str]:
    """Visit directories of a room, sorted by scene number (visit order)."""
    room_dir = os.path.join(root, room)
    scenes = []
    for name in os.listdir(room_dir):
        m = re.fullmatch(r"scene(\d+)", name)
        if m and os.path.isdir(os.path.join(room_dir, name)):
            scenes.append((int(m.group(1)), os.path.join(room_dir, name)))
    return [p for _, p in sorted(scenes)]


def list_planes(visit_dir: str) -> List[int]:
    """Plane indices with a point cloud in this visit, ascending."""
    pdir = os.path.join(visit_dir, "planes")
    out = []
    for name in os.listdir(pdir):
        if name.isdigit() and os.path.exists(
                os.path.join(pdir, name, PCD_NAME)):
            out.append(int(name))
    return sorted(out)


# ── Geometry and rendering ────────────────────────────────────────────────────

def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world pose (OpenCV axes) at ``eye`` looking at ``target``,
    with image 'up' towards world +z (or +y when looking straight down)."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


def project(xyz: np.ndarray, T_wc: np.ndarray, K: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and camera depth z of world points."""
    Xc = (np.asarray(xyz, dtype=np.float64) - T_wc[:3, 3]) @ T_wc[:3, :3]
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, np.nan)
    return K[0, 0] * Xc[:, 0] / zs + K[0, 2], \
        K[1, 1] * Xc[:, 1] / zs + K[1, 2], z


def render_points(xyz: np.ndarray, rgb: Optional[np.ndarray],
                  T_wc: np.ndarray, K: np.ndarray, h: int, w: int,
                  radius: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Z-buffered square splats of (2 radius + 1) pixels. Returns an RGB
    image in [0,1] (white background) and a depth image (0 = empty)."""
    u, v, z = project(xyz, T_wc, K)
    ok = np.isfinite(u) & (z > 1e-3)
    ui, vi, zi = np.round(u[ok]).astype(np.int64), \
        np.round(v[ok]).astype(np.int64), z[ok]
    ci = None if rgb is None else np.asarray(rgb, dtype=np.float32)[ok]
    pix, dep, idx = [], [], []
    src = np.arange(len(zi))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pu, pv = ui + dx, vi + dy
            inside = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
            pix.append(pv[inside] * w + pu[inside])
            dep.append(zi[inside])
            idx.append(src[inside])
    img = np.ones((h, w, 3), dtype=np.float32)
    depth = np.zeros((h, w), dtype=np.float32)
    if not pix or sum(len(p) for p in pix) == 0:
        return img, depth
    pix, dep, idx = np.concatenate(pix), np.concatenate(dep), \
        np.concatenate(idx)
    order = np.lexsort((dep, pix))              # nearest first per pixel
    first = np.ones(len(order), dtype=bool)
    first[1:] = pix[order][1:] != pix[order][:-1]
    win = order[first]
    depth.reshape(-1)[pix[win]] = dep[win]
    if ci is not None:
        img.reshape(-1, 3)[pix[win]] = ci[idx[win]]
    return img, depth


def object_views(xyz: np.ndarray, rgb: np.ndarray, size: int = RENDER_SIZE,
                 n_views: int = N_VIEWS, elev_deg: float = VIEW_ELEV_DEG,
                 dist_factor: float = VIEW_DIST_FACTOR) -> List[np.ndarray]:
    """Renders of an object's coloured points from ``n_views`` azimuths."""
    c = xyz.mean(axis=0)
    radius = max(float(np.percentile(np.linalg.norm(xyz - c, axis=1), 95)),
                 0.02)
    f = 0.5 * size / math.tan(math.radians(25.0))   # 50 deg field of view
    K = np.array([[f, 0, size / 2.0], [0, f, size / 2.0], [0, 0, 1]])
    el = math.radians(elev_deg)
    out = []
    for i in range(n_views):
        az = 2.0 * math.pi * i / n_views
        eye = c + dist_factor * radius * np.array(
            [math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
             math.sin(el)])
        img, _ = render_points(xyz, rgb, look_at(eye, c), K, size, size,
                               radius=1)
        out.append((img * 255).astype(np.uint8))
    return out


# ── Visits and ground truth ───────────────────────────────────────────────────

@dataclass
class VisitObject:
    name: str
    plane: int
    xyz: np.ndarray
    rgb: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        return self.xyz.mean(axis=0)


@dataclass
class Visit:
    k: int
    path: str
    planes: Dict[int, Dict]                  # p -> {'xyz', 'rgb', 'table'}
    objects: Dict[str, VisitObject]
    log: List[str] = field(default_factory=list)


def load_visit(visit_dir: str, k: int) -> Visit:
    """Read one visit: plane clouds, table geometry and the YCB objects
    (excluding objects flagged as lying on the floor)."""
    tables = read_table_txt(os.path.join(visit_dir, "table.txt"))
    planes: Dict[int, Dict] = {}
    found: Dict[str, List[VisitObject]] = {}
    log: List[str] = []
    for p in list_planes(visit_dir):
        pdir = os.path.join(visit_dir, "planes", str(p))
        pcd = read_pcd(os.path.join(pdir, PCD_NAME))
        xyz = pcd["xyz"]
        rgb = pcd.get("rgb", np.full((len(xyz), 3), 0.5, np.float32))
        planes[p] = {"xyz": xyz, "rgb": rgb,
                     "table": tables[p] if p < len(tables) else None}
        for obj in read_gt_anno(os.path.join(pdir, GT_ANNO_NAME)):
            if obj["on_floor"]:
                continue
            idx = obj["indices"]
            idx = idx[(idx >= 0) & (idx < len(xyz))]
            if len(idx) == 0:
                continue
            found.setdefault(obj["name"], []).append(
                VisitObject(obj["name"], p, xyz[idx], rgb[idx]))
    objects: Dict[str, VisitObject] = {}
    for name, parts in found.items():
        if len(parts) == 1:
            objects[name] = parts[0]
            continue
        parts.sort(key=lambda o: -len(o.xyz))
        keep = parts[0]
        for o in parts[1:]:
            if np.linalg.norm(o.centroid - keep.centroid) < MERGE_RADIUS_M:
                keep = VisitObject(name, keep.plane,
                                   np.vstack([keep.xyz, o.xyz]),
                                   np.vstack([keep.rgb, o.rgb]))
            else:
                log.append(f"{os.path.basename(visit_dir)}: '{name}' twice "
                           f"({len(o.xyz)} pts dropped, plane {o.plane})")
        objects[name] = keep
    return Visit(k, visit_dir, planes, objects, log)


def gt_transitions(visits: Sequence[Visit]) -> List[Dict[str, dict]]:
    """Per visit, every YCB name seen in ANY visit of the room:
    {present, centroid, moved_since_prev, appeared_since_prev,
     disappeared_since_prev}. Static = present in both visits and centroid
    distance < MOVED_THRESHOLD_M (the dataset authors' rule)."""
    names = sorted({n for v in visits for n in v.objects})
    out: List[Dict[str, dict]] = []
    last: Dict[str, np.ndarray] = {}
    for k, v in enumerate(visits):
        cur: Dict[str, dict] = {}
        for n in names:
            now = v.objects.get(n)
            prev = visits[k - 1].objects.get(n) if k > 0 else None
            c = now.centroid if now is not None else last.get(n)
            cur[n] = {
                "present": now is not None,
                "centroid": None if c is None else c.tolist(),
                "moved_since_prev": bool(
                    now is not None and prev is not None and
                    np.linalg.norm(now.centroid - prev.centroid)
                    >= MOVED_THRESHOLD_M),
                "appeared_since_prev": bool(k > 0 and now is not None
                                            and prev is None),
                "disappeared_since_prev": bool(k > 0 and now is None
                                               and prev is not None),
            }
            if now is not None:
                last[n] = now.centroid
        out.append(cur)
    return out


# ── Virtual plane cameras (visibility) ────────────────────────────────────────

def _hull_polygon(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ordered convex polygon of projected hull points. table.txt lists the
    plane's hull points in no polygon order, so they are re-ordered here."""
    from scipy.spatial import ConvexHull

    uv = np.stack([u, v], axis=1)
    return uv[ConvexHull(uv).vertices]


def plane_camera(table: Dict, plane_xyz: np.ndarray, w: int = DEPTH_W,
                 h: int = DEPTH_H) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    """A camera CAM_HEIGHT_M above the plane centre looking down at it.
    Returns (T_wc, K, depth); pixels outside the dilated plane hull are NaN
    (never observed), holes inside it get the plane depth."""
    from matplotlib.path import Path

    n = np.asarray(table["plane"][:3], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    c = np.asarray(table["center"], dtype=np.float64)
    T = look_at(c + CAM_HEIGHT_M * n, c)
    hull = np.asarray(table["points"], dtype=np.float64)
    if len(hull) < 3:
        hull = plane_xyz
    rad = hull - hull.mean(axis=0)
    hull = hull + FOOTPRINT_MARGIN_M * rad / (
        np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9)
    Xc = (hull - T[:3, 3]) @ T[:3, :3]
    half_x = np.max(np.abs(Xc[:, 0] / Xc[:, 2]))
    half_y = np.max(np.abs(Xc[:, 1] / Xc[:, 2]))
    f = 0.9 * min((w / 2.0) / max(half_x, 1e-3), (h / 2.0) / max(half_y, 1e-3))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    _, depth = render_points(plane_xyz, None, T, K, h, w, radius=1)
    u, v, _ = project(hull, T, K)
    uv = _hull_polygon(u, v)
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    inside = Path(uv).contains_points(
        np.stack([uu.ravel(), vv.ravel()], axis=1)).reshape(h, w)
    depth = depth.astype(np.float32)
    depth[inside & (depth <= 0)] = CAM_HEIGHT_M
    depth[~inside] = np.nan
    return T, K, depth


# ── Sessions ──────────────────────────────────────────────────────────────────

# embed_fn(list of uint8 HxWx3 renders) -> {"clip": (V, d) unit rows,
#   "label_probs": (V, L) softmax over the vocabulary, "vocab": [L type keys],
#   "dino": (V, d') unit rows or None}
EmbedFn = Callable[[List[np.ndarray]], Dict]


def _unit_mean(E: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalised mean of unit rows and tr(C) = 1 - ||mean||^2."""
    m = np.asarray(E, dtype=np.float64).mean(axis=0)
    disp = float(min(max(1.0 - float(m @ m), 0.0), 1.0))
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32), disp


def make_track(i: int, obj: VisitObject, embed_fn: Optional[EmbedFn],
               rng: np.random.Generator) -> "Track":
    from hiper.evaluation.object_session import Track

    xyz, rgb = obj.xyz, obj.rgb
    sel = (rng.choice(len(xyz), TRACK_POINTS, replace=False)
           if len(xyz) > TRACK_POINTS else np.arange(len(xyz)))
    c = xyz.mean(axis=0)
    cov = np.cov(xyz.T) / max(len(xyz), 1) + CENTROID_SIGMA_M ** 2 * np.eye(3)
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    tr = Track(track_id=i, centroid=c.astype(np.float32),
               centroid_cov=cov.astype(np.float32), n_obs=N_VIEWS,
               clip_mean=None, clip_disp=0.0, clip_label="unknown",
               clip_label_conf=0.0,
               points=np.hstack([xyz[sel], rgb[sel]]).astype(np.float32),
               extent=(hi - lo).astype(np.float32), gt_id=obj.name)
    if embed_fn is not None:
        out = embed_fn(object_views(xyz, rgb))
        tr.clip_mean, tr.clip_disp = _unit_mean(out["clip"])
        probs = np.asarray(out["label_probs"]).mean(axis=0)
        j = int(np.argmax(probs))
        tr.clip_label, tr.clip_label_conf = out["vocab"][j], float(probs[j])
        if out.get("dino") is not None:
            tr.dino_mean, _ = _unit_mean(out["dino"])
    return tr


def build_room_sessions(root: str, room: str,
                        embed_fn: Optional[EmbedFn] = None,
                        seed: int = 0) -> Tuple[List, List[str]]:
    """All visits of ``room`` as SessionData (oracle segmentation), plus a
    log of anomalies. Visits are assumed one day apart."""
    from hiper.evaluation.object_session import SessionData

    visits = [load_visit(d, k) for k, d in
              enumerate(list_visits(root, room))]
    trans = gt_transitions(visits)
    rng = np.random.default_rng(zlib.crc32(room.encode()) + seed)
    sessions, log = [], []
    for k, v in enumerate(visits):
        log += v.log
        tracks = [make_track(i, o, embed_fn, rng)
                  for i, (_, o) in enumerate(sorted(v.objects.items()))]
        Ts, Ds, K = [], [], None
        for p, pl in sorted(v.planes.items()):
            if pl["table"] is None:
                log.append(f"{room} visit {k}: plane {p} has no table entry")
                continue
            T, K_p, depth = plane_camera(pl["table"], pl["xyz"])
            if K is not None and not np.allclose(K, K_p):
                # one K per session: rescale this view's depth to it
                T, K_p, depth = _refit(pl["table"], pl["xyz"], K)
            K = K_p
            for _ in range(FRAMES_PER_PLANE):
                Ts.append(T)
                Ds.append(depth)
        gt = []
        for name, e in trans[k].items():
            if e["centroid"] is None:
                continue
            gt.append({"gt_id": name, "type": name, "group": "pickupable",
                       "position": e["centroid"], "present": e["present"],
                       "moved_since_prev": e["moved_since_prev"],
                       "appeared_since_prev": e["appeared_since_prev"],
                       "disappeared_since_prev": e["disappeared_since_prev"]})
        sessions.append(SessionData(
            scene=f"obchange_{room}", session_index=k, time=k * DAY_S,
            tracks=tracks, T_wc_est=np.array(Ts), K=K,
            depth_stack=np.array(Ds, dtype=np.float16),
            pose_cov=POSE_SIGMA_M ** 2 * np.eye(3), T_wc_gt=np.array(Ts),
            gt_objects=gt,
            meta={"split": "obchange", "room": room,
                  "visit": os.path.basename(v.path), "time_assumed": True,
                  "segmentation": "oracle (GT point indices)",
                  "n_planes": len(v.planes), "citation": CITATION}))
    return sessions, log


def _refit(table: Dict, plane_xyz: np.ndarray, K: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """plane_camera with a fixed intrinsic matrix K (the session's), moving
    the camera along the plane normal until the hull fits the image."""
    from matplotlib.path import Path

    n = np.asarray(table["plane"][:3], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    c = np.asarray(table["center"], dtype=np.float64)
    hull = np.asarray(table["points"], dtype=np.float64)
    if len(hull) < 3:
        hull = plane_xyz
    rad = hull - hull.mean(axis=0)
    hull = hull + FOOTPRINT_MARGIN_M * rad / (
        np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9)
    w, h = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
    height = CAM_HEIGHT_M
    for _ in range(40):
        T = look_at(c + height * n, c)
        u, v, _ = project(hull, T, K)
        if np.all((u > 0) & (u < w) & (v > 0) & (v < h)):
            break
        height *= 1.15
    _, depth = render_points(plane_xyz, None, T, K, h, w, radius=1)
    uv = _hull_polygon(u, v)
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    inside = Path(uv).contains_points(
        np.stack([uu.ravel(), vv.ravel()], axis=1)).reshape(h, w)
    depth = depth.astype(np.float32)
    depth[inside & (depth <= 0)] = height
    depth[~inside] = np.nan
    return T, K, depth
