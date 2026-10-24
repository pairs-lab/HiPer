"""
hiper.evaluation.rscan
~~~~~~~~~~~~~~~~~~~~~~~~~
Adapter for 3RScan (Wald, Avetisyan, Navab, Tombari, Niessner, ICCV 2019;
github.com/WaldJohannaU/3RScan): 478 indoor environments, each scanned
2-12 times with a Tango device (RGB-D sequence with camera poses, a
reconstructed mesh with annotated object instances), with the changes
between the reference scan and every rescan annotated.

Conventions (checked on the public example scans of 3RScan.v2.zip, report
docs/reports/2026-10-24-3rscan-adapter):

  * 3RScan.json: one entry per environment {reference, type, scans:
    [{reference (rescan id), transform, rigid, removed, nonrigid}],
    ambiguity}. 4x4 matrices are stored column-major, in metres.
    ``transform`` maps rescan coordinates to reference coordinates;
    ``rigid[i].transform`` maps the object's reference-scan coordinates to
    its rescan coordinates (scan alignment included). Only changed objects
    are listed; ``removed`` lists reference instances that left. Instance
    ids are shared by all scans of an environment.
  * The official test split carries no transforms or change lists
    (hidden), so it is not used.
  * <scan>/sequence/frame-%06d.{color.jpg,depth.pgm,pose.txt} and
    _info.txt: colour 960x540, depth 224x172 uint16 (m_depthShift per
    metre), separate intrinsics; pose = camera-to-world in the scan frame,
    metres. Frames are stored sideways (portrait device): crops are
    rotated upright by multiples of 90 degrees, using the projected world
    up axis (+z).
  * <scan>/labels.instances.annotated.v2.ply: ascii mesh with per-vertex
    objectId, globalId, NYU40, Eigen13, RIO27; <scan>/semseg.v2.json
    names every objectId.

Protocol (fixed before any 3RScan result):

  * one environment = one sequence; sessions = the reference scan, then
    the rescans in the order listed in 3RScan.json; no timestamps, so
    sessions are assumed one day apart (as for ObChange);
  * splits: official validation -> ``rscan_test`` (evaluated once);
    official train -> ``rscan_val`` (tuning) when sha1(environment id)
    mod 5 == 0, else ``rscan_train`` (priors);
  * objects: every annotated instance except wall, floor and ceiling
    (NYU40 1, 2, 22); world frame: the reference scan frame shifted so
    that its floor (2nd percentile of the floor vertices' height) is at
    z = 0;
  * ground truth from the annotations: a reference object is absent when
    listed in ``removed``, at its rigidly transformed reference position
    when listed in ``rigid``, in place otherwise (``nonrigid`` = deformed
    in place); a reference object that a rescan neither annotates nor
    lists as removed is UNKNOWN there (no label). An object new in a
    rescan is placed at its own reconstructed centroid and is unknown in
    rescans that do not annotate it. Moved = displacement of at least
    MOVED_THRESHOLD_M between consecutive sessions with known positions
    (ObChange's rule);
  * segmentation is ORACLE (as for THOR-Lifelong and ObChange): one Track
    per annotated instance seen in the sampled frames, its masks rendered
    from the annotated mesh at the frame poses; geometry from the
    instance's reconstructed surface in that scan; appearance from upright
    square crops of the colour frames (CLIP + DINOv2, thor_tracks
    encoders); the scan alignment is the dataset's (oracle relocalization,
    small pose covariance).

Parsing, ground truth and rendering are pure numpy; the encoders are
imported lazily by the conversion script.
"""

import hashlib
import json
import os
import zlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

DAY_S = 86400.0
MOVED_THRESHOLD_M = 0.20     # ObChange: static iff displacement < 0.2 m
STRUCTURE_NYU40 = (1, 2, 22)  # wall, floor, ceiling: not map objects
FLOOR_NYU40 = 2
VAL_MOD = 5                  # official train: sha1 % 5 == 0 -> tuning split
FRAME_STRIDE = 5             # sampled frames of a sequence
DEPTH_EVERY = 2              # every 2nd sampled frame in the depth stack
MAX_DEPTH_FRAMES = 60
RENDER_SCALE = 0.5           # instance masks at half the colour resolution
SPLAT_RADIUS = 1             # pixels around each surface sample
SURFACE_SPACING_M = 0.01     # area-weighted samples, about 1 per cm^2
MIN_AREA_FRAC = 0.0005       # as thor_tracks
MAX_CROPS = 32               # largest-mask frames per track
CROP_PAD = 0.15
TRACK_POINTS = 2048
CENTROID_SIGMA_M = 0.01
# A scan reconstructs only part of an object, so the centroid of a static
# object moves between scans (public example: median 5 cm, max 30 cm). The
# track covariance adds (CENTROID_EXTENT_FRAC * extent)^2 per axis, fitted
# on rscan_train only (scripts/rscan_fit_cov.py, 2026-10-24: 11,795 track
# pairs, 917 moved; best balanced accuracy of the relocation test, 0.857).
CENTROID_EXTENT_FRAC = 0.075
POSE_SIGMA_M = 0.02          # dataset alignment (oracle relocalization)
CITATION = ("J. Wald, A. Avetisyan, N. Navab, F. Tombari, M. Niessner, "
            "RIO: 3D object instance re-localization in changing indoor "
            "environments, ICCV 2019 (3RScan)")


# ── 3RScan.json ──────────────────────────────────────────────────────────────

def mat4(values: Sequence[float]) -> np.ndarray:
    """4x4 matrix from 3RScan's 16 column-major values."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T


def load_index(path: str) -> Dict[str, dict]:
    """{environment (reference scan) id: {"type", "reference", "rescans":
    [{"id", "T" (rescan -> reference), "rigid" {instance: M (reference ->
    rescan)}, "removed" set, "nonrigid" set}]}}. Test entries keep their
    rescan ids but have no annotations (T None)."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for e in data:
        rescans = []
        for s in e["scans"]:
            rescans.append({
                "id": s["reference"],
                "T": mat4(s["transform"]) if s.get("transform") else None,
                # test entries list bare instance ids without transforms
                "rigid": {int(r["instance_reference"]): mat4(r["transform"])
                          for r in s.get("rigid", [])
                          if isinstance(r, dict) and r.get("transform")},
                "removed": {int(i) for i in s.get("removed", [])},
                "nonrigid": {int(i) for i in s.get("nonrigid", [])}})
        out[e["reference"]] = {"type": e["type"], "reference": e["reference"],
                               "rescans": rescans}
    return out


def assign_split(env_id: str, official: str) -> Optional[str]:
    """Benchmark split of an environment (see the module docstring)."""
    if official == "validation":
        return "rscan_test"
    if official != "train":
        return None
    h = int(hashlib.sha1(env_id.encode()).hexdigest()[:8], 16)
    return "rscan_val" if h % VAL_MOD == 0 else "rscan_train"


# ── Scan files ───────────────────────────────────────────────────────────────

def read_info(path: str) -> dict:
    """_info.txt -> {"color_K", "depth_K" (3x3), "color_size", "depth_size"
    (w, h), "depth_shift"}."""
    kv = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()

    def K(key):
        return np.asarray(kv[key].split(), np.float64).reshape(4, 4)[:3, :3]
    return {"color_K": K("m_calibrationColorIntrinsic"),
            "depth_K": K("m_calibrationDepthIntrinsic"),
            "color_size": (int(kv["m_colorWidth"]), int(kv["m_colorHeight"])),
            "depth_size": (int(kv["m_depthWidth"]), int(kv["m_depthHeight"])),
            "depth_shift": float(kv.get("m_depthShift", 1000.0))}


def list_frames(seq_dir: str) -> List[int]:
    """Frame indices with a colour image and a finite pose."""
    out = []
    for name in sorted(os.listdir(seq_dir)):
        if name.startswith("frame-") and name.endswith(".color.jpg"):
            i = int(name[6:12])
            if os.path.exists(os.path.join(seq_dir, f"frame-{i:06d}.pose.txt")):
                out.append(i)
    return out


def read_pose(seq_dir: str, i: int) -> np.ndarray:
    return np.loadtxt(os.path.join(seq_dir, f"frame-{i:06d}.pose.txt"))


def read_label_ply(path: str) -> Dict[str, np.ndarray]:
    """ascii labels.instances.annotated.v2.ply -> {"xyz" (N,3), "faces"
    (M,3), and one (N,) int array per label property}."""
    props: List[str] = []
    nv = nf = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format") and "ascii" not in line:
                raise ValueError(f"{path}: only ascii ply is supported")
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line.startswith("property") and nf == 0:
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        V = np.loadtxt(f, max_rows=nv, ndmin=2)
        F = np.loadtxt(f, max_rows=nf, dtype=np.int64, ndmin=2)[:, 1:4]
    out = {"xyz": V[:, :3].astype(np.float64), "faces": F}
    for j, p in enumerate(props):
        if p not in ("x", "y", "z"):
            out[p] = V[:, j].astype(np.int64)
    return out


def read_semseg(path: str) -> Dict[int, str]:
    """{objectId: label} of semseg.v2.json."""
    with open(path) as f:
        return {int(g["objectId"]): str(g["label"])
                for g in json.load(f)["segGroups"]}


# ── Geometry ─────────────────────────────────────────────────────────────────

def surface_samples(xyz: np.ndarray, faces: np.ndarray, vertex_label:
                    np.ndarray, spacing: float = SURFACE_SPACING_M,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted points on the mesh (about one per ``spacing``^2) plus
    the vertices; a point takes the label of its triangle's first vertex."""
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    n = rng.poisson(area / spacing ** 2)
    tri = np.repeat(np.arange(len(faces)), n)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    P = (w[:, :1] * a[tri] + w[:, 1:2] * b[tri] + w[:, 2:] * c[tri])
    L = vertex_label[faces[tri, 0]]
    return np.vstack([xyz, P]), np.concatenate([vertex_label, L])


def render_labels(P: np.ndarray, L: np.ndarray, T_wc: np.ndarray,
                  K: np.ndarray, w: int, h: int,
                  radius: int = SPLAT_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """Z-buffered splat of labelled points (scan frame) into a camera
    (pose T_wc, intrinsics K, w x h): (labels (h, w), 0 = none; depth)."""
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    Xc = (P - t) @ R
    z = Xc[:, 2]
    ok = z > 0.05
    u = K[0, 0] * Xc[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * Xc[ok, 1] / z[ok] + K[1, 2]
    z, L = z[ok], L[ok]
    inb = (u > -radius - 1) & (u < w + radius) & (v > -radius - 1) & \
        (v < h + radius)
    u, v, z, L = u[inb].astype(np.int64), v[inb].astype(np.int64), z[inb], \
        L[inb]
    us, vs, zs, ls = [], [], [], []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            us.append(uu[m]), vs.append(vv[m]), zs.append(z[m]), ls.append(L[m])
    idx = np.concatenate(vs) * w + np.concatenate(us)
    zz, ll = np.concatenate(zs), np.concatenate(ls)
    depth = np.full(h * w, np.inf)
    np.minimum.at(depth, idx, zz)
    win = zz <= depth[idx]
    lab = np.zeros(h * w, dtype=np.int64)
    lab[idx[win]] = ll[win]
    return lab.reshape(h, w), depth.reshape(h, w)


def upright_k(T_wc: np.ndarray) -> int:
    """np.rot90 / torch.rot90 count (counter-clockwise quarter turns) that
    makes the world +z axis point up in the image."""
    up = T_wc[:3, :3].T @ np.array([0.0, 0.0, 1.0])   # camera frame
    x, y = float(up[0]), float(up[1])                 # image right, down
    if abs(y) >= abs(x):
        return 0 if y < 0 else 2
    return 1 if x > 0 else 3


def floor_shift(ply: Dict[str, np.ndarray]) -> np.ndarray:
    """World-from-reference transform: shift z so the floor is at 0."""
    z = ply["xyz"][:, 2]
    nyu = ply.get("NYU40")
    fl = z[nyu == FLOOR_NYU40] if nyu is not None else z[:0]
    z0 = float(np.percentile(fl, 2)) if len(fl) >= 10 else \
        float(np.percentile(z, 1))
    W = np.eye(4)
    W[2, 3] = -z0
    return W


def object_samples(ply: Dict[str, np.ndarray], seed: int = 0
                   ) -> Dict[int, np.ndarray]:
    """{objectId: surface points (scan frame)} of the non-structure
    instances."""
    P, L = surface_samples(ply["xyz"], ply["faces"], ply["objectId"],
                           seed=seed)
    keep = structure_mask(ply)
    out = {}
    order = np.argsort(L, kind="stable")
    Ls = L[order]
    ids, start = np.unique(Ls, return_index=True)
    ends = np.r_[start[1:], len(Ls)]
    for i, s, e in zip(ids.tolist(), start, ends):
        if i > 0 and keep.get(i, False):
            out[i] = P[order[s:e]]
    return out


def structure_mask(ply: Dict[str, np.ndarray]) -> Dict[int, bool]:
    """{objectId: True if it is a map object (majority NYU40 class not
    wall/floor/ceiling)}."""
    oid, nyu = ply["objectId"], ply.get("NYU40")
    out = {}
    for i in np.unique(oid).tolist():
        if i <= 0:
            continue
        if nyu is None:
            out[i] = True
            continue
        cls = np.bincount(nyu[oid == i]).argmax()
        out[i] = int(cls) not in STRUCTURE_NYU40
    return out


def to_world(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


# ── Ground truth ─────────────────────────────────────────────────────────────

def env_ground_truth(ref_cent: Dict[int, np.ndarray],
                     scans: List[dict],
                     moved_threshold: float = MOVED_THRESHOLD_M
                     ) -> List[Dict[int, dict]]:
    """
    Per session {objectId: {"present": True/False/None, "position": world
    xyz or None, "moved_since_prev", "appeared_since_prev",
    "disappeared_since_prev"}}.

    ``ref_cent``: world centroids of the reference objects.
    ``scans[s]`` for s >= 1: {"W" (reference scan -> world), "W_T" (rescan
    -> world, W @ T), "rigid" {id: M}, "removed" set, "annotated" set of
    ids in the rescan, "cent" {id: world centroid from the rescan's own
    geometry}}; ``scans[0]`` is ignored.
    """
    ids: Set[int] = set(ref_cent)
    for sc in scans[1:]:
        ids |= set(sc["annotated"])
    states: List[Dict[int, Tuple[Optional[bool], Optional[np.ndarray]]]] = []
    for s in range(len(scans)):
        cur = {}
        for i in ids:
            if s == 0:
                cur[i] = (i in ref_cent, ref_cent.get(i))
                continue
            sc = scans[s]
            if i in ref_cent:
                if i in sc["removed"]:
                    cur[i] = (False, None)
                elif i in sc["rigid"]:
                    cur[i] = (True, _moved_position(sc, i, ref_cent[i]))
                elif i in sc["annotated"]:
                    cur[i] = (True, ref_cent[i])
                else:
                    cur[i] = (None, None)
            else:
                if i in sc["annotated"]:
                    cur[i] = (True, sc["cent"].get(i))
                else:
                    cur[i] = (None, None)
        states.append(cur)
    out: List[Dict[int, dict]] = []
    last_pos: Dict[int, np.ndarray] = {}
    for s, cur in enumerate(states):
        prev = states[s - 1] if s > 0 else {}
        row = {}
        for i, (pres, pos) in cur.items():
            pp, ppos = prev.get(i, (None, None)) if s > 0 else (None, None)
            moved = bool(pres and pp and pos is not None and ppos is not None
                         and np.linalg.norm(pos - ppos) >= moved_threshold)
            row[i] = {
                "present": pres,
                "position": (pos if pos is not None else last_pos.get(i)),
                "moved_since_prev": moved,
                "appeared_since_prev": bool(s > 0 and pres is True
                                            and pp is False),
                "disappeared_since_prev": bool(s > 0 and pres is False
                                               and pp is True)}
            if pres and pos is not None:
                last_pos[i] = pos
        out.append(row)
    return out


def _moved_position(sc: dict, i: int, c_world_ref: np.ndarray) -> np.ndarray:
    """World position of reference object i after its rigid change:
    world <- rescan (W_T) <- object change M (reference scan -> rescan)
    <- reference scan (inverse of the reference's world shift W)."""
    M = sc["rigid"][i]
    W = sc["W"]
    p = sc["W_T"] @ M @ np.linalg.inv(W) @ np.r_[c_world_ref, 1.0]
    return p[:3]


# ── Sessions (CPU stage; the GPU stage is thor_tracks.build_session) ─────────

def scan_dir(root: str, scan_id: str) -> str:
    return os.path.join(root, scan_id)


def load_scan(root: str, scan_id: str) -> dict:
    """Mesh labels, object surface samples (scan frame), names and frames
    of one scan."""
    d = scan_dir(root, scan_id)
    ply = read_label_ply(os.path.join(d, "labels.instances.annotated.v2.ply"))
    names = read_semseg(os.path.join(d, "semseg.v2.json"))
    keep = structure_mask(ply)
    nyu = ply["NYU40"]
    cls = {i: int(np.bincount(nyu[ply["objectId"] == i]).argmax())
           for i in keep}
    return {"id": scan_id, "dir": d, "ply": ply, "names": names,
            "keep": keep, "nyu40": cls,
            "samples": object_samples(ply, seed=zlib.crc32(scan_id.encode())),
            "info": read_info(os.path.join(d, "sequence", "_info.txt")),
            "frames": list_frames(os.path.join(d, "sequence"))}


def env_scans(env: dict) -> List[Tuple[str, np.ndarray, dict]]:
    """[(scan id, T (scan -> reference), annotation)] in session order:
    the reference, then the annotated rescans as listed."""
    out = [(env["reference"], np.eye(4), {})]
    out += [(r["id"], r["T"], r) for r in env["rescans"] if r["T"] is not None]
    return out


def prepare_env(root: str, env: dict, split: str,
                frame_stride: int = FRAME_STRIDE) -> List[dict]:
    """All sessions of one environment as thor_tracks.build_session inputs
    (without the RGB frames: ``frame_files`` lists them)."""
    from scipy.ndimage import find_objects
    from hiper.evaluation.thor_tracks import square_crop_box

    seq = env_scans(env)
    scans = [load_scan(root, sid) for sid, _, _ in seq]
    W = floor_shift(scans[0]["ply"])
    ref_cent = {i: to_world(W, P).mean(0)
                for i, P in scans[0]["samples"].items()}
    gt_in = [{}]
    for (sid, T, ann), sc in zip(seq[1:], scans[1:]):
        WT = W @ T
        gt_in.append({"W": W, "W_T": WT, "rigid": ann["rigid"],
                      "removed": ann["removed"],
                      "annotated": {i for i, k in sc["keep"].items() if k},
                      "cent": {i: to_world(WT, P).mean(0)
                               for i, P in sc["samples"].items()}})
    gt = env_ground_truth(ref_cent, gt_in)
    names, nyu = {}, {}
    for sc in scans:
        for i, n in sc["names"].items():
            names.setdefault(i, n)
        for i, c in sc["nyu40"].items():
            nyu.setdefault(i, c)
    preps = []
    for k, ((sid, T, _), sc) in enumerate(zip(seq, scans)):
        WT = W @ T
        info = sc["info"]
        sdir = os.path.join(sc["dir"], "sequence")
        frames = sc["frames"][::frame_stride]
        Wc, Hc = info["color_size"]
        w, h = int(round(Wc * RENDER_SCALE)), int(round(Hc * RENDER_SCALE))
        Kr = info["color_K"].copy()
        Kr[:2] *= RENDER_SCALE
        objs = sc["samples"]
        P = np.vstack([objs[i] for i in sorted(objs)]) if objs else \
            np.zeros((0, 3))
        L = np.concatenate([np.full(len(objs[i]), i) for i in sorted(objs)]) \
            if objs else np.zeros(0, np.int64)
        # occluders: walls, floor, ceiling and unlabelled surface render as
        # label -1, so objects behind them stay hidden
        st_ply = sc["ply"]
        st = np.array([not sc["keep"].get(int(i), False)
                       for i in st_ply["objectId"]])
        Pst, Lst = surface_samples(st_ply["xyz"], st_ply["faces"],
                                   np.where(st, -1, 0).astype(np.int64),
                                   seed=zlib.crc32(sid.encode()) + 1)
        Pst = Pst[Lst == -1]
        PP = np.vstack([P, Pst])
        LL = np.concatenate([L, np.full(len(Pst), -1)])
        obs: Dict[int, List[dict]] = {}
        poses = []
        for fi, f in enumerate(frames):
            Tc = read_pose(sdir, f)
            poses.append(Tc)
            if not np.all(np.isfinite(Tc)):
                continue
            lab, _ = render_labels(PP, LL, Tc, Kr, w, h)
            lab = np.where(lab > 0, lab, 0)
            cnt = np.bincount(lab.ravel())
            boxes = find_objects(lab.astype(np.int32))
            rot = upright_k(Tc)
            for i in np.nonzero(cnt >= MIN_AREA_FRAC * w * h)[0].tolist():
                if i == 0 or i not in objs:
                    continue
                sl = boxes[i - 1]
                s_ = 1.0 / RENDER_SCALE
                bb = (sl[1].start * s_, sl[0].start * s_, sl[1].stop * s_,
                      sl[0].stop * s_)
                obs.setdefault(i, []).append({
                    "frame": fi, "area_frac": float(cnt[i] / (w * h)),
                    "crop": square_crop_box(*bb, Wc, Hc, pad=CROP_PAD),
                    "rot": rot})
        rng = np.random.default_rng(zlib.crc32(f"{sid}/points".encode()))
        tracks = []
        for i in sorted(obs):
            ol = sorted(obs[i], key=lambda o: -o["area_frac"])
            used = ol[:MAX_CROPS]
            Pw = to_world(WT, objs[i])
            c = Pw.mean(0)
            cov = np.cov(Pw.T) / max(len(Pw), 1) + \
                CENTROID_SIGMA_M ** 2 * np.eye(3)
            sel = rng.choice(len(Pw), TRACK_POINTS, replace=False) \
                if len(Pw) > TRACK_POINTS else np.arange(len(Pw))
            pts = np.hstack([Pw[sel], np.full((len(sel), 3), 0.5)])
            lo_, hi_ = np.percentile(Pw, [5, 95], axis=0)
            cov = cov + np.diag((CENTROID_EXTENT_FRAC * (hi_ - lo_)) ** 2)
            tracks.append({
                "gt_id": str(i), "centroid": c, "centroid_cov": cov,
                "n_obs": len(ol),
                "pixel_area": float(np.mean([o["area_frac"] for o in ol])),
                "points": pts.astype(np.float32),
                "extent": (hi_ - lo_).astype(np.float32),
                "frames": [o["frame"] for o in used],
                "crops": [o["crop"] for o in used],
                "rots": [o["rot"] for o in used]})
        # depth stack for the in-view test (sensor depth, world poses)
        import cv2
        di = np.unique(np.linspace(0, len(frames) - 1, min(
            MAX_DEPTH_FRAMES, max(1, len(frames) // DEPTH_EVERY))).round()
            .astype(int)) if frames else np.zeros(0, int)
        T_w, depths = [], []
        for j in di:
            if not np.all(np.isfinite(poses[j])):
                continue
            dep = cv2.imread(os.path.join(sdir, f"frame-{frames[j]:06d}"
                                          ".depth.pgm"), cv2.IMREAD_UNCHANGED)
            depths.append((dep.astype(np.float64) / info["depth_shift"])
                          .astype(np.float16))
            T_w.append(WT @ poses[j])
        gt_objs = []
        for i, e in sorted(gt[k].items()):
            if e["position"] is None:
                continue
            gt_objs.append({
                "gt_id": str(i), "type": names.get(i, "unknown"),
                "nyu40": nyu.get(i), "group": "object",
                "position": [float(x) for x in e["position"]],
                "present": e["present"],
                "moved_since_prev": e["moved_since_prev"],
                "appeared_since_prev": e["appeared_since_prev"],
                "disappeared_since_prev": e["disappeared_since_prev"]})
        preps.append({
            "split": split, "scene": env["reference"], "k": k,
            "raw_dir": sdir,
            "gt": {"time": k * DAY_S, "gap_days": 1.0,
                   "pose_cov": (POSE_SIGMA_M ** 2 * np.eye(3)).tolist(),
                   "objects": gt_objs},
            "K": info["color_K"], "K_depth": info["depth_K"],
            "T_wc_est": np.array(T_w).reshape(-1, 4, 4),
            "T_wc_gt": np.array(T_w).reshape(-1, 4, 4),
            "depth_stack": np.array(depths, dtype=np.float16).reshape(
                -1, info["depth_size"][1], info["depth_size"][0]),
            "tracks": tracks, "n_frames": len(frames),
            "frame_files": [os.path.join(sdir, f"frame-{f:06d}.color.jpg")
                            for f in frames],
            "rgb": None,
            "meta": {"scan_id": sid, "env": env["reference"],
                     "official_split": env["type"], "time_assumed": True,
                     "segmentation": "oracle (annotated mesh rendered at "
                                     "the frame poses)",
                     "depth_downsample": 1, "frame_stride": frame_stride,
                     "render_scale": RENDER_SCALE,
                     "centroid_extent_frac": CENTROID_EXTENT_FRAC,
                     "citation": CITATION}})
    return preps


# ── Figures: top-down view of a scan ─────────────────────────────────────────

CEILING_NYU40 = 22


def vertex_colors(scan_path: str, ply: Dict[str, np.ndarray],
                  stride: int = 3, tol: float = 0.05) -> np.ndarray:
    """Mean RGB (uint8) of every mesh vertex over the colour frames in which
    it is visible (sensor depth within ``tol`` of the vertex); grey where it
    is never seen."""
    import cv2
    seq = os.path.join(scan_path, "sequence")
    info = read_info(os.path.join(seq, "_info.txt"))
    Kc, Kd = info["color_K"], info["depth_K"]
    xyz = ply["xyz"]
    acc = np.zeros((len(xyz), 3))
    cnt = np.zeros(len(xyz))
    for f in list_frames(seq)[::stride]:
        T = read_pose(seq, f)
        if not np.all(np.isfinite(T)):
            continue
        Xc = (xyz - T[:3, 3]) @ T[:3, :3]
        z = Xc[:, 2]
        idx = np.nonzero(z > 0.1)[0]
        dep = cv2.imread(os.path.join(seq, f"frame-{f:06d}.depth.pgm"),
                         cv2.IMREAD_UNCHANGED).astype(np.float64) / \
            info["depth_shift"]
        ud = (Kd[0, 0] * Xc[idx, 0] / z[idx] + Kd[0, 2]).astype(np.int64)
        vd = (Kd[1, 1] * Xc[idx, 1] / z[idx] + Kd[1, 2]).astype(np.int64)
        m = (ud >= 0) & (ud < dep.shape[1]) & (vd >= 0) & (vd < dep.shape[0])
        idx, ud, vd = idx[m], ud[m], vd[m]
        d = dep[vd, ud]
        vis = idx[(d > 0) & (np.abs(d - z[idx]) < tol)]
        img = cv2.imread(os.path.join(seq, f"frame-{f:06d}.color.jpg"))[..., ::-1]
        uc = (Kc[0, 0] * Xc[vis, 0] / z[vis] + Kc[0, 2]).astype(np.int64)
        vc = (Kc[1, 1] * Xc[vis, 1] / z[vis] + Kc[1, 2]).astype(np.int64)
        m = (uc >= 0) & (uc < img.shape[1]) & (vc >= 0) & (vc < img.shape[0])
        acc[vis[m]] += img[vc[m], uc[m]]
        cnt[vis[m]] += 1
    col = np.full((len(xyz), 3), 170.0)
    seen = cnt > 0
    col[seen] = acc[seen] / cnt[seen, None]
    return col.astype(np.uint8)


def topdown(scan_path: str, W_T: np.ndarray,
            bounds: Tuple[float, float, float, float], ppm: float = 60.0,
            z_max: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Top-down RGB view of one scan in the world frame (floor at z = 0),
    without the ceiling and everything above ``z_max``: (image, 3x3 H
    mapping world (x, y, 1) to pixels); ``bounds`` = (x0, y0, x1, y1)."""
    ply = read_label_ply(os.path.join(scan_path,
                                      "labels.instances.annotated.v2.ply"))
    col = vertex_colors(scan_path, ply)
    keep_v = ply["NYU40"] != CEILING_NYU40
    P, V = surface_samples(ply["xyz"], ply["faces"],
                           np.arange(len(ply["xyz"])), spacing=0.01)
    Pw = to_world(W_T, P)
    x0, y0, x1, y1 = bounds
    w, h = int(np.ceil((x1 - x0) * ppm)), int(np.ceil((y1 - y0) * ppm))
    u = ((Pw[:, 0] - x0) * ppm).astype(np.int64)
    v = ((y1 - Pw[:, 1]) * ppm).astype(np.int64)
    m = keep_v[V] & (Pw[:, 2] < z_max) & (u >= 0) & (u < w) & (v >= 0) & \
        (v < h)
    order = np.argsort(Pw[m, 2])            # highest surface drawn last
    img = np.full((h, w, 3), 255, np.uint8)
    img[v[m][order], u[m][order]] = col[V[m][order]]
    # fill single-pixel gaps between samples from their neighbours
    import cv2
    empty = np.zeros((h, w), np.uint8)
    empty[v[m], u[m]] = 1
    fill = cv2.dilate(img * empty[..., None], np.ones((3, 3), np.uint8))
    near = cv2.dilate(empty, np.ones((3, 3), np.uint8)).astype(bool)
    gap = near & (empty == 0)
    img[gap] = fill[gap]
    H = np.array([[ppm, 0.0, -x0 * ppm], [0.0, -ppm, y1 * ppm],
                  [0.0, 0.0, 1.0]])
    return img, H


def scan_world_transform(index: Dict[str, dict], root: str, env_id: str,
                         scan_id: str) -> np.ndarray:
    """World-from-scan transform of one scan of an environment (W @ T)."""
    W = floor_shift(read_label_ply(os.path.join(
        scan_dir(root, env_id), "labels.instances.annotated.v2.ply")))
    if scan_id == env_id:
        return W
    T = next(r["T"] for r in index[env_id]["rescans"] if r["id"] == scan_id)
    return W @ T
