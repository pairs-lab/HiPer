"""
hiper.evaluation.thor_tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Raw THOR-Lifelong sessions (thor_world.py output) -> SessionData
(object_session.py), one .npz per session.

Within-session tracking uses the GT instance ids (as in BYE's protocol).
Per session and object (Floor is never in the instance images):
  * a frame counts if the mask covers >= 0.05 % of the image and has valid
    depth (0 < d <= 10 m: through windows AI2-THOR returns the 20 m far
    plane, which would put a window's centroid ~18 m outside); mask pixels (stride 2) are back-projected with the ESTIMATED
    pose; per-frame centroid = per-axis median;
  * centroid = pixel-weighted mean of the per-frame centroids,
    centroid_cov = cov(per-frame centroids) / n + (0.01 m)^2 I;
  * appearance: square crop = mask bbox + 15 % padding per side, 224 px
    (roi_align, zero outside the image); CLIP ViT-B/32 (clip_vocab) and
    DINOv2 ViT-B/14 CLS, both L2-normalised per crop; clip_mean / dino_mean
    = normalised mean, clip_disp = 1 - ||mean of unit embeddings||^2;
    zero-shot label = argmax of the frame-averaged softmax over the THOR
    types (clip_vocab), top-5 in SessionData.meta['clip_top5'];
  * points: xyz (estimated world) + rgb in [0, 1], 1 cm voxel mean,
    random subsample <= 2048; extent = 5-95 percentile box size;
    pixel_area = mean mask fraction; eps_l = 0 (no motion within a session).
  * depth_stack: depth downsampled 4x (block median of valid pixels) to
    160x120 float16, K scaled by 1/4.

CPU stage (``prepare_session``) is pure numpy/cv2 and runs in worker
processes; the GPU stage (``TrackEncoders``) imports torch lazily.
"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hiper.evaluation.object_session import SessionData, Track
from hiper.evaluation.thor_world import backproject, make_rng

MIN_AREA_FRAC = 0.0005
BACKPROJ_STRIDE = 2
MIN_DEPTH_POINTS = 3          # valid-depth pixels at stride 2 per mask
MAX_DEPTH_M = 10.0            # sensor range: windows show the 20 m far plane
CENTROID_STD_FLOOR = 0.01     # m
CROP_PAD = 0.15
CROP_SIZE = 224
VOXEL_M = 0.01
MAX_POINTS = 2048
DEPTH_DS = 4
DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


# ── Pure helpers ─────────────────────────────────────────────────────────────

def square_crop_box(u0: float, v0: float, u1: float, v1: float, W: int,
                    H: int, pad: float = CROP_PAD
                    ) -> Tuple[float, float, float, float]:
    """Square box around a bbox [u0, u1) x [v0, v1) (continuous pixel
    coordinates) enlarged by ``pad`` of the longer side on each side;
    shifted inside the image when it fits, else centred (zero padding)."""
    cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    side = max(u1 - u0, v1 - v0) * (1.0 + 2.0 * pad)
    side = max(side, 2.0)
    x0, y0 = cu - side / 2.0, cv - side / 2.0
    if side <= W:
        x0 = min(max(x0, 0.0), W - side)
    if side <= H:
        y0 = min(max(y0, 0.0), H - side)
    return x0, y0, x0 + side, y0 + side


def voxel_downsample(P: np.ndarray, voxel: float = VOXEL_M) -> np.ndarray:
    """Mean of the rows (xyz + extra columns) falling in each voxel."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 0:
        return P.reshape(0, P.shape[1] if P.ndim == 2 else 6)
    q = np.floor(P[:, :3] / voxel).astype(np.int64)
    q -= q.min(axis=0)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), P.shape[1]))
    for j in range(P.shape[1]):
        out[:, j] = np.bincount(inv, weights=P[:, j], minlength=len(cnt))
    return out / cnt[:, None]


def downsample_depth(depth: np.ndarray, factor: int = DEPTH_DS) -> np.ndarray:
    """Block median of the valid (> 0) pixels (mean of the two middle
    values for an even count); 0 where a block has none."""
    d = np.asarray(depth, dtype=np.float64)
    H, W = d.shape
    h, w = H // factor, W // factor
    b = d[:h * factor, :w * factor].reshape(h, factor, w, factor)
    b = b.transpose(0, 2, 1, 3).reshape(h, w, factor * factor)
    b = np.where(b > 0, b, 0.0)
    s = np.sort(b, axis=2)                 # invalid zeros first
    nz = (b <= 0).sum(axis=2)
    nv = factor * factor - nz
    lo = np.clip(nz + (nv - 1) // 2, 0, factor * factor - 1)
    hi = np.clip(nz + nv // 2, 0, factor * factor - 1)
    med = 0.5 * (np.take_along_axis(s, lo[..., None], 2)[..., 0] +
                 np.take_along_axis(s, hi[..., None], 2)[..., 0])
    return np.where(nv > 0, med, 0.0)


def scale_K(K: np.ndarray, factor: int = DEPTH_DS) -> np.ndarray:
    """Intrinsics of an image downsampled by ``factor`` (pixel (i, j)
    covers [i, i+1) x [j, j+1) in both images)."""
    Ks = np.array(K, dtype=np.float64, copy=True)
    Ks[:2] /= float(factor)
    return Ks


def frame_observations(rgb: np.ndarray, depth: np.ndarray,
                       inst: np.ndarray, T_wc: np.ndarray, K: np.ndarray,
                       stride: int = BACKPROJ_STRIDE,
                       min_area_frac: float = MIN_AREA_FRAC,
                       min_points: int = MIN_DEPTH_POINTS) -> Dict[int, dict]:
    """Per instance index (> 0) of one frame: {'pixels', 'area_frac',
    'bbox' (u0, v0, u1, v1), 'centroid' (median of the back-projected
    stride-``stride`` mask points), 'points' (M, 6) xyz + rgb[0,1]}."""
    from scipy.ndimage import find_objects
    H, W = inst.shape
    inst = np.ascontiguousarray(inst)
    cnt = np.bincount(inst.reshape(-1))
    min_px = min_area_frac * H * W
    cand = [i for i in np.nonzero(cnt >= min_px)[0].tolist() if i > 0]
    if not cand:
        return {}
    boxes = find_objects(inst.astype(np.int32))
    P, pix = backproject(depth, K, T_wc, stride=stride)
    if len(pix) == 0:
        return {}
    sub = inst[pix[:, 0], pix[:, 1]].astype(np.int64)
    order = np.argsort(sub, kind="stable")
    c2 = np.bincount(sub, minlength=len(cnt))
    st = np.concatenate([[0], np.cumsum(c2)])
    col = rgb[pix[:, 0], pix[:, 1]].astype(np.float64) / 255.0
    out = {}
    for i in cand:
        seg = order[st[i]:st[i + 1]]
        if len(seg) < min_points:
            continue
        sl = boxes[i - 1]
        pts = P[seg]
        out[i] = {"pixels": int(cnt[i]), "area_frac": float(cnt[i] / (H * W)),
                  "bbox": (float(sl[1].start), float(sl[0].start),
                           float(sl[1].stop), float(sl[0].stop)),
                  "centroid": np.median(pts, axis=0),
                  "points": np.hstack([pts, col[seg]])}
    return out


def aggregate_centroids(cents: np.ndarray, pixels: np.ndarray,
                        floor_std: float = CENTROID_STD_FLOOR
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-weighted mean of per-frame centroids and
    cov(per-frame centroids) / n + floor_std^2 I."""
    C = np.asarray(cents, dtype=np.float64).reshape(-1, 3)
    w = np.asarray(pixels, dtype=np.float64).reshape(-1)
    c = (w[:, None] * C).sum(0) / w.sum()
    n = len(C)
    cov = np.cov(C.T, ddof=1) / n if n >= 2 else np.zeros((3, 3))
    return c, cov + floor_std ** 2 * np.eye(3)


def unit_mean(E: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """(normalised mean, dispersion 1 - ||mean of unit rows||^2)."""
    E = np.asarray(E, dtype=np.float64)
    if len(E) == 0:
        return None, 0.0
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    m = U.mean(0)
    nm = float(np.linalg.norm(m))
    return (m / max(nm, 1e-12)).astype(np.float32), float(
        min(max(1.0 - nm * nm, 0.0), 1.0))


def extent_of(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or len(points) == 0:
        return None
    return (np.percentile(points[:, :3], 95, axis=0) -
            np.percentile(points[:, :3], 5, axis=0)).astype(np.float32)


# ── CPU stage ────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def prepare_session(raw_dir: str, split: str, scene: str, k: int,
                    keep_rgb: bool = False) -> dict:
    """Decode one raw session and build per-object geometry. Returns a
    picklable dict for the GPU stage (with the RGB frames, uint8
    (n, H, W, 3), if ``keep_rgb``)."""
    import cv2
    cv2.setNumThreads(1)
    gt = _load_json(os.path.join(raw_dir, "gt.json"))
    fdir = os.path.join(raw_dir, "frames")
    n = int(gt["n_frames"])
    obs: Dict[str, List[dict]] = {}
    T_est, T_gt, depths, rgbs = [], [], [], []
    K = None
    for f in range(n):
        fj = _load_json(os.path.join(fdir, f"{f:03d}.json"))
        rgb = np.ascontiguousarray(cv2.imread(
            os.path.join(fdir, f"{f:03d}_rgb.jpg"))[..., ::-1])
        if keep_rgb:
            rgbs.append(rgb)
        dep = cv2.imread(os.path.join(fdir, f"{f:03d}_depth.png"),
                         cv2.IMREAD_UNCHANGED).astype(np.float64) * float(
            fj.get("depth_scale", 0.001))
        inst = cv2.imread(os.path.join(fdir, f"{f:03d}_inst.png"),
                          cv2.IMREAD_UNCHANGED)
        K = np.asarray(fj["K"], dtype=np.float64)
        Te = np.asarray(fj["T_wc_est"], dtype=np.float64)
        T_est.append(Te)
        T_gt.append(np.asarray(fj["T_wc_gt"], dtype=np.float64))
        depths.append(downsample_depth(dep).astype(np.float16))
        dep[dep > MAX_DEPTH_M] = 0.0
        for i, o in frame_observations(rgb, dep, inst, Te, K).items():
            gid = fj["instances"][str(i)]["gt_id"]
            o["frame"] = f
            o["crop"] = square_crop_box(*o["bbox"], inst.shape[1],
                                        inst.shape[0])
            obs.setdefault(gid, []).append(o)
    tracks = []
    for gid in sorted(obs):
        ol = obs[gid]
        cents = np.array([o["centroid"] for o in ol])
        pix = np.array([o["pixels"] for o in ol])
        c, cov = aggregate_centroids(cents, pix)
        pts = voxel_downsample(np.vstack([o["points"] for o in ol]))
        rng = make_rng(split, scene, 0, "points", k, gid)
        if len(pts) > MAX_POINTS:
            pts = pts[np.sort(rng.choice(len(pts), MAX_POINTS,
                                         replace=False))]
        tracks.append({
            "gt_id": gid, "centroid": c, "centroid_cov": cov,
            "n_obs": len(ol), "pixel_area": float(np.mean(
                [o["area_frac"] for o in ol])),
            "points": pts.astype(np.float32), "extent": extent_of(pts),
            "frames": [o["frame"] for o in ol],
            "crops": [o["crop"] for o in ol]})
    return {"split": split, "scene": scene, "k": k, "raw_dir": raw_dir,
            "gt": gt, "K": K, "T_wc_est": np.array(T_est),
            "T_wc_gt": np.array(T_gt), "depth_stack": np.array(depths),
            "tracks": tracks, "n_frames": n,
            "rgb": np.stack(rgbs) if keep_rgb else None}


# ── GPU stage ────────────────────────────────────────────────────────────────

class TrackEncoders:
    """CLIP (clip_vocab) + DINOv2 on square crops of the session frames."""

    def __init__(self, device: Optional[str] = None,
                 types: Optional[Sequence[str]] = None):
        import timm
        import torch
        from hiper.evaluation.clip_vocab import ClipVocab
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.clip = ClipVocab(types=types, device=self.device)
        self.dino = timm.create_model(DINO_MODEL, pretrained=True,
                                      num_classes=0,
                                      img_size=CROP_SIZE).to(
            self.device).eval()
        self.dmean = torch.tensor(DINO_MEAN, device=self.device).view(
            1, 3, 1, 1)
        self.dstd = torch.tensor(DINO_STD, device=self.device).view(
            1, 3, 1, 1)

    def _frames(self, raw_dir: str, n: int):
        """(n, 3, H, W) float16 RGB in [0, 1] on the device."""
        torch = self.torch
        import torchvision
        fdir = os.path.join(raw_dir, "frames")
        data = [torchvision.io.read_file(os.path.join(fdir, f"{f:03d}_rgb.jpg"))
                for f in range(n)]
        try:
            imgs = torchvision.io.decode_jpeg(data, device=self.device)
        except (RuntimeError, TypeError):
            imgs = [torchvision.io.decode_jpeg(d).to(self.device)
                    for d in data]
        return torch.stack(imgs).to(torch.float16) / 255.0

    def encode(self, raw_dir: str, n_frames: int, frames: Sequence[int],
               crops: Sequence[Tuple[float, float, float, float]],
               chunk: int = 128, rgb: Optional[np.ndarray] = None,
               rots: Optional[Sequence[int]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """CLIP and DINOv2 embeddings (L2-normalised) of the square crops.
        Frames stay float16 on the device; each chunk of crops is cut
        (roi_align, float32) from only the frames it needs. ``rots``:
        optional counter-clockwise quarter turns per crop (upright crops
        from sideways frames, 3RScan)."""
        torch = self.torch
        from torchvision.ops import roi_align
        if len(frames) == 0:
            return np.zeros((0, 512), np.float32), np.zeros((0, 768),
                                                            np.float32)
        if rgb is not None:
            imgs = torch.from_numpy(np.ascontiguousarray(rgb)).to(
                self.device).permute(0, 3, 1, 2).to(torch.float16) / 255.0
        else:
            imgs = self._frames(raw_dir, n_frames)
        fr = np.asarray(frames, dtype=np.int64)
        bx = np.asarray(crops, dtype=np.float32).reshape(-1, 4)
        order = np.argsort(fr, kind="stable")
        ce = np.zeros((len(fr), 512), np.float32)
        de = np.zeros((len(fr), 768), np.float32)
        with torch.no_grad():
            for i in range(0, len(order), chunk):
                idx = order[i:i + chunk]
                uf, local = np.unique(fr[idx], return_inverse=True)
                sub = imgs[torch.as_tensor(uf, device=self.device)].float()
                boxes = torch.as_tensor(np.hstack([local[:, None].astype(
                    np.float32), bx[idx]]), device=self.device)
                x = roi_align(sub, boxes, output_size=(CROP_SIZE, CROP_SIZE),
                              spatial_scale=1.0, sampling_ratio=-1,
                              aligned=True).clamp_(0.0, 1.0)
                del sub
                if rots is not None:
                    for j, r in enumerate(np.asarray(rots)[idx].tolist()):
                        if r % 4:
                            x[j] = torch.rot90(x[j], r, dims=(1, 2))
                ce[idx] = self.clip.encode(x)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self.device.startswith("cuda")):
                    d = self.dino((x - self.dmean) / self.dstd)
                d = d.float()
                de[idx] = (d / d.norm(dim=-1, keepdim=True)).cpu().numpy()
                del x, d
        del imgs
        return ce, de


def build_session(prep: dict, enc: TrackEncoders) -> SessionData:
    """GPU stage: appearance + labels, then assemble the SessionData."""
    frames, crops, owner, rots = [], [], [], []
    for ti, t in enumerate(prep["tracks"]):
        frames += t["frames"]
        crops += t["crops"]
        owner += [ti] * len(t["frames"])
        rots += t.get("rots", [0] * len(t["frames"]))
    kw = {"rots": rots} if any(rots) else {}
    clip_e, dino_e = enc.encode(prep["raw_dir"], prep["n_frames"], frames,
                                crops, rgb=prep.get("rgb"), **kw)
    owner = np.asarray(owner)
    tracks, top5 = [], {}
    for ti, t in enumerate(prep["tracks"]):
        sel = owner == ti
        cm, cd = unit_mean(clip_e[sel])
        dm, _ = unit_mean(dino_e[sel])
        label, conf, top = enc.clip.label(clip_e[sel])
        top5[str(ti)] = top
        tracks.append(Track(
            track_id=ti, centroid=np.asarray(t["centroid"], np.float64),
            centroid_cov=np.asarray(t["centroid_cov"], np.float64),
            n_obs=int(t["n_obs"]), clip_mean=cm, clip_disp=cd,
            clip_label=label, clip_label_conf=float(conf), dino_mean=dm,
            points=t["points"], extent=t["extent"],
            pixel_area=float(t["pixel_area"]), eps_l=0.0,
            gt_id=t["gt_id"]))
    gt = prep["gt"]
    from hiper.evaluation.clip_vocab import CLIP_MODEL, CLIP_PRETRAINED
    meta = {"split": prep["split"], "gap_days": gt.get("gap_days"),
            "n_frames": prep["n_frames"], "source": prep["raw_dir"],
            "clip_model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
            "dino_model": DINO_MODEL, "clip_top5": top5,
            "min_area_frac": MIN_AREA_FRAC, "crop_pad": CROP_PAD,
            "max_depth_m": MAX_DEPTH_M,
            "depth_downsample": DEPTH_DS, **prep.get("meta", {})}
    # K_depth: intrinsics of a depth stack stored at its own resolution
    # (3RScan); otherwise the stack is the frames downsampled by DEPTH_DS
    K_depth = prep.get("K_depth")
    return SessionData(
        scene=prep["scene"], session_index=int(prep["k"]),
        time=float(gt["time"]), tracks=tracks,
        T_wc_est=prep["T_wc_est"],
        K=scale_K(prep["K"]) if K_depth is None else np.asarray(K_depth),
        depth_stack=prep["depth_stack"],
        pose_cov=np.asarray(gt["pose_cov"], dtype=np.float64),
        T_wc_gt=prep["T_wc_gt"], gt_objects=gt["objects"], meta=meta)
