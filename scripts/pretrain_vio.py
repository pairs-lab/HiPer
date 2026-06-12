#!/usr/bin/env python3
"""
scripts/pretrain_vio.py
========================
Offline pre-training of the SeLiSLAM DepthNet + PoseNet on a
source domain before the first deployment session.

Mirrors CoVIO / CL-SLAM: pre-train on Cityscapes or your own
UAV footage so the online adaptation starts from a reasonable
initialisation rather than random weights.

Two input modes
---------------
  --bag PATH     : read frames from a ROS bag file (rosbag Python API)
  --img-dir PATH : read a flat directory of sequentially-named JPG/PNG

Usage examples
--------------
  # Pre-train on a ROS bag (1 epoch)
  python3 scripts/pretrain_vio.py \
      --bag /data/cityscapes_flight.bag \
      --image-topic /camera/color/image_raw \
      --depth-topic /camera/depth/image_rect_raw \
      --epochs 25 \
      --lr 1e-4 \
      --out ~/.seliSlam/checkpoints/generalizer.pt

  # Pre-train on an image folder (no depth → monocular mode)
  python3 scripts/pretrain_vio.py \
      --img-dir /data/cityscapes_images/ \
      --epochs 25 \
      --lr 1e-4 \
      --out ~/.seliSlam/checkpoints/generalizer.pt

Output
------
Saves a generalizer.pt checkpoint that vio_frontend_node.py loads
automatically at session start via try_load_checkpoint().
"""

import argparse
import os
import sys
import logging
import random
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="[pretrain_vio] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Allow running without catkin: register src/ as the seliSlam package
# (and stub rospy) via the tests' conftest bootstrap.
_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_folder(img_dir: str):
    """
    Load a sorted list of (H, W, 3) uint8 numpy arrays from a directory.
    Returns list of np.ndarray.
    """
    import cv2
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    paths  = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    log.info(f"Found {len(paths)} images in {img_dir}")
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 192))
        imgs.append(img.astype(np.float32) / 255.0)
    return imgs


def load_bag(bag_path: str, image_topic: str,
             depth_topic: str = None):
    """
    Extract (image, depth) pairs from a ROS bag.
    Returns list of (rgb_np float32 H×W×3, depth_np float32 H×W or None).
    """
    try:
        import rosbag
        from cv_bridge import CvBridge
        import cv2
    except ImportError:
        raise RuntimeError(
            "rosbag / cv_bridge not importable. "
            "Run this script from the catkin workspace with "
            "`source devel/setup.bash`.")

    bridge = CvBridge()
    pairs  = []
    log.info(f"Reading bag: {bag_path}")

    with rosbag.Bag(bag_path, "r") as bag:
        rgb_buf   = {}
        depth_buf = {}

        for topic, msg, t in bag.read_messages(
                topics=[t for t in [image_topic, depth_topic] if t]):
            ts = t.to_sec()
            if topic == image_topic:
                img = bridge.imgmsg_to_cv2(msg, "rgb8")
                img = cv2.resize(img, (640, 192))
                rgb_buf[ts] = img.astype(np.float32) / 255.0
            elif topic == depth_topic:
                d   = bridge.imgmsg_to_cv2(msg, "passthrough")
                d   = cv2.resize(d.astype(np.float32),
                                  (640, 192),
                                  interpolation=cv2.INTER_NEAREST)
                d  /= 1000.0   # mm → m (RealSense convention)
                depth_buf[ts] = d

        # Match nearest depth to each RGB
        depth_times = sorted(depth_buf.keys())
        for ts, rgb in sorted(rgb_buf.items()):
            if depth_times:
                best = min(depth_times, key=lambda x: abs(x - ts))
                d    = depth_buf[best] if abs(best - ts) < 0.1 else None
            else:
                d = None
            pairs.append((rgb, d))

    log.info(f"Extracted {len(pairs)} frames from bag.")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Triplet sampler
# ─────────────────────────────────────────────────────────────────────────────

class TripletDataset:
    """
    Yields triplets of consecutive frames as fake 'frame dicts'
    matching the format expected by DepthPoseNet._loss().
    """

    def __init__(self, frames: list, batch_size: int = 4):
        self._frames      = frames   # list of (rgb np, depth np or None)
        self._batch_size  = batch_size
        self._n           = len(frames)

    def __len__(self):
        return max(0, self._n - 2)

    def random_batch(self, size: int = None):
        """Return a list of triplets (each = list of 3 fake frame dicts)."""
        size     = size or self._batch_size
        n_valid  = self._n - 2
        if n_valid <= 0:
            return []
        indices  = random.sample(range(n_valid), min(size, n_valid))
        batch    = []
        for i in indices:
            triplet = [self._make_frame(i),
                       self._make_frame(i+1),
                       self._make_frame(i+2)]
            batch.append(triplet)
        return batch

    def _make_frame(self, idx: int) -> dict:
        rgb, depth = self._frames[idx]
        return {
            "img_np":    rgb,     # (H,W,3) float32 — used below
            "depth_np":  depth,
            "velocity":  0.0,
            "stamp":     None,
            "img_msg":   _FakeImageMsg(rgb),
            "depth_msg": _FakeImageMsg(
                depth[:,:,np.newaxis].repeat(3,2) if depth is not None
                else np.zeros((192,640,3),np.float32)),
        }


class _FakeImageMsg:
    """
    Minimal duck-type of sensor_msgs/Image so DepthPoseNet._img()
    works without a live ROS master during offline training.
    """
    def __init__(self, rgb_np: np.ndarray):
        # Store RGB float32 (H,W,3); _img() will convert to tensor
        self._rgb = rgb_np
        self.height   = rgb_np.shape[0]
        self.width    = rgb_np.shape[1]
        self.encoding = "rgb8"
        self.data     = (rgb_np * 255).clip(0,255).astype(np.uint8).tobytes()
        self.step     = self.width * 3
        self.header   = type("H", (), {"stamp": None, "frame_id": "cam"})()


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch DepthPoseNet._img for offline use
# ─────────────────────────────────────────────────────────────────────────────

def _patch_depth_pose_net():
    """
    Replace DepthPoseNet._img() so it accepts _FakeImageMsg objects
    without needing cv_bridge (which requires a ROS master).
    """
    import torch
    from seliSlam.vio.depth_pose_net import DepthPoseNet

    def _img_patched(self, msg) -> torch.Tensor:
        rgb = msg._rgb.astype(np.float32)   # already (H,W,3) float32
        t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return t.to(self._dev)

    DepthPoseNet._img = _img_patched
    log.info("DepthPoseNet._img patched for offline training.")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(net, dataset: TripletDataset,
          epochs: int, steps_per_epoch: int,
          batch_size: int, out_path: str):
    """
    Main offline training loop.
    net : DepthPoseNet instance
    """
    log.info(f"Training: epochs={epochs}, "
             f"steps/epoch={steps_per_epoch}, batch={batch_size}")
    best_loss = float("inf")

    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = dataset.random_batch(size=batch_size)
            if not batch:
                break
            loss = net.update_step(batch, n_cycles=1)
            epoch_loss += loss

        avg = epoch_loss / max(steps_per_epoch, 1)
        log.info(f"Epoch {ep:3d}/{epochs}  avg_loss={avg:.5f}")

        if avg < best_loss:
            best_loss = avg
            net.save_generalizer()
            log.info(f"  → checkpoint saved (best so far: {best_loss:.5f})")

    log.info(f"Training complete. Best loss={best_loss:.5f}. "
             f"Checkpoint at {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline VIO pre-training for SeLiSLAM")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag",     help="Path to a ROS bag file")
    src.add_argument("--img-dir", help="Directory of sequential RGB images")

    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic",
                        default="/camera/depth/image_rect_raw")
    parser.add_argument("--epochs",          type=int,   default=25)
    parser.add_argument("--steps-per-epoch", type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--freeze-encoder",  action="store_true",
                        default=True,
                        help="Freeze encoder weights (decoder-only training)")
    parser.add_argument("--out",
                        default=os.path.expanduser(
                            "~/.seliSlam/checkpoints/generalizer.pt"),
                        help="Output checkpoint path")
    parser.add_argument("--fx", type=float, default=385.19)
    parser.add_argument("--fy", type=float, default=385.19)
    parser.add_argument("--cx", type=float, default=321.66)
    parser.add_argument("--cy", type=float, default=239.80)

    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.bag:
        pairs = load_bag(args.bag, args.image_topic, args.depth_topic)
    else:
        imgs  = load_image_folder(args.img_dir)
        pairs = [(img, None) for img in imgs]

    if len(pairs) < 3:
        log.error(f"Need at least 3 frames, got {len(pairs)}.")
        sys.exit(1)

    dataset = TripletDataset(pairs, batch_size=args.batch_size)
    log.info(f"Dataset: {len(dataset)} valid triplets.")

    # ── Build network ─────────────────────────────────────────────────────────
    _patch_depth_pose_net()   # allow offline _img() calls

    from seliSlam.vio.depth_pose_net import DepthPoseNet
    net = DepthPoseNet(freeze_encoder=args.freeze_encoder,
                       device=args.device)
    net.update_K(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    net.try_load_checkpoint()   # resume if a previous checkpoint exists

    # Override learning rate if specified
    if not net._stub:
        for pg in net._optim.param_groups:
            pg["lr"] = args.lr

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(net, dataset,
          epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch,
          batch_size=args.batch_size,
          out_path=args.out)


if __name__ == "__main__":
    main()
