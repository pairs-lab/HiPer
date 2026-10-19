#!/usr/bin/env python3
"""
scripts/run_openloris_eval.py
=============================
Multi-session lifelong SLAM evaluation on OpenLORIS-Scene
(https://lifelong-robotic-vision.github.io/dataset/).

For one scene (e.g. office), runs its sequences in recording order as
lifelong sessions S_1..S_N: the long-term map produced by session i is
loaded at the start of session i+1, exercising cross-session ephemerality,
Beta-belief and sparsification exactly as in deployment. Reports per-session
ATE RMSE and the OpenLORIS C-rate.

Usage
-----
  # Wheel-odometry pose source (no GPU needed) over the office scene
  python3 scripts/run_openloris_eval.py \
      --data-root /data/openloris --scene office --mode odom \
      --map-dir /tmp/hiper_openloris --out results_office.json

  # Continual-VIO pose source (loads ~/.hiper/checkpoints/generalizer.pt)
  python3 scripts/run_openloris_eval.py \
      --data-root /data/openloris --scene office --mode vio --device cuda

Expects extracted 'package'-format sequence directories (color.txt,
aligned_depth.txt, groundtruth.txt, odom.txt). For ROS bags, first run
scripts/openloris_to_sessions.py.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running without catkin: register src/ as the hiper package
_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT / "tests"))
import conftest  # noqa: F401,E402  (registers hiper pkg + rospy stub)

from hiper.evaluation.lifelong_runner import LifelongEvaluator  # noqa: E402
from hiper.evaluation.openloris import (  # noqa: E402
    discover_scene_sessions,
    load_sequence,
)

# Default class-conditioned survival priors (paper Table I)
DEFAULT_CLASS_PRIORS = {
    "building": 1e-7, "wall": 1e-7,
    "tree": 1e-6, "lamppost": 1e-6,
    "bench": 5e-5, "trash can": 5e-5,
    "car": 5e-4, "vehicle": 5e-4,
    "bicycle": 8e-4,
    "person": 2e-3,
    # indoor classes relevant to OpenLORIS scenes
    "table": 5e-5, "sofa": 5e-5, "shelf": 5e-5,
    "chair": 5e-4, "door": 1e-6, "monitor": 5e-5,
}


def build_vio(device: str, fx, fy, cx, cy):
    """Instantiate DepthPoseNet for offline inference (patched _img)."""
    import numpy as np
    import torch
    from hiper.vio.depth_pose_net import DepthPoseNet

    def _img_patched(self, msg):
        rgb = msg._rgb.astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._dev)

    DepthPoseNet._img = _img_patched
    net = DepthPoseNet(freeze_encoder=True, device=device)
    net.update_K(fx=fx, fy=fy, cx=cx, cy=cy)
    net.try_load_checkpoint()
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="Directory containing sequence dirs (office1-1, ...)")
    ap.add_argument("--scene", required=True,
                    help="Scene name (office, corridor, home, cafe, market) "
                         "or scene+site prefix (office1)")
    ap.add_argument("--sessions", type=int, default=0,
                    help="Limit to the first N sessions (0 = all)")
    ap.add_argument("--mode", choices=["odom", "vio"], default="odom")
    ap.add_argument("--stride", type=int, default=1,
                    help="Frame subsampling stride")
    ap.add_argument("--map-dir", default="/tmp/hiper_openloris")
    ap.add_argument("--out", default="openloris_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau-pos", type=float, default=0.3,
                    help="C-rate position-error threshold (m)")
    # D435i color defaults (848x480), overridden by sensors.yaml when present
    ap.add_argument("--fx", type=float, default=611.0)
    ap.add_argument("--fy", type=float, default=611.0)
    ap.add_argument("--cx", type=float, default=424.0)
    ap.add_argument("--cy", type=float, default=240.0)
    args = ap.parse_args()

    session_dirs = discover_scene_sessions(args.data_root, args.scene)
    if not session_dirs:
        print(f"No sequences for scene '{args.scene}' under {args.data_root}")
        sys.exit(1)
    if args.sessions > 0:
        session_dirs = session_dirs[:args.sessions]
    print(f"Scene '{args.scene}': {len(session_dirs)} sessions")

    evaluator = LifelongEvaluator(map_dir=args.map_dir,
                                  class_priors=DEFAULT_CLASS_PRIORS)
    vio = None
    if args.mode == "vio":
        first = load_sequence(session_dirs[0], stride=args.stride)
        intr = first.intrinsics or {"fx": args.fx, "fy": args.fy,
                                    "cx": args.cx, "cy": args.cy}
        # Scale intrinsics from native 848x480 to the 640x192 net input
        sx, sy = 640.0 / 848.0, 192.0 / 480.0
        vio = build_vio(args.device,
                        fx=intr["fx"] * sx, fy=intr["fy"] * sy,
                        cx=intr["cx"] * sx, cy=intr["cy"] * sy)

    rows = []
    for sdir in session_dirs:
        seq = load_sequence(sdir, stride=args.stride)
        res = evaluator.run_session(seq, pose_source=args.mode, vio=vio,
                                    tau_pos=args.tau_pos)
        rows.append({
            "session": res.name, "ate_rmse": res.ate, "c_rate": res.c_rate,
            "n_frames": res.n_frames, "n_objects": res.n_objects,
            "n_pruned": res.n_pruned,
        })
        print(f"  {res.name:14s} ATE={res.ate:7.3f} m  "
              f"C-rate@{args.tau_pos}m={res.c_rate:6.3f}  "
              f"objects={res.n_objects}  pruned={res.n_pruned}")

    finite = [r["ate_rmse"] for r in rows
              if r["ate_rmse"] == r["ate_rmse"]]   # drop NaN
    summary = {
        "scene": args.scene,
        "mode": args.mode,
        "mean_ate": sum(finite) / len(finite) if finite else None,
        "mean_c_rate": sum(r["c_rate"] for r in rows) / len(rows)
            if rows else None,
        "sessions": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMean ATE: {summary['mean_ate']}  "
          f"Mean C-rate: {summary['mean_c_rate']}")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
