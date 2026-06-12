#!/usr/bin/env python3
"""
scripts/openloris_to_sessions.py
================================
Extract OpenLORIS-Scene ROS bags into the TUM-style 'package' directory
layout that scripts/run_openloris_eval.py consumes:

    <out>/<sequence-name>/
        color.txt  color/           (from /d400/color/image_raw)
        aligned_depth.txt aligned_depth/
                                    (from /d400/aligned_depth_to_color/image_raw)
        odom.txt                    (from /odom: pose + linear twist)

groundtruth.txt is shipped separately by the dataset (per-sequence
download) — copy it into the output directory alongside the extracted
files. If your download already contains extracted package directories,
no conversion is needed.

Requires rosbag + cv_bridge (run inside the ROS environment):
    source /opt/ros/noetic/setup.bash
    python3 scripts/openloris_to_sessions.py --bag office1-1.bag --out /data/openloris
"""

import argparse
import os
import sys


def extract_bag(bag_path: str, out_root: str,
                color_topic: str, depth_topic: str, odom_topic: str):
    try:
        import cv2
        import rosbag
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f"rosbag/cv_bridge unavailable ({e}). "
              "Source your ROS environment first.")
        sys.exit(1)

    name = os.path.splitext(os.path.basename(bag_path))[0]
    seq_dir = os.path.join(out_root, name)
    color_dir = os.path.join(seq_dir, "color")
    depth_dir = os.path.join(seq_dir, "aligned_depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    bridge = CvBridge()
    color_lines, depth_lines, odom_lines = [], [], []

    print(f"Extracting {bag_path} -> {seq_dir}")
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(
                topics=[color_topic, depth_topic, odom_topic]):
            ts = msg.header.stamp.to_sec() if hasattr(msg, "header") \
                else t.to_sec()
            if topic == color_topic:
                img = bridge.imgmsg_to_cv2(msg, "bgr8")
                rel = f"color/{ts:.6f}.png"
                cv2.imwrite(os.path.join(seq_dir, rel), img)
                color_lines.append(f"{ts:.6f} {rel}")
            elif topic == depth_topic:
                d = bridge.imgmsg_to_cv2(msg, "passthrough")
                rel = f"aligned_depth/{ts:.6f}.png"
                cv2.imwrite(os.path.join(seq_dir, rel), d)
                depth_lines.append(f"{ts:.6f} {rel}")
            elif topic == odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                v = msg.twist.twist.linear
                w = msg.twist.twist.angular
                odom_lines.append(
                    f"{ts:.6f} {p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w} "
                    f"{v.x} {v.y} {v.z} {w.x} {w.y} {w.z}")

    for fname, lines in [("color.txt", color_lines),
                         ("aligned_depth.txt", depth_lines),
                         ("odom.txt", odom_lines)]:
        with open(os.path.join(seq_dir, fname), "w") as f:
            f.write("# timestamp data\n")
            f.write("\n".join(lines) + "\n")
    print(f"  color: {len(color_lines)}  depth: {len(depth_lines)}  "
          f"odom: {len(odom_lines)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", nargs="+", required=True,
                    help="One or more OpenLORIS-Scene .bag files")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--color-topic", default="/d400/color/image_raw")
    ap.add_argument("--depth-topic",
                    default="/d400/aligned_depth_to_color/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    args = ap.parse_args()

    for bag in args.bag:
        extract_bag(bag, args.out,
                    args.color_topic, args.depth_topic, args.odom_topic)


if __name__ == "__main__":
    main()
