#!/usr/bin/env python3
"""
segmentation_node.py
====================
Open-vocabulary instance segmentation node for SeLiSLAM.

Runs MobileSAM for instance masks and CLIP/DINOv2 for per-instance
semantic embeddings. Supports a socket-bridge mode that offloads
inference to a Python 3.12 server when running on Jetson (ROS Noetic
uses Python 3.8, but heavy VLMs run better on 3.12 with newer CUDA).

Subscribed topics:
  /camera/color/image_raw        sensor_msgs/Image
  /seliSlam/keyframe_pointcloud  sensor_msgs/PointCloud2  (sync trigger)

Published topics:
  /seliSlam/instance_masks       sensor_msgs/Image
      Encoding: 16UC1, each pixel = instance id (0 = background)
  /seliSlam/semantic_embeddings  std_msgs/Float32MultiArray
      Flattened [N, embed_dim] array; metadata in a companion JSON string
  /seliSlam/seg_viz              sensor_msgs/Image  (BGR debug overlay)
"""

import json
import rospy
import numpy as np
import threading

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
import message_filters

from seliSlam.common.params import get_params
from seliSlam.common.ros_utils import make_header, RateMonitor
from seliSlam.segmentation.seg_backend import SegmentationBackend


class SegmentationNode:
    """
    Runs segmentation synchronised with keyframe point clouds.
    Each output carries a stamp matching the triggering keyframe
    so downstream nodes can associate masks with point clouds.
    """

    def __init__(self):
        rospy.init_node("segmentation_node", anonymous=False)
        self.params = get_params()
        sp = self.params.seg

        # ── Backend (handles model loading or socket connection) ──────────────
        self.backend = SegmentationBackend(
            model_name=sp.model_name,
            clip_model=sp.clip_model,
            device=sp.device,
            confidence_threshold=sp.confidence_threshold,
            max_instances=sp.max_instances_per_frame,
            use_socket=sp.use_socket_bridge,
            socket_host=sp.socket_host,
            socket_port=sp.socket_port,
        )

        self._lock = threading.Lock()

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_masks = rospy.Publisher(
            "/seliSlam/instance_masks", Image, queue_size=5)
        self.pub_embeddings = rospy.Publisher(
            "/seliSlam/semantic_embeddings", Float32MultiArray, queue_size=5)
        self.pub_meta = rospy.Publisher(
            "/seliSlam/seg_metadata", String, queue_size=5)
        self.pub_viz = rospy.Publisher(
            "/seliSlam/seg_viz", Image, queue_size=2)

        # ── Subscribers (approximate time sync) ──────────────────────────────
        img_sub = message_filters.Subscriber(
            "/camera/color/image_raw", Image)
        kf_sub  = message_filters.Subscriber(
            "/seliSlam/keyframe_pointcloud", PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, kf_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self._callback)

        self._rate_monitor = RateMonitor("segmentation", window=20)
        rospy.loginfo("[segmentation] Node ready. "
                      f"backend={'socket' if sp.use_socket_bridge else sp.model_name}")

    # ── Main callback ─────────────────────────────────────────────────────────

    def _callback(self, img_msg: Image, kf_msg: PointCloud2):
        """
        Run segmentation on the current keyframe image.
        Outputs:
          - instance mask image (16UC1)
          - per-instance embedding matrix (Float32MultiArray)
          - JSON metadata string: {instance_id: class_label, ...}
        """
        self._rate_monitor.tick()

        with self._lock:
            result = self.backend.process(img_msg)

        if result is None:
            rospy.logwarn_throttle(5.0, "[segmentation] Backend returned None.")
            return

        stamp = img_msg.header.stamp

        # ── Mask image ────────────────────────────────────────────────────────
        mask_msg = Image()
        mask_msg.header = make_header(self.params.camera_frame, stamp)
        mask_msg.height = result["mask"].shape[0]
        mask_msg.width  = result["mask"].shape[1]
        mask_msg.encoding = "16UC1"
        mask_msg.step = mask_msg.width * 2
        mask_msg.data = result["mask"].astype(np.uint16).tobytes()
        self.pub_masks.publish(mask_msg)

        # ── Embeddings ────────────────────────────────────────────────────────
        embeddings = result["embeddings"]   # (N, embed_dim) float32
        if embeddings.size > 0:
            emb_msg = Float32MultiArray()
            d0 = MultiArrayDimension(label="instances",
                                     size=embeddings.shape[0], stride=embeddings.size)
            d1 = MultiArrayDimension(label="embed_dim",
                                     size=embeddings.shape[1], stride=embeddings.shape[1])
            emb_msg.layout.dim = [d0, d1]
            emb_msg.data = embeddings.flatten().tolist()
            self.pub_embeddings.publish(emb_msg)

        # ── Metadata JSON ─────────────────────────────────────────────────────
        meta = {
            "stamp": stamp.to_sec(),
            "instance_classes": result["instance_classes"],
            "instance_scores":  result["instance_scores"],
            "num_instances":    int(result["num_instances"]),
        }
        meta_msg = String(data=json.dumps(meta))
        self.pub_meta.publish(meta_msg)

        # ── Debug visualisation ───────────────────────────────────────────────
        if self.pub_viz.get_num_connections() > 0:
            viz_msg = self.backend.render_overlay(img_msg, result)
            self.pub_viz.publish(viz_msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = SegmentationNode()
    node.spin()
