#!/usr/bin/env python3
"""
ephemerality_node.py
====================
Two-stage semantic ephemerality node for HiPer.

Implements ELite-style local + global ephemerality (Gil et al. 2025)
extended with class-conditioned Bayesian persistence filter priors
(LTVS-BPF, Deng et al. 2023).

Local ephemerality (ε_l):
  Updated within each session via ray-casting Bayesian inference.
  Points with ε_l > tau_l are excluded from pose estimation.

Global ephemerality (ε_g):
  Updated between sessions using the Bayesian persistence filter
  with a class-conditioned exponential survival prior.

Subscribed topics:
  /hiper/keyframe_pointcloud  sensor_msgs/PointCloud2
  /hiper/instance_masks       sensor_msgs/Image
  /hiper/seg_metadata         std_msgs/String  (JSON)
  /hiper/ltm_loaded           hiper/LongTermMap  (on session start)

Published topics:
  /hiper/ephemerality_map     hiper/EphemeralityMap
  /hiper/static_cloud         sensor_msgs/PointCloud2  (ε_l < tau_l)
  /hiper/dynamic_cloud        sensor_msgs/PointCloud2  (ε_l >= tau_l)
"""

import json
import threading

import numpy as np
import rospy
import message_filters

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from hiper.common.params import get_params
from hiper.common.ros_utils import (
    make_header, pointcloud2_to_xyz, xyz_to_pointcloud2, RateMonitor
)
from hiper.ephemerality.local_bpf import LocalEphemeralityBPF
from hiper.ephemerality.global_bpf import GlobalEphemeralityBPF

try:
    from hiper.msg import EphemeralityMap, LongTermMap
except ImportError:
    rospy.logwarn("[ephemerality] Custom msgs not yet compiled — "
                  "run catkin_make first.")
    EphemeralityMap = None
    LongTermMap = None


class EphemeralityNode:

    def __init__(self):
        rospy.init_node("ephemerality_node", anonymous=False)
        self.params = get_params()
        ep = self.params.eph

        # ── BPF engines ───────────────────────────────────────────────────────
        self.local_bpf  = LocalEphemeralityBPF(
            sigma_occupied=ep.sigma_occupied,
            sigma_free=ep.sigma_free,
            alpha=ep.bpf_alpha,
            beta=ep.bpf_beta,
            knn_k=ep.knn_k,
        )
        self.global_bpf = GlobalEphemeralityBPF(
            default_lambda=ep.default_lambda,
            class_priors=ep.class_priors,
            beta_concentration=ep.beta_concentration,
        )

        # ── Internal state ────────────────────────────────────────────────────
        # Combined session point cloud [x, y, z, ε_l]
        self._session_cloud: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        self._lock = threading.Lock()
        self._current_meta: dict = {}
        self._session_id: int = self.params.session_id

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_eph_map   = rospy.Publisher(
            "/hiper/ephemerality_map", EphemeralityMap
            if EphemeralityMap else String, queue_size=2)
        self.pub_static    = rospy.Publisher(
            "/hiper/static_cloud", PointCloud2, queue_size=5)
        self.pub_dynamic   = rospy.Publisher(
            "/hiper/dynamic_cloud", PointCloud2, queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────────
        cloud_sub = message_filters.Subscriber(
            "/hiper/keyframe_pointcloud", PointCloud2)
        mask_sub  = message_filters.Subscriber(
            "/hiper/instance_masks", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, mask_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._cloud_mask_callback)

        rospy.Subscriber("/hiper/seg_metadata", String,
                         self._meta_callback, queue_size=10)

        if LongTermMap is not None:
            rospy.Subscriber("/hiper/ltm_loaded", LongTermMap,
                             self._ltm_callback, queue_size=1)

        self._rate_monitor = RateMonitor("ephemerality", window=30)
        rospy.loginfo("[ephemerality] Node ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _meta_callback(self, msg: String):
        try:
            self._current_meta = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _ltm_callback(self, msg):
        """
        On session start, initialise global BPF from previous LTM.
        Loads per-object ε_g and class survival priors.
        """
        rospy.loginfo(f"[ephemerality] Loading {len(msg.objects)} objects "
                      f"from LTM session {msg.session_id}.")
        if msg.class_survival_priors_json:
            priors = json.loads(msg.class_survival_priors_json)
            self.global_bpf.update_class_priors(priors)
        for obj in msg.objects:
            # Restore the persisted Beta posterior when present (newer
            # maps); 0.0 is the ROS default for unset fields, in which
            # case load_object_state seeds it from the scalar epsilon_g.
            beta_a = float(getattr(obj, "beta_a", 0.0))
            beta_b = float(getattr(obj, "beta_b", 0.0))
            self.global_bpf.load_object_state(
                object_id=obj.object_id,
                epsilon_g=obj.epsilon_g,
                persistence_prob=obj.persistence_prob,
                last_seen=obj.last_seen,
                class_label=obj.class_label,
                beta_a=beta_a if beta_a > 0.0 else None,
                beta_b=beta_b if beta_b > 0.0 else None,
            )

    def _cloud_mask_callback(self, cloud_msg: PointCloud2, mask_msg: Image):
        """
        Per-keyframe callback. Runs local BPF ray-casting update and
        publishes updated static / dynamic split.
        """
        self._rate_monitor.tick()

        pts = pointcloud2_to_xyz(cloud_msg)  # (N, 3)
        if pts.shape[0] == 0:
            return

        # The instance mask is only synchronised here so that keyframes
        # with segmentation get ephemerality updates; per-instance class
        # handling happens downstream in the map manager.

        with self._lock:
            # Update local ephemerality for all points in this keyframe
            eps_l = self.local_bpf.update(
                new_points=pts,
                existing_cloud=self._session_cloud[:, :3]
                if self._session_cloud.shape[0] > 0 else None,
                sensor_origin=np.zeros(3, dtype=np.float32),
            )

            # Accumulate into session cloud
            new_rows = np.concatenate(
                [pts, eps_l.reshape(-1, 1)], axis=1)
            self._session_cloud = np.vstack([self._session_cloud, new_rows]) \
                if self._session_cloud.shape[0] > 0 else new_rows

        # Split into static / dynamic
        tau = self.params.eph.tau_l
        static_pts  = pts[eps_l < tau]
        dynamic_pts = pts[eps_l >= tau]

        stamp = cloud_msg.header.stamp
        frame = self.params.map_frame

        if static_pts.shape[0] > 0:
            self.pub_static.publish(
                xyz_to_pointcloud2(static_pts, frame, stamp))
        if dynamic_pts.shape[0] > 0:
            self.pub_dynamic.publish(
                xyz_to_pointcloud2(dynamic_pts, frame, stamp))

        # Publish ephemerality map (if msg type compiled)
        self._publish_eph_map(stamp)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_eph_map(self, stamp):
        if EphemeralityMap is None:
            return
        with self._lock:
            cloud = self._session_cloud.copy()

        if cloud.shape[0] == 0:
            return

        msg = EphemeralityMap()
        msg.header = make_header(self.params.map_frame, stamp)
        msg.points_with_scores = cloud.flatten().tolist()
        msg.num_points = cloud.shape[0]
        msg.tau_l = self.params.eph.tau_l
        msg.session_id = self._session_id
        self.pub_eph_map.publish(msg)

    def get_global_ephemerality(self, object_id: str,
                                 class_label: str,
                                 current_time: float) -> float:
        """
        Public API called by map_manager_node to get ε_g for an object.
        """
        return self.global_bpf.get_epsilon_g(
            object_id, class_label, current_time)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = EphemeralityNode()
    node.spin()
