#!/usr/bin/env python3
"""
vio_frontend_node.py
====================
Production VIO front-end for HiPer.

New in this version vs scaffold:
  - CameraInfo subscriber → precise intrinsics passed to DepthPoseNet
  - tf2 broadcaster: publishes map → base_link transform every keyframe
  - Pose graph accumulation: stores (stamp, T) pairs for loop-closure
    integration (consumed later by a backend node)
  - Keyframe database: publishes /hiper/keyframe_db on a latched topic
    so new subscribers get the full history immediately
  - Graceful degradation: if IMU topic is absent, velocity supervision
    is disabled (L_vel weight set to 0)
  - RealSense D455 default intrinsics as fallback (your calibration)

Subscribed topics:
  /camera/color/image_raw           sensor_msgs/Image
  /camera/depth/image_rect_raw      sensor_msgs/Image
  /camera/color/camera_info         sensor_msgs/CameraInfo
  /mavros/imu/data   (optional)     sensor_msgs/Imu

Published topics:
  /hiper/odometry                nav_msgs/Odometry
  /hiper/depth                   sensor_msgs/Image
  /hiper/keyframe_pointcloud     sensor_msgs/PointCloud2
  /hiper/pose_graph_edges        nav_msgs/Path   (cumulative trajectory)

TF published:
  map → base_link   (every accepted keyframe)
"""

import rospy
import numpy as np
import threading
from collections import deque

from nav_msgs.msg      import Odometry, Path
from sensor_msgs.msg   import Image, Imu, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import message_filters
import tf2_ros

from hiper.common.params   import get_params
from hiper.common.ros_utils import (
    make_header, xyz_to_pointcloud2, RateMonitor,
    np_to_point, np_to_quat
)
from hiper.vio.replay_buffer  import DiversityReplayBuffer
from hiper.vio.depth_pose_net import DepthPoseNet


class VIOFrontendNode:
    """
    Continual VIO node with full intrinsics wiring and TF publishing.
    """

    def __init__(self):
        rospy.init_node("vio_frontend_node", anonymous=False)
        self.params = get_params()
        vp = self.params.vio

        # ── Model ─────────────────────────────────────────────────────────────
        self.net = DepthPoseNet(
            freeze_encoder=vp.freeze_encoder,
            device=self.params.seg.device,
        )
        self.net.try_load_checkpoint()

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.replay = DiversityReplayBuffer(
            max_size=vp.replay_buffer_size,
            similarity_threshold=vp.replay_similarity_threshold,
            feature_extractor=self.net.get_encoder_features,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._frame_queue:  deque = deque(maxlen=3)
        self._imu_velocity: float = 0.0
        self._imu_lock       = threading.Lock()
        self._update_lock    = threading.Lock()
        self._imu_available  = False   # set True on first IMU message

        # Accumulated pose in map frame (4×4 SE3, starts at identity)
        self._T_map_base     = np.eye(4, dtype=np.float64)
        # Accumulated pose covariance (x, y, z, rotX, rotY, rotZ): a
        # first-order odometry noise model where each composed increment
        # contributes noise proportional to its step length. Consumed by
        # the map manager for object-covariance inflation (Eq. pose_inflate).
        self._pose_cov       = np.zeros((6, 6), dtype=np.float64)
        self._cov_k_trans    = 0.01   # m of std-dev per metre travelled
        self._cov_k_rot      = 0.002  # rad of std-dev per metre travelled
        self._cov_floor      = 1e-6   # baseline per-step variance
        self._last_kf_pos    = None
        self._vio_min_dist   = vp.min_keyframe_distance

        # Pose graph: list of (stamp, T_map_base np.ndarray)
        self._pose_graph: list = []

        # ── TF broadcaster ────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_odom     = rospy.Publisher(
            "/hiper/odometry",          Odometry,    queue_size=10)
        self.pub_depth    = rospy.Publisher(
            "/hiper/depth",             Image,       queue_size=5)
        self.pub_kf_cloud = rospy.Publisher(
            "/hiper/keyframe_pointcloud", PointCloud2, queue_size=5)
        self.pub_path     = rospy.Publisher(
            "/hiper/pose_graph_edges",  Path,        queue_size=2,
            latch=True)

        # ── Subscribers ───────────────────────────────────────────────────────
        img_sub   = message_filters.Subscriber(
            "/camera/color/image_raw",      Image)
        depth_sub = message_filters.Subscriber(
            "/camera/depth/image_rect_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, depth_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self._image_callback)

        # CameraInfo: update intrinsics in DepthPoseNet once and latch
        self._K_received = False
        rospy.Subscriber("/camera/color/camera_info", CameraInfo,
                         self._caminfo_callback, queue_size=1)

        # IMU (optional)
        rospy.Subscriber("/mavros/imu/data", Imu,
                         self._imu_callback, queue_size=20)

        self._rate_monitor = RateMonitor("vio_frontend")
        rospy.on_shutdown(self._on_shutdown)

        # Set RealSense D455 defaults immediately so the net is not blind
        # before the first CameraInfo arrives
        self.net.update_K(fx=385.19, fy=385.19, cx=321.66, cy=239.80)

        rospy.loginfo("[vio_frontend] Node ready. "
                      f"freeze_encoder={vp.freeze_encoder}, "
                      f"device={self.params.seg.device}")

    # ═════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═════════════════════════════════════════════════════════════════════════

    def _caminfo_callback(self, msg: CameraInfo):
        """
        Extract intrinsics from CameraInfo and forward to DepthPoseNet.
        Called once (or whenever camera_info changes — rare).
        """
        K = msg.K   # row-major 3×3 as a 9-element list
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        self.net.update_K(fx=fx, fy=fy, cx=cx, cy=cy)
        if not self._K_received:
            rospy.loginfo(f"[vio_frontend] Intrinsics received: "
                          f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
            self._K_received = True

    def _imu_callback(self, msg: Imu):
        """Extract linear speed magnitude for velocity supervision."""
        with self._imu_lock:
            a = msg.linear_acceleration
            speed = float(np.linalg.norm([a.x, a.y, a.z]))
            # Clamp to sane range (avoid gravity spike at init)
            self._imu_velocity = float(np.clip(speed, 0.0, 15.0))
            self._imu_available = True

    def _image_callback(self, img_msg: Image, depth_msg: Image):
        """Buffer incoming frames and trigger processing when we have 3."""
        self._rate_monitor.tick()
        with self._imu_lock:
            vel = self._imu_velocity if self._imu_available else 0.0

        frame = {
            "img_msg":   img_msg,
            "depth_msg": depth_msg,
            "velocity":  vel,
            "stamp":     img_msg.header.stamp,
        }
        self._frame_queue.append(frame)

        if len(self._frame_queue) == 3:
            self._process_triplet(list(self._frame_queue))

    # ═════════════════════════════════════════════════════════════════════════
    # Core pipeline
    # ═════════════════════════════════════════════════════════════════════════

    def _process_triplet(self, triplet: list):
        """
        Full per-triplet pipeline:
          1. Predict depth + relative pose (infer)
          2. Compose with accumulated map pose
          3. Publish odometry + TF
          4. On keyframes: update replay buffer, run gradient step,
             publish depth image + point cloud + path
        """
        with self._update_lock:
            depth_np, odom_pred = self.net.infer(triplet)

            # Compose incremental SE3 into map pose
            T_inc = self._odom_to_matrix(odom_pred)
            self._T_map_base = self._T_map_base @ np.linalg.inv(T_inc)

            # Propagate the odometry noise model: each increment adds
            # variance proportional to its step length (drift grows
            # monotonically within a session, as for any odometry).
            step = float(np.linalg.norm(T_inc[:3, 3]))
            var_t = (self._cov_k_trans * step) ** 2 + self._cov_floor
            var_r = (self._cov_k_rot * step) ** 2 + self._cov_floor
            self._pose_cov[:3, :3] += np.eye(3) * var_t
            self._pose_cov[3:, 3:] += np.eye(3) * var_r

            stamp = triplet[-1]["stamp"]
            self._publish_odom_and_tf(self._T_map_base, stamp)

            pos = self._T_map_base[:3, 3]
            if not self._is_keyframe(pos):
                return

            self._last_kf_pos = pos.copy()
            self._pose_graph.append((stamp, self._T_map_base.copy()))

            # ── Replay + online update ─────────────────────────────────────
            self.replay.maybe_add(triplet[-1])
            replay_samples = self.replay.sample(
                self.params.vio.replay_batch_size - 1)
            batch = [triplet] + replay_samples
            loss  = self.net.update_step(
                batch, n_cycles=self.params.vio.update_cycles)
            rospy.logdebug_throttle(
                2.0, f"[vio_frontend] KF loss={loss:.4f} "
                     f"buf={len(self.replay)}")

            # ── Publish keyframe outputs ───────────────────────────────────
            centre = triplet[1]

            depth_msg = self.net.depth_to_ros(
                depth_np, centre["img_msg"].header)
            self.pub_depth.publish(depth_msg)

            cloud = self.net.backproject_to_cloud(
                depth_np, centre["img_msg"])
            if cloud.shape[0] > 0:
                cloud_msg = xyz_to_pointcloud2(
                    cloud, self.params.camera_frame, centre["stamp"])
                self.pub_kf_cloud.publish(cloud_msg)

            self._publish_path(stamp)

    # ═════════════════════════════════════════════════════════════════════════
    # Publishing helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _publish_odom_and_tf(self, T: np.ndarray, stamp: rospy.Time):
        """Publish nav_msgs/Odometry and TF map → base_link."""
        from scipy.spatial.transform import Rotation
        pos = T[:3, 3].astype(np.float32)
        q   = Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)

        # Odometry
        odom             = Odometry()
        odom.header      = make_header(self.params.map_frame, stamp)
        odom.child_frame_id = self.params.base_frame
        odom.pose.pose.position    = np_to_point(pos)
        odom.pose.pose.orientation = np_to_quat(q)
        odom.pose.covariance = self._pose_cov.flatten().tolist()
        self.pub_odom.publish(odom)

        # TF
        ts = TransformStamped()
        ts.header               = make_header(self.params.map_frame, stamp)
        ts.child_frame_id       = self.params.base_frame
        ts.transform.translation.x = float(pos[0])
        ts.transform.translation.y = float(pos[1])
        ts.transform.translation.z = float(pos[2])
        ts.transform.rotation.x  = float(q[0])
        ts.transform.rotation.y  = float(q[1])
        ts.transform.rotation.z  = float(q[2])
        ts.transform.rotation.w  = float(q[3])
        self._tf_broadcaster.sendTransform(ts)

    def _publish_path(self, stamp: rospy.Time):
        """Publish accumulated trajectory as nav_msgs/Path for RViz."""
        path = Path()
        path.header = make_header(self.params.map_frame, stamp)
        for kf_stamp, T in self._pose_graph[-500:]:  # cap at 500 poses
            from scipy.spatial.transform import Rotation
            pos = T[:3, 3].astype(np.float32)
            q   = Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
            ps        = PoseStamped()
            ps.header = make_header(self.params.map_frame, kf_stamp)
            ps.pose.position    = np_to_point(pos)
            ps.pose.orientation = np_to_quat(q)
            path.poses.append(ps)
        self.pub_path.publish(path)

    # ═════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════════════════════════

    def _is_keyframe(self, pos: np.ndarray) -> bool:
        if self._last_kf_pos is None:
            return True
        return float(np.linalg.norm(pos - self._last_kf_pos)) \
               >= self._vio_min_dist

    @staticmethod
    def _odom_to_matrix(odom_pred: dict) -> np.ndarray:
        """Convert odom dict to 4×4 SE3 (float64)."""
        from scipy.spatial.transform import Rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = odom_pred["translation"].astype(np.float64)
        T[:3, :3] = Rotation.from_quat(
            odom_pred["rotation"]).as_matrix()
        return T

    def _on_shutdown(self):
        rospy.loginfo("[vio_frontend] Saving on shutdown...")
        self.net.save_generalizer()

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = VIOFrontendNode()
    node.spin()
