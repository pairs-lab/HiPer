"""
seliSlam.common.ros_utils
~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight ROS ↔ numpy conversion helpers.
No heavy dependencies — just rospy, numpy, and std ROS message types.
"""

import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, Pose
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header


# ── Header helpers ────────────────────────────────────────────────────────────

def make_header(frame_id: str, stamp=None) -> Header:
    h = Header()
    h.stamp = stamp if stamp is not None else rospy.Time.now()
    h.frame_id = frame_id
    return h


# ── Geometry conversions ──────────────────────────────────────────────────────

def point_to_np(p: Point) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def np_to_point(v: np.ndarray) -> Point:
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def quat_to_np(q: Quaternion) -> np.ndarray:
    """Returns [x, y, z, w]."""
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def np_to_quat(v: np.ndarray) -> Quaternion:
    """Expects [x, y, z, w]."""
    return Quaternion(x=float(v[0]), y=float(v[1]),
                      z=float(v[2]), w=float(v[3]))


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a ROS Pose to a 4x4 homogeneous transform (float64)."""
    from scipy.spatial.transform import Rotation
    t = point_to_np(pose.position)
    q = quat_to_np(pose.orientation)
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    from scipy.spatial.transform import Rotation
    pose = Pose()
    pose.position = np_to_point(T[:3, 3].astype(np.float32))
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]
    pose.orientation = np_to_quat(q.astype(np.float32))
    return pose


# ── PointCloud2 helpers ───────────────────────────────────────────────────────

def xyz_to_pointcloud2(points: np.ndarray, frame_id: str,
                        stamp=None) -> PointCloud2:
    """
    Convert an (N, 3) float32 numpy array to a PointCloud2 message.
    """
    header = make_header(frame_id, stamp)
    return pc2.create_cloud_xyz32(header, points.tolist())


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """
    Extract an (N, 3) float32 array from a PointCloud2 message.
    Handles organised and unorganised clouds.
    """
    pts = list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                skip_nans=True))
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def xyzrgb_to_pointcloud2(points: np.ndarray, frame_id: str,
                           stamp=None) -> PointCloud2:
    """
    Convert (N, 6) array [x, y, z, r, g, b] where r,g,b in [0,255]
    to a PointCloud2 with XYZ + RGB fields.
    """
    header = make_header(frame_id, stamp)
    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_data = []
    for row in points:
        x, y, z, r, g, b = row
        rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
        import struct
        rgb_float = struct.unpack("f", struct.pack("I", rgb_packed))[0]
        cloud_data.append([x, y, z, rgb_float])
    return pc2.create_cloud(header, fields, cloud_data)


# ── Timing ────────────────────────────────────────────────────────────────────

class RateMonitor:
    """
    Lightweight fps / rate logger for ROS callbacks.
    Usage:
        monitor = RateMonitor("vio_frontend", window=50)
        def cb(msg):
            monitor.tick()
    """
    def __init__(self, name: str, window: int = 100):
        self.name = name
        self.window = window
        self._times = []

    def tick(self):
        now = rospy.Time.now().to_sec()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) == self.window:
            dt = self._times[-1] - self._times[0]
            if dt > 0:
                fps = (self.window - 1) / dt
                rospy.logdebug_throttle(5.0, f"[{self.name}] {fps:.1f} Hz")
