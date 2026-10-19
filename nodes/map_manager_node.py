#!/usr/bin/env python3
"""
map_manager_node.py
===================
Central long-term map (LTM) manager for HiPer.

Responsibilities:
  1. Load the LTM from disk at session start and broadcast it
     to all other nodes via /hiper/ltm_loaded.
  2. Receive ReID results + ephemerality scores and fuse them
     into SemanticObject records.
  3. Run KL-sparse map sparsification (ObVi-SLAM §V-B style)
     after each session to keep the map compact.
  4. Save the updated LTM to disk on shutdown or on request.
  5. Serve the QueryObject service for downstream navigation.

Subscribed topics:
  /hiper/reid_result          hiper/ObjectReIDResult
  /hiper/ephemerality_map     hiper/EphemeralityMap
  /hiper/seg_metadata         std_msgs/String  (JSON)
  /hiper/semantic_embeddings  std_msgs/Float32MultiArray
  /hiper/odometry             nav_msgs/Odometry

Published topics:
  /hiper/ltm_loaded           hiper/LongTermMap  (once at startup)
  /hiper/ltm_objects_viz      visualization_msgs/MarkerArray
  /hiper/ltm_static_cloud     sensor_msgs/PointCloud2

Services:
  /hiper/save_map             hiper/SaveMap
  /hiper/load_map             hiper/LoadMap
  /hiper/query_object         hiper/QueryObject
  /hiper/end_session          std_srvs/Trigger
"""

import json
import threading
import time

import numpy as np
import rospy

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import MarkerArray, Marker
from std_srvs.srv import Trigger, TriggerResponse

from hiper.common.params import get_params
from hiper.common.ros_utils import make_header
from hiper.ephemerality.global_bpf import fuse_and_decay
from hiper.map_io.ltm_io import LTMIO
from hiper.map_io.object_record import ObjectRecord
from hiper.map_io.sparsifier import KLSparsifier

try:
    from hiper.msg import (
        LongTermMap, EphemeralityMap, ObjectReIDResult,
    )
    from hiper.srv import QueryObject, SaveMap, LoadMap
    _MSGS_COMPILED = True
except ImportError:
    rospy.logwarn("[map_manager] Custom msgs not compiled yet — "
                  "limited functionality until catkin_make.")
    _MSGS_COMPILED = False
    LongTermMap = EphemeralityMap = ObjectReIDResult = None
    QueryObject = SaveMap = LoadMap = None


class MapManagerNode:
    """
    The map manager owns the ground truth of the long-term map.
    Every other node either reads from or writes to it through
    ROS topics and services.

    Internal state is a dict: object_id -> ObjectRecord.
    Thread safety: all mutations go through self._lock.
    """

    def __init__(self):
        rospy.init_node("map_manager_node", anonymous=False)
        self.params  = get_params()
        mm           = self.params.map_mgr
        self._lock   = threading.RLock()

        # ── Object registry ───────────────────────────────────────────────────
        # object_id (str) -> ObjectRecord
        self._objects: dict[str, ObjectRecord] = {}
        self._session_id = self.params.session_id

        # Latest robot pose (map frame) and its VIO covariance
        self._robot_pos = np.zeros(3, dtype=np.float32)
        self._robot_pose_cov = None   # (6, 6) or None when unavailable

        # ── I/O helpers ───────────────────────────────────────────────────────
        self._ltm_io     = LTMIO(save_dir=mm.map_save_dir)
        self._sparsifier = KLSparsifier(
            kl_budget=mm.sparsify_kl_budget,
            prune_eg_threshold=mm.prune_eg_threshold,
            min_observations=mm.min_observations,
            prune_var_threshold=self.params.eph.tau_v
                if hasattr(self.params, "eph") else 0.02,
            absent_sessions_threshold=mm.absent_sessions_prune,
        )
        self._min_obs    = mm.min_observations
        self._tau_v      = self.params.eph.tau_v \
            if hasattr(self.params, "eph") else 0.02
        self._eps_radius = mm.eps_assignment_radius

        # Pending observations: object_id -> list of embeddings this session
        self._pending_embeddings: dict[str, list] = {}
        self._pending_scores:     dict[str, list] = {}  # epsilon_l scores

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_ltm_loaded = rospy.Publisher(
            "/hiper/ltm_loaded",
            LongTermMap if _MSGS_COMPILED else String,
            queue_size=1, latch=True)
        self.pub_viz = rospy.Publisher(
            "/hiper/ltm_objects_viz", MarkerArray, queue_size=2)
        self.pub_static_cloud = rospy.Publisher(
            "/hiper/ltm_static_cloud", Float32MultiArray, queue_size=2)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber("/hiper/odometry", Odometry,
                         self._odom_callback, queue_size=10)
        rospy.Subscriber("/hiper/seg_metadata", String,
                         self._meta_callback, queue_size=10)

        if _MSGS_COMPILED:
            rospy.Subscriber("/hiper/reid_result", ObjectReIDResult,
                             self._reid_callback, queue_size=20)
            rospy.Subscriber("/hiper/ephemerality_map", EphemeralityMap,
                             self._eph_callback, queue_size=5)

        # ── Services ──────────────────────────────────────────────────────────
        rospy.Service("/hiper/end_session", Trigger,
                      self._end_session_service)

        if _MSGS_COMPILED:
            rospy.Service("/hiper/save_map",     SaveMap,
                          self._save_map_service)
            rospy.Service("/hiper/load_map",     LoadMap,
                          self._load_map_service)
            rospy.Service("/hiper/query_object", QueryObject,
                          self._query_object_service)

        # ── Timers ────────────────────────────────────────────────────────────
        # Publish visualisation markers every 5 s
        rospy.Timer(rospy.Duration(5.0), self._viz_timer)

        rospy.on_shutdown(self._on_shutdown)

        # ── Boot: load previous LTM and broadcast ────────────────────────────
        self._boot_load_ltm()
        rospy.loginfo("[map_manager] Node ready. "
                      f"Session {self._session_id}, "
                      f"{len(self._objects)} objects loaded.")

    # ═════════════════════════════════════════════════════════════════════════
    # Boot
    # ═════════════════════════════════════════════════════════════════════════

    def _boot_load_ltm(self):
        """
        Try to load the most recent LTM from disk.
        Broadcast it on /hiper/ltm_loaded (latched) so that all
        other nodes initialise correctly even if they started later.
        """
        loaded = self._ltm_io.load_latest()
        if loaded is not None:
            with self._lock:
                self._objects = loaded["objects"]
                if loaded.get("class_priors"):
                    # Store for broadcast; ephemerality_node will consume this
                    self._class_priors = loaded["class_priors"]
                else:
                    self._class_priors = {}
            rospy.loginfo(f"[map_manager] Loaded {len(self._objects)} "
                          f"objects from {self._ltm_io.latest_path}.")
        else:
            self._class_priors = {}
            rospy.loginfo("[map_manager] No previous LTM found — "
                          "starting fresh.")

        # Broadcast regardless (empty LTM is valid for session 0)
        self._broadcast_ltm()

    # ═════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═════════════════════════════════════════════════════════════════════════

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self._robot_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        # Keep the VIO pose covariance for Eq. (pose_inflate): observations
        # made under uncertain motion inflate their object's covariance.
        cov = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        self._robot_pose_cov = cov if np.any(cov) else None

    def _meta_callback(self, msg: String):
        # Cache latest segmentation metadata for use in reid_callback
        try:
            self._latest_meta = json.loads(msg.data)
        except Exception:
            pass

    def _reid_callback(self, msg):
        """
        Receive a ReID result from object_encoder_node.
        If confident, fuse the query embedding into the matched object.
        If no match (or low confidence), create a pending new object.
        """
        if not msg.is_confident or not msg.matched_object_id:
            return

        oid = msg.matched_object_id
        emb = np.array(msg.query_embedding, dtype=np.float32)

        with self._lock:
            if oid not in self._objects:
                rospy.logwarn_throttle(
                    10.0,
                    f"[map_manager] ReID matched '{oid}' but it is not "
                    f"in the current object registry.")
                return

            # Accumulate embedding for this session
            if oid not in self._pending_embeddings:
                self._pending_embeddings[oid] = []
            self._pending_embeddings[oid].append(emb)

            # Update observation count + timestamp
            obj = self._objects[oid]
            obj.observation_count += 1
            obj.last_seen = msg.header.stamp.to_sec()
            obj.last_robot_pos = self._robot_pos.copy()
            # Pose-uncertainty inflation (Eq. pose_inflate): the object
            # covariance absorbs the VIO pose covariance at observation
            # time, so objects mapped under well-constrained motion
            # tighten faster than those seen under uncertain flight.
            if self._robot_pose_cov is not None:
                obj.inflate_position_covariance(self._robot_pose_cov)

    def _eph_callback(self, msg):
        """
        Receive updated ephemerality map.
        Assign each scored point to its nearest map object (within
        eps_assignment_radius) and buffer the per-object mean ε_l for
        the end-of-session global ε_g update. Objects with no nearby
        evidence receive no entry — absence is handled separately at
        session end.
        """
        pts = np.array(msg.points_with_scores,
                       dtype=np.float32).reshape(-1, 4)
        if pts.shape[0] == 0:
            return

        with self._lock:
            if not self._objects:
                return
            oids = list(self._objects.keys())
            centers = np.stack(
                [self._objects[o].position for o in oids])     # (M, 3)
            # Nearest object per point, gated by the assignment radius
            d2 = ((pts[:, None, :3] - centers[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)                    # (N,)
            within = d2[np.arange(len(pts)), nearest] \
                <= self._eps_radius ** 2
            for j, oid in enumerate(oids):
                sel = within & (nearest == j)
                if not np.any(sel):
                    continue
                self._pending_scores.setdefault(oid, []).append(
                    float(pts[sel, 3].mean()))

    # ═════════════════════════════════════════════════════════════════════════
    # Services
    # ═════════════════════════════════════════════════════════════════════════

    def _end_session_service(self, req):
        """
        Called at the end of a deployment session.
        1. Fuse accumulated embeddings into object records.
        2. Update global ephemerality via BPF.
        3. Run map sparsification.
        4. Save updated LTM to disk.
        5. Increment session counter.
        """
        rospy.loginfo("[map_manager] End-of-session processing started.")
        t0 = time.time()

        with self._lock:
            # 1. Fuse embeddings (running average) + embedding-covariance update
            for oid, emb_list in self._pending_embeddings.items():
                if oid not in self._objects:
                    continue
                obj = self._objects[oid]
                new_embs = np.stack(emb_list, axis=0)   # (K, D)
                mean_emb = new_embs.mean(axis=0)
                # Welford update of the running embedding covariance trace,
                # feeding this session's mean fused embedding (Sec. embedding
                # uncertainty). This maintains u_j = trace(C_j)/d.
                obj.update_embedding_uncertainty(mean_emb)
                # Exponential moving average with existing embedding
                alpha = 1.0 / max(obj.observation_count, 1)
                obj.embedding = (
                    (1 - alpha) * obj.embedding + alpha * mean_emb
                    if obj.embedding is not None else mean_emb
                )
                obj.embedding /= (np.linalg.norm(obj.embedding) + 1e-8)

            # 2. Update global ephemerality via the class-conditioned BPF
            #    coupled to the per-object Beta posterior (second-order
            #    ephemerality). epsilon_g is re-derived as 1 - E[p].
            current_time = rospy.Time.now().to_sec()
            for oid, score_list in self._pending_scores.items():
                if oid not in self._objects:
                    continue
                obj = self._objects[oid]
                mean_eps_l = float(np.mean(score_list))
                eps_g_hat = self._bayesian_eg_update(
                    obj.epsilon_g, mean_eps_l,
                    obj.class_label, current_time, obj.last_seen,
                )
                obj.update_beta(eps_g_hat)

            # 3. Absence bookkeeping. Unseen objects receive class-
            #    conditioned transience-only evidence (absence must never
            #    add persistence mass) and their consecutive-absence
            #    counter advances; seen objects reset it.
            seen_ids = (set(self._pending_embeddings.keys())
                        | set(self._pending_scores.keys()))
            for oid, obj in self._objects.items():
                if oid in seen_ids:
                    obj.sessions_absent = 0
                    continue
                lam = self._class_priors.get(
                    obj.class_label, self.params.eph.default_lambda)
                delta_t = max(0.0, current_time - obj.last_seen)
                obj.update_beta_absence(1.0 - float(np.exp(-lam * delta_t)))
                obj.sessions_absent += 1

            # 4. Sparsify (KL divergence minimisation)
            self._objects = self._sparsifier.sparsify(self._objects)

            # 5. Clear pending buffers
            self._pending_embeddings.clear()
            self._pending_scores.clear()
            self._session_id += 1

        # 6. Save
        path = self._ltm_io.save(
            objects=self._objects,
            session_id=self._session_id,
            class_priors=self._class_priors,
        )
        elapsed = time.time() - t0
        rospy.loginfo(f"[map_manager] Session end complete in "
                      f"{elapsed:.2f}s. Saved to {path}. "
                      f"Objects: {len(self._objects)}.")

        # Broadcast updated LTM
        self._broadcast_ltm()
        return TriggerResponse(success=True,
                               message=f"Session {self._session_id - 1} "
                                       f"finalised. {len(self._objects)} "
                                       f"objects saved.")

    def _save_map_service(self, req):
        from hiper.srv import SaveMapResponse
        with self._lock:
            path = req.filepath or self._ltm_io.default_path(self._session_id)
            self._ltm_io.save(self._objects, self._session_id,
                              self._class_priors, filepath=path)
        return SaveMapResponse(success=True, message=f"Saved to {path}")

    def _load_map_service(self, req):
        from hiper.srv import LoadMapResponse
        loaded = self._ltm_io.load(req.filepath)
        if loaded is None:
            return LoadMapResponse(success=False,
                                   message="File not found.",
                                   num_objects_loaded=0)
        with self._lock:
            self._objects = loaded["objects"]
        self._broadcast_ltm()
        n = len(self._objects)
        return LoadMapResponse(success=True,
                               message=f"Loaded {n} objects.",
                               num_objects_loaded=n)

    def _query_object_service(self, req):
        from hiper.srv import QueryObjectResponse
        results = []
        with self._lock:
            for obj in self._objects.values():
                if req.class_label and obj.class_label != req.class_label:
                    continue
                if obj.epsilon_g > req.epsilon_g_max:
                    continue
                results.append(obj.to_ros_msg())
        return QueryObjectResponse(objects=results, success=True,
                                   message=f"{len(results)} objects found.")

    # ═════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _broadcast_ltm(self):
        """Publish a latched LongTermMap message (consumed by all nodes)."""
        if not _MSGS_COMPILED:
            self.pub_ltm_loaded.publish(
                String(data=json.dumps({"session_id": self._session_id,
                                        "num_objects": len(self._objects)})))
            return

        msg = LongTermMap()
        msg.header = make_header(self.params.map_frame)
        msg.session_id     = self._session_id
        msg.total_sessions = self._session_id
        msg.map_frame      = self.params.map_frame
        msg.class_survival_priors_json = json.dumps(self._class_priors)

        with self._lock:
            for obj in self._objects.values():
                # Commit gate: enough observations AND a confident Beta
                # posterior (Var[p] < tau_v) — tentative objects stay
                # out of the broadcast map until evidence accumulates.
                if obj.observation_count >= self._min_obs \
                        and obj.is_committed(self._tau_v):
                    msg.objects.append(obj.to_ros_msg())

        self.pub_ltm_loaded.publish(msg)
        rospy.loginfo(f"[map_manager] Broadcast LTM: "
                      f"{len(msg.objects)} objects, "
                      f"session {self._session_id}.")

    def _bayesian_eg_update(self, eps_g: float, eps_l: float,
                             class_label: str, current_time: float,
                             last_seen: float) -> float:
        """
        One class-conditioned BPF step (shared implementation in
        hiper.ephemerality.global_bpf.fuse_and_decay).
        """
        lam = self._class_priors.get(class_label,
                                     self.params.eph.default_lambda)
        return fuse_and_decay(eps_g, eps_l, lam, current_time - last_seen)

    def _viz_timer(self, _event):
        """Publish MarkerArray for RViz visualisation."""
        if self.pub_viz.get_num_connections() == 0:
            return
        markers = MarkerArray()
        stamp = rospy.Time.now()
        with self._lock:
            objects = list(self._objects.values())

        for i, obj in enumerate(objects):
            if obj.observation_count < self._min_obs:
                continue

            m = Marker()
            m.header = make_header(self.params.map_frame, stamp)
            m.ns     = "hiper_objects"
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(obj.position[0])
            m.pose.position.y = float(obj.position[1])
            m.pose.position.z = float(obj.position[2])
            m.pose.orientation.w = 1.0

            # Scale by ellipsoid axes
            axes = obj.axes if obj.axes is not None else np.array([0.5]*3)
            m.scale.x = float(axes[0] * 2)
            m.scale.y = float(axes[1] * 2)
            m.scale.z = float(axes[2] * 2)

            # Colour: green = permanent (low ε_g), red = ephemeral (high ε_g)
            eg = float(obj.epsilon_g)
            m.color.r = eg
            m.color.g = 1.0 - eg
            m.color.b = 0.2
            m.color.a = 0.6

            m.lifetime = rospy.Duration(6.0)
            markers.markers.append(m)

        self.pub_viz.publish(markers)

    def _on_shutdown(self):
        if self.params.map_mgr.autosave_on_shutdown:
            rospy.loginfo("[map_manager] Autosaving LTM on shutdown...")
            with self._lock:
                self._ltm_io.save(self._objects, self._session_id,
                                  self._class_priors)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = MapManagerNode()
    node.spin()
