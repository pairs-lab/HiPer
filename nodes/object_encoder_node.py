#!/usr/bin/env python3
"""
object_encoder_node.py
======================
Per-scene contrastive point cloud encoder for SeLiSLAM.
Implements BYE (Huang et al. 2025) with UAV-specific augmentations:
  - altitude-invariant contrastive training
  - top-down viewpoint augmentation
  - VLM ensemble for association confidence

At session start:
  - Loads the per-scene encoder checkpoint (trained in the previous session)
  - Builds an in-memory object embedding memory bank from the LTM

At runtime (per keyframe):
  - Encodes each partial object point cloud
  - Performs KNN search against the memory bank
  - Publishes ObjectReIDResult for each detected instance

After session end:
  - Triggers a background re-training of the encoder on the new session data

Subscribed topics:
  /seliSlam/static_cloud         sensor_msgs/PointCloud2
  /seliSlam/instance_masks       sensor_msgs/Image
  /seliSlam/semantic_embeddings  std_msgs/Float32MultiArray
  /seliSlam/ltm_loaded           seliSlam/LongTermMap

Published topics:
  /seliSlam/reid_result          seliSlam/ObjectReIDResult
  /seliSlam/encoder_status       std_msgs/String  (JSON status)
"""

import json
import threading
import numpy as np
import rospy
import message_filters

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String

from seliSlam.common.params import get_params
from seliSlam.common.ros_utils import (
    make_header, pointcloud2_to_xyz, RateMonitor
)
from seliSlam.object_encoder.association import ensemble_associate
from seliSlam.object_encoder.bye_encoder import BYEEncoder
from seliSlam.object_encoder.memory_bank import ObjectMemoryBank
from seliSlam.object_encoder.uav_augment import UAVPointCloudAugmentor

try:
    from seliSlam.msg import ObjectReIDResult, LongTermMap
except ImportError:
    ObjectReIDResult = None
    LongTermMap = None


class ObjectEncoderNode:

    def __init__(self):
        rospy.init_node("object_encoder_node", anonymous=False)
        self.params = get_params()
        ep = self.params.encoder

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = BYEEncoder(
            embed_dim=ep.embed_dim,
            backbone=ep.backbone,
            num_points=ep.num_points,
            knn_k=ep.knn_k,
            device=self.params.seg.device,
        )
        self.encoder.try_load_checkpoint()

        # ── Memory banks ──────────────────────────────────────────────────────
        # Two modality-separate banks (paper Eq. 13 is a SCORE-level
        # ensemble): CLIP embeddings persist in the LTM across sessions;
        # BYE embeddings come from the per-scene encoder and live in a
        # sidecar archive written after each session's retraining.
        self.clip_bank = ObjectMemoryBank(
            embed_dim=512,
            knn_k=ep.memory_bank_k,
            sigma_u=ep.sigma_u,
        )
        self.bye_bank = ObjectMemoryBank(
            embed_dim=ep.embed_dim,
            knn_k=ep.memory_bank_k,
            sigma_u=ep.sigma_u,
        )

        # ── UAV augmentor (used during re-training, not inference) ────────────
        self.augmentor = UAVPointCloudAugmentor(
            altitude_jitter_range=ep.altitude_jitter_range,
            angle_jitter_deg=ep.angle_jitter_deg,
        )

        # ── Session point cloud accumulator (for end-of-session retraining) ──
        self._session_observations: list = []  # list of (pts, object_id)
        self._lock = threading.Lock()
        self._current_embeddings: np.ndarray = np.zeros((0, ep.embed_dim),
                                                         dtype=np.float32)
        self._current_classes: dict = {}

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_reid   = rospy.Publisher(
            "/seliSlam/reid_result",
            ObjectReIDResult if ObjectReIDResult else String,
            queue_size=10)
        self.pub_status = rospy.Publisher(
            "/seliSlam/encoder_status", String, queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────────
        cloud_sub = message_filters.Subscriber(
            "/seliSlam/static_cloud", PointCloud2)
        mask_sub  = message_filters.Subscriber(
            "/seliSlam/instance_masks", Image)
        emb_sub   = message_filters.Subscriber(
            "/seliSlam/semantic_embeddings", Float32MultiArray)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, mask_sub, emb_sub], queue_size=10, slop=0.15)
        self.ts.registerCallback(self._callback)

        if LongTermMap is not None:
            rospy.Subscriber("/seliSlam/ltm_loaded", LongTermMap,
                             self._ltm_callback, queue_size=1)

        # Service to trigger end-of-session retraining
        from std_srvs.srv import Trigger
        rospy.Service("/seliSlam/retrain_encoder", Trigger,
                      self._retrain_service)

        self._rate_monitor = RateMonitor("object_encoder", window=20)
        rospy.loginfo("[object_encoder] Node ready. "
                      f"backbone={ep.backbone}, embed_dim={ep.embed_dim}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _ltm_callback(self, msg):
        """Build the in-memory embedding banks from LTM objects."""
        rospy.loginfo(f"[object_encoder] Building memory banks from "
                      f"{len(msg.objects)} LTM objects.")
        self.clip_bank.clear()
        for obj in msg.objects:
            emb = np.array(obj.embedding, dtype=np.float32)
            if emb.size == 0:
                continue
            u_j = float(getattr(obj, "embedding_uncertainty", 0.0))
            self.clip_bank.add(obj.object_id, emb, obj.class_label,
                               uncertainty=u_j)
        self._load_bye_bank()
        self.pub_status.publish(String(data=json.dumps({
            "status": "memory_bank_ready",
            "num_objects": len(msg.objects),
        })))

    def _bye_bank_path(self) -> str:
        import os
        return os.path.join(self.params.map_mgr.map_save_dir,
                            "bye_bank_latest.npz")

    def _load_bye_bank(self):
        """Load the BYE embeddings saved after the previous session's
        retraining; absent in session 0 (the CLIP score then carries the
        full ensemble weight)."""
        import os
        self.bye_bank.clear()
        path = self._bye_bank_path()
        if not os.path.exists(path):
            rospy.loginfo("[object_encoder] No previous BYE bank — "
                          "CLIP-only association this session.")
            return
        data = np.load(path, allow_pickle=True)
        ids = [i.decode() if isinstance(i, bytes) else str(i)
               for i in data["object_ids"]]
        for i, oid in enumerate(ids):
            self.bye_bank.add(oid, data["embeddings"][i].astype(np.float32),
                              class_label="")
        rospy.loginfo(f"[object_encoder] Loaded BYE bank "
                      f"({len(ids)} objects) from {path}.")

    def _callback(self, cloud_msg: PointCloud2,
                  mask_msg: Image,
                  emb_msg: Float32MultiArray):
        """
        Per-keyframe: for each instance in the mask,
        extract the partial point cloud, encode it, and query the bank.
        """
        self._rate_monitor.tick()

        pts_all = pointcloud2_to_xyz(cloud_msg)   # (N, 3)
        if pts_all.shape[0] == 0:
            return

        mask = np.frombuffer(mask_msg.data, dtype=np.uint16).reshape(
            mask_msg.height, mask_msg.width)

        # Reconstruct embedding matrix from Float32MultiArray
        if len(emb_msg.layout.dim) >= 2:
            n_inst  = emb_msg.layout.dim[0].size
            e_dim   = emb_msg.layout.dim[1].size
            vlm_emb = np.array(emb_msg.data, dtype=np.float32).reshape(n_inst, e_dim)
        else:
            vlm_emb = np.zeros((0, 512), dtype=np.float32)

        stamp = cloud_msg.header.stamp
        instance_ids = [i for i in np.unique(mask) if i > 0]

        # ── Gather all instances of this keyframe ─────────────────────────────
        ep = self.params.encoder
        kept_ids, kept_pts, bye_rows, clip_rows = [], [], [], []
        for inst_id in instance_ids:
            inst_mask = mask == inst_id
            if inst_mask.sum() < 20:   # too few points to encode reliably
                continue
            partial_pts = self._extract_instance_cloud(pts_all, inst_mask,
                                                        mask_msg)
            if partial_pts.shape[0] < 10:
                continue
            # Zero-centre the point cloud (BYE paper §IV-B)
            partial_pts = partial_pts - partial_pts.mean(axis=0)
            with self._lock:
                scene_emb = self.encoder.encode(partial_pts)  # (embed_dim,)
            clip = (self._l2norm(vlm_emb[inst_id - 1])
                    if inst_id - 1 < vlm_emb.shape[0]
                    else np.zeros(vlm_emb.shape[1] or 512, dtype=np.float32))
            kept_ids.append(int(inst_id))
            kept_pts.append(partial_pts.copy())
            bye_rows.append(scene_emb)
            clip_rows.append(clip)

        if not kept_ids:
            return

        # ── Joint score-level ensemble + Hungarian (paper Eq. 13) ─────────────
        results = ensemble_associate(
            bye_queries=np.stack(bye_rows) if len(self.bye_bank) else None,
            clip_queries=(np.stack(clip_rows)
                          if len(self.clip_bank) and vlm_emb.shape[0] > 0
                          else None),
            bank_ids=(self.clip_bank.object_ids()
                      if len(self.clip_bank) else self.bye_bank.object_ids()),
            bye_bank=self._bank_rows(self.bye_bank, self.clip_bank),
            clip_bank=(self.clip_bank.embeddings_matrix()
                       if len(self.clip_bank) else None),
            bank_uncertainties=(self.clip_bank.uncertainties()
                                if len(self.clip_bank) else None),
            w=ep.vlm_ensemble_weight,
            sigma_u=ep.sigma_u,
            tau_u=ep.tau_u,
            score_threshold=ep.reid_confidence_threshold,
        )

        for res in results:
            inst_id = kept_ids[res.instance_index]
            pts = kept_pts[res.instance_index]
            # Label observations with the matched LTM id when confident so
            # the end-of-session retraining and the BYE sidecar bank carry
            # cross-session identities.
            label = res.object_id if res.is_confident \
                else f"inst_{inst_id}"
            self._session_observations.append((pts, label))
            self._publish_reid(res.object_id, res.score,
                               clip_rows[res.instance_index],
                               stamp, is_confident=res.is_confident)

    def _bank_rows(self, bye_bank, clip_bank):
        """
        BYE bank rows aligned with the CLIP bank id order (the association
        operates on one shared candidate list). Missing BYE entries are
        zero rows, which contribute zero cosine and let the re-normalised
        CLIP score carry the ensemble for those candidates.
        """
        if len(clip_bank) == 0:
            return bye_bank.embeddings_matrix()
        if len(bye_bank) == 0:
            return None
        dim = bye_bank.embeddings_matrix().shape[1]
        rows = np.zeros((len(clip_bank), dim), dtype=np.float32)
        for i, oid in enumerate(clip_bank.object_ids()):
            emb = bye_bank.get_embedding(oid)
            if emb is not None:
                rows[i] = emb
        return rows

    def _extract_instance_cloud(self, pts_all: np.ndarray,
                                  inst_mask_2d: np.ndarray,
                                  mask_msg: Image) -> np.ndarray:
        """
        Project 2D instance mask pixels to the 3D point cloud.
        This is a simple bounding-box crop from the organised cloud;
        a proper implementation would use the depth image projection.
        """
        H, W = mask_msg.height, mask_msg.width
        N = pts_all.shape[0]
        if N != H * W:
            # Unorganised cloud: fall back to spatial cropping not possible
            # Return all points (no per-instance split from unorganised cloud)
            return pts_all[:min(N, self.params.encoder.num_points)]

        flat_mask = inst_mask_2d.flatten()
        inst_pts  = pts_all[flat_mask > 0]
        return inst_pts

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _publish_reid(self, match_id: str, score: float,
                       embedding: np.ndarray, stamp,
                       is_confident: bool = False):
        """`embedding` is the CLIP embedding of the query instance, so the
        LTM record fused downstream stays in CLIP space (open-vocabulary
        semantics, map-alignment gate)."""
        if ObjectReIDResult is None:
            msg = String(data=json.dumps({
                "matched_object_id": match_id,
                "match_score": float(score),
                "is_confident": bool(is_confident),
            }))
            self.pub_reid.publish(msg)
            return

        msg = ObjectReIDResult()
        msg.header = make_header(self.params.camera_frame, stamp)
        msg.matched_object_id = match_id if match_id else ""
        msg.match_score       = float(score)
        msg.is_confident      = bool(is_confident)
        msg.query_embedding   = embedding.tolist()
        msg.observation_frame = self.params.camera_frame
        self.pub_reid.publish(msg)

    def _retrain_service(self, req):
        """
        ROS service handler. Spawns a background thread to retrain
        the BYE encoder on this session's accumulated observations.
        """
        from std_srvs.srv import TriggerResponse
        n = len(self._session_observations)
        if n < 10:
            return TriggerResponse(success=False,
                                   message=f"Too few observations ({n}) to retrain.")
        t = threading.Thread(target=self._retrain_background, daemon=True)
        t.start()
        return TriggerResponse(success=True,
                               message=f"Retraining started on {n} observations.")

    def _retrain_background(self):
        rospy.loginfo(f"[object_encoder] Retraining BYE encoder on "
                      f"{len(self._session_observations)} observations.")
        with self._lock:
            obs = list(self._session_observations)
        self.encoder.train_session(obs, augmentor=self.augmentor)
        self.encoder.save_checkpoint()
        self._save_bye_bank(obs)
        rospy.loginfo("[object_encoder] Retraining complete. Checkpoint saved.")
        self.pub_status.publish(String(data=json.dumps(
            {"status": "retrain_complete"})))

    def _save_bye_bank(self, observations):
        """
        Re-encode this session's observations with the freshly retrained
        encoder and persist one mean BYE embedding per object id; the
        next session loads this as its BYE memory bank.
        """
        import os
        per_object: dict = {}
        with self._lock:
            for pts, label in observations:
                emb = self.encoder.encode(pts)
                per_object.setdefault(str(label), []).append(emb)
        if not per_object:
            return
        ids, embs = [], []
        for oid, emb_list in per_object.items():
            mean = np.mean(np.stack(emb_list), axis=0)
            mean /= (np.linalg.norm(mean) + 1e-8)
            ids.append(oid.encode())
            embs.append(mean.astype(np.float32))
        path = self._bye_bank_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path,
                            object_ids=np.array(ids, dtype=object),
                            embeddings=np.stack(embs))
        rospy.loginfo(f"[object_encoder] BYE bank saved: "
                      f"{len(ids)} objects -> {path}")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = ObjectEncoderNode()
    node.spin()
