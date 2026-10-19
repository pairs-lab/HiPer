"""
hiper.evaluation.lifelong_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offline multi-session (lifelong) evaluation harness.

Drives the same long-term map machinery used by the ROS nodes —
LTMIO, ObjectRecord, GlobalEphemeralityBPF, KLSparsifier — over an
ordered list of dataset sessions (e.g. the per-scene sequences of
OpenLORIS-Scene) without a ROS master. For every session it:

  1. loads the previous long-term map (session N-1) from disk,
  2. estimates the trajectory (wheel odometry or the continual VIO net),
  3. optionally ingests object observations,
  4. computes ATE RMSE and the OpenLORIS C-rate against ground truth,
  5. runs the end-of-session pipeline: per-object Beta/BPF update,
     unseen-object decay, KL-sparse pruning, and LTM save,

so that map state — including the unified uncertainty belief
(Beta posterior, embedding uncertainty, position covariance) —
propagates from one session to the next exactly as in deployment.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from hiper.ephemerality.global_bpf import GlobalEphemeralityBPF
from hiper.evaluation.openloris import OpenLorisSequence
from hiper.evaluation.trajectory_metrics import (
    associate_timestamps,
    ate_rmse,
    correct_rate,
)
from hiper.map_io.ltm_io import LTMIO
from hiper.map_io.object_record import ObjectRecord
from hiper.map_io.sparsifier import KLSparsifier


@dataclass
class Observation:
    """One object observation produced during a session."""
    position: np.ndarray                 # (3,) map frame
    embedding: Optional[np.ndarray] = None
    class_label: str = "unknown"
    epsilon_l: float = 0.5               # local ephemerality evidence
    stamp: float = 0.0
    pose_covariance: Optional[np.ndarray] = None   # (3,3) VIO/odom covariance


@dataclass
class SessionResult:
    name: str = ""
    ate: float = float("nan")
    c_rate: float = float("nan")
    n_frames: int = 0
    n_objects: int = 0
    n_pruned: int = 0
    est_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    est_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


class LifelongEvaluator:
    """
    Multi-session evaluation driver.

    Parameters
    ----------
    map_dir : str
        Directory where ltm_session_*.npz files are written/read.
    class_priors : dict
        class_label -> lambda_c survival decay rates.
    """

    def __init__(
        self,
        map_dir: str,
        class_priors: Optional[Dict[str, float]] = None,
        default_lambda: float = 1e-4,
        beta_concentration: float = 10.0,
        tau_v: float = 0.02,
        tau_g: float = 0.95,
        min_observations: int = 5,
        kl_budget: int = 500,
        assoc_dist: float = 1.5,
        assoc_cos: float = 0.6,
    ):
        self._ltm_io = LTMIO(save_dir=map_dir)
        self._class_priors = dict(class_priors or {})
        self._bpf = GlobalEphemeralityBPF(
            default_lambda=default_lambda,
            class_priors=self._class_priors,
            beta_concentration=beta_concentration,
        )
        self._sparsifier = KLSparsifier(
            kl_budget=kl_budget,
            prune_eg_threshold=tau_g,
            min_observations=min_observations,
            prune_var_threshold=tau_v,
        )
        self._assoc_dist = assoc_dist
        self._assoc_cos = assoc_cos
        self._objects: Dict[str, ObjectRecord] = {}
        self._session_id = 0
        self.results: List[SessionResult] = []

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self):
        """Load the latest LTM and seed the BPF states from it."""
        loaded = self._ltm_io.load_latest()
        if loaded is not None:
            self._objects = loaded["objects"]
            self._session_id = loaded.get("session_id", self._session_id)
            if loaded.get("class_priors"):
                self._class_priors.update(loaded["class_priors"])
                self._bpf.update_class_priors(loaded["class_priors"])
            for oid, obj in self._objects.items():
                self._bpf.load_object_state(
                    object_id=oid,
                    epsilon_g=obj.epsilon_g,
                    persistence_prob=obj.persistence_prob,
                    last_seen=obj.last_seen,
                    class_label=obj.class_label,
                    beta_a=obj.beta_a if obj.beta_a != 1.0 or obj.beta_b != 1.0
                        else None,
                    beta_b=obj.beta_b if obj.beta_a != 1.0 or obj.beta_b != 1.0
                        else None,
                )
        self._pending_eps_l: Dict[str, List[float]] = {}
        self._seen_this_session: set = set()

    def ingest(self, obs: Observation) -> str:
        """
        Associate one observation with the map (nearest neighbour with a
        distance gate plus an embedding-cosine gate when both sides have
        embeddings), update the matched record or create a new one.
        Returns the object id.
        """
        oid = self._associate(obs)
        if oid is None:
            rec = ObjectRecord()
            rec.class_label = obs.class_label
            rec.position = obs.position.astype(np.float32)
            rec.embedding = (obs.embedding.astype(np.float32)
                             if obs.embedding is not None else None)
            rec.first_seen = obs.stamp
            oid = rec.object_id
            self._objects[oid] = rec

        rec = self._objects[oid]
        rec.observation_count += 1
        rec.last_seen = obs.stamp
        # Position running mean
        n = rec.observation_count
        rec.position = ((n - 1) * rec.position + obs.position) / n
        rec.position = rec.position.astype(np.float32)
        # Pose-uncertainty inflation (Eq. 16) when covariance is available
        if obs.pose_covariance is not None:
            rec.inflate_position_covariance(obs.pose_covariance)
        # Embedding fusion + Welford uncertainty
        if obs.embedding is not None:
            rec.update_embedding_uncertainty(obs.embedding)
            if rec.embedding is None:
                rec.embedding = obs.embedding.astype(np.float32)
            else:
                alpha = 1.0 / max(n, 1)
                emb = (1 - alpha) * rec.embedding + alpha * obs.embedding
                rec.embedding = (emb / (np.linalg.norm(emb) + 1e-8)) \
                    .astype(np.float32)

        self._pending_eps_l.setdefault(oid, []).append(float(obs.epsilon_l))
        self._seen_this_session.add(oid)
        return oid

    def end_session(self, current_time: float) -> Dict[str, int]:
        """
        Run the end-of-session pipeline and persist the LTM.
        Returns {"n_objects": ..., "n_pruned": ...}.
        """
        # 1. Per-object class-conditioned BPF + Beta update (seen objects)
        for oid, scores in self._pending_eps_l.items():
            if oid not in self._objects:
                continue
            obj = self._objects[oid]
            mean_eps_l = float(np.mean(scores))
            eps_g = self._bpf.update(oid, obj.class_label,
                                     mean_eps_l, current_time)
            state = self._bpf.get_all_states()[oid]
            obj.epsilon_g = eps_g
            obj.beta_a = state.beta_a
            obj.beta_b = state.beta_b
            obj.persistence_prob = state.persistence_mean()

        # 2. Absence bookkeeping: unseen objects gain class-conditioned
        #    transience-only evidence and advance their absence counter.
        states = self._bpf.get_all_states()
        for oid, obj in self._objects.items():
            if oid in self._seen_this_session:
                obj.sessions_absent = 0
                continue
            self._bpf.mark_not_observed(oid, current_time)
            obj.sessions_absent += 1
            if oid in states:
                st = states[oid]
                obj.epsilon_g = st.epsilon_g
                obj.persistence_prob = st.persistence_prob
                obj.beta_a, obj.beta_b = st.beta_a, st.beta_b

        # 3. KL-sparse pruning
        n_before = len(self._objects)
        self._objects = self._sparsifier.sparsify(self._objects)
        n_pruned = n_before - len(self._objects)

        # 4. Save
        self._session_id += 1
        self._ltm_io.save(self._objects, self._session_id, self._class_priors)
        return {"n_objects": len(self._objects), "n_pruned": n_pruned}

    # ── Trajectory estimation ─────────────────────────────────────────────────

    @staticmethod
    def trajectory_from_odometry(seq: OpenLorisSequence):
        """Wheel-odometry trajectory (baseline pose source)."""
        return seq.odom_t.copy(), seq.odom_xyz.copy()

    @staticmethod
    def trajectory_from_vio(seq: OpenLorisSequence, vio,
                            image_size=(640, 192)):
        """
        Integrate the continual VIO network over consecutive frame pairs.
        `vio` is a DepthPoseNet (its _img method must accept the offline
        duck-typed image message — see scripts/run_openloris_eval.py).
        Metric scale comes from the IMU/odometry speed channel.
        """
        from hiper.evaluation.openloris import load_frame_images
        from scipy.spatial.transform import Rotation

        if len(seq.frames) < 3:
            return np.zeros(0), np.zeros((0, 3))

        T = np.eye(4)
        ts = [seq.frames[0]["stamp"]]
        xyz = [T[:3, 3].copy()]
        prev = None
        for frame in seq.frames:
            cur = load_frame_images(frame, size=image_size)
            cur["img_msg"] = _OfflineImageMsg(cur["img_np"])
            if prev is not None:
                _, pose = vio.infer([prev, prev, cur])
                dT = np.eye(4)
                dT[:3, :3] = Rotation.from_quat(pose["rotation"]).as_matrix()
                dT[:3, 3] = pose["translation"]
                # Metric scale from odometry speed
                dt = cur["stamp"] - prev["stamp"]
                v = float(cur.get("velocity", 0.0))
                norm = np.linalg.norm(dT[:3, 3])
                if v > 1e-3 and norm > 1e-6 and dt > 0:
                    dT[:3, 3] *= (v * dt) / norm
                T = T @ dT
                ts.append(cur["stamp"])
                xyz.append(T[:3, 3].copy())
            prev = cur
        return np.array(ts), np.stack(xyz, axis=0)

    # ── Full session run ──────────────────────────────────────────────────────

    def run_session(
        self,
        seq: OpenLorisSequence,
        pose_source: str = "odom",
        vio=None,
        observation_fn: Optional[
            Callable[[OpenLorisSequence, np.ndarray, np.ndarray],
                     List[Observation]]] = None,
        tau_pos: float = 0.3,
    ) -> SessionResult:
        """
        Run one full session: trajectory + observations + map lifecycle.
        observation_fn(seq, est_t, est_xyz) -> List[Observation] lets the
        caller plug in any perception front-end (segmentation-based or
        synthetic for unit tests).
        """
        self.start_session()

        if pose_source == "vio":
            if vio is None:
                raise ValueError("pose_source='vio' requires a DepthPoseNet")
            est_t, est_xyz = self.trajectory_from_vio(seq, vio)
        else:
            est_t, est_xyz = self.trajectory_from_odometry(seq)

        if observation_fn is not None:
            for obs in observation_fn(seq, est_t, est_xyz):
                self.ingest(obs)

        res = SessionResult(name=seq.name, est_t=est_t, est_xyz=est_xyz,
                            n_frames=len(seq.frames))
        if len(seq.gt_t) > 0 and len(est_t) > 0:
            idx_e, idx_g = associate_timestamps(est_t, seq.gt_t, max_dt=0.1)
            if len(idx_e) >= 3:
                res.ate = ate_rmse(est_xyz[idx_e], seq.gt_xyz[idx_g])
            res.c_rate = correct_rate(est_t, est_xyz,
                                      seq.gt_t, seq.gt_xyz, tau_pos=tau_pos)

        end_time = (float(seq.frames[-1]["stamp"]) if seq.frames
                    else float(est_t[-1]) if len(est_t) else 0.0)
        stats = self.end_session(end_time)
        res.n_objects = stats["n_objects"]
        res.n_pruned = stats["n_pruned"]
        self.results.append(res)
        return res

    # ── Internals ─────────────────────────────────────────────────────────────

    def _associate(self, obs: Observation) -> Optional[str]:
        """Nearest-neighbour association with distance + embedding gates."""
        best_oid, best_d = None, self._assoc_dist
        for oid, rec in self._objects.items():
            d = float(np.linalg.norm(rec.position - obs.position))
            if d >= best_d:
                continue
            if (obs.embedding is not None and rec.embedding is not None
                    and rec.embedding.shape == obs.embedding.shape):
                cos = float(np.dot(rec.embedding, obs.embedding) /
                            (np.linalg.norm(rec.embedding) *
                             np.linalg.norm(obs.embedding) + 1e-8))
                if cos < self._assoc_cos:
                    continue
            best_oid, best_d = oid, d
        return best_oid


class _OfflineImageMsg:
    """Duck-typed sensor_msgs/Image for offline VIO inference."""

    def __init__(self, rgb_np: np.ndarray):
        self._rgb = rgb_np
        self.height = rgb_np.shape[0]
        self.width = rgb_np.shape[1]
        self.encoding = "rgb8"
        self.header = type("H", (), {"stamp": None, "frame_id": "cam"})()
