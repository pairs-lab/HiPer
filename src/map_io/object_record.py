"""
seliSlam.map_io.object_record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ObjectRecord: the canonical in-memory representation of a
tracked semantic object in the long-term map.

Every field maps 1-to-1 to SemanticObject.msg so that
serialisation is a simple copy — no transformation needed.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
import uuid


@dataclass
class ObjectRecord:
    # Identity
    object_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_label: str   = "unknown"

    # Geometry (map frame)
    position:    np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    axes:        np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float32))
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32))

    # Uncertainty (upper-triangular covariance, 6 elements)
    position_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([1, 1, 1]).flatten()[[0,1,2,4,5,8]]
            .astype(np.float32))

    # Semantics
    embedding:   Optional[np.ndarray] = None  # (embed_dim,) float32

    # Ephemerality
    epsilon_g:        float = 0.5
    epsilon_l:        float = 0.5
    persistence_prob: float = 0.5

    # Second-order ephemerality: Beta(a, b) posterior over persistence prob.
    # epsilon_g is reported as 1 - E[p] = b / (a + b); the variance encodes
    # how confident that estimate is. Uninformative prior a = b = 1.
    beta_a: float = 1.0
    beta_b: float = 1.0

    # Embedding uncertainty: scalar u_j = trace(C_j) / d, where C_j is the
    # running covariance of the object's fused embeddings (Welford recursion).
    # The running statistics below are persisted so accumulation continues
    # across sessions; only the scalar u_j is broadcast over ROS.
    embedding_uncertainty: float = 0.0
    _emb_mean:  Optional[np.ndarray] = None   # (embed_dim,)
    _emb_m2:    Optional[np.ndarray] = None   # (embed_dim,) sum of squared deltas
    _emb_count: int = 0

    # Observation statistics
    observation_count: int   = 0
    first_seen:        float = 0.0
    last_seen:         float = 0.0
    # Consecutive sessions in which this object was not re-observed
    # (pruning rule iii); reset to 0 whenever the object is seen.
    sessions_absent:   int   = 0
    # Observation count at the last sparsification, so the 1/sqrt(n)
    # prior shrink is applied relative to the original prior instead of
    # compounding geometrically across sessions.
    last_sparsify_count: int = 0

    # Internal: latest robot position when this object was observed
    last_robot_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    # ── Unified belief state (theta_j) helpers ────────────────────────────────

    def persistence_mean(self) -> float:
        """Posterior mean E[p] = a / (a + b)."""
        s = self.beta_a + self.beta_b
        return float(self.beta_a / s) if s > 0 else 0.5

    def persistence_var(self) -> float:
        """Posterior variance ab / ((a+b)^2 (a+b+1))."""
        a, b = self.beta_a, self.beta_b
        s = a + b
        if s <= 0:
            return 0.25
        return float(a * b / (s * s * (s + 1.0)))

    def update_beta(self, eps_g_hat: float):
        """
        Update the Beta posterior from a fused-and-decayed instantaneous
        ephemerality estimate (the \\hat{epsilon_g} of the paper).
        Persistence evidence is (1 - eps_g_hat); transience evidence is
        eps_g_hat. epsilon_g is then re-derived as the posterior mean of
        ephemerality, 1 - E[p].
        """
        eps = float(np.clip(eps_g_hat, 0.0, 1.0))
        self.beta_a += (1.0 - eps)
        self.beta_b += eps
        self.persistence_prob = self.persistence_mean()
        self.epsilon_g = 1.0 - self.persistence_prob

    def update_beta_absence(self, transience: float):
        """
        Absence update of the Beta posterior: not re-observing an object
        is (class-conditioned, time-scaled) evidence of transience only —
        it must never add persistence mass to a.
        """
        self.beta_b += float(np.clip(transience, 0.0, 1.0))
        self.persistence_prob = self.persistence_mean()
        self.epsilon_g = 1.0 - self.persistence_prob

    def seed_beta_from_scalar(self, eps_g: float, concentration: float = 10.0):
        """
        Seed the Beta posterior so its mean matches a scalar epsilon_g with a
        given pseudo-count. Used when an object is loaded from disk, where the
        scalar represents accumulated (not uninformative) evidence.
        """
        eps = float(np.clip(eps_g, 0.0, 1.0))
        k = max(concentration, 1e-3)
        self.beta_a = k * (1.0 - eps)
        self.beta_b = k * eps
        self.persistence_prob = self.persistence_mean()
        self.epsilon_g = 1.0 - self.persistence_prob

    def is_committed(self, tau_v: float) -> bool:
        """True once the persistence belief is confident enough to act on."""
        return self.persistence_var() < tau_v

    def update_embedding_uncertainty(self, embedding: np.ndarray):
        """
        Online Welford update of the running embedding covariance trace.
        Maintains per-dimension mean and M2 and exposes the scalar
        u_j = trace(C_j)/d. Cheap O(d) memory, no full d x d matrix.
        """
        e = embedding.astype(np.float64).flatten()
        if self._emb_mean is None or self._emb_count == 0:
            self._emb_mean  = e.copy()
            self._emb_m2    = np.zeros_like(e)
            self._emb_count = 1
            self.embedding_uncertainty = 0.0
            return
        self._emb_count += 1
        delta  = e - self._emb_mean
        self._emb_mean += delta / self._emb_count
        delta2 = e - self._emb_mean
        self._emb_m2 += delta * delta2
        if self._emb_count > 1:
            trace = float(self._emb_m2.sum() / (self._emb_count - 1))
            self.embedding_uncertainty = trace / max(e.shape[0], 1)

    def inflate_position_covariance(self, pose_cov: np.ndarray,
                                    jacobian: Optional[np.ndarray] = None):
        """
        Pose-uncertainty inflation: Sigma_j <- Sigma_j + J Sigma_pose J^T.
        pose_cov is the 3x3 (or 6x6 reduced to translation block) VIO pose
        covariance at observation time; jacobian defaults to identity
        (camera-to-world translation Jacobian for a point landmark).
        Operates on the stored upper-triangular 6-vector representation.
        """
        Sigma = self._cov_vec_to_mat(self.position_covariance)
        Sp = np.asarray(pose_cov, dtype=np.float64)
        if Sp.shape == (6, 6):
            # ROS pose covariance ordering is (x, y, z, rotX, rotY, rotZ):
            # the translation block is the top-left 3x3.
            Sp = Sp[:3, :3]
        if Sp.shape != (3, 3):
            return
        J = np.eye(3) if jacobian is None else np.asarray(jacobian, np.float64)
        Sigma = Sigma + J @ Sp @ J.T
        self.position_covariance = self._cov_mat_to_vec(Sigma)

    @staticmethod
    def _cov_vec_to_mat(v: np.ndarray) -> np.ndarray:
        """Upper-triangular 6-vector [xx,xy,xz,yy,yz,zz] -> symmetric 3x3."""
        v = np.asarray(v, dtype=np.float64).flatten()
        M = np.zeros((3, 3))
        M[0, 0], M[0, 1], M[0, 2] = v[0], v[1], v[2]
        M[1, 1], M[1, 2]         = v[3], v[4]
        M[2, 2]                  = v[5]
        M = M + M.T - np.diag(np.diag(M))
        return M

    @staticmethod
    def _cov_mat_to_vec(M: np.ndarray) -> np.ndarray:
        return np.array([M[0, 0], M[0, 1], M[0, 2],
                         M[1, 1], M[1, 2], M[2, 2]], dtype=np.float32)

    # ── ROS conversion ────────────────────────────────────────────────────────

    def to_ros_msg(self):
        """Serialise to seliSlam/SemanticObject ROS message."""
        try:
            from seliSlam.msg import SemanticObject
            from geometry_msgs.msg import Point, Quaternion
        except ImportError:
            return None

        msg = SemanticObject()
        msg.object_id   = self.object_id
        msg.class_label = self.class_label

        msg.embedding = (self.embedding.tolist()
                         if self.embedding is not None else [])

        msg.position = Point(
            x=float(self.position[0]),
            y=float(self.position[1]),
            z=float(self.position[2]),
        )
        msg.axes = [float(self.axes[0]),
                    float(self.axes[1]),
                    float(self.axes[2])]
        msg.orientation = Quaternion(
            x=float(self.orientation[0]),
            y=float(self.orientation[1]),
            z=float(self.orientation[2]),
            w=float(self.orientation[3]),
        )
        msg.position_covariance = [float(v)
                                   for v in self.position_covariance]
        msg.epsilon_g        = float(self.epsilon_g)
        msg.epsilon_l        = float(self.epsilon_l)
        msg.persistence_prob = float(self.persistence_prob)
        msg.beta_a           = float(self.beta_a)
        msg.beta_b           = float(self.beta_b)
        msg.embedding_uncertainty = float(self.embedding_uncertainty)
        msg.observation_count = int(self.observation_count)
        msg.first_seen       = float(self.first_seen)
        msg.last_seen        = float(self.last_seen)
        return msg

    @classmethod
    def from_ros_msg(cls, msg) -> "ObjectRecord":
        """Deserialise from seliSlam/SemanticObject ROS message."""
        rec = cls()
        rec.object_id    = msg.object_id
        rec.class_label  = msg.class_label
        rec.embedding    = (np.array(msg.embedding, dtype=np.float32)
                            if len(msg.embedding) > 0 else None)
        rec.position     = np.array([msg.position.x,
                                      msg.position.y,
                                      msg.position.z], dtype=np.float32)
        rec.axes         = np.array(list(msg.axes), dtype=np.float32)
        rec.orientation  = np.array([msg.orientation.x,
                                      msg.orientation.y,
                                      msg.orientation.z,
                                      msg.orientation.w], dtype=np.float32)
        rec.position_covariance = np.array(
            list(msg.position_covariance), dtype=np.float32)
        rec.epsilon_g        = float(msg.epsilon_g)
        rec.epsilon_l        = float(msg.epsilon_l)
        rec.persistence_prob = float(msg.persistence_prob)
        # Beta params are strictly positive when set; a 0.0 from an old
        # message (ROS default for unset float fields) means "absent".
        beta_a = float(getattr(msg, "beta_a", 0.0))
        beta_b = float(getattr(msg, "beta_b", 0.0))
        rec.beta_a = beta_a if beta_a > 0.0 else 1.0
        rec.beta_b = beta_b if beta_b > 0.0 else 1.0
        rec.embedding_uncertainty = float(getattr(msg, "embedding_uncertainty", 0.0))
        rec.observation_count = int(msg.observation_count)
        rec.first_seen       = float(msg.first_seen)
        rec.last_seen        = float(msg.last_seen)
        return rec

    # ── numpy serialisation (for .npz files) ─────────────────────────────────

    def to_numpy_dict(self) -> dict:
        return {
            "object_id":           np.bytes_(self.object_id),
            "class_label":         np.bytes_(self.class_label),
            "position":            self.position,
            "axes":                self.axes,
            "orientation":         self.orientation,
            "position_covariance": self.position_covariance,
            "embedding":           self.embedding
                                   if self.embedding is not None
                                   else np.zeros(0, dtype=np.float32),
            "epsilon_g":           np.float32(self.epsilon_g),
            "epsilon_l":           np.float32(self.epsilon_l),
            "persistence_prob":    np.float32(self.persistence_prob),
            "beta_a":              np.float32(self.beta_a),
            "beta_b":              np.float32(self.beta_b),
            "embedding_uncertainty": np.float32(self.embedding_uncertainty),
            "emb_mean":            (self._emb_mean.astype(np.float32)
                                    if self._emb_mean is not None
                                    else np.zeros(0, dtype=np.float32)),
            "emb_m2":              (self._emb_m2.astype(np.float32)
                                    if self._emb_m2 is not None
                                    else np.zeros(0, dtype=np.float32)),
            "emb_count":           np.int32(self._emb_count),
            "observation_count":   np.int32(self.observation_count),
            "first_seen":          np.float64(self.first_seen),
            "last_seen":           np.float64(self.last_seen),
            "sessions_absent":     np.int32(self.sessions_absent),
            "last_sparsify_count": np.int32(self.last_sparsify_count),
        }

    @classmethod
    def from_numpy_dict(cls, d: dict) -> "ObjectRecord":
        rec = cls()
        rec.object_id    = d["object_id"].decode()
        rec.class_label  = d["class_label"].decode()
        rec.position     = d["position"].astype(np.float32)
        rec.axes         = d["axes"].astype(np.float32)
        rec.orientation  = d["orientation"].astype(np.float32)
        rec.position_covariance = d["position_covariance"].astype(np.float32)
        emb = d["embedding"]
        rec.embedding    = emb.astype(np.float32) if emb.size > 0 else None
        rec.epsilon_g    = float(d["epsilon_g"])
        rec.epsilon_l    = float(d["epsilon_l"])
        rec.persistence_prob  = float(d["persistence_prob"])
        rec.beta_a       = float(d["beta_a"]) if "beta_a" in d else 1.0
        rec.beta_b       = float(d["beta_b"]) if "beta_b" in d else 1.0
        rec.embedding_uncertainty = (float(d["embedding_uncertainty"])
                                     if "embedding_uncertainty" in d else 0.0)
        if "emb_mean" in d and d["emb_mean"].size > 0:
            rec._emb_mean  = d["emb_mean"].astype(np.float64)
            rec._emb_m2    = d["emb_m2"].astype(np.float64)
            rec._emb_count = int(d["emb_count"])
        rec.observation_count = int(d["observation_count"])
        rec.first_seen   = float(d["first_seen"])
        rec.last_seen    = float(d["last_seen"])
        rec.sessions_absent = int(d.get("sessions_absent", 0))
        rec.last_sparsify_count = int(d.get("last_sparsify_count", 0))
        return rec
