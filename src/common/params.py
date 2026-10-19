"""
hiper.common.params
~~~~~~~~~~~~~~~~~~~~~~
Centralised ROS parameter loading with typed defaults and
a single validation step. Every node calls get_params() and
gets back a plain dataclass — no rospy.get_param() scattered
around the codebase.
"""

import rospy
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class VIOParams:
    # Replay buffer
    replay_buffer_size: int = 100
    replay_similarity_threshold: float = 0.95
    replay_batch_size: int = 3
    update_cycles: int = 5
    # Loss weights
    loss_weight_smoothness: float = 0.001
    loss_weight_velocity: float = 0.05
    # Encoder frozen (only decoders updated — Jetson-friendly)
    freeze_encoder: bool = True
    # Minimum distance between keyframes (metres)
    min_keyframe_distance: float = 0.2


@dataclass
class SegmentationParams:
    model_name: str = "mobile_sam"          # "mobile_sam" | "sam2"
    clip_model: str = "ViT-B/32"
    device: str = "cuda"                    # "cuda" | "cpu"
    confidence_threshold: float = 0.5
    max_instances_per_frame: int = 32
    # Socket bridge address for Python 3.12 inference server
    use_socket_bridge: bool = False
    socket_host: str = "localhost"
    socket_port: int = 9876


@dataclass
class EphemeralityParams:
    # Local ephemerality (within-session ray-casting BPF)
    tau_l: float = 0.5               # threshold: below = static
    sigma_occupied: float = 0.1      # Gaussian sigma for occupied space
    sigma_free: float = 0.3          # Gaussian sigma for free space
    bpf_alpha: float = 0.5
    bpf_beta: float = 0.1
    knn_k: int = 5                   # neighbours for ephemerality propagation
    # Global ephemerality (cross-session persistence)
    tau_g: float = 0.7               # threshold for static map extraction
    # Second-order ephemerality (Beta belief over persistence probability)
    beta_concentration: float = 10.0 # pseudo-count when seeding Beta from a
                                     # loaded/initial scalar epsilon_g
    tau_v: float = 0.02              # max Beta variance to commit/prune (confidence gate)
    # Class-conditioned survival prior defaults (overridden by loaded priors).
    # Rates are in s^-1 so that 1/lambda is the characteristic lifetime:
    # building ~30 yr, tree ~4 mo, bench ~4 d, car ~3 h, person ~8 min.
    default_lambda: float = 1e-4     # exponential decay rate (s^-1)
    class_priors: Dict[str, float] = field(default_factory=lambda: {
        "building":   1e-9,
        "wall":       1e-9,
        "tree":       1e-7,
        "lamppost":   1e-7,
        "bench":      3e-6,
        "trash can":  3e-6,
        "car":        1e-4,
        "vehicle":    1e-4,
        "bicycle":    2e-4,
        "person":     2e-3,
    })


@dataclass
class ObjectEncoderParams:
    embed_dim: int = 256
    backbone: str = "dgcnn"          # "dgcnn" | "pointnet"
    num_points: int = 1024
    knn_k: int = 10
    # UAV augmentation
    altitude_jitter_range: float = 5.0   # metres
    angle_jitter_deg: float = 30.0
    # Memory bank KNN search
    memory_bank_k: int = 10
    reid_confidence_threshold: float = 0.75
    # VLM ensemble weight
    vlm_ensemble_weight: float = 0.5
    # Embedding uncertainty gating (Sec. embedding uncertainty)
    sigma_u: float = 0.25            # scale in gamma = exp(-(u_i+u_j)/2 sigma_u^2)
    tau_u: float = 0.5               # combined-uncertainty ceiling for a firm match


@dataclass
class MapManagerParams:
    map_frame: str = "map"
    map_save_dir: str = "/tmp/hiper_maps"
    autosave_on_shutdown: bool = True
    # Sparsification: KL divergence budget
    sparsify_kl_budget: float = 0.01
    # Object merge distance threshold (metres)
    merge_distance_m: float = 0.5
    # Minimum observations before an object enters the LTM
    min_observations: int = 5
    # Pruning rule (i): epsilon_g ceiling (with the Beta-variance gate)
    prune_eg_threshold: float = 0.95
    # Pruning rule (iii): prune after this many consecutive absent sessions
    absent_sessions_prune: int = 3
    # Radius (m) for assigning ephemerality points to map objects
    eps_assignment_radius: float = 2.0


@dataclass
class HiPerParams:
    vio:        VIOParams        = field(default_factory=VIOParams)
    seg:        SegmentationParams = field(default_factory=SegmentationParams)
    eph:        EphemeralityParams = field(default_factory=EphemeralityParams)
    encoder:    ObjectEncoderParams = field(default_factory=ObjectEncoderParams)
    map_mgr:    MapManagerParams = field(default_factory=MapManagerParams)
    # Shared
    map_frame:  str = "map"
    base_frame: str = "base_link"
    camera_frame: str = "camera_color_optical_frame"
    session_id: int = 0


def get_params(ns: str = "~") -> HiPerParams:
    """
    Load all parameters from the ROS parameter server.
    Resolution order per key: private node namespace (~/key), then the
    shared /hiper namespace the launch files load the YAML into,
    then the dataclass default.
    """
    def _get(key, default):
        sentinel = object()
        val = rospy.get_param(f"{ns}/{key}", sentinel)
        if val is not sentinel:
            return val
        return rospy.get_param(f"/hiper/{key}", default)

    p = HiPerParams()

    # VIO
    p.vio.replay_buffer_size         = _get("vio/replay_buffer_size", p.vio.replay_buffer_size)
    p.vio.replay_similarity_threshold= _get("vio/replay_similarity_threshold", p.vio.replay_similarity_threshold)
    p.vio.replay_batch_size          = _get("vio/replay_batch_size", p.vio.replay_batch_size)
    p.vio.update_cycles              = _get("vio/update_cycles", p.vio.update_cycles)
    p.vio.freeze_encoder             = _get("vio/freeze_encoder", p.vio.freeze_encoder)
    p.vio.min_keyframe_distance      = _get("vio/min_keyframe_distance", p.vio.min_keyframe_distance)

    # Segmentation
    p.seg.model_name                 = _get("seg/model_name", p.seg.model_name)
    p.seg.clip_model                 = _get("seg/clip_model", p.seg.clip_model)
    p.seg.device                     = _get("seg/device", p.seg.device)
    p.seg.confidence_threshold       = _get("seg/confidence_threshold", p.seg.confidence_threshold)
    p.seg.use_socket_bridge          = _get("seg/use_socket_bridge", p.seg.use_socket_bridge)
    p.seg.socket_port                = _get("seg/socket_port", p.seg.socket_port)

    # Ephemerality
    p.eph.tau_l                      = _get("eph/tau_l", p.eph.tau_l)
    p.eph.tau_g                      = _get("eph/tau_g", p.eph.tau_g)
    p.eph.sigma_occupied             = _get("eph/sigma_occupied", p.eph.sigma_occupied)
    p.eph.sigma_free                 = _get("eph/sigma_free", p.eph.sigma_free)
    p.eph.knn_k                      = _get("eph/knn_k", p.eph.knn_k)
    p.eph.default_lambda             = _get("eph/default_lambda", p.eph.default_lambda)
    p.eph.beta_concentration         = _get("eph/beta_concentration", p.eph.beta_concentration)
    p.eph.tau_v                      = _get("eph/tau_v", p.eph.tau_v)
    loaded_priors                    = _get("eph/class_priors", {})
    if isinstance(loaded_priors, dict) and loaded_priors:
        p.eph.class_priors.update(
            {str(k): float(v) for k, v in loaded_priors.items()})
    # Object encoder
    p.encoder.embed_dim              = _get("encoder/embed_dim", p.encoder.embed_dim)
    p.encoder.backbone               = _get("encoder/backbone", p.encoder.backbone)
    p.encoder.reid_confidence_threshold = _get("encoder/reid_confidence_threshold",
                                               p.encoder.reid_confidence_threshold)
    p.encoder.vlm_ensemble_weight    = _get("encoder/vlm_ensemble_weight", p.encoder.vlm_ensemble_weight)
    p.encoder.sigma_u                = _get("encoder/sigma_u", p.encoder.sigma_u)
    p.encoder.tau_u                  = _get("encoder/tau_u", p.encoder.tau_u)

    # Map manager
    p.map_mgr.map_frame              = _get("map_mgr/map_frame", p.map_mgr.map_frame)
    p.map_mgr.map_save_dir           = _get("map_mgr/map_save_dir", p.map_mgr.map_save_dir)
    p.map_mgr.min_observations       = _get("map_mgr/min_observations", p.map_mgr.min_observations)
    p.map_mgr.prune_eg_threshold     = _get("map_mgr/prune_eg_threshold", p.map_mgr.prune_eg_threshold)
    p.map_mgr.absent_sessions_prune  = _get("map_mgr/absent_sessions_prune", p.map_mgr.absent_sessions_prune)
    p.map_mgr.eps_assignment_radius  = _get("map_mgr/eps_assignment_radius", p.map_mgr.eps_assignment_radius)

    # Shared frames
    p.map_frame                      = _get("map_frame", p.map_frame)
    p.base_frame                     = _get("base_frame", p.base_frame)
    p.camera_frame                   = _get("camera_frame", p.camera_frame)
    p.session_id                     = _get("session_id", p.session_id)

    rospy.loginfo(f"[hiper] Parameters loaded. Session {p.session_id}, "
                  f"device={p.seg.device}, backbone={p.encoder.backbone}")
    return p
