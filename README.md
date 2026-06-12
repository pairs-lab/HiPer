# SeLiSLAM — Semantic Lifelong SLAM for UAVs

**S**emantic **Li**felong **SLAM** is a ROS Noetic pipeline that combines
continual visual-inertial odometry, open-vocabulary semantic segmentation,
two-stage Bayesian ephemerality modelling, and contrastive object
re-identification into a single long-term map management system designed for UAV deployment.

It additionally maintains a **unified per-object uncertainty model**: each
object carries a Beta posterior over its persistence probability (second-order
ephemerality), a running embedding covariance, and pose-inflated geometric
covariance. The Beta variance gates cross-session association and pruning, so
objects seen too few times to judge are retained as tentative rather than
committed or deleted prematurely.

---

## Architecture at a glance

```
Camera + IMU
    │
    ▼
vio_frontend_node     ← CoVIO continual adaptation, TF publisher
    │ /seliSlam/keyframe_pointcloud
    ▼
segmentation_node     ← MobileSAM + CLIP (direct or socket-bridge)
    │ /seliSlam/instance_masks  /seliSlam/semantic_embeddings
    ▼
ephemerality_node     ← Local BPF (per-session) + global BPF (cross-session)
    │ /seliSlam/static_cloud  /seliSlam/dynamic_cloud
    ▼
object_encoder_node   ← BYE contrastive encoder + memory bank re-ID
    │ /seliSlam/reid_result
    ▼
map_manager_node      ← Long-term map: fuse, sparsify, save/load
    │ /seliSlam/ltm_objects_viz  /seliSlam/ltm_loaded (latched)
    ▼
  disk: ~/.seliSlam/maps/ltm_session_NNNN.npz
```

---

## Prerequisites

### ROS + system packages

```bash
sudo apt install -y \
  ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-tf2-ros ros-noetic-pcl-ros ros-noetic-message-filters \
  ros-noetic-rviz ros-noetic-realsense2-camera
```

### Python packages (desktop)

```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121   # adjust for your CUDA

pip install \
  open-clip-torch opencv-python-headless Pillow scipy numpy pytest \
  git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Python packages (Jetson AGX Xavier — JetPack 5.x)

```bash
# Install PyTorch from NVIDIA NGC (pre-built for aarch64 + CUDA):
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
pip install open-clip-torch opencv-python-headless Pillow scipy numpy pytest
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# For the Python 3.12 inference server:
python3.12 -m pip install torch open-clip-torch Pillow numpy
```

---

## Build

```bash
# The seliSlam package lives directly under your catkin workspace's src/
# (e.g. ~/lifelong_ws/src/pairs_selislam/seliSlam)
cd ~/lifelong_ws
catkin_make
source devel/setup.bash
```

---

## Workflow: first deployment (session 0)

### Step 1 — Pre-train the VIO network

```bash
# On a bag recorded earlier (or Cityscapes images):
python3 scripts/pretrain_vio.py \
    --bag /data/your_flight.bag \
    --image-topic /camera/color/image_raw \
    --depth-topic  /camera/depth/image_rect_raw \
    --epochs 25 --lr 1e-4 --device cuda

# Checkpoint saved to: ~/.seliSlam/checkpoints/generalizer.pt
```

### Step 2 — Launch the full pipeline

**Desktop:**
```bash
roslaunch seliSlam seliSlam_desktop.launch \
    bag:=/data/flight01.bag session_id:=0
```

**Jetson (two terminals):**
```bash
# Terminal 1: Python 3.12 inference server
python3.12 scripts/inference_server.py \
    --port 9876 --device cuda --debug

# Terminal 2: ROS pipeline
roslaunch seliSlam seliSlam_jetson.launch session_id:=0
```

### Step 3 — End the session and save the map

```bash
rosservice call /seliSlam/end_session
# → runs: embedding fusion, global BPF update, KL sparsification,
#   disk save to ~/.seliSlam/maps/ltm_session_0001.npz,
#   re-broadcast of updated LTM
```

### Step 4 — Retrain the BYE object encoder on session 0 data

```bash
rosservice call /seliSlam/retrain_encoder
# Trains in the background; checkpoint → ~/.seliSlam/checkpoints/bye_encoder.pt
```

---

## Workflow: subsequent sessions

```bash
# Session 1 loads the previous LTM automatically
roslaunch seliSlam seliSlam_desktop.launch session_id:=1
# ...fly, collect data...
rosservice call /seliSlam/end_session
rosservice call /seliSlam/retrain_encoder
```

The VIO generalizer weights accumulate across sessions.  
The BYE encoder is re-trained from scratch each session on the new exploration data.  
The LTM persists and grows across all sessions.

---

## Map queries

```bash
# Query all static objects of class "bench"
rosservice call /seliSlam/query_object \
    '{class_label: "bench", epsilon_g_max: 0.7}'

# Save the current map to a custom path
rosservice call /seliSlam/save_map '{filepath: "/tmp/my_map.npz"}'

# Load a specific session map
rosservice call /seliSlam/load_map '{filepath: "/tmp/ltm_session_0003.npz"}'
```

---

## Running the tests

```bash
cd seliSlam
python -m pytest tests/ -v
# Expected: 130 passed in < 3 s (no ROS master, no GPU required)
```

---

## RViz visualisation

RViz opens automatically on desktop. Key topics:

| Topic | Meaning |
|---|---|
| `/seliSlam/ltm_objects_viz` | Ellipsoid markers: green = permanent, red = ephemeral |
| `/seliSlam/static_cloud` | Points classified as static this session (ε_l < τ_l) |
| `/seliSlam/dynamic_cloud` | Dynamic points excluded from mapping |
| `/seliSlam/pose_graph_edges` | Accumulated UAV trajectory |
| `/seliSlam/seg_viz` | Colour-coded instance segmentation overlay |

---

## Parameter reference

All parameters live under the `/seliSlam` namespace.  
See `config/params_desktop.yaml` and `config/params_jetson.yaml` for defaults.

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `vio/replay_buffer_size` | 100 | Max frames in diversity replay buffer |
| `vio/update_cycles` | 5 | Gradient steps per keyframe |
| `eph/tau_l` | 0.5 | Local ephemerality threshold |
| `eph/tau_g` | 0.7 | Global ephemerality threshold for static map |
| `eph/beta_concentration` | 10.0 | Pseudo-count when seeding the Beta posterior from a loaded scalar ε_g |
| `eph/tau_v` | 0.02 | Max Beta variance to commit/prune an object (confidence gate) |
| `encoder/backbone` | `dgcnn` | `dgcnn` (better) or `pointnet` (lighter) |
| `encoder/reid_confidence_threshold` | 0.75 | Min cosine sim for a confident re-ID |
| `encoder/sigma_u` | 0.25 | Scale in the uncertainty down-weight γ = exp(−(u_i+u_j)/2σ_u²) |
| `encoder/tau_u` | 0.5 | Combined-uncertainty ceiling above which a match is held tentative |
| `seg/use_socket_bridge` | `false` | `true` on Jetson (Python 3.12 server) |
| `map_mgr/min_observations` | 5 | Min keyframe obs before object enters LTM |

---

## Citation

If you use SeLiSLAM in your research, please cite the following papers
that this system is built upon:

- Vödisch et al., *CoVIO: Online Continual Learning for Visual-Inertial Odometry*, CVPR Workshop 2023
- Gil et al., *Ephemerality meets LiDAR-based Lifelong Mapping (ELite)*, arXiv 2025
- Huang et al., *BYE: Build Your Encoder With One Sequence of Exploration Data*, RA-L 2025
- Adkins et al., *ObVi-SLAM: Long-Term Object-Visual SLAM*, RA-L 2024
- Deng et al., *Long-Term Visual SLAM Using a Bayesian Persistence Filter*, RA-M 2023
- Li et al., *Learn to Memorize and to Forget: A Continual Learning Perspective of Dynamic SLAM*, ECCV 2024
