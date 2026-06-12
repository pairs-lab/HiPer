# SemanticObject.msg
# Represents a single semantically-labelled object in the long-term map.
# Pose is given as the ellipsoid centre in the map frame.

std_msgs/Header header

# Unique object id (persists across sessions)
string object_id

# Semantic class label (open-vocabulary string, e.g. "tree", "bench")
string class_label

# CLIP/DINOv2 embedding vector (length = embed_dim, typically 512 or 768)
float32[] embedding

# 3D position of ellipsoid centre (map frame)
geometry_msgs/Point position

# Ellipsoid half-axes [a, b, c] in metres
float32[3] axes

# Orientation quaternion of ellipsoid (map frame)
geometry_msgs/Quaternion orientation

# Flattened upper-triangular covariance of position (6 elements, row-major)
float32[6] position_covariance

# Global ephemerality score in [0, 1]. 0 = permanent, 1 = highly transient.
float32 epsilon_g

# Local ephemerality score for the current session (updated per-session)
float32 epsilon_l

# Bayesian persistence probability P(delta_i = 1 | Y_{1:N})
float32 persistence_prob

# Second-order ephemerality: Beta(a, b) posterior over persistence prob.
# epsilon_g is reported as 1 - a/(a+b); the variance encodes confidence.
float32 beta_a
float32 beta_b

# Scalar embedding uncertainty u_j = trace(C_j)/d (running embedding covariance)
float32 embedding_uncertainty

# Number of sessions this object has been observed across
uint32 observation_count

# Unix timestamp of first and last observation
float64 first_seen
float64 last_seen
