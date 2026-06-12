# EphemeralityMap.msg
# Published after each ephemerality update pass.
# Carries the full point cloud with per-point epsilon_l scores.

std_msgs/Header header

# Flattened point cloud: each point = [x, y, z, epsilon_l]
# Length = 4 * N where N is number of map points
float32[] points_with_scores

# Number of points
uint32 num_points

# Threshold used to separate static from dynamic points this session
float32 tau_l

# Session index (increments per deployment)
uint32 session_id
