# ObjectReIDResult.msg
# Published by object_encoder_node after matching a new partial
# point cloud observation to an object in the long-term map.

std_msgs/Header header

# The id of the matched long-term map object ("" if no match)
string matched_object_id

# Cosine similarity score of the best match (0 to 1)
float32 match_score

# Whether the match exceeds the configured confidence threshold
bool is_confident

# The new partial point cloud's embedding (for debugging / logging)
float32[] query_embedding

# Frame in which the observation was made
string observation_frame
