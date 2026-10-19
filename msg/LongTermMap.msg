# LongTermMap.msg
# Transmitted once at session start (loaded from disk) and
# once at session end (to be saved to disk).

std_msgs/Header header

# All tracked semantic objects
hiper/SemanticObject[] objects

# Map frame id (e.g. "map" or "world")
string map_frame

# Session index that produced this map version
uint32 session_id

# Total number of sessions accumulated
uint32 total_sessions

# Serialised class-conditioned survival prior table (JSON string).
# Keys: class_label -> {lambda: float, shape: float}
string class_survival_priors_json
