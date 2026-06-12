cmake_minimum_required(VERSION 3.0.2)
project(seliSlam)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  image_transport
  cv_bridge
  pcl_ros
  pcl_conversions
  message_generation
)

# ── Custom message types ──────────────────────────────────────────────────────
add_message_files(FILES
  SemanticObject.msg
  EphemeralityMap.msg
  LongTermMap.msg
  ObjectReIDResult.msg
)

# The .srv files live in msg/ alongside the messages; catkin's default
# service directory is srv/, so it must be set explicitly here.
add_service_files(DIRECTORY msg FILES
  QueryObject.srv
  SaveMap.srv
  LoadMap.srv
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

# ── Catkin package export ─────────────────────────────────────────────────────
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    tf2_ros
    message_runtime
)

# ── Python package install ────────────────────────────────────────────────────
catkin_python_setup()

# ── Node scripts (make executable) ───────────────────────────────────────────
catkin_install_python(PROGRAMS
  nodes/vio_frontend_node.py
  nodes/segmentation_node.py
  nodes/ephemerality_node.py
  nodes/object_encoder_node.py
  nodes/map_manager_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ── Config / launch install ───────────────────────────────────────────────────
install(DIRECTORY launch/  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY config/  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)
