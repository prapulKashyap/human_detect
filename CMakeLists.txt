cmake_minimum_required(VERSION 2.8.3)
project(human_detect)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  sensor_msgs
  pcl_conversions
  pcl_msgs
  visualization_msgs
)



###################################
## catkin specific configuration ##
###################################

catkin_package(

)

###########
## Build ##
###########


include_directories(
  ${catkin_INCLUDE_DIRS}
)



## Declare a cpp executable
add_executable(pcl_node_w_nodelets src/pcl_node_w_nodelets.cpp)



## Specify libraries to link a library or executable target against
target_link_libraries(pcl_node_w_nodelets ${catkin_LIBRARIES})
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html



#############
## Testing ##
#############


