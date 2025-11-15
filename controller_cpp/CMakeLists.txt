cmake_minimum_required(VERSION 3.8)
project(controller_cpp)

# C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# 필요 시
find_package(f110_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(
  include
)

add_executable(controller_manager_cpp
  src/controller_manager_cpp.cpp
  src/pp.cpp
  src/map.cpp
  src/lookup_steer_angle.cpp
  src/ftg.cpp
)

target_include_directories(controller_manager_cpp PRIVATE include)

ament_target_dependencies(controller_manager_cpp
  rclcpp rcl_interfaces ament_index_cpp yaml-cpp
  ackermann_msgs geometry_msgs nav_msgs sensor_msgs visualization_msgs std_msgs
  f110_msgs tf2 tf2_geometry_msgs
)

target_link_libraries(controller_manager_cpp yaml-cpp)

install(TARGETS controller_manager_cpp
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY cfg
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
