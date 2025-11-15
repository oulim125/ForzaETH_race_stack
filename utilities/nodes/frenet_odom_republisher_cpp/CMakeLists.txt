cmake_minimum_required(VERSION 3.8)
project(frenet_odom_republisher_cpp)

# C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic -O2)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(frenet_conversion_cpp REQUIRED)

# Executable
add_executable(frenet_odom_republisher src/frenet_odom_republisher.cpp)
ament_target_dependencies(frenet_odom_republisher
  rclcpp
  f110_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  frenet_conversion_cpp        # include 경로 + 링크 정보 전파
)

# Install
install(TARGETS frenet_odom_republisher
  DESTINATION lib/${PROJECT_NAME})

# launch 디렉토리 설치
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
