cmake_minimum_required(VERSION 3.8)
project(opponent_publisher_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)


add_executable(opponent_publisher src/opponent_publisher.cpp)
ament_target_dependencies(opponent_publisher
  rclcpp
  ackermann_msgs
  nav_msgs
  f110_msgs
)

add_executable(collision_detector src/collision_detector.cpp)
ament_target_dependencies(collision_detector
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  f110_msgs
  tf2
  tf2_geometry_msgs
)

install(TARGETS
  opponent_publisher
  collision_detector
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
