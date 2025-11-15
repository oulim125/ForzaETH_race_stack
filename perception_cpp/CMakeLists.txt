cmake_minimum_required(VERSION 3.8)
project(perception_cpp)

# 기본 C++ 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 필수 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(f110_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)            
find_package(frenet_conversion_cpp REQUIRED)  

# 컴파일 옵션 (선택적으로 최적화/경고)
add_compile_options(-Wall -Wextra -Wpedantic -O2)

# -------------------
# Executables
# -------------------

# detect 노드
add_executable(detect src/detect.cpp)
ament_target_dependencies(detect
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  f110_msgs
  yaml-cpp
  frenet_conversion_cpp 
)
target_link_libraries(detect
  Eigen3::Eigen
  yaml-cpp
)

# tracking 노드
add_executable(tracking src/tracking.cpp)
ament_target_dependencies(tracking
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  f110_msgs
  yaml-cpp 
)
target_link_libraries(tracking
  Eigen3::Eigen
  yaml-cpp
)

# -------------------
# Install
# -------------------
install(TARGETS
  detect
  tracking
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/)

ament_package()
