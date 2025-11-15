cmake_minimum_required(VERSION 3.8)
project(frenet_conversion_cpp)

# ---- C++ standard ----
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ---- Warnings ----
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Dependencies ----
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)  # header-only, but keep as an imported target

# ---- Library ----
add_library(${PROJECT_NAME} SHARED
  src/frenet_converter_cpp.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link (Eigen3 is header-only, but target_link_libraries is harmless and keeps usage consistent)
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)

# (선택) PIC 보장 - 일부 플랫폼/툴체인에서 유용
set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# ---- Export to consumers ----
# export용 별도 이름 권장 (ROS2 패턴)
set(EXPORT_NAME export_${PROJECT_NAME})
ament_export_targets(${EXPORT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(Eigen3)
ament_export_libraries(${PROJECT_NAME})

# ---- Install ----
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${EXPORT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ---- Tests / Linters ----
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
