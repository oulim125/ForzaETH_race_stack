#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class FTG_Controller {
public:
  // 파라미터는 파이썬과 동일
  FTG_Controller(rclcpp::Node* node,
                 bool mapping,
                 bool debug,
                 double safety_radius,
                 double max_lidar_dist,
                 double max_speed,
                 double range_offset,
                 double track_width);

  // LiDAR ranges -> (speed, steering_angle)
  std::pair<double,double> process_lidar(const std::vector<float>& ranges);

  // 외부에서 속도 세팅(맵핑 모드 radius 계산에 사용)
  void set_vel(double v) { velocity_ = v; }

private:
  // ===== 내부 상수 (파이썬과 동일) =====
  static constexpr int PREPROCESS_CONV_SIZE = 3;
  static constexpr double STRAIGHTS_STEERING_ANGLE = M_PI / 18.0;   // 10 deg
  static constexpr double MILD_CURVE_ANGLE        = M_PI / 6.0;     // 30 deg
  static constexpr double ULTRASTRAIGHTS_ANGLE    = M_PI / 60.0;    // 3 deg

  // ===== 입력 파라미터/상태 =====
  rclcpp::Node* node_{nullptr};
  rclcpp::Logger logger_;
  bool mapping_{false};
  bool DEBUG_{false};
  int SAFETY_RADIUS_{3};
  double MAX_LIDAR_DIST_{10.0};
  double MAX_SPEED_{3.0};
  int range_offset_{180}; // 배열 양끝에서 자르는 개수(파이썬 동일 의미)
  double track_width_{1.0};

  // 속도 스케일
  double CORNERS_SPEED_{0.0};
  double MILD_CORNERS_SPEED_{0.0};
  double STRAIGHTS_SPEED_{0.0};
  double ULTRASTRAIGHTS_SPEED_{0.0};

  // 런타임 상태
  double velocity_{0.0};
  double radians_per_elem_{0.0};

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr best_pnt_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scan_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr best_gap_pub_;

  // ===== 내부 구현 =====
  std::vector<double> preprocess_lidar(const std::vector<float>& ranges);
  std::pair<double,double> get_best_range_point(const std::vector<double>& proc_ranges);
  std::pair<int,int> find_largest_gap(const std::vector<double>& ranges, double radius);
  double get_radius() const;
  double get_steer_angle(double x, double y) const;
  std::vector<double> safety_border(const std::vector<double>& ranges);

  void publish_gap_markers(int gap_left, int gap_right, double radius);
  void publish_best_point(double bx, double by);
  void publish_scan_points(const std::vector<double>& proc_ranges);
  void delete_gap_markers();
};
