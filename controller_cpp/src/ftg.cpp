#include "ftg.hpp"

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

FTG_Controller::FTG_Controller(rclcpp::Node* node,
                               bool mapping,
                               bool debug,
                               double safety_radius,
                               double max_lidar_dist,
                               double max_speed,
                               double range_offset,
                               double track_width)
: node_(node),
  logger_(node_->get_logger()),
  mapping_(mapping),
  DEBUG_(debug),
  SAFETY_RADIUS_(safety_radius),
  MAX_LIDAR_DIST_(max_lidar_dist),
  MAX_SPEED_(max_speed),
  range_offset_(range_offset),
  track_width_(track_width)
{
  // Speed params (파이썬 scale=0.6 고정)
  const double scale = 0.6;
  CORNERS_SPEED_        = 0.3  * MAX_SPEED_ * scale;
  MILD_CORNERS_SPEED_   = 0.45 * MAX_SPEED_ * scale;
  STRAIGHTS_SPEED_      = 0.8  * MAX_SPEED_ * scale;
  ULTRASTRAIGHTS_SPEED_ = 1.0  * MAX_SPEED_ * scale;

  best_pnt_pub_ = node_->create_publisher<Marker>("/best_points/marker", 10);
  scan_pub_     = node_->create_publisher<MarkerArray>("/scan_proc/markers", 10);
  best_gap_pub_ = node_->create_publisher<MarkerArray>("/best_gap/markers", 10);
}

std::pair<double,double>
FTG_Controller::process_lidar(const std::vector<float>& ranges)
{
  if (ranges.size() < static_cast<size_t>(2*range_offset_ + PREPROCESS_CONV_SIZE)) {
    // 유효하지 않은 입력 방어
    return {0.0, 0.0};
  }

  // 1) Preprocess
  auto proc = preprocess_lidar(ranges);

  // 2) Safety bubble
  proc = safety_border(proc);

  // 3) 디버그: 전처리 스캔 마커
  if (DEBUG_) publish_scan_points(proc);

  // 4) 최대 갭의 중앙점(best point)
  auto [bx, by] = get_best_range_point(proc);

  // 5) 조향각
  const double steering = get_steer_angle(bx, by);

  // 6) 속도 결정
  double speed = 0.0;
  if (mapping_) {
    speed = 1.5;
  } else {
    const double a = std::abs(steering);
    if (a > MILD_CURVE_ANGLE)            speed = CORNERS_SPEED_;
    else if (a > STRAIGHTS_STEERING_ANGLE)  speed = MILD_CORNERS_SPEED_;
    else if (a > ULTRASTRAIGHTS_ANGLE)      speed = STRAIGHTS_SPEED_;
    else                                    speed = ULTRASTRAIGHTS_SPEED_;
  }

  return {speed, steering};
}

// ===== 내부 구현 =====

std::vector<double>
FTG_Controller::preprocess_lidar(const std::vector<float>& ranges)
{
  const int N = static_cast<int>(ranges.size());
  radians_per_elem_ = (1.5 * M_PI) / static_cast<double>(N); // -135~+135 deg 스팬 가정

  // 뒤쪽 범위 제외 및 앞/뒤 range_offset 만큼 제거
  std::vector<double> cut;
  cut.reserve(N - 2*range_offset_);
  for (int i = range_offset_; i < N - range_offset_; ++i) {
    cut.push_back(static_cast<double>(ranges[i]));
  }

  // 이동 평균(길이 PREPROCESS_CONV_SIZE)
  std::vector<double> smoothed;
  const int W = PREPROCESS_CONV_SIZE;
  if ((int)cut.size() >= W) {
    smoothed.resize(cut.size() - W + 1);
    for (size_t i=0; i<smoothed.size(); ++i) {
      double acc = 0.0;
      for (int k=0;k<W;++k) acc += cut[i+k];
      smoothed[i] = acc / (double)W;
    }
  } else {
    smoothed = cut; // 너무 짧으면 그대로
  }

  // 클리핑 [0, MAX_LIDAR_DIST]
  for (auto& v : smoothed) {
    if (v < 0.0) v = 0.0;
    if (v > MAX_LIDAR_DIST_) v = MAX_LIDAR_DIST_;
  }

  // LiDAR 오른->왼 정렬을 뒤집기
  std::reverse(smoothed.begin(), smoothed.end());
  return smoothed;
}

double FTG_Controller::get_steer_angle(double x, double y) const {
  const double angle = std::atan2(y, x);
  // [-0.4, 0.4] 제한
  return std::clamp(angle, -0.4, 0.4);
}

std::pair<double,double>
FTG_Controller::get_best_range_point(const std::vector<double>& proc_ranges)
{
  // 버블 반경
  const double radius = get_radius();

  // 최대 갭 찾기
  auto [gap_left, gap_right] = find_largest_gap(proc_ranges, radius);

  // Python 코드의 인덱스 보정(시각화를 위해 범용적으로 맞춤)
  // gap_middle를 각도로 환산 → 레이저 프레임에서 좌표
  // 역변환(보정)은 RViz 마커에서만 사용, 좌표 계산은 단순 중앙 인덱스 기준
  const int gap_middle = (gap_left + gap_right) / 2;

  // 디버그 마커
  if (DEBUG_) {
    delete_gap_markers();
    publish_gap_markers(gap_left, gap_right, radius);
  }

  const double best_y = std::cos(gap_middle * radians_per_elem_) * radius;
  const double best_x = std::sin(gap_middle * radians_per_elem_) * radius;

  if (DEBUG_) publish_best_point(best_x, best_y);

  return {best_x, best_y};
}

std::pair<int,int>
FTG_Controller::find_largest_gap(const std::vector<double>& ranges, double radius)
{
  const int n = static_cast<int>(ranges.size());
  if (n < 2) return {0, 0};

  // threshold 이진화
  std::vector<int> bin(n, 0);
  for (int i=0;i<n;++i) bin[i] = (ranges[i] >= radius) ? 1 : 0;

  // 변화점 찾기
  std::vector<int> diff(n-1, 0);
  for (int i=0;i<n-1;++i) diff[i] = std::abs(bin[i+1] - bin[i]);

  if (!diff.empty()) {
    diff.front() = 1;
    diff.back()  = 1;
  }

  std::vector<int> idxs;
  for (int i=0;i<(int)diff.size();++i) if (diff[i] != 0) idxs.push_back(i);

  if (idxs.size() < 2) {
    // 모두 같다면 전체를 갭으로 취급
    return {0, n-1};
  }

  // 후보 구간들 중 평균이 0.5보다 큰(=대부분 1) 구간의 길이를 사용
  int best_left=0, best_right=0, best_width=0;
  for (size_t i=0;i+1<idxs.size();++i) {
    int L = idxs[i];
    int R = idxs[i+1];
    int width = R - L;
    if (width <= 0) continue;

    // 평균 > 0.5 ?
    int sum = 0;
    for (int j=L; j<R; ++j) sum += bin[j];
    const double mean = static_cast<double>(sum) / static_cast<double>(width);
    if (mean > 0.5 && width > best_width) {
      best_width = width;
      best_left  = L;
      best_right = R;
    }
  }

  if (best_width == 0) {
    // 유효 구간 없으면 전체에서 가장 긴 1 구간 스캔
    int curL=-1;
    for (int i=0;i<n;++i) {
      if (bin[i]==1 && curL==-1) curL=i;
      if ((bin[i]==0 || i==n-1) && curL!=-1) {
        int curR = (bin[i]==0) ? i : (i+1);
        if (curR-curL > best_width) {
          best_width = curR-curL;
          best_left  = curL;
          best_right = curR;
        }
        curL = -1;
      }
    }
  }

  return {best_left, best_right};
}

double FTG_Controller::get_radius() const {
  // min(5, track_width/2 + 2 * (v/MAX_SPEED))
  return std::min(5.0, track_width_/2.0 + 2.0 * ( (MAX_SPEED_ > 1e-9) ? (velocity_/MAX_SPEED_) : 0.0 ));
}

std::vector<double>
FTG_Controller::safety_border(const std::vector<double>& ranges)
{
  std::vector<double> filtered = ranges;
  const int n = static_cast<int>(ranges.size());
  // 정방향
  for (int i=0; i<n-1; ) {
    if (ranges[i+1] - ranges[i] > 0.5) {
      for (int j=0; j<SAFETY_RADIUS_; ++j) {
        if (i+j < n) filtered[i+j] = ranges[i];
      }
      i += std::max(1, SAFETY_RADIUS_-2);
    } else {
      ++i;
    }
  }
  // 역방향
  for (int i=n-1; i>0; ) {
    if (ranges[i-1] - ranges[i] > 0.5) {
      for (int j=0; j<SAFETY_RADIUS_; ++j) {
        if (i-j >= 0) filtered[i-j] = ranges[i];
      }
      i -= std::max(1, SAFETY_RADIUS_-2);
    } else {
      --i;
    }
  }
  return filtered;
}

void FTG_Controller::publish_scan_points(const std::vector<double>& proc_ranges)
{
  MarkerArray ma;
  const auto now = node_->now();
  for (int i=0;i<(int)proc_ranges.size();++i) {
    Marker m;
    m.header.frame_id = "car_state/laser";
    m.header.stamp = now;
    m.type = Marker::SPHERE;
    m.scale.x = m.scale.y = m.scale.z = 0.05;
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.b = 1.0;
    m.id = i;
    m.pose.position.x = std::sin(i * radians_per_elem_) * proc_ranges[i];
    m.pose.position.y = std::cos(i * radians_per_elem_) * proc_ranges[i];
    m.pose.orientation.w = 1.0;
    ma.markers.push_back(std::move(m));
  }
  scan_pub_->publish(ma);
}

void FTG_Controller::publish_gap_markers(int gap_left, int gap_right, double radius)
{
  MarkerArray ma;
  const auto now = node_->now();
  for (int i=gap_left; i<gap_right; ++i) {
    Marker m;
    m.header.frame_id = "car_state/laser";
    m.header.stamp = now;
    m.type = Marker::SPHERE;
    m.scale.x = m.scale.y = m.scale.z = 0.05;
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.g = 1.0;
    m.id = i - gap_left;
    m.pose.position.y = std::cos(i * radians_per_elem_) * radius;
    m.pose.position.x = std::sin(i * radians_per_elem_) * radius;
    m.pose.orientation.w = 1.0;
    ma.markers.push_back(std::move(m));
  }
  best_gap_pub_->publish(ma);
}

void FTG_Controller::publish_best_point(double bx, double by)
{
  Marker m;
  m.header.frame_id = "car_state/laser";
  m.header.stamp = node_->now();
  m.type = Marker::SPHERE;
  m.scale.x = m.scale.y = m.scale.z = 0.2;
  m.color.a = 1.0;
  m.color.b = 1.0;
  m.color.g = 1.0;
  m.id = 0;
  m.pose.position.x = bx;
  m.pose.position.y = by;
  m.pose.orientation.w = 1.0;
  best_pnt_pub_->publish(m);
}

void FTG_Controller::delete_gap_markers()
{
  MarkerArray del;
  Marker m;
  m.header.frame_id = "car_state/laser";
  m.header.stamp = node_->now();
  m.action = Marker::DELETEALL;
  m.id = 0;
  del.markers.push_back(m);
  best_gap_pub_->publish(del);
}
