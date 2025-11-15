#include "map.hpp"
#include <limits>
#include <cmath>
#include <algorithm>

MAP_Controller::MAP_Controller(double t_clip_min_,
                               double t_clip_max_,
                               double m_l1_,
                               double q_l1_,
                               double speed_lookahead_,
                               double lat_err_coeff_,
                               double acc_scaler_for_steer_,
                               double dec_scaler_for_steer_,
                               double start_scale_speed_,
                               double end_scale_speed_,
                               double downscale_factor_,
                               double speed_lookahead_for_steer_,

                               bool   prioritize_dyn_,
                               double trailing_gap_,
                               double trailing_p_gain_,
                               double trailing_i_gain_,
                               double trailing_d_gain_,
                               double blind_trailing_speed_,
                               double trailing_to_gbtrack_speed_scale_,

                               double loop_rate_,
                               const std::string& LUT_name,

                               Logger logger_info_,
                               Logger logger_warn_)
: t_clip_min(t_clip_min_), t_clip_max(t_clip_max_),
  m_l1(m_l1_), q_l1(q_l1_),
  speed_lookahead(speed_lookahead_), lat_err_coeff(lat_err_coeff_),
  acc_scaler_for_steer(acc_scaler_for_steer_), dec_scaler_for_steer(dec_scaler_for_steer_),
  start_scale_speed(start_scale_speed_), end_scale_speed(end_scale_speed_),
  downscale_factor(downscale_factor_), speed_lookahead_for_steer(speed_lookahead_for_steer_),
  prioritize_dyn(prioritize_dyn_),
  trailing_gap(trailing_gap_), trailing_p_gain(trailing_p_gain_),
  trailing_i_gain(trailing_i_gain_), trailing_d_gain(trailing_d_gain_),
  blind_trailing_speed(blind_trailing_speed_),
  trailing_to_gbtrack_speed_scale(trailing_to_gbtrack_speed_scale_),
  loop_rate(loop_rate_),
  logger_info(logger_info_), logger_warn(logger_warn_),
  LUT_name_(LUT_name),
  steer_lookup_(LUT_name_, logger_info_)
{}

MAP_Controller::Output
MAP_Controller::main_loop(const std::string& state,
                          const std::optional<Pose3>& position_in_map,
                          const std::vector<WpRow>& waypoint_array_in_map,
                          double speed_now,
                          const std::optional<Opp5>& opponent,
                          const std::optional<Fren4>& position_in_map_frenet,
                          const std::vector<double>& acc_now,
                          double track_length)
{
  // 업데이트
  state_         = state;
  if (position_in_map) pose_ = *position_in_map;
  wp_            = waypoint_array_in_map;
  speed_now_     = speed_now;
  opp_           = opponent;
  frenet_        = position_in_map_frenet;
  acc_now_       = acc_now;
  track_length_  = track_length;

  // PREPROCESS
  const double yaw = pose_[2];
  const Vec2 v{ std::cos(yaw)*speed_now_, std::sin(yaw)*speed_now_ };

  // lateral error (from frenet d)
  auto [lat_e_norm, lateral_error] = calc_lateral_error_norm();
  auto [L1_point, L1_distance] = calc_L1_point(lateral_error);


  // LONGITUDINAL
  const double adv_ts_sp = speed_lookahead;
  const Vec2 la_pos{ pose_[0] + v[0]*adv_ts_sp, pose_[1] + v[1]*adv_ts_sp };
  const int idx_la = nearest_waypoint(la_pos, wp_);
  double global_speed = (idx_la >= 0) ? wp_[idx_la][2] : 0.0;

  if (state_ == "StateType.TRAILING" && opp_ && frenet_) {
    speed_command_ = trailing_controller(global_speed);
  } else if (state_ == "StateType.TRAILING_TO_GBTRACK") {
    speed_command_ = global_speed * trailing_to_gbtrack_speed_scale;
  } else {
    i_gap_ = 0.0;
    speed_command_ = global_speed;
  }

  // 곡률 최신값을 반영해 감속
  speed_command_ = speed_adjust_lat_err(speed_command_, lat_e_norm);
  speed_command_ = speed_adjust_heading(speed_command_);

  // POSTPROCESS ...
  double speed = std::isfinite(speed_command_) ? std::max(speed_command_, 0.0) : 0.0;

  // LATERAL (이미 계산한 L1을 재사용)
  double steering_angle = calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v);

  return {speed, 0.0, 0.0, steering_angle, L1_point, L1_distance, idx_nearest_wp_};
}

int MAP_Controller::nearest_waypoint(const Vec2& position, const std::vector<WpRow>& waypoints) const {
  if (waypoints.empty()) return -1;
  int best = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i=0;i<(int)waypoints.size();++i) {
    double dx = position[0] - waypoints[i][0];
    double dy = position[1] - waypoints[i][1];
    double d2 = dx*dx + dy*dy;
    if (d2 < best_d2) { best_d2 = d2; best = i; }
  }
  return best;
}

MAP_Controller::Vec2
MAP_Controller::waypoint_at_distance_before_car(double distance,
                                                const std::vector<WpRow>& waypoints,
                                                int idx_waypoint_behind_car) const
{
  if (distance <= 0.0) distance = t_clip_min;
  const double waypoints_distance = 0.1;
  const int d_index = static_cast<int>(distance/waypoints_distance + 0.5);
  const int idx = std::min((int)waypoints.size()-1, idx_waypoint_behind_car + d_index);
  return Vec2{ waypoints[idx][0], waypoints[idx][1] };
}

std::pair<double,double> MAP_Controller::calc_lateral_error_norm() const {
  double lateral_error = 0.0;
  if (frenet_) lateral_error = std::abs((*frenet_)[1]); // d
  const double max_lat_e = 0.5, min_lat_e = 0.0;
  const double lat_e_clip = clip(lateral_error, min_lat_e, max_lat_e);
  const double lat_e_norm = 0.5 * ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e));
  return {lat_e_norm, lateral_error};
}

double MAP_Controller::speed_adjust_lat_err(double global_speed, double lat_e_norm) const {
  const double lat_e_coeff = clip(lat_err_coeff, 0.0, 1.0);
  const double lat_e = lat_e_norm * 2.0;
  const double curv = clip(curvature_waypoints_ / 0.28, 0.0, 1.0); // 0.28 하드코딩 된 것이므로 추후 개선 필요 
  return global_speed * (1.0 - lat_e_coeff + lat_e_coeff*std::exp(-lat_e*curv));
}

// double MAP_Controller::speed_adjust_heading(double speed_command) const {
//   if (idx_nearest_wp_ < 0 || idx_nearest_wp_ >= (int)wp_.size()) return speed_command;
//   const double heading = pose_[2];
//   const double map_heading = wp_[idx_nearest_wp_][6];
//   double diff = std::abs(heading - map_heading);
//   if (diff > M_PI) diff = 2*M_PI - diff;

//   if (diff < M_PI/9.0) return speed_command;
//   double scaler = (diff < M_PI/2.0)
//     ? 1.0 - 0.5 * (diff/(M_PI/2.0))
//     : 0.5;
//   logger_info("[MAP] heading error decreasing velocity");
//   return speed_command * scaler;
// }

double MAP_Controller::speed_adjust_heading(double speed_command) const {
  if (wp_.empty()) return speed_command;

  // 최근접 웨이포인트 인덱스를 내부에서 계산 (순서 의존성 제거)
  const int idx = nearest_waypoint({pose_[0], pose_[1]}, wp_);
  if (idx < 0 || idx >= static_cast<int>(wp_.size())) return speed_command;

  const double heading     = pose_[2];
  const double map_heading = wp_[idx][6];

  // 래핑된 각도 차이 dpsi ∈ [-pi, pi]
  const double dpsi = std::atan2(std::sin(heading - map_heading), std::cos(heading - map_heading));
  const double heading_error = std::abs(dpsi);

  // 임계값 및 최소 스케일
  const double ok_thresh   = M_PI / 15.0; // ≈ 12 degrees
  const double hard_thresh = M_PI / 2.0;  // 90 degrees
  const double min_scale   = 0.5;         // 최소 50%

  double scale = 1.0;
  if (heading_error <= ok_thresh) {
    scale = 1.0;
  } else if (heading_error < hard_thresh) {
    // 12°~90° 구간에서 1.0 → 0.5 선형 감소
    const double t = (heading_error - ok_thresh) / (hard_thresh - ok_thresh); // ∈ (0,1)
    scale = min_scale + (1.0 - min_scale) * (1.0 - t);
  } else {
    scale = min_scale;
  }

  // 로깅 (deg 표기)
  const double deg = heading_error * 180.0 / M_PI;
  // logger_info(("[MAP] heading error=" + std::to_string(deg) + " deg, speed scale=" + std::to_string(scale)).c_str());

  return speed_command * scale;
}

double MAP_Controller::acc_scaling(double steer) const {
  double mean_acc = 0.0;
  if (!acc_now_.empty()) {
    for (double a : acc_now_) mean_acc += a;
    mean_acc /= (double)acc_now_.size();
  }
  if (mean_acc >= 1.0)      steer *= acc_scaler_for_steer;
  else if (mean_acc <= -1.0)steer *= dec_scaler_for_steer;
  return steer;
}

double MAP_Controller::speed_steer_scaling(double steer, double speed) const {
  const double speed_diff = std::max(0.1, end_scale_speed - start_scale_speed);
  const double factor = 1.0 - clip((speed - start_scale_speed)/speed_diff, 0.0, 1.0) * downscale_factor;
  return steer * factor;
}

double MAP_Controller::trailing_controller(double global_speed) {
  const double opp_s      = (*opp_)[0];
  const double opp_vs     = (*opp_)[2];
  const bool   opp_visible= ((*opp_)[4] > 0.5);

  const double my_s  = (*frenet_)[0];
  const double my_vs = (*frenet_)[2];

  gap_ = std::fmod(opp_s - my_s + track_length_, track_length_);
  gap_actual_ = gap_;
  gap_should_ = trailing_gap;
  gap_error_  = gap_should_ - gap_actual_;
  v_diff_     = my_vs - opp_vs;
  i_gap_ = clip(i_gap_ + gap_error_/loop_rate, -10.0, 10.0);

  const double p_value = gap_error_ * trailing_p_gain;
  const double d_value = v_diff_    * trailing_d_gain;
  const double i_value = i_gap_     * trailing_i_gain;

  trailing_command_ = clip(opp_vs - p_value - i_value - d_value, 0.0, global_speed);
  if (!opp_visible && gap_actual_ > gap_should_) {
    trailing_command_ = std::max(blind_trailing_speed, trailing_command_);
  }
  return trailing_command_;
}

std::pair<MAP_Controller::Vec2,double>
MAP_Controller::calc_L1_point(double lateral_error) {
  idx_nearest_wp_ = nearest_waypoint({pose_[0], pose_[1]}, wp_);
  if (idx_nearest_wp_ < 0) idx_nearest_wp_ = 0;

  if ((int)wp_.size() - idx_nearest_wp_ > 2) {
    double sum=0.0; int cnt=0;
    for (int i=idx_nearest_wp_; i<(int)wp_.size(); ++i) { sum += std::abs(wp_[i][5]); ++cnt; }
    curvature_waypoints_ = (cnt>0) ? (sum/cnt) : 0.0;
  }

  double L1_distance = q_l1 + speed_now_ * m_l1;
  const double lower_bound = std::max(t_clip_min, std::sqrt(2.0)*std::abs(lateral_error));
  L1_distance = clip(L1_distance, lower_bound, t_clip_max);

  Vec2 L1_point = waypoint_at_distance_before_car(L1_distance, wp_, idx_nearest_wp_);
  return {L1_point, L1_distance};
}

double MAP_Controller::calc_steering_angle(const Vec2& L1_point,
                                           double L1_distance,
                                           double yaw,
                                           double lat_e_norm,
                                           const Vec2& v)
{
  // lookahead for steer
  double speed_la_for_lu = 0.0;
  if (state_ == "StateType.TRAILING" && opp_) {
    speed_la_for_lu = speed_now_;
  } else {
    const double adv_ts_st = speed_lookahead_for_steer;
    const Vec2 la_pos{ pose_[0] + v[0]*adv_ts_st, pose_[1] + v[1]*adv_ts_st };
    const int idx = nearest_waypoint(la_pos, wp_);
    speed_la_for_lu = (idx>=0) ? wp_[idx][2] : speed_now_;
  }
  const double speed_for_lu = speed_adjust_lat_err(speed_la_for_lu, lat_e_norm);

  // eta
  const Vec2 L1_vec{ L1_point[0] - pose_[0], L1_point[1] - pose_[1] };
  const double L1_norm = std::hypot(L1_vec[0], L1_vec[1]);
  double eta = 0.0;
  if (L1_norm <= 1e-8) {
    logger_warn("[MAP] norm(L1_vector)==0, eta=0");
  } else {
    const double dot = (-std::sin(yaw))*L1_vec[0] + ( std::cos(yaw))*L1_vec[1];
    eta = std::asin( clip(dot / L1_norm, -1.0, 1.0) );
  }

  // lat_acc = 2 * v^2 / L1 * sin(eta)
  double lat_acc = 0.0;
  if (L1_distance <= 1e-8 || std::abs(std::sin(eta)) <= 1e-12) {
    lat_acc = 0.0;
    logger_warn("[MAP] L1 or sin(eta) ~ 0, lat_acc=0");
  } else {
    lat_acc = 2.0 * speed_for_lu * speed_for_lu / L1_distance * std::sin(eta);
  }

  // LUT로 조향각 계산
  double steer = steer_lookup_.lookup_steer_angle(lat_acc, speed_for_lu);

  // 가감속/속도 기반 스케일링
  steer = acc_scaling(steer);
  steer = speed_steer_scaling(steer, speed_for_lu);
  steer *= clip(1.0 + (speed_now_/10.0), 1.0, 1.25);

  // rate limit
  const double threshold = 0.4;
  if (std::abs(steer - curr_steering_angle_) > threshold) {
    // logger_info("[MAP] steering angle clipped");
  }
  steer = clip(steer, curr_steering_angle_ - threshold, curr_steering_angle_ + threshold);
  curr_steering_angle_ = steer;

  return steer;
}
