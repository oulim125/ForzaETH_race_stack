#pragma once
#include <array>
#include <vector>
#include <string>
#include <tuple>
#include <functional>
#include <cmath>
#include <algorithm>
#include <optional>

class PP_Controller {
public:
  using Vec2  = std::array<double,2>;
  using Pose3 = std::array<double,3>;   // x, y, yaw
  using Fren4 = std::array<double,4>;   // s, d, vs, vd
  using Opp5  = std::array<double,5>;   // s, d, vs, is_static(0/1), is_visible(0/1)
  // waypoint row: [x, y, v, share, s, kappa, psi, ax]
  using WpRow = std::array<double,8>;

  using Logger = std::function<void(const std::string&)>;

  PP_Controller(double t_clip_min,
                double t_clip_max,
                double m_l1,
                double q_l1,
                double speed_lookahead,
                double lat_err_coeff,
                double acc_scaler_for_steer,
                double dec_scaler_for_steer,
                double start_scale_speed,
                double end_scale_speed,
                double downscale_factor,
                double speed_lookahead_for_steer,

                bool   prioritize_dyn,
                double trailing_gap,
                double trailing_p_gain,
                double trailing_i_gain,
                double trailing_d_gain,
                double blind_trailing_speed,
                double trailing_to_gbtrack_speed_scale,

                double loop_rate,
                double wheelbase,

                Logger logger_info  = [](auto const&){},
                Logger logger_warn  = [](auto const&){});

  // Python과 동일한 반환 순서:
  // (speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint)
  using Output = std::tuple<double,double,double,double,Vec2,double,int>;

  Output main_loop(const std::string& state,
                   const std::optional<Pose3>& position_in_map,
                   const std::vector<WpRow>& waypoint_array_in_map,
                   double speed_now,
                   const std::optional<Opp5>& opponent,
                   const std::optional<Fren4>& position_in_map_frenet,
                   const std::vector<double>& acc_now,
                   double track_length);

  // 동적 파라미터 갱신을 위해 public 멤버 그대로 노출 (Python과 동일)
  double t_clip_min, t_clip_max, m_l1, q_l1, speed_lookahead, lat_err_coeff;
  double acc_scaler_for_steer, dec_scaler_for_steer;
  double start_scale_speed, end_scale_speed, downscale_factor, speed_lookahead_for_steer;
  bool   prioritize_dyn;
  double trailing_gap, trailing_p_gain, trailing_i_gain, trailing_d_gain, blind_trailing_speed, trailing_to_gbtrack_speed_scale;
  double loop_rate, wheelbase;

  // 상태 플래그 (원본 코드와 호환)
  bool flag1{false};

  // 내부 로깅 함수 교체 가능
  Logger logger_info, logger_warn;

private:
  // 내부 상태
  std::string state_;
  Pose3 pose_{0,0,0};
  std::vector<WpRow> wp_;
  double speed_now_{0.0};
  std::optional<Opp5> opp_;
  std::optional<Fren4> frenet_;
  std::vector<double> acc_now_;
  double track_length_{0.0};

  // controller 내부 파라미터
  std::vector<double> lateral_error_list_; // (미사용: 추후 분석 시 유지)
  double curr_steering_angle_{0.0};
  int idx_nearest_wp_{-1};
  double curvature_waypoints_{0.0};
  double max_curvature_{0.0};

  std::array<double,10> d_vs_{{0}};
  double acceleration_command_{0.0};

  // trailing 상태
  double gap_{0.0}, gap_should_{0.0}, gap_error_{0.0}, gap_actual_{0.0}, v_diff_{0.0};
  double i_gap_{0.0};
  double trailing_command_{2.0};
  double speed_command_{0.0};
  double trailing_speed_{0.0};

  // 내부 유틸
  static inline double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
  }
  static inline double norm2(double x, double y) { return std::sqrt(x*x + y*y); }

  int nearest_waypoint(const Vec2& position, const std::vector<WpRow>& waypoints) const;
  Vec2 waypoint_at_distance_before_car(double distance,
                                       const std::vector<WpRow>& waypoints,
                                       int idx_waypoint_behind_car) const;

  std::pair<double,double> calc_lateral_error_norm() const; // (lat_e_norm, lateral_error)
  double speed_adjust_lat_err(double global_speed, double lat_e_norm) const;
  double speed_adjust_heading(double speed_command) const;

  double acc_scaling(double steer) const;
  double speed_steer_scaling(double steer, double speed) const;

  double trailing_controller(double global_speed);

  std::pair<Vec2,double> calc_L1_point(double lateral_error);
  double calc_steering_angle(const Vec2& L1_point, double L1_distance,
                             double yaw, double lat_e_norm,
                             const Vec2& v);
};
