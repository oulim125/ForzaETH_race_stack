#pragma once
#include <string>
#include <vector>
#include <functional>
#include <optional>

class LookupSteerAngle {
public:
  using Logger = std::function<void(const std::string&)>;

  // model_name: 예) "NUC2_hangar_pacejka" (확장자/접미사 제외)
  // logger: RCLCPP_INFO/WARN 래핑 람다 등을 넘겨주세요. (없으면 기본 무시)
  explicit LookupSteerAngle(const std::string& model_name,
                            Logger logger = [](const std::string&){});

  // lat_accel [m/s^2], speed [m/s] -> steer [rad]
  double lookup_steer_angle(double accel, double vel) const;

  bool loaded() const { return loaded_; }

private:
  // CSV 로드 결과 lu[r][c]:
  //   r=0, c>=1: 속도 컬럼 헤더 (v1, v2, ...)
  //   r>=1, c=0: 스티어 헤더 (steer1, steer2, ...)
  //   r>=1, c>=1: 해당 (steer, v)에서의 +가속도 값 (또는 가속도 그리드)
  //
  // 원본 파이썬과 동일하게, 열 헤더는 속도, 행 헤더는 조향각,
  // 각 셀은 양의 가속도 그리드라고 가정하고 선형보간합니다.
  std::vector<std::vector<double>> lu_;
  bool loaded_{false};
  Logger logger_;

  // 내부 유틸
  static double clip(double v, double lo, double hi);
  static std::optional<std::pair<double,int>> find_nearest(const std::vector<double>& arr, double value);
  static std::tuple<double,int,double,int> find_closest_neighbors(const std::vector<double>& col, double value);
  static bool load_csv(const std::string& file, std::vector<std::vector<double>>& out);
};
