#include "lookup_steer_angle.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

static inline double lerp(double x, double x0, double x1, double y0, double y1) {
  if (std::abs(x1 - x0) < 1e-12) return y0;
  const double t = (x - x0) / (x1 - x0);
  return y0 + t * (y1 - y0);
}

double LookupSteerAngle::clip(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

std::optional<std::pair<double,int>>
LookupSteerAngle::find_nearest(const std::vector<double>& arr, double value) {
  if (arr.empty()) return std::nullopt;
  int idx = 0;
  double best = std::numeric_limits<double>::infinity();
  for (int i=0;i<(int)arr.size();++i) {
    double d = std::abs(arr[i] - value);
    if (d < best) { best = d; idx = i; }
  }
  return std::make_pair(arr[idx], idx);
}

std::tuple<double,int,double,int>
LookupSteerAngle::find_closest_neighbors(const std::vector<double>& col, double value) {
  // NaN 이전까지만 사용 (파이썬 로직 동일)
  int valid_len = (int)col.size();
  for (int i=0;i<(int)col.size();++i) {
    if (std::isnan(col[i])) { valid_len = i; break; }
  }
  std::vector<double> a(col.begin(), col.begin() + valid_len);
  if (a.empty()) {
    return {0.0, 0, 0.0, 0};
  }
  auto near = find_nearest(a, value);
  int closest_idx = near ? near->second : 0;
  double closest = near ? near->first : a.front();
  if (closest_idx == 0) {
    return {a[0], 0, a[0], 0};
  } else if (closest_idx == (int)a.size()-1) {
    return {a.back(), (int)a.size()-1, a.back(), (int)a.size()-1};
  } else {
    // 이웃 두 개 중 근접한 것을 다시 선택
    std::vector<double> neigh{ a[closest_idx-1], a[closest_idx+1] };
    auto near2 = find_nearest(neigh, value);
    int second_idx = (near2 && near2->second==1) ? closest_idx+1 : closest_idx-1;
    double second = a[second_idx];
    return {closest, closest_idx, second, second_idx};
  }
}

bool LookupSteerAngle::load_csv(const std::string& file, std::vector<std::vector<double>>& out) {
  std::ifstream ifs(file);
  if (!ifs.is_open()) return false;
  out.clear();
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      try {
        // 공백 trim
        size_t start = cell.find_first_not_of(" \t");
        size_t end   = cell.find_last_not_of(" \t");
        std::string tok = (start==std::string::npos) ? "" : cell.substr(start, end-start+1);
        if (tok.empty()) { row.push_back(std::numeric_limits<double>::quiet_NaN()); }
        else             { row.push_back(std::stod(tok)); }
      } catch (...) {
        row.push_back(std::numeric_limits<double>::quiet_NaN());
      }
    }
    out.push_back(std::move(row));
  }
  return !out.empty();
}

LookupSteerAngle::LookupSteerAngle(const std::string& model_name, Logger logger)
: logger_(std::move(logger))
{
  // forza_controller/share/forza_controller/cfg/<model>_lookup_table.csv 를 읽음
  // const auto share = ament_index_cpp::get_package_share_directory("forza_controller");
  const std::string share = "/home/misys/forza_ws/race_stack/controller_cpp";
  const std::string file = share + "/cfg/" + model_name + "_lookup_table.csv";
  if (!load_csv(file, lu_)) {
    logger_("LookupSteerAngle: failed to load csv: " + file);
    loaded_ = false;
    return;
  }
  loaded_ = true;
}

double LookupSteerAngle::lookup_steer_angle(double accel, double vel) const {
  if (!loaded_ || lu_.size() < 2 || lu_[0].size() < 2) return 0.0;

  // 부호는 마지막에 적용; 내부 보간은 |accel|
  const double sign_accel = (accel >= 0.0) ? 1.0 : -1.0;
  accel = std::abs(accel);

  // 헤더
  // 첫 행: [NaN, v1, v2, ...]
  // 첫 열: [NaN; steer1; steer2; ...]
  const std::vector<double> &lu_vs = lu_[0];        // size C
  std::vector<double> vs;
  for (size_t c=1;c<lu_vs.size();++c) vs.push_back(lu_vs[c]);

  std::vector<double> steers;
  for (size_t r=1;r<lu_.size();++r) {
    if (lu_[r].empty()) break;
    steers.push_back(lu_[r][0]);
  }

  if (vs.empty() || steers.empty()) return 0.0;

  // 속도에 가장 가까운 컬럼 선택
  auto nv = find_nearest(vs, vel);
  int c_v_idx = nv ? nv->second : 0;

  // 해당 속도 컬럼에서 (가속도) 가장 가까운 두 점 찾기
  // 열 인덱스는 +1 (헤더 보정)
  std::vector<double> acc_col;
  acc_col.reserve(steers.size());
  for (size_t r=1; r<lu_.size(); ++r) {
    if (c_v_idx+1 < (int)lu_[r].size())
      acc_col.push_back(lu_[r][c_v_idx+1]);
    else
      acc_col.push_back(std::numeric_limits<double>::quiet_NaN());
  }

  auto [c_a, c_a_idx, s_a, s_a_idx] = find_closest_neighbors(acc_col, accel);
  double steer_angle = 0.0;
  if (c_a_idx == s_a_idx) {
    steer_angle = steers[c_a_idx];
  } else {
    // 가속도 축 기준 선형 보간 -> 스티어값도 그에 맞춰 선형 보간
    steer_angle = lerp(accel, c_a, s_a, steers[c_a_idx], steers[s_a_idx]);
  }
  return steer_angle * sign_accel;
}
