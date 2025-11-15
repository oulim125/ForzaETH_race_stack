// tracking_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <f110_msgs/msg/wpnt.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <f110_msgs/msg/obstacle.hpp>
#include <f110_msgs/msg/obstacle_array.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <deque>
#include <optional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>

using std::placeholders::_1;

static inline double wrap_track(double s, double L) {
  double m = std::fmod(s, L);
  if (m < 0) m += L;
  if (m > L*0.5) m -= L;
  return m;
}
template<typename T> static inline T clamp(T v, T lo, T hi){ return std::max(lo, std::min(hi, v)); }

struct ObsSD {
  int id{0};
  std::vector<double> meas_s, meas_d;
  std::array<double,2> mean{0.0,0.0}; // (s,d)
  int static_count{0}, total_count{0}, nb_meas{0};
  int ttl{3};
  bool isInFront{true};
  int current_lap{0};
  std::optional<bool> staticFlag;
  double size{0.0};
  int nb_detection{0};
  bool isVisible{true};

  static int    MIN_NB_MEAS;   // 5
  static int    TTL_DEFAULT;   // 3
  static double MIN_STD;       // 0.16
  static double MAX_STD;       // 0.2

  ObsSD(int _id, double s, double d, int lap, double sz, bool vis)
  : id(_id), mean{s,d}, current_lap(lap), size(sz), isVisible(vis) {
    meas_s.push_back(s); meas_d.push_back(d);
    ttl = TTL_DEFAULT;
  }

  void update_mean(double track_len){
    if (nb_meas==0) { mean = {meas_s.back(), meas_d.back()}; return; }
    mean[1] = (mean[1]*nb_meas + meas_d.back())/(nb_meas+1);

    double prev_ang = mean[0] * 2*M_PI/track_len;
    double meas_ang = meas_s.back() * 2*M_PI/track_len;
    double c = (std::cos(prev_ang)*nb_meas + std::cos(meas_ang))/(nb_meas+1);
    double s = (std::sin(prev_ang)*nb_meas + std::sin(meas_ang))/(nb_meas+1);
    double ang = std::atan2(s,c);
    double smean = ang * track_len / (2*M_PI);
    mean[0] = (smean>=0 ? smean : smean + track_len);
  }

  double std_s(double track_len) const {
    double mu = mean[0], acc=0.0;
    for(double s:meas_s) acc += std::pow(wrap_track(s-mu, track_len), 2);
    return std::sqrt(acc/(double)meas_s.size());
  }
  double std_d() const {
    if (meas_d.empty()) return 0.0;
    double mu = std::accumulate(meas_d.begin(),meas_d.end(),0.0)/(double)meas_d.size();
    double acc=0.0; for(double v:meas_d) acc += (v-mu)*(v-mu);
    return std::sqrt(acc/(double)meas_d.size());
  }

  void classify_static(double track_len){
    if (nb_meas > MIN_NB_MEAS) {
      double ss = std_s(track_len), sd = std_d();
      if (ss < MIN_STD && sd < MIN_STD)      static_count++;
      else if (ss > MAX_STD || sd > MAX_STD) static_count=0;
      total_count++;
      staticFlag = (static_count/(double)std::max(total_count,1)) >= 0.5;
    } else {
      staticFlag.reset();
    }
  }
};
int    ObsSD::MIN_NB_MEAS = 5;
int    ObsSD::TTL_DEFAULT = 3;
double ObsSD::MIN_STD     = 0.16;
double ObsSD::MAX_STD     = 0.20;

// 간이 Cartesian 변환(시각화/가시성 판정용)
struct SimpleFrenet {
  std::vector<double> wx, wy, psi, s_acc;
  double L{0.0};
  void set_path(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& PSI){
    wx=X; wy=Y; psi=PSI;
    const int N=wx.size();
    s_acc.assign(N,0.0);
    for(int i=1;i<N;++i) s_acc[i]=s_acc[i-1]+std::hypot(wx[i]-wx[i-1], wy[i]-wy[i-1]);
    L = N? s_acc.back() : 0.0;
  }
  std::array<double,2> get_cartesian(double s, double d) const {
    if (wx.empty()) return {0,0};
    auto it = std::lower_bound(s_acc.begin(), s_acc.end(), s);
    int i0 = (it==s_acc.begin()?0:int(it - s_acc.begin()-1));
    int i1 = std::min(i0+1, (int)wx.size()-1);
    double t = (s_acc[i1]-s_acc[i0] > 1e-6) ? (s - s_acc[i0])/(s_acc[i1]-s_acc[i0]) : 0.0;
    double x = wx[i0] + t*(wx[i1]-wx[i0]);
    double y = wy[i0] + t*(wy[i1]-wy[i0]);
    double th = psi[i0];
    double nx = -std::sin(th), ny = std::cos(th);
    return { x + nx*d, y + ny*d };
  }
};

// EKF(x=[s,vs,d,vd])
struct OppEKF {
  Eigen::Vector4d x;            // [s, vs, d, vd]
  Eigen::Matrix4d P;
  Eigen::Matrix4d Q;
  Eigen::Matrix<double,4,4> H;
  Eigen::Matrix4d R;
  Eigen::Matrix4d F;
  Eigen::Matrix4d B;

  double dt{0.025};
  double L{1.0}; // track length
  bool initialised{false};
  int ttl{40};
  double size{0.0};
  int id{0};
  bool useTargetVel{false};

  std::vector<double> vs_hist;
  std::deque<double> vs_filt, vd_filt; // 길이 5 이동평균 버퍼

  double P_vs{0.2}, P_d{0.02}, P_vd{0.2};
  double ratio_to_glob{0.6};

  const std::vector<f110_msgs::msg::Wpnt>* waypoints{nullptr};

  void configure(double _dt, double track_len,
                 double meas_var_s, double meas_var_vs,
                 double meas_var_d, double meas_var_vd,
                 double proc_var_vs, double proc_var_vd,
                 int    _ttl,
                 double _Pvs, double _Pd, double _Pvd,
                 double _ratio,
                 const std::vector<f110_msgs::msg::Wpnt>* glb) {

    dt = _dt; L = track_len; ttl=_ttl;
    P_vs=_Pvs; P_d=_Pd; P_vd=_Pvd; ratio_to_glob=_ratio; waypoints=glb;

    F.setIdentity();
    F(0,1)=dt;
    F(2,3)=dt;

    // Q: discrete white noise (accel noise) 블록
    auto make_q = [&](double var){
      Eigen::Matrix2d q;
      double dt2 = dt*dt;
      double dt3 = dt2*dt;
      double dt4 = dt2*dt2;
      q << dt4/4.0, dt3/2.0,
           dt3/2.0, dt2;
      return var * q;
    };
    Eigen::Matrix2d q_s = make_q(proc_var_vs);
    Eigen::Matrix2d q_d = make_q(proc_var_vd);
    Q.setZero();
    Q.block<2,2>(0,0) = q_s; // [s,vs]
    Q.block<2,2>(2,2) = q_d; // [d,vd]

    H.setIdentity();
    R.setZero();
    R(0,0)=meas_var_s;
    R(1,1)=meas_var_vs;
    R(2,2)=meas_var_d;
    R(3,3)=meas_var_vd;

    P.setIdentity();
    P(0,0)=meas_var_s;  P(1,1)=proc_var_vs;
    P(2,2)=meas_var_d;  P(3,3)=proc_var_vd;

    B.setIdentity();
    initialised=false;
    vs_hist.clear();
    vs_filt.clear();
    vd_filt.clear();
  }

  double target_velocity() const {
    if (!waypoints || waypoints->empty()) return 0.0;
    int idx = (int)std::fmod(std::max(0.0, x(0)*10.0), L);
    idx = clamp<int>(idx, 0, (int)waypoints->size()-1);
    return ratio_to_glob * waypoints->at(idx).vx_mps;
  }

  static Eigen::Vector4d residual_h(const Eigen::Vector4d& a, const Eigen::Vector4d& b, double L){
    Eigen::Vector4d y = a - b;
    y(0)=wrap_track(y(0), L);
    return y;
  }

  void predict(){
    Eigen::Vector4d u = Eigen::Vector4d::Zero();
    if (useTargetVel) {
      u(1) = P_vs*(target_velocity() - x(1));
      u(2) = -P_d * x(2);
      u(3) = -P_vd* x(3);
    } else {
      u(2) = -P_d * x(2);
      u(3) = -P_vd* x(3);
    }
    x = F*x + B*u;
    P = F*P*F.transpose() + Q;
    x(0) = wrap_track(x(0), L);
  }

  void update(const Eigen::Vector4d& z){
    // Hx
    Eigen::Vector4d hx = x;
    hx(0) = wrap_track(hx(0), L);
    // 잔차
    Eigen::Vector4d y = residual_h(z, hx, L);
    Eigen::Matrix4d S = H*P*H.transpose() + R;
    Eigen::Matrix4d K = P*H.transpose()*S.inverse();
    x = x + K*y;
    P = (Eigen::Matrix4d::Identity() - K*H)*P;
    x(0) = wrap_track(x(0), L);

    // 속도 히스토리 & 5샘플 필터
    vs_hist.push_back(x(1));
    if (vs_hist.size() > 20) vs_hist.erase(vs_hist.begin(), vs_hist.end()-10);
    vs_filt.push_back(x(1)); if (vs_filt.size()>5) vs_filt.pop_back();
    vd_filt.push_front(x(3)); if (vd_filt.size()>5) vd_filt.pop_back();
  }

  double mean_vs() const {
    if (vs_filt.empty()) return 0.0;
    return std::accumulate(vs_filt.begin(), vs_filt.end(), 0.0) / (double)vs_filt.size();
  }
  double mean_vd() const {
    if (vd_filt.empty()) return 0.0;
    return std::accumulate(vd_filt.begin(), vd_filt.end(), 0.0) / (double)vd_filt.size();
  }
};

class TrackingNode : public rclcpp::Node {
public:
  TrackingNode()
  : Node("tracking",
         rclcpp::NodeOptions()
           .allow_undeclared_parameters(true)
           .automatically_declare_parameters_from_overrides(true))
  {
    // -------- 파라미터(고정) --------
    rate_ = this->declare_parameter<int>("rate", 40);
    P_vs_ = this->declare_parameter<double>("P_vs", 0.2);
    P_d_  = this->declare_parameter<double>("P_d", 0.02);
    P_vd_ = this->declare_parameter<double>("P_vd", 0.2);
    meas_var_s_  = this->declare_parameter<double>("measurment_var_s", 0.002);
    meas_var_d_  = this->declare_parameter<double>("measurment_var_d", 0.002);
    meas_var_vs_ = this->declare_parameter<double>("measurment_var_vs", 0.2);
    meas_var_vd_ = this->declare_parameter<double>("measurment_var_vd", 0.2);
    proc_var_vs_ = this->declare_parameter<int>("process_var_vs", 2);
    proc_var_vd_ = this->declare_parameter<int>("process_var_vd", 8);
    max_dist_    = this->declare_parameter<double>("max_dist", 0.5);
    var_pub_     = this->declare_parameter<int>("var_pub", 1);

    // 동적(런타임) 파라미터 초기값
    opp_ttl_ = this->declare_parameter<int>("ttl_dynamic", 40);
    ratio_to_glob_ = this->declare_parameter<double>("ratio_to_glob_path", 0.6);
    ObsSD::TTL_DEFAULT = this->declare_parameter<int>("ttl_static", 3);
    ObsSD::MIN_NB_MEAS = this->declare_parameter<int>("min_nb_meas", 5);
    ObsSD::MIN_STD     = this->declare_parameter<double>("min_std", 0.16);
    ObsSD::MAX_STD     = this->declare_parameter<double>("max_std", 0.20);
    dist_deletion_     = this->declare_parameter<double>("dist_deletion", 7.0);
    dist_infront_      = this->declare_parameter<double>("dist_infront", 8.0);
    vs_reset_          = this->declare_parameter<double>("vs_reset", 0.1);
    aggro_multi_       = this->declare_parameter<double>("aggro_multi", 2.0);
    debug_mode_        = this->declare_parameter<bool>("debug_mode", false);
    publish_static_    = this->declare_parameter<bool>("publish_static", true);
    no_memory_         = this->declare_parameter<bool>("noMemoryMode", false);

    // 동적 파라미터 콜백
    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&TrackingNode::on_set_params, this, std::placeholders::_1));

    // 구독
    obs_sub_  = this->create_subscription<f110_msgs::msg::ObstacleArray>(
      "/perception/detection/raw_obstacles", 10, std::bind(&TrackingNode::raw_obs_cb, this, _1));
    wpts_sub_ = this->create_subscription<f110_msgs::msg::WpntArray>(
      "/global_waypoints", 10, std::bind(&TrackingNode::wpts_cb, this, _1));
    frenet_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/car_state/frenet/odom", 10, std::bind(&TrackingNode::odom_frenet_cb, this, _1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/car_state/odom", 10, std::bind(&TrackingNode::odom_cb, this, _1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&TrackingNode::scan_cb, this, _1));

    // 퍼블리셔
    obstacles_pub_ = this->create_publisher<f110_msgs::msg::ObstacleArray>("/perception/obstacles", 5);
    raw_opponent_pub_ = this->create_publisher<f110_msgs::msg::ObstacleArray>("/perception/raw_obstacles", 5);
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/static_dynamic_marker_pub", 5);

    // 타이머
    const double period = (rate_>0)? 1.0/double(rate_): 0.025;
    timer_ = this->create_wall_timer(std::chrono::duration<double>(period), std::bind(&TrackingNode::loop, this));

    RCLCPP_INFO(this->get_logger(), "[Tracking] C++ node ready. rate=%d", rate_);
  }

private:
  // 콜백들
  void raw_obs_cb(const f110_msgs::msg::ObstacleArray::SharedPtr msg){
    meas_obstacles_ = msg->obstacles;
    current_stamp_ = msg->header.stamp;
  }

  void wpts_cb(const f110_msgs::msg::WpntArray::SharedPtr msg){
    if (track_len_ > 0.0) return;
    if (msg->wpnts.empty()) return;
    glb_ = msg->wpnts;

    // 파이썬과 동일: track_length = 마지막 wp의 s_m
    track_len_ = glb_.back().s_m;

    // Cartesian 변환용 보조(옵션)
    std::vector<double> X,Y,PSI;
    X.reserve(glb_.size()); Y.reserve(glb_.size()); PSI.reserve(glb_.size());
    for (auto &w: glb_) { X.push_back(w.x_m); Y.push_back(w.y_m); PSI.push_back(w.psi_rad); }
    frenet_.set_path(X,Y,PSI);

    ekf_.configure(1.0/double(rate_), track_len_,
                   meas_var_s_, meas_var_vs_, meas_var_d_, meas_var_vd_,
                   (double)proc_var_vs_, (double)proc_var_vd_,
                   opp_ttl_, P_vs_, P_d_, P_vd_, ratio_to_glob_, &glb_);
    RCLCPP_INFO(this->get_logger(), "[Tracking] received global path (track_length=%.3f)", track_len_);
  }

  void odom_frenet_cb(const nav_msgs::msg::Odometry::SharedPtr msg){
    car_s_ = msg->pose.pose.position.x;
    if (!last_car_s_) last_car_s_ = car_s_;
  }

  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg){
    car_xy_[0] = msg->pose.pose.position.x;
    car_xy_[1] = msg->pose.pose.position.y;
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.pose.orientation, q);
    double r,p,y; tf2::Matrix3x3(q).getRPY(r,p,y);
    car_dir_[0] = std::cos(y); car_dir_[1] = std::sin(y);
  }

  void scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg){
    scans_ = msg->ranges;
    scan_min_ = msg->angle_min;
    scan_max_ = msg->angle_max;
    scan_inc_ = msg->angle_increment;
  }

  // 메인 루프
  void loop(){
    if (track_len_ <= 0.0 || !car_s_.has_value()) return;

    // 랩 업데이트
    if (last_car_s_) {
      if (car_s_.value() - last_car_s_.value() < -track_len_/2.0) current_lap_++;
      last_car_s_ = car_s_;
    }

    if (ekf_.initialised) ekf_.predict();

    update_tracking_();
    publish_obstacles_();
    //publish_markers_();
  }

  void update_tracking_(){
    if (meas_obstacles_.empty()) {
      age_and_prune_();
      return;
    }

    auto meas_copy = meas_obstacles_;
    std::vector<ObsSD*> to_remove;

    for (auto &trk : tracked_) {
      const f110_msgs::msg::Obstacle* meas = verify_position_(trk, meas_copy);
      bool tracked = (meas != nullptr);

      if (tracked) {
        // 갱신
        trk.meas_s.push_back(meas->s_center);
        trk.meas_d.push_back(meas->d_center);
        if (trk.meas_s.size() > 30) {
          trk.meas_s.erase(trk.meas_s.begin(), trk.meas_s.end()-20);
          trk.meas_d.erase(trk.meas_d.begin(), trk.meas_d.end()-20);
        }
        trk.update_mean(track_len_);
        trk.nb_meas += 1;
        trk.isInFront = true;
        trk.isVisible = true;
        trk.current_lap = current_lap_;
        trk.size = meas->size;
        trk.classify_static(track_len_);
        trk.ttl = ObsSD::TTL_DEFAULT;

        // 동적이면 EKF
        if (trk.staticFlag.has_value() && !trk.staticFlag.value()) {
          if (ekf_.initialised) {
            ekf_.useTargetVel = false;

            // vs_reset 조건: 평균 속도가 매우 작고, 히스토리 충분, static 퍼블리시이면 동적 중단
            if (ekf_.vs_hist.size() > 10 && publish_static_ && ekf_.mean_vs() < vs_reset_) {
              ekf_.initialised = false;
              trk.static_count = 0;
              trk.total_count  = 0;
              trk.nb_meas      = 0;
              trk.staticFlag   = true;
            } else {
              if (trk.meas_s.size() >= 3) {
                double vs = ((2.0/3.0)*(trk.meas_s[trk.meas_s.size()-1] - trk.meas_s[trk.meas_s.size()-2]) * rate_
                           + (1.0/3.0)*(trk.meas_s[trk.meas_s.size()-2] - trk.meas_s[trk.meas_s.size()-3]) * rate_);
                if (!(vs>-1.0 && vs<8.0)) {
                  ekf_.initialised=false;
                } else {
                  double vd = 0.0;
                  if (trk.meas_d.size()>=2) vd = (trk.meas_d.back() - trk.meas_d[trk.meas_d.size()-2]) * rate_;
                  Eigen::Vector4d z;
                  z << std::fmod(trk.meas_s.back(), track_len_), vs, trk.meas_d.back(), vd;
                  ekf_.update(z);
                  ekf_.id = trk.id; ekf_.ttl = opp_ttl_; ekf_.size = trk.size;
                }
              }
            }
          } else {
            // EKF 초기화
            if (trk.meas_s.size()>=2 && trk.meas_d.size()>=2) {
              ekf_.x <<
                trk.meas_s.back(),
                (trk.meas_s.back()-trk.meas_s[trk.meas_s.size()-2])*rate_,
                trk.meas_d.back(),
                (trk.meas_d.back()-trk.meas_d[trk.meas_d.size()-2])*rate_;
              ekf_.initialised = true;
              ekf_.id = trk.id; ekf_.ttl = opp_ttl_; ekf_.size = trk.size;
              ekf_.vs_hist.clear();
              ekf_.vs_filt.clear(); ekf_.vd_filt.clear();
            }
          }
        }

        // 사용한 측정 제거
        auto it = std::find_if(meas_copy.begin(), meas_copy.end(),
          [&](const f110_msgs::msg::Obstacle& o){return (&o)==meas;});
        if (it!=meas_copy.end()) meas_copy.erase(it);
      }
      else {
        // 미연관: TTL/가시성
        if (trk.ttl <= 0) {
          if (trk.staticFlag.has_value() && !trk.staticFlag.value())
            ekf_.useTargetVel = true;
          to_remove.push_back(&trk);
        } else if (!trk.staticFlag.has_value()) {
          trk.ttl -= 1;
        } else {
          trk.isInFront = check_in_front_(trk, car_s_.value());
          double dist_s = dist_obs_car_s_(trk, car_s_.value());
          if (trk.staticFlag.value() && no_memory_) {
            trk.ttl -= 1;
          } else if (trk.staticFlag.value() && dist_s < dist_deletion_) {
            auto xy = frenet_.get_cartesian(trk.mean[0], trk.mean[1]);
            double vx = xy[0]-car_xy_[0], vy = xy[1]-car_xy_[1];
            trk.isVisible = check_fov_(vx, vy);
            if (trk.isVisible) trk.ttl -= 1;
          } else if (!trk.staticFlag.value()) {
            trk.ttl -= 1;
          } else {
            trk.isVisible = false;
          }
        }
      }
    }

    // EKF TTL
    if (ekf_.initialised) {
      if (ekf_.ttl <= 0) { ekf_.initialised=false; ekf_.useTargetVel=false; }
      else ekf_.ttl -= 1;
    }

    // 제거
    if (!to_remove.empty()) {
      tracked_.erase(std::remove_if(tracked_.begin(), tracked_.end(),
                  [&](const ObsSD& x){ return std::find(to_remove.begin(), to_remove.end(), &const_cast<ObsSD&>(x))!=to_remove.end(); }),
                  tracked_.end());
    }

    // 신규 트랙 생성
    for (auto &m : meas_copy) {
      tracked_.emplace_back(next_id_++, m.s_center, m.d_center, current_lap_, m.size, true);
    }
  }

  void age_and_prune_(){
    for (auto &t : tracked_) {
      if (t.ttl>0) t.ttl -= 1;
    }
    if (ekf_.initialised) {
      if (ekf_.ttl>0) ekf_.ttl -= 1; else { ekf_.initialised=false; ekf_.useTargetVel=false; }
    }
    tracked_.erase(std::remove_if(tracked_.begin(), tracked_.end(),
                  [&](const ObsSD& t){ return t.ttl<=0; }),
                  tracked_.end());
  }

  const f110_msgs::msg::Obstacle* verify_position_(ObsSD& trk, const std::vector<f110_msgs::msg::Obstacle>& meas_list){
    if (meas_list.empty()) return nullptr;
    double maxd = max_dist_;
    std::array<double,2> pos; // (s,d)
    if (trk.staticFlag.has_value() && !trk.staticFlag.value() && ekf_.initialised) {
      pos = { std::fmod(std::max(0.0, ekf_.x(0)), track_len_), ekf_.x(2) };
      maxd *= aggro_multi_;
    } else {
      pos = trk.mean;
    }
    const f110_msgs::msg::Obstacle* best=nullptr;
    double bestdist=1e9;
    for (auto &m : meas_list) {
      double d = std::hypot(pos[0]-m.s_center, pos[1]-m.d_center);
      if (d < maxd && d < bestdist) { best=&m; bestdist=d; }
    }
    if (!best && trk.staticFlag.has_value() && !trk.staticFlag.value()) {
      for (auto &m : meas_list) {
        double d = std::hypot(trk.mean[0]-m.s_center, trk.mean[1]-m.d_center);
        if (d < maxd && d < bestdist) { best=&m; bestdist=d; }
      }
    }
    return best;
  }

  bool check_in_front_(const ObsSD& trk, double car_s) const {
    double ds = wrap_track(trk.meas_s.back() - car_s, track_len_);
    return 0.0 < ds && ds < dist_infront_;
  }
  double dist_obs_car_s_(const ObsSD& trk, double car_s) const {
    double d = std::fmod(trk.meas_s.back() - car_s, track_len_);
    if (d<0) d += track_len_;
    return d;
  }
  bool check_fov_(double vx, double vy) const {
    // 차량 헤딩 기준 회전 (car frame)
    double cx = car_dir_[0], cy = car_dir_[1];
    double x =  cx*vx +  cy*vy;
    double y = -cy*vx +  cx*vy;
    double ang = std::atan2(y,x);
    if (ang > scan_max_ || ang < scan_min_) return false;
    if (scans_.empty() || scan_inc_<=0) return false;
    int idx = (int)std::llround( (ang - scan_min_) / scan_inc_ );
    if (idx<0 || idx>=(int)scans_.size()) return false;
    double dist = std::hypot(vx,vy);
    int lo = std::max(0, idx-4), hi = std::min((int)scans_.size(), idx+4);
    double mind = 1e9;
    for (int i=lo;i<hi;++i) mind = std::min(mind, (double)scans_[i]);
    return dist < mind;
  }

  void publish_obstacles_(){
    f110_msgs::msg::ObstacleArray est, raw;
    est.header.frame_id = "map";
    est.header.stamp = current_stamp_;
    raw.header = est.header;

    std::vector<f110_msgs::msg::Obstacle> static_or_conf;
    std::vector<f110_msgs::msg::Obstacle> raw_opps;

    for (auto &t : tracked_) {
      f110_msgs::msg::Obstacle m;
      m.id = t.id;
      m.size = t.size;
      m.vs = 0.0; m.vd = 0.0;
      m.is_static = true;
      m.is_actually_a_gap = false;
      m.is_visible = t.isVisible;

      if (!t.staticFlag.has_value()) {
        m.s_center = std::fmod(t.meas_s.back(), track_len_);
        m.d_center = t.meas_d.back();
      } else if (t.staticFlag.value()) {
        m.s_center = t.mean[0];
        m.d_center = t.mean[1];
      } else {
        m.s_center = std::fmod(t.meas_s.back(), track_len_);
        m.d_center = t.meas_d.back();
      }
      m.s_start = std::fmod(m.s_center - m.size*0.5, track_len_);
      if (m.s_start<0) m.s_start += track_len_;
      m.s_end   = std::fmod(m.s_center + m.size*0.5, track_len_);
      m.d_right = m.d_center - m.size*0.5;
      m.d_left  = m.d_center + m.size*0.5;

      if (!t.staticFlag.has_value() && publish_static_) static_or_conf.push_back(m);
      else if (t.staticFlag.value() && publish_static_) static_or_conf.push_back(m);
      else raw_opps.push_back(m);
    }

    if (ekf_.initialised) {
      if (ekf_.P(0,0) < (double)var_pub_) {
        f110_msgs::msg::Obstacle m;
        m.id = ekf_.id;
        m.size = ekf_.size;
        m.vs = ekf_.mean_vs();
        m.vd = ekf_.mean_vd();
        m.is_static = false;
        m.is_actually_a_gap = false;
        m.is_visible = true;
        m.s_center = std::fmod(std::max(0.0, ekf_.x(0)), track_len_);
        m.d_center = ekf_.x(2);
        m.s_start = std::fmod(m.s_center - m.size*0.5, track_len_); if (m.s_start<0) m.s_start += track_len_;
        m.s_end   = std::fmod(m.s_center + m.size*0.5, track_len_);
        m.d_right = m.d_center - m.size*0.5;
        m.d_left  = m.d_center + m.size*0.5;
        static_or_conf.push_back(m);
      }
    }

    est.obstacles = static_or_conf;
    obstacles_pub_->publish(est);
    raw.obstacles = raw_opps;
    raw_opponent_pub_->publish(raw);
  }


  // 동적 파라미터 콜백
  rcl_interfaces::msg::SetParametersResult
  on_set_params(const std::vector<rclcpp::Parameter>& ps){
    for (auto &p: ps){
      const auto& n = p.get_name();
      if (n=="ttl_dynamic")             opp_ttl_ = p.as_int();
      else if (n=="ratio_to_glob_path") ratio_to_glob_ = p.as_double();
      else if (n=="ttl_static")         ObsSD::TTL_DEFAULT = p.as_int();
      else if (n=="min_nb_meas")        ObsSD::MIN_NB_MEAS = p.as_int();
      else if (n=="min_std")            ObsSD::MIN_STD = p.as_double();
      else if (n=="max_std")            ObsSD::MAX_STD = p.as_double();
      else if (n=="dist_deletion")      dist_deletion_ = p.as_double();
      else if (n=="dist_infront")       dist_infront_ = p.as_double();
      else if (n=="vs_reset")           vs_reset_ = p.as_double();
      else if (n=="aggro_multi")        aggro_multi_ = p.as_double();
      else if (n=="debug_mode")         debug_mode_ = p.as_bool();
      else if (n=="publish_static")     publish_static_ = p.as_bool();
      else if (n=="noMemoryMode")       no_memory_ = p.as_bool();
    }
    if (track_len_>0.0) {
      ekf_.configure(1.0/double(rate_), track_len_,
                     meas_var_s_, meas_var_vs_, meas_var_d_, meas_var_vd_,
                     (double)proc_var_vs_, (double)proc_var_vd_,
                     opp_ttl_, P_vs_, P_d_, P_vd_, ratio_to_glob_, &glb_);
    }
    rcl_interfaces::msg::SetParametersResult res; res.successful=true; return res;
  }

private:
  // 파라미터들
  int rate_{40};
  double P_vs_{0.2}, P_d_{0.02}, P_vd_{0.2};
  double meas_var_s_{0.002}, meas_var_d_{0.002}, meas_var_vs_{0.2}, meas_var_vd_{0.2};
  int    proc_var_vs_{2}, proc_var_vd_{8};
  double max_dist_{0.5};
  int var_pub_{1};

  int    opp_ttl_{40};
  double ratio_to_glob_{0.6};
  double dist_deletion_{7.0}, dist_infront_{8.0}, vs_reset_{0.1}, aggro_multi_{2.0};
  bool   debug_mode_{false}, publish_static_{true}, no_memory_{false};

  // 상태
  std::vector<f110_msgs::msg::Obstacle> meas_obstacles_;
  rclcpp::Time current_stamp_;
  std::vector<ObsSD> tracked_;
  int next_id_{1};

  // 경로/프레네
  std::vector<f110_msgs::msg::Wpnt> glb_;
  SimpleFrenet frenet_;
  double track_len_{-1.0};

  // 차량 상태/스캔
  std::optional<double> car_s_;
  std::optional<double> last_car_s_;
  int current_lap_{0};
  double car_xy_[2]{0,0};
  double car_dir_[2]{1,0};
  std::vector<float> scans_;
  double scan_min_{0.0}, scan_max_{0.0}, scan_inc_{0.0};

  // EKF
  OppEKF ekf_;

  // ROS I/O
  rclcpp::Subscription<f110_msgs::msg::ObstacleArray>::SharedPtr obs_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr wpts_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr frenet_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<f110_msgs::msg::ObstacleArray>::SharedPtr obstacles_pub_;
  rclcpp::Publisher<f110_msgs::msg::ObstacleArray>::SharedPtr raw_opponent_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrackingNode>());
  rclcpp::shutdown();
  return 0;
}
