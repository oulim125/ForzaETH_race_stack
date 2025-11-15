#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>
#include <f110_msgs/msg/obstacle_array.hpp>
#include <f110_msgs/msg/obstacle.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <optional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <tuple>
#include <tf2/time.h>
#include <limits>
#include <memory>


#include <frenet_conversion_cpp/frenet_converter_cpp.hpp>

using std::placeholders::_1;

struct Obstacle {
  float cx{0}, cy{0}, size{0}, theta{0};
  int id{-1};
};

// ---------- helpers ----------
static inline float normalize_s(float x, float track_len) {
  float m = std::fmod(x, track_len);
  if (m < 0) m += track_len;
  if (m > track_len * 0.5f) m -= track_len;
  return m;
}

template<typename T>
static inline T clamp(T v, T lo, T hi) { return std::max(lo, std::min(hi, v)); }

// ---------- 메인 노드 ----------
class DetectNode : public rclcpp::Node {
public:
  DetectNode()
  // : rclcpp::Node("detection",
  //         rclcpp::NodeOptions()
  //           .allow_undeclared_parameters(true)
  //           .automatically_declare_parameters_from_overrides(true))
  : rclcpp::Node("detection")
  {
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    // QoS (latched equiv)
    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local();

    // 파라미터
    measuring_ = this->declare_parameter<bool>("measure", false);
    from_bag_  = this->declare_parameter<bool>("from_bag", false);

    rate_ = this->declare_parameter<int>("rate", 40);
    lambda_deg_ = this->declare_parameter<int>("lambda", 10);
    sigma_ = this->declare_parameter<double>("sigma", 0.03);
    min_2_points_dist_ = this->declare_parameter<double>("min_2_points_dist", 0.01);

    rate_ = this->get_parameter("rate").as_int();
    lambda_deg_ = this->get_parameter("lambda").as_int();
    sigma_ = this->get_parameter("sigma").as_double();
    min_2_points_dist_ = this->get_parameter("min_2_points_dist").as_double();

    min_obs_size_ = this->declare_parameter<int>("min_obs_size", 10);
    max_obs_size_ = this->declare_parameter<double>("max_obs_size", 0.5);
    max_viewing_distance_ = this->declare_parameter<double>("max_viewing_distance", 9.0);
    boundaries_inflation_ = this->declare_parameter<double>("boundaries_inflation", 0.1);

    // 동적 파라미터 콜백
    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&DetectNode::on_set_params, this, std::placeholders::_1));
    
    // 구독
    using rclcpp::SensorDataQoS;
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10, std::bind(&DetectNode::scan_cb, this, _1));
    glb_wpnts_sub_ = this->create_subscription<f110_msgs::msg::WpntArray>(
      "/global_waypoints", 10, std::bind(&DetectNode::path_cb, this, _1));
    frenet_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/car_state/frenet/odom", 10, std::bind(&DetectNode::frenet_cb, this, _1));

    // 발행
    breakpoints_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/breakpoints_markers", 5);
    boundaries_pub_  = this->create_publisher<visualization_msgs::msg::Marker>("/perception/detect_bound", latched_qos);
    raw_obstacles_pub_ = this->create_publisher<f110_msgs::msg::ObstacleArray>("/perception/detection/raw_obstacles", 5);
    obstacles_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/obstacles_markers_new", 5);

    // 타이머
    const double period = (rate_>0) ? 1.0/static_cast<double>(rate_) : 0.05;
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(period),
      std::bind(&DetectNode::loop, this));

    RCLCPP_INFO(this->get_logger(), "DetectNode up. rate=%d, lambda=%d deg", rate_, lambda_deg_);
  }

private:
  // ---- 콜백들 ----
  void scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    scan_ = msg;
    const int n = (int)msg->ranges.size();
    const auto meta = std::make_tuple(n, msg->angle_min, msg->angle_increment);
    if (!scan_meta_ || *scan_meta_ != meta) {
      scan_meta_ = meta;
      angles_.resize(n);
      cos_.resize(n);
      sin_.resize(n);
      for (int i=0;i<n;++i) {
        const float ang = msg->angle_min + msg->angle_increment * i;
        angles_[i] = ang; cos_[i]=std::cos(ang); sin_[i]=std::sin(ang);
      }
    }
  }

  void path_cb(const f110_msgs::msg::WpntArray::SharedPtr msg) {
    if (!msg || msg->wpnts.empty()) return;

    // 처음 들어오면 프레네 변환 준비
    if (!converter_ready_) {
      std::vector<double> wx, wy, psi;
      wx.reserve(msg->wpnts.size());
      wy.reserve(msg->wpnts.size());
      psi.reserve(msg->wpnts.size());
      for (const auto &w : msg->wpnts) {
        wx.push_back(static_cast<double>(w.x_m));
        wy.push_back(static_cast<double>(w.y_m));
        psi.push_back(static_cast<double>(w.psi_rad));
      }

      // ⬇⬇⬇ FrenetConverter 생성
      converter_ = std::make_unique<FrenetConverter>(wx, wy, psi);

      // 트랙 길이 (메시지의 s_m을 쓰거나, 라이브러리 계산값을 써도 됨)
      track_length_ = static_cast<float>(msg->wpnts.back().s_m);
      // track_length_ = static_cast<float>(converter_->raceline_length());

      // s, d 경계 전처리
      s_array_.clear(); d_right_.clear(); d_left_.clear();
      s_array_.reserve(msg->wpnts.size());
      d_right_.reserve(msg->wpnts.size());
      d_left_.reserve(msg->wpnts.size());
      for (const auto &w : msg->wpnts) {
        s_array_.push_back((float)w.s_m);
        d_right_.push_back((float)(w.d_right - boundaries_inflation_));
        d_left_.push_back ((float)(w.d_left  - boundaries_inflation_));
      }
      float mn = +1e9f, mx = -1e9f;
      for (size_t i=0;i<msg->wpnts.size();++i) {
        mn = std::min(mn, std::min(d_right_[i], d_left_[i]));
        mx = std::max(mx, std::max(d_right_[i], d_left_[i]));
      }
      smallest_d_ = mn;
      biggest_d_  = mx;
      converter_ready_ = true;

      RCLCPP_INFO(this->get_logger(), "[Opponent Detection]: received global path (%zu pts)", msg->wpnts.size());
    }

    // ✅ 여기부터 추가: 경계 마커 생성/퍼블리시
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = this->now();
    m.ns = "detect_bound";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1;
    m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0;

    m.points.clear();
    m.points.reserve(msg->wpnts.size() * 2);
    for (const auto &w : msg->wpnts) {
      // 오른쪽 경계 (-d_right)
      auto pR = converter_->get_cartesian((double)w.s_m, (double)(-w.d_right + boundaries_inflation_));
      // auto pR = converter_->get_cartesian((double)w.s_m, (double)( w.d_right + boundaries_inflation_));
      geometry_msgs::msg::Point p1; p1.x = pR.first; p1.y = pR.second; p1.z = 0.0;
      m.points.push_back(p1);

      // 왼쪽 경계 (+d_left)
      auto pL = converter_->get_cartesian((double)w.s_m, (double)( w.d_left  - boundaries_inflation_));
      // auto pL = converter_->get_cartesian((double)w.s_m, (double)(-w.d_left  - boundaries_inflation_));
      geometry_msgs::msg::Point p2; p2.x = pL.first; p2.y = pL.second; p2.z = 0.0;
      m.points.push_back(p2);
    }
    boundaries_pub_->publish(m);
  }

  void frenet_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if (!msg) return;
    car_s_ = (float)msg->pose.pose.position.x;
  }

  // ---- 메인 루프 ----
  void loop() {
    if (!converter_ready_ || !scan_) return;

    // 중복 스캔 스킵 (메시지 stamp 그대로 비교)
    if (last_scan_stamp_ &&
        last_scan_stamp_->sec    == scan_->header.stamp.sec &&
        last_scan_stamp_->nanosec== scan_->header.stamp.nanosec) {
      return;
    }
    last_scan_stamp_ = scan_->header.stamp;

    // ============================== TF 조회 (스캔 시각 우선, 실패 시 최신으로 재시도) ==============================
    std::optional<geometry_msgs::msg::TransformStamped> tr;  // 비어 있으면 downstream에서 알아서 처리

    // 1) 스캔 시각으로 시도
    try {
      auto tr_scan = tf_buffer_->lookupTransform(
        "map",
        scan_->header.frame_id,
        rclcpp::Time(scan_->header.stamp),                 // 스캔 타임스탬프
        rclcpp::Duration::from_seconds(0.3)                // 타임아웃
      );
      tr = tr_scan;
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "[TF] map <- %s @ scan stamp failed: %s. Retrying with latest...",
        scan_->header.frame_id.c_str(), e.what());

      // 2) 최신 시각으로 재시도
      try {
        auto tr_latest = tf_buffer_->lookupTransform(
          "map",
          scan_->header.frame_id,
          tf2::TimePointZero,                               // 최신
          tf2::durationFromSec(0.3)                         // 타임아웃
        );
        tr = tr_latest;
      } catch (const tf2::TransformException &e2) {
        RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "[TF] Latest transform also failed map <- %s: %s",
          scan_->header.frame_id.c_str(), e2.what());
      }
    }

    const auto clusters = segment_laser_points_(scan_, tr);
    auto obstacles = fit_obstacles_pca_(clusters, tr);
    filter_and_publish_(obstacles);
  }

  // ---- 구현 세부 ----
  std::vector<Eigen::MatrixXf> segment_laser_points_(
      const sensor_msgs::msg::LaserScan::SharedPtr& scan,
      const std::optional<geometry_msgs::msg::TransformStamped>& /*tr*/) {
    std::vector<Eigen::MatrixXf> clusters;
    if (!scan || cos_.empty()) return clusters;

    const int N = (int)scan->ranges.size();
    std::vector<float> ranges(N);
    for (int i=0;i<N;++i) ranges[i] = std::isfinite(scan->ranges[i]) ? (float)scan->ranges[i] : std::numeric_limits<float>::quiet_NaN();

    // 레이저 좌표(x,y)
    std::vector<float> x(N), y(N);
    std::vector<uint8_t> valid(N, 0);
    for (int i=0;i<N;++i) {
      float r = ranges[i];
      if (std::isfinite(r) && r>0.f) { x[i]=r*cos_[i]; y[i]=r*sin_[i]; valid[i]=1; }
      else { x[i]=0.f; y[i]=0.f; valid[i]=0; }
    }

    // 경계 판단
    const float lambda = (float)lambda_deg_ * (float)M_PI/180.f;
    const float dphi   = scan->angle_increment;
    const float denom  = std::sin(lambda - dphi);
    const float div_const = (std::abs(denom)>1e-6) ? (std::sin(dphi)/denom) : 1e6f;

    std::vector<uint8_t> boundary(N,0);
    boundary[0]=1; boundary[N-1]=1;

    for (int i=1;i<N;++i) {
      if (!valid[i] || !valid[i-1]) { boundary[i]=1; continue; }
      const float dmax_i = ranges[i]*div_const + 3.f*(float)sigma_;
      const float dx = x[i]-x[i-1];
      const float dy = y[i]-y[i-1];
      const float dd = std::hypot(dx,dy);
      if (dd >= dmax_i) boundary[i]=1;
    }

    // 경계 인덱스 → 슬라이스
    int start = 0;
    for (int i=1;i<N;++i) {
      if (boundary[i]) {
        if (i - start > 0) {
          // pts [start, i)
          const int npts = i - start;
          Eigen::MatrixXf pts(npts,2);
          int w=0;
          for (int k=start;k<i;++k) {
            if (valid[k]) {
              pts(w,0)=x[k]; pts(w,1)=y[k]; ++w;
            }
          }
          if (w>=min_obs_size_) {
            pts.conservativeResize(w,2);
            clusters.push_back(std::move(pts));
          }
        }
        start = i;
      }
    }
    return clusters;
  }

  // PCA로 중심/방향/두께 추정 + TF(map)
  std::vector<Obstacle> fit_obstacles_pca_(
      const std::vector<Eigen::MatrixXf>& clusters,
      const std::optional<geometry_msgs::msg::TransformStamped>& tr) {
    std::vector<Obstacle> out;
    if (clusters.empty()) return out;

    // TF 2D 구성
    Eigen::Matrix2f R2 = Eigen::Matrix2f::Identity();
    Eigen::Vector2f t2 = Eigen::Vector2f::Zero();
    float yaw = 0.f;

    if (tr) {
      const auto &q = tr->transform.rotation;
      tf2::Quaternion tq(q.x,q.y,q.z,q.w);
      double r,p,y;
      tf2::Matrix3x3(tq).getRPY(r,p,y);
      yaw = (float)y;
      R2 << std::cos(yaw), -std::sin(yaw),
            std::sin(yaw),  std::cos(yaw);
      t2 << (float)tr->transform.translation.x,
            (float)tr->transform.translation.y;
    }

    for (const auto& pts : clusters) {
      if (pts.rows() < std::max(3, min_obs_size_)) continue;

      // 평균
      Eigen::RowVector2f mu = pts.colwise().mean();
      Eigen::MatrixXf X = pts.rowwise() - mu;
      const int n = X.rows();

      // 공분산 (2x2), 고유분해
      Eigen::Matrix2f C = (X.adjoint() * X) / std::max(1, n-1);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
      if (es.info()!=Eigen::Success) continue;

      // 주/부축
      Eigen::Vector2f evals = es.eigenvalues();
      Eigen::Matrix2f evecs = es.eigenvectors(); // 열: 고유벡터(오름차순)
      int i_max = (evals(1)>evals(0)) ? 1 : 0;
      int i_min = 1 - i_max;
      Eigen::Vector2f v_major = evecs.col(i_max);
      Eigen::Vector2f v_minor = evecs.col(i_min);

      // 투영 후 길이
      Eigen::VectorXf p_major = X * v_major;
      Eigen::VectorXf p_minor = X * v_minor;
      float len_major = p_major.maxCoeff() - p_major.minCoeff();
      float len_minor = p_minor.maxCoeff() - p_minor.minCoeff();

      // 레이저 프레임 중심
      Eigen::Vector2f c_l = mu.transpose();
      // 맵 프레임
      Eigen::Vector2f c_m = R2 * c_l + t2;
      float theta_l = std::atan2(v_major.y(), v_major.x());
      float theta_m = theta_l + yaw;

      Obstacle ob;
      ob.cx = c_m.x();
      ob.cy = c_m.y();
      ob.theta = theta_m;
      ob.size = std::max((float)min_2_points_dist_, len_minor);
      out.push_back(ob);
    }
    return out;
  }

  // 트랙 내부 판단 + 메시지 퍼블리시
  void filter_and_publish_(std::vector<Obstacle>& obs) {
    if (!converter_) return;

    // 트랙 내부 검사
    std::vector<Obstacle> kept;
    kept.reserve(obs.size());
    for (auto &o : obs) {
      if (o.size > (float)max_obs_size_) continue; // 크기 필터
      if (!laser_point_on_track_(o.cx, o.cy)) continue;
      kept.push_back(o);
    }
    // id 부여
    for (size_t i=0;i<kept.size();++i) kept[i].id = (int)i;

    // 메시지 publish
    f110_msgs::msg::ObstacleArray arr;
    arr.header.stamp = scan_->header.stamp; 
    arr.header.frame_id = "map";
    if (kept.empty()) { 
      raw_obstacles_pub_->publish(arr); 
      publish_markers_(kept);
      return; }

    // (cx, cy) → (s,d) : FrenetConverter 사용
    std::vector<double> xd(kept.size()), yd(kept.size());
    for (size_t i=0;i<kept.size();++i) { xd[i]=static_cast<double>(kept[i].cx); yd[i]=static_cast<double>(kept[i].cy); }
    auto sd = converter_->get_frenet(xd, yd);
    const auto &S = sd.first;
    const auto &D = sd.second;

    for (size_t i=0;i<kept.size();++i) {
      f110_msgs::msg::Obstacle m;
      m.id = kept[i].id;
      m.s_center = static_cast<float>(S[i]);
      m.d_center = static_cast<float>(D[i]);
      m.size = kept[i].size;
      m.s_start = m.s_center - kept[i].size*0.5f;
      m.s_end   = m.s_center + kept[i].size*0.5f;
      m.d_left  = m.d_center + kept[i].size*0.5f;
      m.d_right = m.d_center - kept[i].size*0.5f;
      arr.obstacles.push_back(m);
    }
    raw_obstacles_pub_->publish(arr);

    // (옵션) 마커도 원하면 주석 해제해서 퍼블리시
    publish_markers_(kept);
  }

  bool laser_point_on_track_(float x, float y) {
    if (!converter_ready_ || s_array_.empty() || !converter_) return false;

    // 단일 포인트 (x,y) → (s,d)
    auto sd = converter_->get_frenet(static_cast<double>(x), static_cast<double>(y));
    const float ss = static_cast<float>(sd.first);
    const float dd = static_cast<float>(sd.second);

    if (normalize_s(ss - car_s_, track_length_) > (float)max_viewing_distance_) return false;
    if (std::abs(dd) >= biggest_d_)  return false;
    if (std::abs(dd) <= smallest_d_) return true;

    // s_array_에서 ss 위치의 인덱스
    auto it = std::lower_bound(s_array_.begin(), s_array_.end(), ss);
    int idx = (it==s_array_.begin()) ? 0 : (int)(it - s_array_.begin() - 1);
    idx = clamp<int>(idx, 0, (int)s_array_.size()-1);

    if (dd <= -d_right_[idx] || dd >= d_left_[idx]) return false;
    return true;

    // if (dd <= -d_left_[idx] || dd >= d_right_[idx]) return false;
    // return true;
  }

  void publish_markers_(const std::vector<Obstacle>& kept) {
    visualization_msgs::msg::MarkerArray arr;
    // clear
    {
      visualization_msgs::msg::Marker del;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
    }
    // cubes
    for (auto &o : kept) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map";
      m.header.stamp = this->now();
      m.ns = "obstacles";
      m.id = o.id;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.scale.x = o.size; m.scale.y = o.size; m.scale.z = o.size;
      m.color.a = 0.5f; m.color.g = 1.0f; m.color.r = 0.0f; m.color.b = 1.0f;
      m.pose.position.x = o.cx; m.pose.position.y = o.cy; m.pose.position.z = 0.0;
      tf2::Quaternion q; q.setRPY(0,0,o.theta);
      m.pose.orientation = tf2::toMsg(q);
      arr.markers.push_back(m);
    }
    obstacles_marker_pub_->publish(arr);
  }

  // ---- 동적 파라미터 ----
  rcl_interfaces::msg::SetParametersResult
  on_set_params(const std::vector<rclcpp::Parameter>& params) {
    bool need_update_bounds=false;
    for (auto &p : params) {
      if (p.get_name()=="min_obs_size")         min_obs_size_ = p.as_int();
      else if (p.get_name()=="max_obs_size")    max_obs_size_ = p.as_double();
      else if (p.get_name()=="max_viewing_distance") max_viewing_distance_ = p.as_double();
      else if (p.get_name()=="boundaries_inflation") { boundaries_inflation_ = p.as_double(); need_update_bounds=true; }
    }
    if (need_update_bounds && !s_array_.empty()) {
      for (size_t i=0;i<s_array_.size();++i) {
        d_right_[i] = d_right_[i] - /*old*/0.0 + (-boundaries_inflation_);
        d_left_[i]  = d_left_[i]  - /*old*/0.0 + (-boundaries_inflation_);
      }
      float mn=1e9f,mx=-1e9f;
      for (size_t i=0;i<s_array_.size();++i){
        mn=std::min(mn,std::min(d_right_[i],d_left_[i]));
        mx=std::max(mx,std::max(d_right_[i],d_left_[i]));
      }
      smallest_d_=mn; biggest_d_=mx;
    }
    rcl_interfaces::msg::SetParametersResult res; res.successful=true; return res;
  }

private:
  // 파라미터
  bool measuring_{false}, from_bag_{false};
  int rate_{40}, lambda_deg_{10}, min_obs_size_{10};
  double sigma_{0.03}, min_2_points_dist_{0.01};
  double max_obs_size_{0.5}, max_viewing_distance_{9.0}, boundaries_inflation_{0.1};

  // 구독/발행
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr glb_wpnts_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr frenet_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr breakpoints_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr boundaries_pub_;
  rclcpp::Publisher<f110_msgs::msg::ObstacleArray>::SharedPtr raw_obstacles_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 스캔/메타
  sensor_msgs::msg::LaserScan::SharedPtr scan_;
  std::optional<std::tuple<int,float,float>> scan_meta_;
  std::vector<float> angles_, cos_, sin_;
  std::optional<builtin_interfaces::msg::Time> last_scan_stamp_;

  // 프레네 / 전역 경계
  std::unique_ptr<FrenetConverter> converter_;
  bool converter_ready_{false};
  float track_length_{0.f};
  std::vector<float> s_array_, d_right_, d_left_;
  float smallest_d_{0.f}, biggest_d_{0.f};
  float car_s_{0.f};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DetectNode>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
