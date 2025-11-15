#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <f110_msgs/msg/obstacle_array.hpp>
#include <f110_msgs/msg/wpnt_array.hpp>

#include <tf2/LinearMath/Quaternion.h>

#include <vector>
#include <cmath>
#include <limits>
#include <optional>

using std::placeholders::_1;

class CollisionDetector : public rclcpp::Node {
public:
  CollisionDetector() : rclcpp::Node("collisiond_detector")
  {
    // Publishers
    pub_collision_ = create_publisher<std_msgs::msg::Bool>("/opponent_collision", 10);
    pub_distance_  = create_publisher<std_msgs::msg::Float32>("/opponent_dist", 10);
    pub_markers_   = create_publisher<visualization_msgs::msg::MarkerArray>("/collision_marker", 10);

    // Subscribers
    sub_obstacles_ = create_subscription<f110_msgs::msg::ObstacleArray>(
        "/perception/obstacles", 10, std::bind(&CollisionDetector::obstaclesCB, this, _1));

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/odom_frenet", 10, std::bind(&CollisionDetector::odomCB, this, _1));

    sub_global_waypoints_ = create_subscription<f110_msgs::msg::WpntArray>(
        "/global_waypoints", 10, std::bind(&CollisionDetector::glbWpntsCB, this, _1));

    // 초기 상태
    first_visualization_ = true;
    x_viz_ = 0.0;
    y_viz_ = 0.0;
    viz_counter_ = 0;
    viz_q_.x = viz_q_.y = viz_q_.z = 0.0;
    viz_q_.w = 1.0;

    RCLCPP_INFO(get_logger(), "CollisionDetector constructed. Waiting for /global_waypoints...");

    // 파이썬 코드에서는 waypoints 수신까지 spin_once로 대기했지만,
    // C++에선 수신되면 timer를 시작하는 방식으로 동일 효과를 냅니다.
  }

private:
  // ==== Callbacks ====
  void obstaclesCB(const f110_msgs::msg::ObstacleArray::SharedPtr msg)
  {
    obs_arr_ = *msg;  // 전체 복사 (필요 시 이동/스마트 처리 가능)
  }

  void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    car_odom_ = msg->pose.pose;
  }

  void glbWpntsCB(const f110_msgs::msg::WpntArray::SharedPtr msg)
  {
    glb_wpnts_ = msg->wpnts;
    if (glb_wpnts_.size() < 2) {
      RCLCPP_WARN(get_logger(), "Received too few global waypoints: %zu", glb_wpnts_.size());
      return;
    }

    if (!timer_) {
      // 50 Hz (0.02 s)
      timer_ = create_wall_timer(std::chrono::milliseconds(20),
                                 std::bind(&CollisionDetector::loop, this));
      RCLCPP_INFO(get_logger(), "Global waypoints received. Starting collision loop at 50 Hz.");
    }
  }

  // ==== Main loop ====
  void loop()
  {
    if (glb_wpnts_.size() < 2) {
      // 아직 웨이포인트 못 받았으면 대기
      static int warn_count = 0;
      if (warn_count++ % 100 == 0) {
        RCLCPP_INFO(get_logger(), "Waiting for global waypoints...");
      }
      return;
    }

    // 충돌 판정
    bool collision_bool;
    double min_dist_s, min_dist_d;
    std::tie(collision_bool, min_dist_s, min_dist_d) = collisionCheck(obs_arr_, car_odom_);

    // 기존 텍스트 Clear 타이머
    if (viz_counter_ > 0) {
      viz_counter_--;
      if (viz_counter_ == 0) {
        vizCollision(/*dist_s=*/0.0, /*dist_d=*/0.0, /*clear=*/true);
      }
    }

    // /opponent_collision
    std_msgs::msg::Bool col_msg;
    col_msg.data = collision_bool;
    pub_collision_->publish(col_msg);

    // 시각화 및 유지시간 (2초 = 100 tick)
    if (collision_bool) {
      viz_counter_ = 100;
      vizCollision(min_dist_s, min_dist_d, /*clear=*/false);
    }

    // /opponent_dist  (euclidean in s-d space)
    std_msgs::msg::Float32 dist_msg;
    dist_msg.data = static_cast<float>(std::sqrt(min_dist_s * min_dist_s + min_dist_d * min_dist_d));
    pub_distance_->publish(dist_msg);
  }

  // ==== Collision logic ====
  std::tuple<bool, double, double>
  collisionCheck(const f110_msgs::msg::ObstacleArray &obs_arr, const geometry_msgs::msg::Pose &car_odom)
  {
    if (glb_wpnts_.size() < 2) {
      return std::make_tuple(false, 100.0, 100.0);
    }
    // 파이썬 코드에서 glb_waypoints[-2].s_m를 트랙 길이로 사용
    const double track_len = glb_wpnts_[glb_wpnts_.size() - 2].s_m;

    const double car_s = car_odom.position.x;
    const double car_d = car_odom.position.y;

    for (const auto & obs : obs_arr.obstacles) {
      const double od_s = obs.s_center;
      const double od_d = obs.d_center;

      // (od_s - car_s) % track_len
      double forward_gap = fmodSafe(od_s - car_s, track_len);
      if (forward_gap < 0.0) forward_gap += track_len;

      // (car_s - od_s) % track_len
      double backward_gap = fmodSafe(car_s - od_s, track_len);
      if (backward_gap < 0.0) backward_gap += track_len;

      if (forward_gap < 0.55 && std::fabs(car_d - od_d) < 0.35) {
        return std::make_tuple(true, forward_gap, std::fabs(car_d - od_d));
      }
      if (backward_gap < 0.25 && std::fabs(car_d - od_d) < 0.30) {
        return std::make_tuple(true, backward_gap, std::fabs(car_d - od_d));
      }
    }
    return std::make_tuple(false, 100.0, 100.0);
  }

  // ==== Visualization ====
  void vizCollision(double dist_s, double dist_d, bool clear)
  {
    // 최초 호출 시 기준 위치 계산 (파이썬 로직과 동일)
    if (first_visualization_ && !clear && glb_wpnts_.size() >= 3) {
      first_visualization_ = false;
      const size_t idx = glb_wpnts_.size() / 4;

      const double x0 = glb_wpnts_[idx].x_m;
      const double y0 = glb_wpnts_[idx].y_m;
      const double x1 = glb_wpnts_[idx + 1].x_m;
      const double y1 = glb_wpnts_[idx + 1].y_m;

      // 노멀 벡터: -(y1 - y0, -(x1 - x0)) = (-(y1-y0), x0-x1)
      const double nx = -(y1 - y0);
      const double ny = (x0 - x1);
      const double norm = std::hypot(nx, ny);
      if (norm > 1e-9) {
        const double d_left = glb_wpnts_[idx].d_left;  // 메시지에 존재한다고 가정
        const double scale = 1.75 * d_left / norm;
        const double vx = nx * scale;
        const double vy = ny * scale;

        const double yaw = std::atan2(vy, vx);
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);

        viz_q_.x = q.x(); viz_q_.y = q.y(); viz_q_.z = q.z(); viz_q_.w = q.w();
        x_viz_ = x0 + vx;
        y_viz_ = y0 + vy;
      }
    }

    visualization_msgs::msg::MarkerArray arr;

    visualization_msgs::msg::Marker text;
    text.header.frame_id = "map";
    text.header.stamp = now();
    text.ns = "collision_detector";
    text.id = 0;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;

    text.scale.z = 1.0;

    text.color.r = 1.0f;
    text.color.g = 0.0f;
    text.color.b = 0.0f;
    text.color.a = 1.0f;

    text.pose.orientation = viz_q_;
    text.pose.position.x = x_viz_;
    text.pose.position.y = y_viz_;
    text.pose.position.z = 0.0;

    if (clear) {
      text.text = "";
    } else {
      char buf[128];
      std::snprintf(buf, sizeof(buf), "COLLISION: dist_s :%.1f, dist_d :%.1f m", dist_s, dist_d);
      text.text = std::string(buf);
    }

    arr.markers.push_back(text);
    pub_markers_->publish(arr);
  }

  static double fmodSafe(double a, double b)
  {
    if (b == 0.0) return a;
    return std::fmod(a, b);
  }

private:
  // Pubs/Subs
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_collision_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_distance_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

  rclcpp::Subscription<f110_msgs::msg::ObstacleArray>::SharedPtr sub_obstacles_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr sub_global_waypoints_;

  rclcpp::TimerBase::SharedPtr timer_{nullptr};

  // State
  f110_msgs::msg::ObstacleArray obs_arr_;
  geometry_msgs::msg::Pose car_odom_;
  std::vector<f110_msgs::msg::Wpnt> glb_wpnts_;

  // Viz state
  bool first_visualization_{true};
  double x_viz_{0.0}, y_viz_{0.0};
  geometry_msgs::msg::Quaternion viz_q_;
  int viz_counter_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CollisionDetector>());
  rclcpp::shutdown();
  return 0;
}
