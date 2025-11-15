// frenet_odom_republisher.cpp
#include <rclcpp/rclcpp.hpp>

#include <f110_msgs/msg/wpnt_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <vector>
#include <optional>
#include <string>

// ✅ 이전에 작성한 FrenetConverter 헤더
#include <frenet_conversion_cpp/frenet_converter_cpp.hpp>

class FrenetOdomRepublisher : public rclcpp::Node {
public:
  FrenetOdomRepublisher()
  : rclcpp::Node("frenet_odom_republisher")
  {
    // Publishers (동일 토픽/큐 사이즈)
    frenet_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/car_state/frenet/odom", 10);
    frenet_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/car_state/frenet/pose", 10);

    state_odom_pub_  = this->create_publisher<nav_msgs::msg::Odometry>("/car_state/odom", 10);
    state_pose_pub_  = this->create_publisher<geometry_msgs::msg::PoseStamped>("/car_state/pose", 10);

    // Subscribers
    wpnts_sub_ = this->create_subscription<f110_msgs::msg::WpntArray>(
        "/global_waypoints", 10,
        std::bind(&FrenetOdomRepublisher::globalTrajectoryCallback, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pf/viz/inferred_pose", 10,
        std::bind(&FrenetOdomRepublisher::poseCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/early_fusion/odom", 10,
        std::bind(&FrenetOdomRepublisher::odomCallback, this, std::placeholders::_1));
  }

private:
  // ===== Callbacks =====

  void globalTrajectoryCallback(const f110_msgs::msg::WpntArray::SharedPtr msg) {
    const auto &wps = msg->wpnts;
    if (wps.empty()) return;

    std::vector<double> xs; xs.reserve(wps.size());
    std::vector<double> ys; ys.reserve(wps.size());
    std::vector<double> psis; psis.reserve(wps.size());

    for (const auto &w : wps) {
      xs.push_back(static_cast<double>(w.x_m));
      ys.push_back(static_cast<double>(w.y_m));
      psis.push_back(static_cast<double>(w.psi_rad));
    }

    try {
      converter_ = std::make_unique<FrenetConverter>(xs, ys, psis);
      has_global_trajectory_ = true;
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to build FrenetConverter: %s", e.what());
      has_global_trajectory_ = false;
      converter_.reset();
    }
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    latest_pose_ = *msg;  // 최신 PF pose 저장
    has_latest_pose_ = true;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    // pose or trajectory 미보유 시 skip (Python과 동일 로직)
    if (!has_latest_pose_ || !has_global_trajectory_ || !converter_) return;

    // ---------- 최신 PF pose에서 위치/자세 추출 ----------
    const auto &pose_msg = latest_pose_; // 복사 보관해 둔 최신 값
    const auto &pose = pose_msg.pose.position;
    const auto &quat_msg = pose_msg.pose.orientation;

    tf2::Quaternion q;
    tf2::fromMsg(quat_msg, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw); // theta

    // ---------- velocity 준비 (EKF 속도) ----------
    const auto &vel = odom_msg->twist.twist;

    // ---------- 최종 카테시안 상태 퍼블리시 ----------
    // /car_state/pose: PF Pose 그대로
    geometry_msgs::msg::PoseStamped carstate_pose;
    carstate_pose.header = pose_msg.header;   // 보통 frame_id="map"
    carstate_pose.pose   = pose_msg.pose;
    state_pose_pub_->publish(carstate_pose);

    // /car_state/odom: Pose=PF, Twist=EKF
    nav_msgs::msg::Odometry carstate_odom;
    carstate_odom.header = pose_msg.header;
    carstate_odom.child_frame_id = "base_link";
    carstate_odom.pose.pose   = pose_msg.pose;
    carstate_odom.twist.twist = vel;
    state_odom_pub_->publish(carstate_odom);

    // ---------- Frenet 변환 ----------
    // 단일 포인트 변환(C++ FrenetConverter의 single-point API)
    auto sd = converter_->get_frenet(static_cast<double>(pose.x),
                                     static_cast<double>(pose.y));
    const double s = sd.first;
    const double d = sd.second;

    auto sddot = converter_->get_frenet_velocities(
        static_cast<double>(vel.linear.x),
        static_cast<double>(vel.linear.y),
        yaw);
    const double s_dot = sddot.first;
    const double d_dot = sddot.second;

    // 가장 가까운 웨이포인트 인덱스(문자열)
    const int closest_idx = converter_->get_closest_index(
        static_cast<double>(pose.x), static_cast<double>(pose.y));

    // ---------- Frenet Odometry 메시지 ----------
    nav_msgs::msg::Odometry fr_odom;
    fr_odom.header = pose_msg.header;
    fr_odom.child_frame_id = std::to_string(closest_idx);

    // 프레네 위치: x=s, y=d (Python과 동일 매핑)
    fr_odom.pose.pose.position.x = s;
    fr_odom.pose.pose.position.y = d;

    // orientation은 기존 코드 유지(= PF quaternion)
    fr_odom.pose.pose.orientation = quat_msg;

    // 프레네 속도: s_dot, d_dot
    fr_odom.twist.twist.linear.x = s_dot;
    fr_odom.twist.twist.linear.y = d_dot;

    // ---------- Frenet PoseStamped 메시지 ----------
    geometry_msgs::msg::PoseStamped fr_pose;
    fr_pose.header = pose_msg.header;
    fr_pose.pose.position.x = s;
    fr_pose.pose.position.y = d;
    fr_pose.pose.orientation = quat_msg; // Python은 그대로 뒀으므로 동일

    // ---------- Publish ----------
    frenet_odom_pub_->publish(fr_odom);
    frenet_pose_pub_->publish(fr_pose);
  }

private:
  // 상태
  bool has_global_trajectory_{false};
  bool has_latest_pose_{false};
  geometry_msgs::msg::PoseStamped latest_pose_;
  std::unique_ptr<FrenetConverter> converter_;

  // pubs/subs
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr frenet_odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr frenet_pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr state_pose_pub_;

  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr wpnts_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FrenetOdomRepublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
