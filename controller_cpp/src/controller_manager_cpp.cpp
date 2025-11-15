// src/controller_manager.cpp
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <future>
#include <typeinfo>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/srv/get_parameters.hpp"
#include "rcl_interfaces/msg/parameter_event.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/floating_point_range.hpp"
#include "rclcpp/parameter_events_filter.hpp"

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// 프로젝트 메시지들(필요 시 경로 조정)
#include "f110_msgs/msg/wpnt_array.hpp"
#include "f110_msgs/msg/obstacle_array.hpp"
#include "f110_msgs/msg/pid_data.hpp"

// ament share dir
#include "ament_index_cpp/get_package_share_directory.hpp"

// yaml-cpp
#include <yaml-cpp/yaml.h>

// tf2 (Euler→Quat)
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "map.hpp"
#include "pp.hpp"
#include "ftg.hpp"

#include "rclcpp/executors/single_threaded_executor.hpp"

using namespace std::chrono_literals;

class Controller : public rclcpp::Node {
public:
  Controller()
  : rclcpp::Node(
        "controller_manager",
        rclcpp::NodeOptions()
          .allow_undeclared_parameters(true)
          .automatically_declare_parameters_from_overrides(true))
  {
    // 원격 파라미터 조회
    map_path_          = get_remote_parameter<std::string>("global_parameters", "map_path");
    racecar_version_   = get_remote_parameter<std::string>("global_parameters", "racecar_version");
    sim_               = get_remote_parameter<bool>("global_parameters", "sim");
    state_machine_rate_= get_remote_parameter<double>("state_machine", "rate_hz");


    // 로컬 파라미터
    LUT_name_ = this->get_parameter("LU_table").as_string();
    RCLCPP_INFO(get_logger(), "Using LUT: %s", LUT_name_.c_str());
    mode_     = this->get_parameter("mode").as_string();        // "MAP" | "PP" | "FTG"
    mapping_  = this->get_parameter("mapping").as_bool();

    rate_hz_ = 40.0;
    state_ = "GB_TRACK";

    // 퍼블리셔
    drive_pub_      = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    steering_pub_   = create_publisher<visualization_msgs::msg::Marker>("steering", 10);
    lookahead_pub_  = create_publisher<visualization_msgs::msg::Marker>("lookahead_point", 10);
    trailing_pub_   = create_publisher<visualization_msgs::msg::Marker>("trailing_opponent_marker", 10);
    waypoint_pub_   = create_publisher<visualization_msgs::msg::MarkerArray>("my_waypoints", 10);
    l1_dist_pub_    = create_publisher<geometry_msgs::msg::Point>("l1_distance", 10);
    gap_data_pub_   = create_publisher<f110_msgs::msg::PidData>("/trailing/gap_data", 10);

    // 모드에 따라 컨트롤러 초기화
    if (mode_ == "MAP") {
      RCLCPP_INFO(get_logger(), "Initializing MAP controller");
      init_map_controller();
      prioritize_dyn_ = l1_params_["prioritize_dyn"].as<bool>();
    } else if (mode_ == "PP") {
      RCLCPP_INFO(get_logger(), "Initializing PP controller");
      init_pp_controller();
      prioritize_dyn_ = l1_params_["prioritize_dyn"].as<bool>();
    } else if (mode_ == "FTG") {
      RCLCPP_INFO(get_logger(), "Initializing FTG controller");
      init_ftg_controller();
      prioritize_dyn_ = false;
    } else {
      RCLCPP_ERROR(get_logger(), "Invalid mode: %s", mode_.c_str());
      throw std::runtime_error("Invalid mode parameter");
    }

    // 서브스크립션
    state_sub_ = create_subscription<std_msgs::msg::String>(
        "/state", 10,
        [this](std_msgs::msg::String::SharedPtr msg){ state_ = msg->data; });

    glb_wpnts_sub_ = create_subscription<f110_msgs::msg::WpntArray>(
        "/global_waypoints", 10,
        std::bind(&Controller::track_length_cb, this, std::placeholders::_1));

    local_wpnts_sub_ = create_subscription<f110_msgs::msg::WpntArray>(
        "/local_waypoints", 10,
        std::bind(&Controller::local_waypoint_cb, this, std::placeholders::_1));

    obstacles_sub_ = create_subscription<f110_msgs::msg::ObstacleArray>(
        "/perception/obstacles", 10,
        std::bind(&Controller::obstacle_cb, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/odom", 10,
        [this](nav_msgs::msg::Odometry::SharedPtr msg){ speed_now_ = msg->twist.twist.linear.x; });

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/car_state/pose", 10,
        std::bind(&Controller::car_state_cb, this, std::placeholders::_1));

    frenet_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/car_state/frenet/odom", 10,
        std::bind(&Controller::car_state_frenet_cb, this, std::placeholders::_1));

    // 연산량 절감: LiDAR는 기본 비활성(원본 주석과 동일)
    // scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    //     "/scan", 10, std::bind(&Controller::scan_cb, this, std::placeholders::_1));

    // 필수 메시지 대기
    wait_for_messages_();

    // 동적 파라미터(자체 노드 파라미터 업데이트 반영)
    declare_update_params_and_watch_();

    // 메인 루프 타이머
    timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate_hz_),
        std::bind(&Controller::control_loop, this));

    RCLCPP_INFO(get_logger(), "Controller ready");
  }

private:
  // -------------------- 원격 파라미터 헬퍼 --------------------
  template<typename T>
  T get_remote_parameter(const std::string &remote_node, const std::string &name) {
    using Service = rcl_interfaces::srv::GetParameters;

    // 절대 이름으로 서비스 지정 (네임스페이스 문제 회피)
    const std::string service_name = "/" + remote_node + "/get_parameters";

    auto client = this->create_client<Service>(service_name);
    // 서비스 올라올 때까지 대기
    if (!client->wait_for_service(std::chrono::seconds(5))) {
      throw std::runtime_error("Timeout waiting for service: " + service_name);
    }

    auto req = std::make_shared<Service::Request>();
    req->names = {name};

    // 여기서 중요한 포인트: executor를 만들어 노드를 추가하고 spin_until_future_complete 호출
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(this->get_node_base_interface());

    auto future = client->async_send_request(req);
    auto rc = exec.spin_until_future_complete(future, std::chrono::seconds(5));
    if (rc != rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error("Failed to get parameter '" + name + "' from " + remote_node);
    }

    auto res = future.get();
    if (res->values.empty()) {
      throw std::runtime_error("Empty response for parameter '" + name + "'");
    }

    const auto &p = res->values[0];
    using PT = rcl_interfaces::msg::ParameterType;

    if constexpr (std::is_same_v<T, std::string>) {
      if (p.type == PT::PARAMETER_STRING) return p.string_value;
    } else if constexpr (std::is_same_v<T, bool>) {
      if (p.type == PT::PARAMETER_BOOL)   return p.bool_value;
      if (p.type == PT::PARAMETER_STRING) return (p.string_value == "true");  // 관용적 폴백
    } else if constexpr (std::is_same_v<T, double>) {
      if (p.type == PT::PARAMETER_DOUBLE)  return p.double_value;
      if (p.type == PT::PARAMETER_INTEGER) return static_cast<double>(p.integer_value);
    } else if constexpr (std::is_integral_v<T>) {
      if (p.type == PT::PARAMETER_INTEGER) return static_cast<T>(p.integer_value);
      if (p.type == PT::PARAMETER_DOUBLE)  return static_cast<T>(p.double_value);
    }

    std::stringstream ss;
    ss << "Type mismatch getting '" << name << "' from " << remote_node
      << " (got=" << static_cast<int>(p.type) << ")";
    throw std::runtime_error(ss.str());
  }


  // -------------------- 초기화 루틴 --------------------
  void init_map_controller() {
    load_l1_params_from_yaml_();
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/sensors/imu/raw", 10, std::bind(&Controller::imu_cb, this, std::placeholders::_1));
    acc_now_.assign(10, 0.0);

    map_controller_ = std::make_unique<MAP_Controller>(
      l1_params_["t_clip_min"].as<double>(),
      l1_params_["t_clip_max"].as<double>(),
      l1_params_["m_l1"].as<double>(),
      l1_params_["q_l1"].as<double>(),
      l1_params_["speed_lookahead"].as<double>(),
      l1_params_["lat_err_coeff"].as<double>(),
      l1_params_["acc_scaler_for_steer"].as<double>(),
      l1_params_["dec_scaler_for_steer"].as<double>(),
      l1_params_["start_scale_speed"].as<double>(),
      l1_params_["end_scale_speed"].as<double>(),
      l1_params_["downscale_factor"].as<double>(),
      l1_params_["speed_lookahead_for_steer"].as<double>(),
      l1_params_["prioritize_dyn"].as<bool>(),
      l1_params_["trailing_gap"].as<double>(),
      l1_params_["trailing_p_gain"].as<double>(),
      l1_params_["trailing_i_gain"].as<double>(),
      l1_params_["trailing_d_gain"].as<double>(),
      l1_params_["blind_trailing_speed"].as<double>(),
      l1_params_["trailing_to_gbtrack_speed_scale"].as<double>(),
      rate_hz_,
      LUT_name_,

      // 로거 람다 (Python의 logger_info/warn 대체)
      [this](const std::string &s){ RCLCPP_INFO(this->get_logger(), "%s", s.c_str());},
      [this](const std::string &s){ RCLCPP_WARN(this->get_logger(), "%s", s.c_str());}
    );
  }

  void init_pp_controller() {
    load_l1_params_from_yaml_();

    // wheelbase
    double wheelbase = 0.33; // fallback
    const auto share_dir = ament_index_cpp::get_package_share_directory("stack_master");
    if (sim_) {
      const std::string cfg = share_dir + "/config/" + racecar_version_ + "/sim_params.yaml";
      YAML::Node y = YAML::LoadFile(cfg);
      wheelbase = y["lr"].as<double>() + y["lf"].as<double>();
    } else {
      const std::string cfg = share_dir + "/config/" + racecar_version_ + "/vesc.yaml";
      YAML::Node y = YAML::LoadFile(cfg);
      wheelbase = y["vesc_to_odom_node"]["ros__parameters"]["wheelbase"].as<double>();
    }

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/sensors/imu/raw", 10, std::bind(&Controller::imu_cb, this, std::placeholders::_1));
    acc_now_.assign(10, 0.0);

    pp_controller_ = std::make_unique<PP_Controller>(
      l1_params_["t_clip_min"].as<double>(),
      l1_params_["t_clip_max"].as<double>(),
      l1_params_["m_l1"].as<double>(),
      l1_params_["q_l1"].as<double>(),
      l1_params_["speed_lookahead"].as<double>(),
      l1_params_["lat_err_coeff"].as<double>(),
      l1_params_["acc_scaler_for_steer"].as<double>(),
      l1_params_["dec_scaler_for_steer"].as<double>(),
      l1_params_["start_scale_speed"].as<double>(),
      l1_params_["end_scale_speed"].as<double>(),
      l1_params_["downscale_factor"].as<double>(),
      l1_params_["speed_lookahead_for_steer"].as<double>(),
      l1_params_["prioritize_dyn"].as<bool>(),
      l1_params_["trailing_gap"].as<double>(),
      l1_params_["trailing_p_gain"].as<double>(),
      l1_params_["trailing_i_gain"].as<double>(),
      l1_params_["trailing_d_gain"].as<double>(),
      l1_params_["blind_trailing_speed"].as<double>(),
      l1_params_["trailing_to_gbtrack_speed_scale"].as<double>(),
      rate_hz_,
      wheelbase,
      // 로거 람다 (Python의 logger_info/warn 대체)
      [this](const std::string &s){ RCLCPP_INFO(this->get_logger(), "%s", s.c_str());},
      [this](const std::string &s){ RCLCPP_WARN(this->get_logger(), "%s", s.c_str());}
    );
  }

  void init_ftg_controller() {
    // 원격 파라미터
    state_machine_debug_         = get_remote_parameter<bool>("state_machine", "debug");
    state_machine_safety_radius_ = get_remote_parameter<double>("state_machine", "safety_radius");
    state_machine_max_lidar_dist_= get_remote_parameter<double>("state_machine", "max_lidar_dist");
    state_machine_max_speed_     = get_remote_parameter<double>("state_machine", "max_speed");
    state_machine_range_offset_  = get_remote_parameter<double>("state_machine", "range_offset");
    state_machine_track_width_   = get_remote_parameter<double>("state_machine", "track_width");

    RCLCPP_INFO(get_logger(),
      "FTG params: debug=%d, safety=%.3f, maxLidar=%.3f, maxSpeed=%.3f, offset=%.3f, trackWidth=%.3f",
      state_machine_debug_, state_machine_safety_radius_, state_machine_max_lidar_dist_,
      state_machine_max_speed_, state_machine_range_offset_, state_machine_track_width_);

    ftg_controller_ = std::make_unique<FTG_Controller>(
      this,                               
      mapping_,                           
      state_machine_debug_,               
      state_machine_safety_radius_,  
      state_machine_max_lidar_dist_,      
      state_machine_max_speed_,           
      state_machine_range_offset_,   
      state_machine_track_width_          
    );
  }

  void load_l1_params_from_yaml_() {
    const std::string share_dir = ament_index_cpp::get_package_share_directory("stack_master");
    const std::string cfg = share_dir + "/config/" + racecar_version_ + "/l1_params.yaml";
    YAML::Node y = YAML::LoadFile(cfg);
    const auto node = y["controller"]["ros__parameters"];

    // 필요한 키만 복사
    const std::vector<std::string> keys = {
      "t_clip_min","t_clip_max","m_l1","q_l1","speed_lookahead","lat_err_coeff",
      "acc_scaler_for_steer","dec_scaler_for_steer","start_scale_speed","end_scale_speed",
      "downscale_factor","speed_lookahead_for_steer","prioritize_dyn","trailing_gap",
      "trailing_p_gain","trailing_i_gain","trailing_d_gain","blind_trailing_speed", "trailing_to_gbtrack_speed_scale"
    };
    l1_params_.clear();
    for (const auto &k : keys) l1_params_[k] = node[k];
  }

  // -------------------- 메시지 대기 --------------------
  void wait_for_messages_() {
    RCLCPP_INFO(get_logger(), "Controller Manager waiting for messages...");
    bool track_length_ok = false, wpnts_ok = false, state_ok = false;

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(this->get_node_base_interface());

    rclcpp::Rate r(50.0);
    while (rclcpp::ok() && (!track_length_ok || !wpnts_ok || !state_ok)) {
      exec.spin_some();  // ← 안전
      if (track_length_ && !track_length_ok) {
        RCLCPP_INFO(get_logger(), "Received track length");
        track_length_ok = true;
      }
      if (!waypoint_array_in_map_.empty() && !wpnts_ok) {
        RCLCPP_INFO(get_logger(), "Received waypoint array");
        wpnts_ok = true;
      }
      if (speed_now_.has_value() && position_in_map_.has_value() &&
          position_in_map_frenet_.has_value() && !state_ok) {
        RCLCPP_INFO(get_logger(), "Received car state messages");
        state_ok = true;
      }
      r.sleep();
    }
    RCLCPP_INFO(get_logger(), "All required messages received. Continuing...");
  }

  // -------------------- 동적 파라미터 선언/감시 --------------------
  void declare_update_params_and_watch_() {
    using rcl_interfaces::msg::ParameterDescriptor;
    using rcl_interfaces::msg::FloatingPointRange;

    auto fp_range = [](double a, double b, double step){
      FloatingPointRange r; r.from_value=a; r.to_value=b; r.step=step; return r;
    };
    auto decl = [&](const std::string& name, const YAML::Node& v, const ParameterDescriptor& d){
      if (v.IsScalar()) {
        if (v.Tag() == "!" || v.as<std::string>().find_first_not_of("0123456789.-") != std::string::npos) {
          // 문자열/불리언 추정
          if (v.as<std::string>()=="true" || v.as<std::string>()=="false")
            this->declare_parameter(name, v.as<bool>(), d);
          else
            this->declare_parameter(name, v.as<std::string>(), d);
        } else {
          // 숫자
          this->declare_parameter(name, v.as<double>(), d);
        }
      } else {
        // bool로 가정
        this->declare_parameter(name, v.as<bool>(), d);
      }
    };

    // 선언
    decl("t_clip_min", l1_params_["t_clip_min"], param_desc(fp_range(0.0,1.5,0.01)));
    decl("t_clip_max", l1_params_["t_clip_max"], param_desc(fp_range(0.0,10.0,0.01)));
    decl("m_l1", l1_params_["m_l1"],         param_desc(fp_range(0.0,1.0,0.001)));
    decl("q_l1", l1_params_["q_l1"],         param_desc(fp_range(-1.0,1.0,0.001)));
    decl("speed_lookahead", l1_params_["speed_lookahead"], param_desc(fp_range(0.0,1.0,0.01)));
    decl("lat_err_coeff",   l1_params_["lat_err_coeff"],   param_desc(fp_range(0.0,1.0,0.01)));
    decl("acc_scaler_for_steer", l1_params_["acc_scaler_for_steer"], param_desc(fp_range(0.0,1.5,0.01)));
    decl("dec_scaler_for_steer", l1_params_["dec_scaler_for_steer"], param_desc(fp_range(0.0,1.5,0.01)));
    decl("start_scale_speed", l1_params_["start_scale_speed"], param_desc(fp_range(0.0,10.0,0.01)));
    decl("end_scale_speed",   l1_params_["end_scale_speed"],   param_desc(fp_range(0.0,10.0,0.01)));
    decl("downscale_factor",  l1_params_["downscale_factor"],  param_desc(fp_range(0.0,0.5,0.01)));
    decl("speed_lookahead_for_steer", l1_params_["speed_lookahead_for_steer"], param_desc(fp_range(0.0,0.2,0.01)));
    declare_parameter("prioritize_dyn", prioritize_dyn_);
    decl("trailing_gap", l1_params_["trailing_gap"], param_desc(fp_range(0.0,3.0,0.1)));
    decl("trailing_p_gain", l1_params_["trailing_p_gain"], param_desc(fp_range(0.0,3.0,0.01)));
    decl("trailing_i_gain", l1_params_["trailing_i_gain"], param_desc(fp_range(0.0,0.5,0.001)));
    decl("trailing_d_gain", l1_params_["trailing_d_gain"], param_desc(fp_range(0.0,1.0,0.01)));
    decl("blind_trailing_speed", l1_params_["blind_trailing_speed"], param_desc(fp_range(0.0,3.0,0.01)));
    decl("trailing_to_gbtrack_speed_scale", l1_params_["trailing_to_gbtrack_speed_scale"], param_desc(fp_range(0.0,1.0,0.01)));

    // 파라미터 변경 감시
    param_event_sub_ = create_subscription<rcl_interfaces::msg::ParameterEvent>(
      "/parameter_events", 10,
      std::bind(&Controller::on_parameter_event_, this, std::placeholders::_1));
  }

  rcl_interfaces::msg::ParameterDescriptor param_desc(
      const rcl_interfaces::msg::FloatingPointRange &rng) {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
    d.floating_point_range = {rng};
    return d;
  }

  void on_parameter_event_(const rcl_interfaces::msg::ParameterEvent::SharedPtr event) {
    // 이 노드의 파라미터만 처리
    if (event->node != this->get_fully_qualified_name() || mode_ == "FTG")
      return;

    // 업데이트(간단히 get_parameter 사용)
    if (mode_ == "MAP" && map_controller_) {
      map_controller_->t_clip_min               = get_parameter("t_clip_min").as_double();
      map_controller_->t_clip_max               = get_parameter("t_clip_max").as_double();
      map_controller_->m_l1                     = get_parameter("m_l1").as_double();
      map_controller_->q_l1                     = get_parameter("q_l1").as_double();
      map_controller_->speed_lookahead          = get_parameter("speed_lookahead").as_double();
      map_controller_->lat_err_coeff            = get_parameter("lat_err_coeff").as_double();
      map_controller_->acc_scaler_for_steer     = get_parameter("acc_scaler_for_steer").as_double();
      map_controller_->dec_scaler_for_steer     = get_parameter("dec_scaler_for_steer").as_double();
      map_controller_->start_scale_speed        = get_parameter("start_scale_speed").as_double();
      map_controller_->end_scale_speed          = get_parameter("end_scale_speed").as_double();
      map_controller_->downscale_factor         = get_parameter("downscale_factor").as_double();
      map_controller_->speed_lookahead_for_steer= get_parameter("speed_lookahead_for_steer").as_double();
      map_controller_->prioritize_dyn           = get_parameter("prioritize_dyn").as_bool();
      map_controller_->trailing_gap             = get_parameter("trailing_gap").as_double();
      map_controller_->trailing_p_gain          = get_parameter("trailing_p_gain").as_double();
      map_controller_->trailing_i_gain          = get_parameter("trailing_i_gain").as_double();
      map_controller_->trailing_d_gain          = get_parameter("trailing_d_gain").as_double();
      map_controller_->blind_trailing_speed     = get_parameter("blind_trailing_speed").as_double();
      map_controller_->trailing_to_gbtrack_speed_scale = get_parameter("trailing_to_gbtrack_speed_scale").as_double();
    } else if (mode_ == "PP" && pp_controller_) {
      pp_controller_->t_clip_min                = get_parameter("t_clip_min").as_double();
      pp_controller_->t_clip_max                = get_parameter("t_clip_max").as_double();
      pp_controller_->m_l1                      = get_parameter("m_l1").as_double();
      pp_controller_->q_l1                      = get_parameter("q_l1").as_double();
      pp_controller_->speed_lookahead           = get_parameter("speed_lookahead").as_double();
      pp_controller_->lat_err_coeff             = get_parameter("lat_err_coeff").as_double();
      pp_controller_->acc_scaler_for_steer      = get_parameter("acc_scaler_for_steer").as_double();
      pp_controller_->dec_scaler_for_steer      = get_parameter("dec_scaler_for_steer").as_double();
      pp_controller_->start_scale_speed         = get_parameter("start_scale_speed").as_double();
      pp_controller_->end_scale_speed           = get_parameter("end_scale_speed").as_double();
      pp_controller_->downscale_factor          = get_parameter("downscale_factor").as_double();
      pp_controller_->speed_lookahead_for_steer = get_parameter("speed_lookahead_for_steer").as_double();
      pp_controller_->prioritize_dyn            = get_parameter("prioritize_dyn").as_bool();
      pp_controller_->trailing_gap              = get_parameter("trailing_gap").as_double();
      pp_controller_->trailing_p_gain           = get_parameter("trailing_p_gain").as_double();
      pp_controller_->trailing_i_gain           = get_parameter("trailing_i_gain").as_double();
      pp_controller_->trailing_d_gain           = get_parameter("trailing_d_gain").as_double();
      pp_controller_->blind_trailing_speed      = get_parameter("blind_trailing_speed").as_double();
      pp_controller_->trailing_to_gbtrack_speed_scale = get_parameter("trailing_to_gbtrack_speed_scale").as_double();
    }
    RCLCPP_INFO(get_logger(), "Updated parameters");
  }

  // -------------------- 콜백들 --------------------
  void scan_cb(const sensor_msgs::msg::LaserScan::SharedPtr msg) { scan_ = msg; }

  void track_length_cb(const f110_msgs::msg::WpntArray::SharedPtr msg) {
    if (!msg->wpnts.empty()) {
      track_length_ = msg->wpnts.back().s_m;
      waypoints_.clear();
      waypoints_.reserve(msg->wpnts.size());
      for (const auto &w : msg->wpnts) {
        waypoints_.push_back({w.x_m, w.y_m, w.psi_rad});
      }
    }
  }

  void obstacle_cb(const f110_msgs::msg::ObstacleArray::SharedPtr msg) {
    if (!msg->obstacles.empty() && position_in_map_frenet_.has_value() && track_length_.has_value()) {
      bool static_flag = false;
      double closest_opp = track_length_.value();
      std::optional<std::array<double,5>> best;
      for (const auto &ob : msg->obstacles) {
        const double opponent_dist = std::fmod(ob.s_start - position_in_map_frenet_.value()[0] + track_length_.value(), track_length_.value());
        if (opponent_dist < closest_opp || (static_flag && !ob.is_static)) {
          closest_opp = opponent_dist;
          best = {ob.s_center, ob.d_center, ob.vs, static_cast<double>(ob.is_static), static_cast<double>(ob.is_visible)};
          static_flag = ob.is_static ? l1_params_["prioritize_dyn"].as<bool>() : false;
        }
      }
      opponent_ = best;
    } else {
      opponent_.reset();
    }
  }

  void car_state_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    const auto &p = msg->pose.position;
    // Euler yaw
    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.orientation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    position_in_map_ = std::array<double,3>{p.x, p.y, yaw};
  }

  void local_waypoint_cb(const f110_msgs::msg::WpntArray::SharedPtr msg) {
    waypoint_array_in_map_.clear();
    waypoint_array_in_map_.reserve(msg->wpnts.size());
    for (const auto &w : msg->wpnts) {
      double share = 0.0;
      if (w.d_right + w.d_left != 0.0)
        share = std::min(w.d_left, w.d_right) / (w.d_right + w.d_left);

      waypoint_array_in_map_.push_back({
        w.x_m, w.y_m, w.vx_mps, share, w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2
      });
    }
    waypoint_safety_counter_ = 0;
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    // rolling buffer, (-acc_y) == long_acc
    for (size_t i=acc_now_.size()-1; i>0; --i) acc_now_[i] = acc_now_[i-1];
    acc_now_[0] = msg->linear_acceleration.x;
  }

  void car_state_frenet_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // pose.x->s, pose.y->d, twist.linear.x->vs, twist.linear.y->vd
    position_in_map_frenet_ = std::array<double,4>{
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->twist.twist.linear.x,
      msg->twist.twist.linear.y
    };
  }

  // -------------------- 시각화 유틸 --------------------
  void visualize_steering(double theta) {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, theta);

    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = "car_state/base_link";
    mk.header.stamp = now();
    mk.type = visualization_msgs::msg::Marker::CUBE;
    mk.id = 50;
    mk.scale.x = 0.6;
    mk.scale.y = 0.05;
    mk.scale.z = 0.01;
    mk.color.r = 1.0f; mk.color.g = 0.0f; mk.color.b = 0.0f; mk.color.a = 1.0f;
    mk.pose.orientation = tf2::toMsg(q);
    steering_pub_->publish(mk);
  }

  void set_waypoint_markers_(const std::vector<std::array<double,8>>& wps) {
    visualization_msgs::msg::MarkerArray arr;
    int id = 1;
    for (const auto &w : wps) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "map";
      m.header.stamp = now();
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.id = id++;
      m.scale.x = m.scale.y = m.scale.z = 0.1;
      m.color.b = 1.0f; m.color.a = 1.0f;
      m.pose.position.x = w[0];
      m.pose.position.y = w[1];
      m.pose.orientation.w = 1.0;
      arr.markers.push_back(m);
    }
    waypoint_pub_->publish(arr);
  }

  void set_lookahead_marker_(const std::array<double,2>& pt, int id) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = now();
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.id = id;
    m.scale.x = m.scale.y = m.scale.z = 0.35;
    m.color.g = 1.0f; m.color.a = 1.0f;
    m.pose.position.x = pt[0];
    m.pose.position.y = pt[1];
    m.pose.orientation.w = 1.0;
    lookahead_pub_->publish(m);
  }

  // -------------------- 메인 루프 --------------------
  void control_loop() {
    double speed = 0.0, steer = 0.0;

    if (mode_ == "MAP" && map_controller_) {
      std::tie(speed, steer) = map_cycle_();
    } else if (mode_ == "PP" && pp_controller_) {
      std::tie(speed, steer) = pp_cycle_();
    } else if (mode_ == "FTG" && ftg_controller_) {
      std::tie(speed, steer) = ftg_cycle_();
    }

    ackermann_msgs::msg::AckermannDriveStamped ack;
    ack.header.stamp = this->get_clock()->now();
    ack.header.frame_id = "base_link";
    ack.drive.steering_angle = steer;
    ack.drive.speed = speed;
    drive_pub_->publish(ack);
  }

  std::pair<double,double> map_cycle_() {
    // MAP_Controller::main_loop는 Python과 동일 인터페이스라고 가정
    auto out = map_controller_->main_loop(
      state_,
      position_in_map_,                 // std::optional<std::array<double,3>>
      waypoint_array_in_map_,          // std::vector<std::array<double,8>>
      speed_now_.value_or(0.0),
      opponent_,                       // std::optional<std::array<double,5>>
      position_in_map_frenet_,         // std::optional<std::array<double,4>>
      acc_now_,                        // std::vector<double> size 10
      track_length_.value_or(0.0)
    );
    double speed = std::get<0>(out);
    double steer = std::get<3>(out);

    waypoint_safety_counter_++;
    if (waypoint_safety_counter_ >= static_cast<int>(rate_hz_ / state_machine_rate_ * 10.0)) {
      RCLCPP_WARN(get_logger(), "[controller_manager] Received no local wpnts. STOPPING!!");
      speed = 0.0; steer = 0.0;
    }
    return {speed, steer};
  }

  std::pair<double,double> pp_cycle_() {
    auto out = pp_controller_->main_loop(
      state_,
      position_in_map_,
      waypoint_array_in_map_,
      speed_now_.value_or(0.0),
      opponent_,
      position_in_map_frenet_,
      acc_now_,
      track_length_.value_or(0.0)
    );
    double speed = std::get<0>(out);
    double steer = std::get<3>(out);

    waypoint_safety_counter_++;
    if (waypoint_safety_counter_ >= static_cast<int>(rate_hz_ / state_machine_rate_ * 10.0)) {
      RCLCPP_WARN(get_logger(), "[controller_manager] Received no local wpnts. STOPPING!!");
      speed = 0.0; steer = 0.0;
    }
    return {speed, steer};
  }

  std::pair<double,double> ftg_cycle_() {
    if (!scan_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "FTGONLY!!! (no scan yet)");
      return {0.0, 0.0};
    }
    const auto &ranges = scan_->ranges;
    auto out = ftg_controller_->process_lidar(ranges);
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "[STATE MACHINE] FTGONLY!!!");
    return out;
  }

private:
  // 상태/설정
  std::string map_path_;
  std::string racecar_version_;
  bool sim_{false};
  double state_machine_rate_{10.0};

  std::string LUT_name_;
  std::string mode_;
  bool mapping_{false};
  double rate_hz_{30.0};
  std::string state_;

  // 데이터 보관
  std::optional<double> track_length_;
  std::vector<std::array<double,3>> waypoints_; // x,y,psi
  std::vector<std::array<double,8>> waypoint_array_in_map_; // x,y,v,share,s,kappa,psi,ax
  int waypoint_safety_counter_{0};

  std::optional<double> speed_now_;
  std::optional<std::array<double,3>> position_in_map_;
  std::optional<std::array<double,4>> position_in_map_frenet_;
  std::optional<std::array<double,5>> opponent_;

  std::vector<double> acc_now_;
  bool prioritize_dyn_{false};

  // FTG 원격 파라미터
  bool   state_machine_debug_{false};
  double state_machine_safety_radius_{0.0};
  double state_machine_max_lidar_dist_{0.0};
  double state_machine_max_speed_{0.0};
  double state_machine_range_offset_{0.0};
  double state_machine_track_width_{0.0};

  // YAML 로드된 L1 파라미터 (키→YAML::Node)
  std::unordered_map<std::string, YAML::Node> l1_params_;

  // ROS I/O
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr steering_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr lookahead_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trailing_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoint_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr l1_dist_pub_;
  rclcpp::Publisher<f110_msgs::msg::PidData>::SharedPtr gap_data_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr glb_wpnts_sub_;
  rclcpp::Subscription<f110_msgs::msg::WpntArray>::SharedPtr local_wpnts_sub_;
  rclcpp::Subscription<f110_msgs::msg::ObstacleArray>::SharedPtr obstacles_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr frenet_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<rcl_interfaces::msg::ParameterEvent>::SharedPtr param_event_sub_;

  sensor_msgs::msg::LaserScan::SharedPtr scan_;

  // 컨트롤러들
  std::unique_ptr<MAP_Controller> map_controller_;
  std::unique_ptr<PP_Controller>  pp_controller_;
  std::unique_ptr<FTG_Controller> ftg_controller_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Controller>());
  rclcpp::shutdown();
  return 0;
}
