#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import rclpy
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.client import Client
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import (
    ParameterValue, ParameterType, ParameterDescriptor,
    FloatingPointRange
)
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from ament_index_python import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import (ObstacleArray, PidData, WpntArray)
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from controller.pp import PP_Controller
from stack_master.parameter_event_handler import ParameterEventHandler


def return_param_value(p: ParameterValue):
    if p.type == ParameterType.PARAMETER_NOT_SET:   return None
    if p.type == ParameterType.PARAMETER_BOOL:      return p.bool_value
    if p.type == ParameterType.PARAMETER_INTEGER:   return p.integer_value
    if p.type == ParameterType.PARAMETER_DOUBLE:    return p.double_value
    if p.type == ParameterType.PARAMETER_STRING:    return p.string_value
    if p.type == ParameterType.PARAMETER_BYTE_ARRAY:return p.byte_array_value
    if p.type == ParameterType.PARAMETER_BOOL_ARRAY:return p.bool_array_value
    if p.type == ParameterType.PARAMETER_INTEGER_ARRAY: return p.integer_array_value
    if p.type == ParameterType.PARAMETER_DOUBLE_ARRAY:  return p.double_array_value
    if p.type == ParameterType.PARAMETER_STRING_ARRAY:  return p.string_array_value
    raise TypeError(f"Unexpected parameter type: {p.type}")


class Controller(Node):
    """Pure Pursuit-only controller manager with publish gate."""

    def __init__(self):
        super().__init__(
            'controller_manager_small',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        # --- Remote/global params ---
        self.racecar_version = self.get_remote_parameter('global_parameters', 'racecar_version')
        self.sim = self.get_remote_parameter('global_parameters', 'sim')
        self.state_machine_rate = self.get_remote_parameter('state_machine', 'rate_hz')

        # --- Internal state ---
        self.rate = 40
        self.state = "GB_TRACK"
        self.LUT_name = self.get_parameter('LU_table').value if self.has_parameter('LU_table') else 'default'

        # --- Gate parameter: publish only when True ---
        # 기본값 False: 계산은 계속하되 /drive 퍼블리시는 차단
        self.declare_parameter('drive_publish_enabled', False)
        self.drive_publish_enabled = bool(self.get_parameter('drive_publish_enabled').value)

        # --- Publishers (drive & viz) ---
        self.publish_topic = '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, 'my_waypoints', 10)
        self.l1_pub = self.create_publisher(Point, 'l1_distance', 10)
        self.gap_data = self.create_publisher(PidData, '/trailing/gap_data', 10)

        # --- Runtime buffers ---
        self.track_length = None
        self.opponent = None
        self.waypoint_array_in_map = None
        self.speed_now = None
        self.position_in_map = None
        self.position_in_map_frenet = None
        self.waypoint_safety_counter = 0
        self.waypoint_array_buf = MarkerArray()
        self.markers_buf = [Marker() for _ in range(1000)]
        self.scan = LaserScan()

        # --- Pure Pursuit setup ---
        self.get_logger().info("Initializing PP controller")
        self.init_pp_controller()

        # --- Subscribers ---
        self.create_subscription(String, '/state', self.state_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.track_length_cb, 10)
        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacle_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self.local_waypoint_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/car_state/pose', self.car_state_cb, 10)
        self.create_subscription(Odometry, '/car_state/frenet/odom', self.car_state_frenet_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Wait until initial messages arrive
        self.wait_for_messages()

        # --- Dynamic param update (L1 params) ---
        self.param_handler = ParameterEventHandler(self)
        self.callback_handle = self.param_handler.add_parameter_event_callback(
            callback=self.l1_param_cb
        )

        # --- Declare PP dynamic params & seed values ---
        self.declare_and_seed_dyn_params()

        # --- On-set callback for gate param (and any others you might add later) ---
        self.add_on_set_parameters_callback(self._on_set_params)

        # --- Main loop ---
        self.create_timer(1 / self.rate, self.control_loop)
        self.get_logger().info("Pure Pursuit Controller ready (publish gated by 'drive_publish_enabled')")

    # -------------------- Pure Pursuit setup --------------------

    def init_pp_controller(self):
        stack_master_path = get_package_share_directory('stack_master')
        l1_cfg = Path(stack_master_path) / 'config' / self.racecar_version / 'l1_params.yaml'
        with open(l1_cfg, 'r') as f:
            self.l1_params = yaml.safe_load(f)['controller']['ros__parameters']

        if self.sim:
            sim_cfg = Path(stack_master_path) / 'config' / self.racecar_version / 'sim_params.yaml'
            with open(sim_cfg, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params['lr'] + car_params['lf']
        else:
            vesc_cfg = Path(stack_master_path) / 'config' / self.racecar_version / 'vesc.yaml'
            with open(vesc_cfg, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params['vesc_to_odom_node']['ros__parameters']['wheelbase']

        self.create_subscription(Imu, '/vesc/sensors/imu/raw', self.imu_cb, 10)
        self.acc_now = np.zeros(10)

        self.pp_controller = PP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["speed_lookahead_for_steer"],
            self.l1_params["prioritize_dyn"],
            self.l1_params["trailing_gap"],
            self.l1_params["trailing_p_gain"],
            self.l1_params["trailing_i_gain"],
            self.l1_params["trailing_d_gain"],
            self.l1_params["blind_trailing_speed"],
            self.rate,
            self.wheelbase,
            logger_info=self.get_logger().info,
            logger_warn=self.get_logger().warning,
        )

    def declare_and_seed_dyn_params(self):
        specs = [
            ('t_clip_min',         self.l1_params["t_clip_min"],          (0.0, 1.5, 0.01)),
            ('t_clip_max',         self.l1_params["t_clip_max"],          (0.0, 10.0, 0.01)),
            ('m_l1',               self.l1_params["m_l1"],                (0.0, 1.0, 0.001)),
            ('q_l1',               self.l1_params["q_l1"],                (-1.0, 1.0, 0.001)),
            ('speed_lookahead',    self.l1_params["speed_lookahead"],     (0.0, 1.0, 0.01)),
            ('lat_err_coeff',      self.l1_params["lat_err_coeff"],       (0.0, 1.0, 0.01)),
            ('acc_scaler_for_steer', self.l1_params["acc_scaler_for_steer"], (0.0, 1.5, 0.01)),
            ('dec_scaler_for_steer', self.l1_params["dec_scaler_for_steer"], (0.0, 1.5, 0.01)),
            ('start_scale_speed',  self.l1_params["start_scale_speed"],   (0.0, 10.0, 0.01)),
            ('end_scale_speed',    self.l1_params["end_scale_speed"],     (0.0, 10.0, 0.01)),
            ('downscale_factor',   self.l1_params["downscale_factor"],    (0.0, 0.5, 0.01)),
            ('speed_lookahead_for_steer', self.l1_params["speed_lookahead_for_steer"], (0.0, 0.2, 0.01)),
            ('prioritize_dyn',     self.l1_params["prioritize_dyn"],      None),
            ('trailing_gap',       self.l1_params["trailing_gap"],        (0.0, 3.0, 0.1)),
            ('trailing_p_gain',    self.l1_params["trailing_p_gain"],     (0.0, 3.0, 0.01)),
            ('trailing_i_gain',    self.l1_params["trailing_i_gain"],     (0.0, 0.5, 0.001)),
            ('trailing_d_gain',    self.l1_params["trailing_d_gain"],     (0.0, 1.0, 0.01)),
            ('blind_trailing_speed', self.l1_params["blind_trailing_speed"], (0.0, 3.0, 0.01)),
        ]
        to_set = []
        for name, default, rng in specs:
            if rng is None:
                desc = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)
            else:
                desc = ParameterDescriptor(
                    type=ParameterType.PARAMETER_DOUBLE,
                    floating_point_range=[FloatingPointRange(from_value=rng[0], to_value=rng[1], step=rng[2])]
                )
            to_set.append(self.declare_parameter(name, default, desc))
        self.set_parameters(to_set)

    # -------------------- Wait for initial msgs --------------------

    def wait_for_messages(self):
        self.get_logger().info('Controller Manager waiting for initial messages...')
        got_track = got_wps = got_state = False
        while not (got_track and got_wps and got_state):
            rclpy.spin_once(self)
            if self.track_length is not None and not got_track:
                self.get_logger().info('Received track length'); got_track = True
            if self.waypoint_array_in_map is not None and not got_wps:
                self.get_logger().info('Received waypoint array'); got_wps = True
            if self.speed_now is not None and self.position_in_map is not None and \
               self.position_in_map_frenet is not None and not got_state:
                self.get_logger().info('Received car state messages'); got_state = True
        self.get_logger().info('All required messages received. Continuing...')

    # -------------------- Remote params --------------------

    def get_remote_parameter(self, remote_node_name: str, param_name: str):
        cli: Client = self.create_client(GetParameters, f'{remote_node_name}/get_parameters')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for remote parameter service...')
        req = GetParameters.Request(); req.names = [param_name]
        future = cli.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                res = future.result()
                return return_param_value(res.values[0])

    # -------------------- Param callbacks --------------------

    def _on_set_params(self, params):
        """Handles non-L1 params like the publish gate."""
        for p in params:
            if p.name == 'drive_publish_enabled':
                if p.type_ == Parameter.Type.BOOL:
                    self.drive_publish_enabled = bool(p.value)
                    self.get_logger().info(f"[controller_manager] drive_publish_enabled = {self.drive_publish_enabled}")
        return SetParametersResult(successful=True)

    def l1_param_cb(self, _evt):
        """Update PP controller when L1 dynamic params change."""
        self.pp_controller.t_clip_min = self.get_parameter('t_clip_min').value
        self.pp_controller.t_clip_max = self.get_parameter('t_clip_max').value
        self.pp_controller.m_l1 = self.get_parameter('m_l1').value
        self.pp_controller.q_l1 = self.get_parameter('q_l1').value
        self.pp_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
        self.pp_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
        self.pp_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
        self.pp_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
        self.pp_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
        self.pp_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
        self.pp_controller.downscale_factor = self.get_parameter('downscale_factor').value
        self.pp_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value
        self.pp_controller.prioritize_dyn = self.get_parameter('prioritize_dyn').value
        self.pp_controller.trailing_gap = self.get_parameter('trailing_gap').value
        self.pp_controller.trailing_p_gain = self.get_parameter('trailing_p_gain').value
        self.pp_controller.trailing_i_gain = self.get_parameter('trailing_i_gain').value
        self.pp_controller.trailing_d_gain = self.get_parameter('trailing_d_gain').value
        self.pp_controller.blind_trailing_speed = self.get_parameter('blind_trailing_speed').value
        self.get_logger().info("PP dynamic parameters updated")

    # -------------------- Subscribers --------------------

    def scan_cb(self, data: LaserScan):
        self.scan = data

    def track_length_cb(self, data: WpntArray):
        self.track_length = data.wpnts[-1].s_m
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])

    def obstacle_cb(self, data: ObstacleArray):
        if len(data.obstacles) > 0 and \
           self.position_in_map_frenet is not None and len(self.position_in_map_frenet) and \
           self.track_length is not None:
            closest_opp = self.track_length
            static_flag = False
            for obstacle in data.obstacles:
                opponent_dist = (obstacle.s_start - self.position_in_map_frenet[0]) % self.track_length
                if opponent_dist < closest_opp or (static_flag and not obstacle.is_static):
                    closest_opp = opponent_dist
                    opponent_static = obstacle.is_static
                    opponent_s = obstacle.s_center
                    opponent_d = obstacle.d_center
                    opponent_vs = obstacle.vs
                    opponent_visible = obstacle.is_visible
                    static_flag = False if not obstacle.is_static else self.pp_controller.prioritize_dyn
                    self.opponent = [opponent_s, opponent_d, opponent_vs, opponent_static, opponent_visible]
        else:
            self.opponent = None

    def odom_cb(self, data: Odometry):
        self.speed_now = data.twist.twist.linear.x

    def car_state_cb(self, data: PoseStamped):
        x = data.pose.position.x
        y = data.pose.position.y
        rot = Rotation.from_quat([
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ])
        theta = rot.as_euler('xyz', degrees=False)[2]
        self.position_in_map = np.array([x, y, theta])[np.newaxis]

    def local_waypoint_cb(self, data: WpntArray):
        arr = []
        for w in data.wpnts:
            if w.d_right + w.d_left != 0:
                arr.append([w.x_m, w.y_m, w.vx_mps,
                            min(w.d_left, w.d_right) / (w.d_right + w.d_left),
                            w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2])
            else:
                arr.append([w.x_m, w.y_m, w.vx_mps, 0, w.s_m, w.kappa_radpm, w.psi_rad, w.ax_mps2])
        self.waypoint_array_in_map = np.array(arr)
        self.waypoint_safety_counter = 0

    def imu_cb(self, data: Imu):
        self.acc_now[1:] = self.acc_now[:-1]
        self.acc_now[0] = -data.linear_acceleration.y

    def car_state_frenet_cb(self, data: Odometry):
        self.position_in_map_frenet = np.array([
            data.pose.pose.position.x,   # s
            data.pose.pose.position.y,   # d
            data.twist.twist.linear.x,   # vs
            data.twist.twist.linear.y    # vd
        ])

    def state_cb(self, data: String):
        self.state = data.data

    # -------------------- PP compute cycle --------------------

    def pp_cycle(self):
        speed, acc, jerk, steer, L1_point, L1_dist, idx = self.pp_controller.main_loop(
            self.state,
            self.position_in_map,
            self.waypoint_array_in_map,
            self.speed_now,
            self.opponent,
            self.position_in_map_frenet,
            self.acc_now,
            self.track_length
        )

        self.set_lookahead_marker(L1_point, 100)

        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate / self.state_machine_rate * 10:
            self.get_logger().warning("[controller_manager] No local waypoints. STOP.")
            speed, steer = 0.0, 0.0
        self.pp_controller.flag1 = False
        return speed, steer

    # -------------------- Main loop --------------------

    def control_loop(self):
        speed, steer = self.pp_cycle()

        # publish gate: only publish when drive_publish_enabled is True
        if self.drive_publish_enabled:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.drive.steering_angle = steer
            msg.drive.speed = speed
            self.drive_pub.publish(msg)

    # -------------------- Viz utils --------------------

    def set_waypoint_markers(self, waypoints):
        wpnt_id = 0
        for waypoint in waypoints:
            m = self.markers_buf[wpnt_id]
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = 2
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; m.color.a = 1.0
            m.pose.position.x = waypoint[0]
            m.pose.position.y = waypoint[1]
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.id = wpnt_id + 1
            wpnt_id += 1
        self.waypoint_array_buf.markers = self.markers_buf[:wpnt_id]
        self.waypoint_pub.publish(self.waypoint_array_buf)

    def set_lookahead_marker(self, lookahead_point, mid):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = 2
        m.id = mid
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
        m.pose.position.x = lookahead_point[0]
        m.pose.position.y = lookahead_point[1]
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        self.lookahead_pub.publish(m)


def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    rclpy.shutdown()

