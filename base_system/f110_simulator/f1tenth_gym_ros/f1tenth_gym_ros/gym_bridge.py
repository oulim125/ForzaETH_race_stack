# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import yaml
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import ParameterType
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

from sensor_msgs.msg import Imu

import gymnasium as gym
import numpy as np
from transforms3d import euler
import os.path


class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge',
                         automatically_declare_parameters_from_overrides=True)

        # 1. IMU 토픽 이름 파라미터 등록
        self.set_descriptor(name='ego_imu_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))

        self.set_descriptor(name='ego_namespace', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_pose_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_opp_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_scan_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='ego_drive_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_namespace', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_ego_odom_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_scan_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='opp_drive_topic', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))

        self.set_descriptor(name='scan_distance_to_base_link', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Transforms related"))
        self.set_descriptor(name='scan_fov', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="laserscan related"))
        self.set_descriptor(name='scan_beams', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER, description="laserscan related"))

        self.set_descriptor(name='map_path', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))
        self.set_descriptor(name='map_img_ext', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING))

        self.set_descriptor(name='num_agent', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER))

        self.set_descriptor(name='sim_params', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING, description="The path to the sim_params yaml file"))

        self.set_descriptor(name='sx', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting X position of ego"))
        self.set_descriptor(name='sy', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting Y position of ego"))
        self.set_descriptor(name='stheta', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting heading of ego"))

        self.set_descriptor(name='sx1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting X position of opponent"))
        self.set_descriptor(name='sy1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting Y position of opponent"))
        self.set_descriptor(name='stheta1', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="Starting heading of opponent"))

        self.set_descriptor(name='kb_teleop', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL, description="Whether teleop is enabled"))

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # get sim params
        sim_params_yaml = self.get_parameter('sim_params').value
        sim_param_data = yaml.safe_load(open(sim_params_yaml, 'r'))
        sim_params = {key: float(value)
                      for key, value in sim_param_data.items()}

        # env backend
        map_yaml_path = os.path.abspath(self.get_parameter('map_path').value)
        self.env = gym.make('f110_gym:f110-v1',
                            map=map_yaml_path.split('.')[0],
                            map_ext=self.get_parameter('map_img_ext').value,
                            params=sim_params,
                            num_agents=num_agents,
                            disable_env_checker=True)

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / (scan_beams-1)
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_odom_topic = self.ego_namespace + '/' + \
            self.get_parameter('ego_odom_topic').value
        ego_pose_topic = self.ego_namespace + '/' + \
            self.get_parameter('ego_pose_topic').value
        self.scan_distance_to_base_link = self.get_parameter(
            'scan_distance_to_base_link').value
        self.ts = self.get_clock().now().to_msg()

        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])

            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + \
                self.get_parameter('opp_odom_topic').value
            opp_drive_topic = self.get_parameter('opp_drive_topic').value

            ego_opp_odom_topic = self.ego_namespace + '/' + \
                self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + \
                self.get_parameter('opp_ego_odom_topic').value
        else:
            self.has_opp = False
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        # sim physical step timer
        cb_group1= ReentrantCallbackGroup()
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback, callback_group=cb_group1)
        # topic publishing timer
        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=cb_group1)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.ego_pose_pub = self.create_publisher(PoseStamped, ego_pose_topic, 10)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(
                LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(
                Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(
                Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(
                Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False
        
        # 2. IMU 퍼블리셔 생성
        imu_topic = self.get_parameter('ego_imu_topic').value
        self.ego_imu_pub = self.create_publisher(Imu, imu_topic, 20)

        # 3. 가속도 계산을 위한 상태 변수
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_time = self.get_clock().now()

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            10)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)

    def drive_callback(self, drive_msg):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y']
                        [1], self.obs['poses_theta'][1]]
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[rx, ry, rtheta], opp_pose]))
        else:
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([[rx, ry, rtheta]]))

    def opp_reset_callback(self, pose_msg):
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            self.obs, _, self.done, _ = self.env.reset(
                poses=np.array([list(self.ego_pose), [rx, ry, rtheta]]))

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.ego_drive_published and not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(
                np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array(
                [[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        self.ts = self.get_clock().now().to_msg()
        self._update_sim_state()

    def timer_callback(self):
        # pub scans
        scan = LaserScan()
        scan.header.stamp = self.ts
        # scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.0
        scan.range_max = 30.0
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = self.ts
            # opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

        # pub tf
        self._publish_odom(self.ts)
        self._publish_transforms(self.ts)
        self._publish_laser_transforms(self.ts)
        self._publish_wheel_transforms(self.ts)

        # imu 메시지
        imu_msg = Imu()
        imu_msg.header.stamp = self.ts
        imu_msg.header.frame_id = self.ego_namespace + '/base_link'

        # 선형 가속도(시뮬레이터는 평면 주행 -> z=0)
        imu_msg.linear_acceleration.x = float(self.ego_accel_x)
        imu_msg.linear_acceleration.y = float(self.ego_accel_y)
        imu_msg.linear_acceleration.z = 0.0  # 2D

        # 각속도는 시뮬레이터가 이미 제공하는 yaw_rate 사용
        imu_msg.angular_velocity.x = 0.0
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = float(self.ego_speed[2])

        # 자세(orientation, yaw만 신뢰)
        quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        imu_msg.orientation.x = quat[1]
        imu_msg.orientation.y = quat[2]
        imu_msg.orientation.z = quat[3]
        imu_msg.orientation.w = quat[0]
       
        # 아주 작은 값(비대각/기타)
        eps = 1e-9

        # 3x3 행렬을 1차원으로 채우기 (row-major)
        imu_msg.orientation_covariance         = [eps] * 9
        imu_msg.angular_velocity_covariance    = [eps] * 9
        imu_msg.linear_acceleration_covariance = [eps] * 9

        # 대각 성분(variance)만 0.0으로 설정
        # indices: 0(xx), 4(yy), 8(zz)
        imu_msg.orientation_covariance[0] = 0.0  # roll
        imu_msg.orientation_covariance[4] = 0.0  # pitch
        imu_msg.orientation_covariance[8] = 0.0  # yaw

        imu_msg.angular_velocity_covariance[0] = 0.0  # wx
        imu_msg.angular_velocity_covariance[4] = 0.0  # wy
        imu_msg.angular_velocity_covariance[8] = 0.0  # wz

        imu_msg.linear_acceleration_covariance[0] = 0.0  # ax
        imu_msg.linear_acceleration_covariance[4] = 0.0  # ay
        imu_msg.linear_acceleration_covariance[8] = 0.0  # az

        self.ego_imu_pub.publish(imu_msg)
        # ---------------------------------------------------

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0] = self.obs['poses_x'][1]
            self.opp_pose[1] = self.obs['poses_y'][1]
            self.opp_pose[2] = self.obs['poses_theta'][1]
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

        # -----선형 가속도 계산(finite difference)-----
        now = self.get_clock().now()
        dt = (now - self._prev_time).nanoseconds * 1e-9
        if dt > 0:    # 첫 루프 보호
            self.ego_accel_x = (self.ego_speed[0] - self._prev_vx) / dt
            self.ego_accel_y = (self.ego_speed[1] - self._prev_vy) / dt
        else:
            self.ego_accel_x = 0.0
            self.ego_accel_y = 0.0
        
        self._prev_vx = self.ego_speed[0]
        self._prev_vy = self.ego_speed[1]
        self._prev_time = now

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]

            # 기본값: 전부 1e-4
        ego_odom.pose.covariance = [1e-4]*36
        ego_odom.twist.covariance = [1e-4]*36

        # Pose covariance
        ego_odom.pose.covariance[0]  = 0.25   # x
        ego_odom.pose.covariance[7]  = 0.5    # y
        ego_odom.pose.covariance[35] = 0.4    # yaw

        # Twist covariance
        ego_odom.twist.covariance[0]  = 0.02  # vx
        ego_odom.twist.covariance[7]  = 0.05  # vy
        ego_odom.twist.covariance[35] = 0.0   # vaz     

        self.ego_odom_pub.publish(ego_odom)
        
        # publish pose
        pose_msg = PoseStamped()
        pose_msg.header = ego_odom.header
        pose_msg.pose = ego_odom.pose.pose
        self.ego_pose_pub.publish(pose_msg)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = euler.euler2quat(
                0.0, 0.0, self.opp_pose[2], axes='sxyz')
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(
                0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = float(self.scan_distance_to_base_link or 0.27)
        ego_scan_ts.transform.translation.y = 0.0
        ego_scan_ts.transform.translation.z = 0.0
        ego_scan_ts.transform.rotation.w = 1.0 # 단위 회전
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        self.br.sendTransform(ego_scan_ts)

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = float(self.scan_distance_to_base_link or 0.27)
            opp_scan_ts.transform.translation.y = 0.0
            opp_scan_ts.transform.translation.z = 0.0
            opp_scan_ts.transform.rotation.w = 1.0
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            self.br.sendTransform(opp_scan_ts)



def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    
    executor = MultiThreadedExecutor()
    executor.add_node(gym_bridge)

    try:
        executor.spin()
    except KeyboardInterrupt:
        gym_bridge.get_logger().info('Exiting gym_bridge')
    

    gym_bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
