from __future__ import annotations
import rclpy
from rclpy.node import Node

from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from frenet_conversion.frenet_converter import FrenetConverter
import numpy as np
from tf_transformations import euler_from_quaternion


class FrenetOdomRepublisher(Node):
    def __init__(self):
        super().__init__('frenet_odom_republisher')

        # ----------------- General parameters -----------------
        self.has_global_trajectory = False
        self.latest_pose: PoseStamped | None = None

        # ----------------- Publishers -----------------
        # Frenet 토픽
        self.frenet_odom_pub = self.create_publisher(
            Odometry, '/car_state/frenet/odom', 10)
        self.frenet_pose_pub = self.create_publisher(
            PoseStamped, '/car_state/frenet/pose', 10)

        # 최종 카테시안 상태 토픽
        self.state_odom_pub = self.create_publisher(
            Odometry, '/car_state/odom', 10)
        self.state_pose_pub = self.create_publisher(
            PoseStamped, '/car_state/pose', 10)

        # ----------------- Subscribers -----------------
        self.create_subscription(
            WpntArray, '/global_waypoints',
            self.global_trajectory_callback, 10)

        # Particle Filter가 퍼블리시하는 Pose
        self.create_subscription(
            PoseStamped, '/pf/viz/inferred_pose',
            self.pose_callback, 10)

        # EKF Odometry (속도용)
        self.create_subscription(
            Odometry, '/early_fusion/odom',
            self.odom_callback, 10)

    # ============ Callbacks ============

    def global_trajectory_callback(self, msg: WpntArray):
        wps = msg.wpnts
        xs = [w.x_m for w in wps]
        ys = [w.y_m for w in wps]
        psis = [w.psi_rad for w in wps]

        self.converter = FrenetConverter(
            np.array(xs), np.array(ys), np.array(psis))
        self.has_global_trajectory = True

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg

    def odom_callback(self, odom_msg: Odometry):
        # ---------- pose 준비 ----------
        if self.latest_pose is None or not self.has_global_trajectory:
            return

        # 최신 PF pose에서 위치/자세 추출
        pose_msg = self.latest_pose
        pose = pose_msg.pose.position
        quat = pose_msg.pose.orientation
        theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

        # ---------- velocity 준비 (EKF 속도) ----------
        vel = odom_msg.twist.twist  # linear + angular 모두 포함됨

        # ---------- 최종 카테시안 상태 퍼블리시 ----------
        # /car_state/pose
        carstate_pose = PoseStamped()
        carstate_pose.header = pose_msg.header           # 보통 frame_id="map"
        carstate_pose.pose = pose_msg.pose               # PF Pose 그대로 사용
        self.state_pose_pub.publish(carstate_pose)

        # /car_state/odom  (Pose=PF, Twist=EKF)
        carstate_odom = Odometry()
        carstate_odom.header = pose_msg.header
        carstate_odom.child_frame_id = 'base_link'       
        carstate_odom.pose.pose = pose_msg.pose
        carstate_odom.twist.twist = vel
        self.state_odom_pub.publish(carstate_odom)

        # ---------- Frenet 변환 ----------
        fr_pos = self.converter.get_frenet([pose.x], [pose.y])
        fr_vel = self.converter.get_frenet_velocities(vel.linear.x, vel.linear.y, theta)

        # ---------- Frenet Odometry 메시지 ----------
        fr_odom = Odometry()
        fr_odom.header = pose_msg.header
        # 기존 코드의 child_frame_id 동작 유지 (가까운 인덱스를 문자열로)
        fr_odom.child_frame_id = str(
            self.converter.get_closest_index([pose.x], [pose.y])[0])

        fr_odom.pose.pose.position.x = fr_pos[0, 0]
        fr_odom.pose.pose.position.y = fr_pos[1, 0]
        # 프레네 포즈의 orientation은 소비자에 따라 의미가 모호할 수 있으나 기존 동작 유지
        fr_odom.pose.pose.orientation = quat
        fr_odom.twist.twist.linear.x = fr_vel[0][0]
        fr_odom.twist.twist.linear.y = fr_vel[1][0]

        # ---------- Frenet PoseStamped 메시지 ----------
        fr_pose = PoseStamped()
        fr_pose.header = pose_msg.header
        fr_pose.pose.position.x = fr_pos[0, 0]
        fr_pose.pose.position.y = fr_pos[1, 0]
        # 필요 시 orientation=0으로 명시할 수도 있으나 기존 유지

        # ---------- Publish ----------
        self.frenet_odom_pub.publish(fr_odom)
        self.frenet_pose_pub.publish(fr_pose)


def main(args=None):
    rclpy.init(args=args)
    node = FrenetOdomRepublisher()
    rclpy.spin(node)
    node.destroy_node()