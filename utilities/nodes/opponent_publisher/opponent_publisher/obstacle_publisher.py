import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from tf_transformations import quaternion_from_euler
import numpy as np
import math
from frenet_conversion.frenet_converter import FrenetConverter

class Opponent_Publisher(Node):
    def __init__(self):
        super().__init__('opponent_follower')

        self.declare_parameter('lookahead_s', 0.5)
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('wheelbase', 0.2)
        self.declare_parameter('odom_topic', '/opp_racecar/odom')

        self.lookahead_s = self.get_parameter('lookahead_s').value
        self.speed = self.get_parameter('speed').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.global_waypoints = None
        self.track_length = None
        self.converter = None
        self.current_s = None

        self.current_x = None
        self.current_y = None
        self.current_theta = None

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)

        self.create_subscription(WpntArray, '/global_waypoints', self.waypoints_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)

        self.timer = self.create_timer(1/3, self.timer_cb)
        self.get_logger().info("Opponent Publisher ready")


        self.get_waypoints_flag = False


    def waypoints_cb(self, msg: WpntArray):
        if self.get_waypoints_flag:
            return
        else:
            self.global_waypoints = msg
            self.track_length = msg.wpnts[-1].s_m
            waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in msg.wpnts])
            self.converter = FrenetConverter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])
            self.get_logger().info(f"Waypoints loaded, track length = {self.track_length:.2f} m")
            self.get_waypoints_flag = True

    def odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        self.current_x = pose.position.x
        self.current_y = pose.position.y
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
        self.current_theta = math.atan2(siny_cosp, cosy_cosp)

        if self.converter is not None and self.track_length is not None:
            s_arr, _ = self.converter.get_frenet([self.current_x], [self.current_y])
            self.current_s = float(s_arr[0]) % self.track_length

    def timer_cb(self):
        if self.global_waypoints is None or self.converter is None or self.current_s is None:
            return

        target_s = (self.current_s + self.lookahead_s) % self.track_length
        target_xy = self.converter.get_cartesian([target_s], [0.0])
        tx, ty = float(target_xy[0][0]), float(target_xy[1][0])

        lx, ly = tx - self.current_x, ty - self.current_y
        local_x = math.cos(-self.current_theta) * lx - math.sin(-self.current_theta) * ly
        local_y = math.sin(-self.current_theta) * lx + math.cos(-self.current_theta) * ly

        if abs(local_x) < 1e-6 and abs(local_y) < 1e-6:
            steering = 0.0
        else:
            alpha = math.atan2(local_y, local_x)
            Ld = math.hypot(local_x, local_y)
            steering = math.atan2(2 * self.wheelbase * math.sin(alpha), Ld) if Ld > 1e-6 else 0.0 

        ack = AckermannDriveStamped()
        ack.header.stamp = self.get_clock().now().to_msg()
        ack.header.frame_id = 'base_link'
        ack.drive.speed = self.speed
        ack.drive.steering_angle = steering
        self.drive_pub.publish(ack)


    def publish_pose_and_odom(self, x, y, theta, speed):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        q = quaternion_from_euler(0, 0, theta)
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = speed
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Opponent_Publisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()