#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import tf.transformations
import pinocchio as pin
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from IK_gazebo import (
    model,
    data,
    q_max,
    q_min,
    damping,
    dt,
    solve_qp,
)

import matplotlib.pyplot as plt
from std_msgs.msg import Float32, Float64MultiArray


class VisualServoing:
    def __init__(self):
        rospy.init_node("visual_servoing", anonymous=True)
        self.listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.pub = rospy.Publisher(
            "/body_controller/command", JointTrajectory, queue_size=10
        )

        self.q = np.zeros(8)
        self.errors = []

        self.old_gray = None
        self.mask = None
        self.points = []
        self.p0 = None
        self.colors = np.random.randint(0, 255, (100, 3))
        self.z = 0
        self.s = np.zeros((4, 2))
        self.end_effector_frame = model.getFrameId("camera_optical_link1")

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        # rospy.Subscriber("/camera_gripper_left/image_raw", Image, self.process_image)
        rospy.Subscriber("/body_controller/state", JointTrajectoryControllerState, self.callback)
        rospy.Subscriber("/image_depth", Float32, self.depth_callback)
        rospy.Subscriber("/detected_corners", Float64MultiArray, self.corners)
        self.setup_transforms()

    def setup_transforms(self):
        try:
            self.listener.waitForTransform(
                "camera_link1", "Link_6", rospy.Time(0), rospy.Duration(1.0)
            )
            (trans, rot) = self.listener.lookupTransform(
                "camera_link1", "Link_6", rospy.Time(0)
            )
            self.Re_c = tf.transformations.quaternion_matrix(rot)[:3, :3]
            self.de_c = trans
            self.S_de_c = np.matrix(
                [
                    [0, -self.de_c[2], self.de_c[1]],
                    [self.de_c[2], 0, -self.de_c[0]],
                    [-self.de_c[1], self.de_c[0], 0],
                ]
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logerr("Could not get transform between camera_link2 and Link_6")

        try:
            self.listener.waitForTransform(
                "Link_6", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            (transb, rotb) = self.listener.lookupTransform(
                "Link_6", "base_link", rospy.Time(0)
            )
            self.Rb_e = tf.transformations.quaternion_matrix(rotb)[:3, :3]
            self.transb = transb
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logerr("Could not get transform between Link_6 and base_link")

    def publish_joint_angles(self, joint_angles):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            "Joint_1",
            "Joint_2",
            "Joint_3",
            "Joint_4",
            "Joint_5",
            "Joint_6",
        ]
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(0.1)
        trajectory_msg.points = [point]
        self.pub.publish(trajectory_msg)

    def callback(self, msg: JointTrajectoryControllerState):
        n = msg.actual.positions
        for i in range(0, len(n)):
            self.q[i] = n[i]
        # print("current angles", self.q)

    def depth_callback(self, msg: Float32):
        if msg.data > 0:
            self.z = msg.data

    def corners(self, msg: Float64MultiArray):
        for i in range(0, 4):
            self.s[i] = [msg.data[2 * i], msg.data[(2 * i) + 1]]
        # print("current coordinates" ,self.s)

    def compute_jacobian(self, ee_b):
        pin.forwardKinematics(model, data, self.q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, self.q)

        J = pin.computeFrameJacobian(
            model, data, self.q, self.end_effector_frame, pin.ReferenceFrame.LOCAL
        )
        desired_twist = ee_b.A1

        H = J.T @ J + damping * np.eye(model.nv)
        g = J.T @ desired_twist

        theta_dot_max = 1.0 * np.ones(model.nv)
        theta_dot_min = -1.0 * np.ones(model.nv)

        q_upper_violation = (q_max - self.q) / dt
        q_lower_violation = (q_min - self.q) / dt

        C = np.vstack(
            [np.eye(model.nv), -np.eye(model.nv), np.eye(model.nv), -np.eye(model.nv)]
        )
        b = np.hstack(
            [theta_dot_min, -theta_dot_max, q_lower_violation, -q_upper_violation]
        )

        theta_dot = solve_qp(H, g, C.T, b)[0]
        angles = pin.integrate(model, self.q, theta_dot * 0.01)
        self.publish_joint_angles(angles)

    def compute_ee_velocity_base(self, ee_b, rotation_matrix):
        zero_matrix = np.zeros((3, 3))
        ee_b = (
            np.block([[rotation_matrix, zero_matrix], [zero_matrix, rotation_matrix]])
            @ ee_b
        )
        self.compute_jacobian(ee_b)

    def compute_ee_velocity(self, camera_velocity, rotation_matrix, skew_matrix):
        zero_matrix = np.zeros((3, 3))
        m = np.block(
            [
                [rotation_matrix, skew_matrix @ rotation_matrix],
                [zero_matrix, rotation_matrix],
            ]
        )
        ee_b = m @ camera_velocity
        self.compute_ee_velocity_base(ee_b, self.Rb_e)

    def compute_interaction_matrix(self, points, focal_length, depth):
        n = len(points)
        Le = []
        for i in range(n):
            u = points[i][0]
            v = points[i][1]
            Lp = np.matrix(
                [
                    [
                        -focal_length / depth,
                        0,
                        u / depth,
                        u * v / focal_length,
                        -(focal_length**2 + u**2) / focal_length,
                        v,
                    ],
                    [
                        0,
                        -focal_length / depth,
                        v / depth,
                        (focal_length**2 + v**2) / focal_length,
                        -u * v / focal_length,
                        -u,
                    ],
                ]
            )
            Le.append(Lp)
        return np.vstack(Le)

    def calculate_area(self, points):
        if len(points) < 4:
            return 0
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]
        return abs((y3 - y1) * (x2 - x1))

    def final_velocity(self):
        if self.z > 0:
            Kwz = 0.01
            Kvz = 0.002
            Kp = 0.15

            points = self.s
            sigma = (self.calculate_area(points)) ** 0.5

            sd = np.array([[415, 510], [560, 510], [415, 666], [560, 666]])
            area = (self.calculate_area(sd)) ** 0.5

            error = sd - points
            error = error.reshape(8, 1)
            wz = 0
            vz = Kvz * (area - sigma)
            zi_z = np.array([[vz]])
            zi_z.reshape((1,1))

            width = 800
            fov = 1.3962634
            focal_length = width / (2 * np.tan(fov / 2))
            depth = self.z
            rospy.loginfo(f"depth is {depth}")
        
            Le = self.compute_interaction_matrix(points, focal_length, depth)

            Lz = Le[:, 2]  # 8x1
            Lxy = np.hstack([Le[:, 0:2], Le[:, 3:6]])  # 8x5
            L_in = np.linalg.pinv(Lxy)  # 5x8
            z_mat = Lz@zi_z  # 8x1
            print(z_mat.shape)
            s_dot = -(Kp * error)
            post = s_dot - z_mat
            zi_xy = L_in @ post
            zi_cam = np.vstack(
                [zi_xy[0],zi_xy[1],zi_z[0], zi_xy[2],zi_xy[3],zi_xy[4]]
            )
            print(zi_cam)
            print(np.linalg.norm(zi_cam))
            if np.linalg.norm(zi_cam)>0.1:
                self.compute_jacobian(zi_cam)
                self.errors.append(np.linalg.norm(error))
                print("error ", np.linalg.norm(error))
            else:
                plt.plot(self.errors)
                plt.title("Error over Time")
                plt.xlabel("Time Step")
                plt.ylabel("Error (Euclidean Distance)")
                plt.show()
                rospy.signal_shutdown("Reached position")
                cv2.destroyAllWindows()

    # def select_points(self):
    #     for _ in range(3):
    #         x = int(input("Enter x coordinate: "))
    #         y = int(input("Enter y coordinate: "))
    #         self.points.append((x, y))
    #         print(f"Point selected: {x}, {y}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.setup_transforms()
            self.final_velocity()
            rate.sleep()
        cv2.destroyAllWindows()
        plt.plot(self.errors)
        plt.title("Error over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Error (Euclidean Distance)")
        plt.show()


if __name__ == "__main__":
    visual_servoing = VisualServoing()
    visual_servoing.run()