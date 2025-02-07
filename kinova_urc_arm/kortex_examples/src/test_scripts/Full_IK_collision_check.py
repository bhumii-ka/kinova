#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import os
from pinocchio.visualize import MeshcatVisualizer
import time

# Load the robot model
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the URDF file relative to the script's location
urdf_path = os.path.join(script_dir, "../../../kortex_description/robots/gen3.urdf")  # Replace with the path to your URDF file
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
geom_model = pin.buildGeomFromUrdf(
    model, urdf_path, pin.GeometryType.COLLISION
)
geom_model.addAllCollisionPairs()
print("num collision pairs:", len(geom_model.collisionPairs))
geom_data = pin.GeometryData(geom_model)

# Visualization setup
visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
viz = MeshcatVisualizer(model, geom_model, visual_model)
viz.initViewer(open=True, loadModel=True)
viz.loadViewerModel()

# End-effector frame
end_effector_frame = model.getFrameId("end_effector_link")  # Replace with your end-effector frame name

# Initialize joint configuration
q = pin.neutral(model)
viz.display(q)

# Velocity scaling
velocity_scale = 0.1  # Adjust this for desired velocity magnitude
dt = 0.05  # Time step for integration
damping = 1e-6  # Regularization factor

# Joint limits
q_min = model.lowerPositionLimit
q_max = model.upperPositionLimit

# Key-to-twist mapping
key_twist_mapping = {
    Qt.Key_W: np.array([velocity_scale, 0, 0, 0, 0, 0]),  # Forward
    Qt.Key_S: np.array([-velocity_scale, 0, 0, 0, 0, 0]), # Backward
    Qt.Key_A: np.array([0, velocity_scale, 0, 0, 0, 0]), # Left
    Qt.Key_D: np.array([0, -velocity_scale, 0, 0, 0, 0]), # Right
    Qt.Key_Q: np.array([0, 0, velocity_scale, 0, 0, 0]), # Up
    Qt.Key_E: np.array([0, 0, -velocity_scale, 0, 0, 0]), # Down
    Qt.Key_J: np.array([0, 0, 0, -velocity_scale, 0, 0]), # Rotate around x towards left
    Qt.Key_L: np.array([0, 0, 0, velocity_scale, 0, 0]), # Rotate around x towards right
    Qt.Key_I: np.array([0, 0, 0, 0, velocity_scale, 0]), # Rotate around y down
    Qt.Key_K: np.array([0, 0, 0, 0, -velocity_scale, 0]), # Rotate around y up
    Qt.Key_U: np.array([0, 0, 0, 0, 0, velocity_scale]), # Yaw left
    Qt.Key_O: np.array([0, 0, 0, 0, 0, -velocity_scale]), # Yaw right
}

class VelocityIKController(QWidget):
    def __init__(self):
        super().__init__()
        self.pressed_keys = set()
        self.timer = QTimer()
        self.timer.timeout.connect(self.control_loop)
        self.timer.start(int(dt * 100))             # Timeout every 5 ms instead of 50ms
        self.prev_theta_dot = np.zeros(model.nv)
        
    def keyPressEvent(self, event):
        self.pressed_keys.add(event.key())

    def keyReleaseEvent(self, event):
        self.pressed_keys.discard(event.key())

    def compute_desired_twist(self):
        desired_twist = np.zeros(6)
        for key in self.pressed_keys:
            if key in key_twist_mapping:
                desired_twist += key_twist_mapping[key]
        return desired_twist

    def collision_constraint(self, q_new):
        # Compute the distance between collision pairs
        pin.computeCollisions(model, data, geom_model, geom_data, q_new, False)
        distances = []
        for k in range(len(geom_model.collisionPairs)):
            cr = geom_data.collisionResults[k]
            cp = geom_model.collisionPairs[k]
            if cr.isCollision() and cp.second - cp.first != 1:
                distances.append(-1)  # Negative distance indicates collision
            else:
                distances.append(1)
        return np.array(distances)

    def control_loop(self):
        global q

        # Compute forward kinematics and Jacobian
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        J = pin.computeFrameJacobian(model, data, q, end_effector_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Get the desired twist from key input
        desired_twist = self.compute_desired_twist()

        if np.linalg.norm(desired_twist) > 1e-6:  # If there is a desired motion
            # Objective function: minimize the difference between desired and actual twist
            def objective(theta_dot):
                return np.linalg.norm(J @ theta_dot - desired_twist) ** 2 + damping * np.linalg.norm(theta_dot) ** 2

            # Joint velocity limits
            theta_dot_max = 1.0 * np.ones(model.nv)
            theta_dot_min = -1.0 * np.ones(model.nv)

            # Joint position limits
            q_upper_violation = (q_max - q) / dt
            q_lower_violation = (q_min - q) / dt

            # Linear constraints for joint velocity and position limits
            linear_constraints = LinearConstraint(
                np.vstack([np.eye(model.nv), np.eye(model.nv)]),
                np.hstack([theta_dot_min, q_lower_violation]),
                np.hstack([theta_dot_max, q_upper_violation])
            )

            # Nonlinear constraints for collision avoidance
            def collision_constraint_fun(theta_dot):
                q_new = pin.integrate(model, q, theta_dot * dt)
                return self.collision_constraint(q_new)

            nonlinear_constraints = NonlinearConstraint(
                collision_constraint_fun,
                lb=0.0,  # Ensure distances are positive (no collision)
                ub=np.inf
            )

            # Solve the optimization problem
            start = time.perf_counter()
            result = minimize(
                objective,
                x0=self.prev_theta_dot,
                constraints=[linear_constraints, nonlinear_constraints],
                method='SLSQP',
                # options={"maxiter": 10}
            )
            end = time.perf_counter()

            # print(f"Optimization time: {end - start:.3f} s")
            if result.success:
                theta_dot = result.x
                self.prev_theta_dot = theta_dot
                q = pin.integrate(model, q, theta_dot * dt)
                # print(q)
                viz.display(q)

if __name__ == "__main__":
    app = QApplication([])
    controller = VelocityIKController()
    controller.show()
    app.exec_()