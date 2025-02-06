#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys, select, termios, tty
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
# from std_msgs.msg import Float64MultiArray

# Define the keys and their corresponding joint indices
joint_names = ['joint_1', 'joint_2', 'joint_3','joint_4', 'joint_5', 'joint_6','joint_7']
key_mapping = {
    'a': 0, 'z': 0,  # Increase/decrease joint_0
    's': 1, 'x': 1,  # Increase/decrease joint_1
    'd': 2, 'c': 2,  # Increase/decrease joint_2
    'f': 3, 'v': 3,  # Increase/decrease joint_3 
    'g': 4, 'b': 4,  # Increase/decrease joint_0
    'h': 5, 'n': 5,  # Increase/decrease joint_1
    'j': 6, 'm': 6,  # Increase/decrease joint_2
}

# Initial joint angles
def angles(msg: JointTrajectoryControllerState):
    global joint_angles
    joint_angles = [msg.actual.positions[0],msg.actual.positions[1],msg.actual.positions[2],msg.actual.positions[3],msg.actual.positions[4],msg.actual.positions[5],msg.actual.positions[6]]

joint_step = 0.01  # Step size for changing joint angles

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    rospy.init_node('teleop_gripper')
    pub = rospy.Publisher('/my_gen3/gen3_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Teleoperation node started. Use keys to control the gripper.")

    while not rospy.is_shutdown():
        keys = ''
        while True:
            key = get_key()
            if key == '':
                break
            keys += key

        command_executed = False

        for key in keys:
            if key in key_mapping:
                joint_index = key_mapping[key]
                if key in 'asdfghj':  # Increase joint angle
                    joint_angles[joint_index] += joint_step
                elif key in 'zxcvbnm':  # Decrease joint angle
                    joint_angles[joint_index] -= joint_step

                # Clamp the angles within a range if needed

                command_executed = True
                rospy.loginfo(f"Key '{key}' pressed. Adjusting Finger {joint_index} to {joint_angles[joint_index]} radians.")

        if '\x03' in keys:  # Ctrl+C
            rospy.loginfo("Shutting down teleoperation node.")
            break

        # Create the JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = joint_names
        
        # Create a single point in the trajectory
        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(0.1)  # Small duration to indicate immediate movement
        
        trajectory_msg.points = [point]

        # Publish the trajectory message
        pub.publish(trajectory_msg)
        
        if command_executed:
            rospy.loginfo(f"Published joint angles: {joint_angles}")

        rate.sleep()

if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    try:
        rospy.Subscriber('/my_gen3/gen3_joint_trajectory_controller/state',JointTrajectoryControllerState,angles)
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)