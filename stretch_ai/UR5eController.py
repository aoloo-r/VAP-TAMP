#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import rospy
import moveit_commander
import geometry_msgs.msg
import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from moveit_commander.conversions import pose_to_list
from control_msgs.msg import *
from geometry_msgs.msg import Pose, PoseStamped, Transform
from trajectory_msgs.msg import *
import tf


GRIPPER_WIDTH_M = 0.087

# ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
INITIAL_JOINT = [0.20, -2.19, 2.04, -1.42, -1.57, 0.19]
JOINT_FOR_SEARCH = [0, -0.9, 0.5, -1.57, -1.57, 0]
INITIAL_JOINT_FOR_PICKUP = [0, -0.9, 0.5, -1.2, -1.57, 0]
JOINT_FOR_SEARCH_KNIFE = [0.26, -0.94, 0.81, -1.72, -1.64, 0.34]
INITIAL_JOINT_FOR_PICKUP_KNIFE = [0.26, -0.94, 0.81, -1.2, -1.64, 0.34]
Z_OFFSET_FROM_TABLE = 1.1
Z_OFFSET_FROM_OBJECT = 0.97
#Z_offset = 0.2
Z_offset = 0.03


class RobotiqGripper(object):
    def __init__(self):
        # self.cur_status = None
        # self.status_sub = rospy.Subscriber('Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self._status_cb)
        self.cmd_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=10)

    # def _status_cb(self, msg):
    #     self.cur_status = msg
        
    # def wait_for_connection(self, timeout=-1):
    #     rospy.sleep(0.1)
    #     r = rospy.Rate(30)
    #     start_time = rospy.get_time()
    #     while not rospy.is_shutdown():
    #         if (timeout >= 0. and rospy.get_time() - start_time > timeout):
    #             return False
    #         if self.cur_status is not None:
    #             return True
    #         r.sleep()
    #     return False

    # def is_ready(self):
    #     return self.cur_status.gSTA == 3 and self.cur_status.gACT == 1

    # def is_reset(self):
    #     return self.cur_status.gSTA == 0 or self.cur_status.gACT == 0

    # def is_moving(self):
    #     return self.cur_status.gGTO == 1 and self.cur_status.gOBJ == 0

    # def is_stopped(self):
    #     return self.cur_status.gOBJ != 0

    # def object_detected(self):
    #     return self.cur_status.gOBJ == 1 or self.cur_status.gOBJ == 2

    # def get_fault_status(self):
    #     return self.cur_status.gFLT

    # def get_pos(self):
    #     po = self.cur_status.gPO
    #     return np.clip(GRIPPER_WIDTH_M/(13.-230.)*(po-230.), 0, GRIPPER_WIDTH_M)

    # def get_req_pos(self):
    #     pr = self.cur_status.gPR
    #     return np.clip(GRIPPER_WIDTH_M/(13.-230.)*(pr-230.), 0, GRIPPER_WIDTH_M)

    # def is_closed(self):
    #     return self.cur_status.gPO >= 230

    # def is_opened(self):
    #     return self.cur_status.gPO <= 13

    # # in mA
    # def get_current(self):
    #     return self.cur_status.gCU * 0.1

    # # if timeout is negative, wait forever
    # def wait_until_stopped(self, timeout=-1):
    #     r = rospy.Rate(30)
    #     start_time = rospy.get_time()
    #     while not rospy.is_shutdown():
    #         if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
    #             return False
    #         if self.is_stopped():
    #             return True
    #         r.sleep()
    #     return False

    # def wait_until_moving(self, timeout=-1):
    #     r = rospy.Rate(30)
    #     start_time = rospy.get_time()
    #     while not rospy.is_shutdown():
    #         if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
    #             return False
    #         if not self.is_stopped():
    #             return True
    #         r.sleep()
    #     return False

    def reset(self):
        cmd = outputMsg.Robotiq2FGripper_robot_output()
        cmd.rACT = 0
        self.cmd_pub.publish(cmd)

    def activate(self, timeout=-1):
        cmd = outputMsg.Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rGTO = 1
        # cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        self.cmd_pub.publish(cmd)
        # r = rospy.Rate(30)
        # start_time = rospy.get_time()
        # while not rospy.is_shutdown():
        #     if timeout >= 0. and rospy.get_time() - start_time > timeout:
        #         return False
        #     if self.is_ready():
        #         return True
        #     r.sleep()
        # return False

    def auto_release(self):
        cmd = outputMsg.Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rATR = 1
        self.cmd_pub.publish(cmd)

    ##
    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, GRIPPER_WIDTH_M]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, block=False, timeout=-1):
        cmd = outputMsg.Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13.-230.)/GRIPPER_WIDTH_M * pos + 230., 0, 255))
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        self.cmd_pub.publish(cmd)
        # rospy.sleep(0.1)
        # if block:
        #     if not self.wait_until_moving(timeout):
        #         return False
        #     return self.wait_until_stopped(timeout)
        # return True

    def deactivate(self, block=False, timeout=-1):
        cmd = outputMsg.Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rGTO = 0
        self.cmd_pub.publish(cmd)
        # rospy.sleep(0.1)
        # if block:
        #     return self.wait_until_stopped(timeout)
        # return True

    def open(self, vel=0.1, force=100, block=False, timeout=-1):
        # if self.is_opened():
        #     return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.1, force=100, block=False, timeout=-1):
        # if self.is_closed():
        #     return True
        return self.goto(-1.0, vel, force, block=block, timeout=timeout)


class UR5eController(object):
    def __init__(self):
        self.previous_pose = None
        
        group_name = "manipulator"
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        
        # display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
        #                                             moveit_msgs.msg.DisplayTrajectory,
        #                                             queue_size=20)
        
        self.gripper = RobotiqGripper()
        self.gripper.activate()
        rospy.sleep(0.1)
        # success = self.gripper.wait_for_connection(timeout=60)
        # if not success:
        #     rospy.loginfo("Could not connect to the gripper.")
        # if self.gripper.is_reset():
        #     self.gripper.reset()
        #     self.gripper.activate()
        
        self.listener = tf.TransformListener()
        self.rate = rospy.Rate(10.0)

        # Process tracking
        self.camera_process = None
        self.gripper_process = None


    def _all_close(self, goal, actual, tolerance):
        """
          Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
          @param: goal       A list of floats, a Pose or a PoseStamped
          @param: actual     A list of floats, a Pose or a PoseStamped
          @param: tolerance  A float
          @returns: bool
        """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self._all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self._all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def _move_to_target_by_joint(self, target_joint):
        """
        Move to the target pose by using joint
        Args:
            target_joint (list[int, int, int, int, int, int])
        """
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.move_group.go(target_joint, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return self._all_close(target_joint, current_joints, 0.01)

    def _move_to_target_by_pose(self, target_pose):
        current_pose = self.move_group.get_current_pose().pose
        # set orientation to be top-down grasp
        target_pose.orientation = current_pose.orientation
        
        print("====================== Current Position =======================")
        print(current_pose.position)
        print("====================== Current Orientation ====================")
        print(current_pose.orientation)
        
        print("====================== Target Position ========================")
        print(target_pose.position)
        print("====================== Target Orientation =====================")
        print(target_pose.orientation)
        
        self.move_group.set_pose_target(target_pose)

        ## Now, we call the planner to compute the plan and execute it.
        plan = self.move_group.go(wait=True)

        print(plan)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        return self._all_close(target_pose, current_pose, 0.01)
        
    def _get_object_position(self):
        rospy.sleep(3.0)
        while not rospy.is_shutdown():
            try:
                position, orientation = self.listener.lookupTransform('/ur5e_base_link', '/camera_marker', rospy.Time(0))
                print('position:{}'.format(position))
                print('oritention:{}'.format(orientation))
                if position:
                    break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        return position, orientation
    
    def log(self, success, message_template):
        if success:
            current_pose = self.move_group.get_current_pose().pose
            rospy.loginfo("{} is done!".format(message_template))
            rospy.loginfo("{}".format(current_pose))
        else:
            rospy.logwarn("{} is failed...".format(message_template))
            
    def move_to_init(self):      
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
             
    def pickup(self, object_info, retry=3):
        """
        pick up an object
        
        Args: 
            object_info (dict): Object information that is a form of:
                {
                    "object_name": "bowl",
                    "object_target_xy": [x, y]
                    "object_offset: [-0.05, -0.0, 0.13]
                }
        """
        
        offset = object_info["offset"]
            
        #################################################
        # Open the gripper to grasp the object
        #################################################
        self.gripper.open()
        rospy.sleep(1)
        self.gripper.open()
        
            
        #################################################
        # Move to the initial pose
        #################################################            
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_joint(INITIAL_JOINT)
            self.log(success, message_template="move to the initial pose")
            count += 1
            rospy.sleep(1)
        print("moved to init pose")
        #################################################
        # Move to the pose to observe objects on a table
        #################################################
        success = self._move_to_target_by_joint(JOINT_FOR_SEARCH)
        self.log(success, message_template="move to the pose for observation")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_joint(JOINT_FOR_SEARCH)
            self.log(success, message_template="move to the pose for observation")
            count += 1
            rospy.sleep(1)
        print("moved to observation pose")
        #################################################
        # Get object position in real time
        #################################################
        position, orientation = self._get_object_position()
        rospy.sleep(1)
        
        #################################################
        # Move to the pose for grasping (look down)
        #################################################
        success = self._move_to_target_by_joint(INITIAL_JOINT_FOR_PICKUP)
        self.log(success, message_template="move to the pose for looking down")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_joint(INITIAL_JOINT_FOR_PICKUP)
            self.log(success, message_template="move to the pose for looking down")
            count += 1
            rospy.sleep(1)
        
        #################################################
        # Compute the target pose corresponding to 
        # above the object and move to the pose
        #################################################
        target_pose= self.move_group.get_current_pose().pose
        target_pose.position.x = -position[1] - offset[0]
        target_pose.position.y = position[0] - offset[1]
        target_pose.position.z = Z_OFFSET_FROM_TABLE
        
        success = self._move_to_target_by_pose(target_pose)
        self.log(success, message_template="move to above the object")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_pose(target_pose)
            self.log(success, message_template="move to above the object")
            count += 1
            rospy.sleep(1)
        print("moved to target pose")
        #################################################
        # Compute the target pose corresponding to 
        # the grasping point and move to the pose
        #################################################
        current_pose = self.move_group.get_current_pose().pose
        target_pose = current_pose
        target_pose.position.z = Z_OFFSET_FROM_TABLE - offset[2]

        success = self._move_to_target_by_pose(target_pose)
        self.log(success, message_template="move to the grasp point")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_pose(target_pose)
            self.log(success, message_template="move to the grasp point")
            count += 1
            rospy.sleep(1)
        
        #################################################
        # Close the gripper to grasp the object
        #################################################
        self.gripper.close()
        rospy.sleep(1)
        
        #################################################
        # Remember the current pose to place the object
        #################################################
        self.previous_pose = self.move_group.get_current_pose().pose
        print("place joint reference", self.move_group.get_current_joint_values())
        
        #################################################
        # Move to the initial pose
        #################################################   
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_joint(INITIAL_JOINT)
            self.log(success, message_template="move to the initial pose")
            count += 1
            rospy.sleep(1)
        
        print("moved to init pose")
        
    def place(self, object_info, retry=3):
        """
        place an object
        
        Args: 
            object_info (dict): Object information that is a form of:
                {
                    "object_name": "bowl",
                    "object_target_xy": [x, y]
                }
        """
        
        #################################################
        # Open the gripper to grasp the object
        #################################################
        self.gripper.close()
        rospy.sleep(1)
            
        #################################################
        # Move to the previous pose
        #################################################    
        # TODO
        #target_pose= self.previous_pose commented by Z and the next two lines are added by Z
        
        target_pose = Pose()
        target_pose.position.x = object_info["target"]["x"]
        target_pose.position.y = object_info["target"]["y"]
        target_pose.position.z = object_info["target"]["z"]
        target_pose.orientation.x = object_info["target"]["ox"]
        target_pose.orientation.y = object_info["target"]["oy"]
        target_pose.orientation.z = object_info["target"]["oz"]
        
        offset = object_info["offset"]
        
        target_pose.position.x -= object_info["offset"][0]
        target_pose.position.y -= object_info["offset"][1]
        target_pose.position.z -= object_info["offset"][2]
        
        success = self._move_to_target_by_pose(target_pose)
        self.log(success, message_template="move to pose for place")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_pose(target_pose)
            self.log(success, message_template="move to pose for place")
            count += 1
            rospy.sleep(1)
                
        #################################################
        # Close the gripper to grasp the object
        #################################################
        self.gripper.open()
        rospy.sleep(1)
        
        ##############################################
        target_pose.position.z -= offset[2]###
        # Move to the initial pose
        #################################################   
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:            
            success = self._move_to_target_by_joint(INITIAL_JOINT)
            self.log(success, message_template="move to the initial pose")
            count += 1
            rospy.sleep(1)
        
        
             
    def place_to_marker(self, object_info, retry=3):
        """
        place an object to marker

        Args:
            object_info (dict): Object information that is a form of:
                {
                    "object_name": "bowl",
                    "object_target_xy": [x, y]
                    "object_offset: [-0.05, -0.0, 0.13]
                }
        """

        offset = object_info["offset"]

        #################################################
        # Ensure gripper is closed (holding object)
        #################################################
        self.gripper.close()
        rospy.sleep(1)
        self.gripper.close()


        #################################################
        # Move to the initial pose
        #################################################
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_joint(INITIAL_JOINT)
            self.log(success, message_template="move to the initial pose")
            count += 1
            rospy.sleep(1)
        print("moved to init pose")
        #################################################
        # Move to the pose to observe marker on a table
        #################################################
        success = self._move_to_target_by_joint(JOINT_FOR_SEARCH)
        self.log(success, message_template="move to the pose for observation")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_joint(JOINT_FOR_SEARCH)
            self.log(success, message_template="move to the pose for observation")
            count += 1
            rospy.sleep(1)
        print("moved to observation pose")
        #################################################
        # Get object position in real time
        #################################################
        position, orientation = self._get_object_position()
        rospy.sleep(1)

        #################################################
        # Move to the pose for placement (look down)
        #################################################
        success = self._move_to_target_by_joint(INITIAL_JOINT_FOR_PICKUP)
        self.log(success, message_template="move to the pose for looking down")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_joint(INITIAL_JOINT_FOR_PICKUP)
            self.log(success, message_template="move to the pose for looking down")
            count += 1
            rospy.sleep(1)

        #################################################
        # Compute the target pose corresponding to
        # above the object and move to the pose
        #################################################
        target_pose= self.move_group.get_current_pose().pose
        target_pose.position.x = -position[1] - offset[0]
        target_pose.position.y = position[0] - offset[1]
        target_pose.position.z = Z_OFFSET_FROM_TABLE

        success = self._move_to_target_by_pose(target_pose)
        self.log(success, message_template="move to above the object")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_pose(target_pose)
            self.log(success, message_template="move to above the object")
            count += 1
            rospy.sleep(1)
        print("moved to target pose")
        #################################################
        # Compute the target pose corresponding to
        # the placement point and move to the pose
        #################################################
        current_pose = self.move_group.get_current_pose().pose
        target_pose = current_pose
        target_pose.position.z = Z_OFFSET_FROM_TABLE - offset[2]

        success = self._move_to_target_by_pose(target_pose)
        self.log(success, message_template="move to the place point")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_pose(target_pose)
            self.log(success, message_template="move to the place point")
            count += 1
            rospy.sleep(1)

        #################################################
        # Open the gripper to release the object
        #################################################
        self.gripper.open()
        rospy.sleep(1)

        #################################################
        # Remember the current pose to place the object
        #################################################
        self.previous_pose = self.move_group.get_current_pose().pose

        #################################################
        # Move to the initial pose
        #################################################
        success = self._move_to_target_by_joint(INITIAL_JOINT)
        self.log(success, message_template="move to the initial pose")
        rospy.sleep(1)
        count = 1
        while not success and count < retry:
            success = self._move_to_target_by_joint(INITIAL_JOINT)
            self.log(success, message_template="move to the initial pose")
            count += 1
            rospy.sleep(1)

        print("moved to init pose")
    