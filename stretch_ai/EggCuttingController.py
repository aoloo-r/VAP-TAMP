#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DKPrompt Task Execution Framework
# For VLM-based situation handling and task recovery on Stretch robot with UR5e arm
# Supports: Task 1 (Bottle Collection), Task 2 (Halve an Egg), Task 3 (Firewood Storage)

import rospy
from geometry_msgs.msg import Point, Quaternion

from SegbotController import SegbotController
from UR5eController import UR5eController


class DKPromptExecutor:
    """Execute manipulation tasks with VLM-based situation handling."""

    def __init__(self):
        """Initialize robot controllers."""
        rospy.init_node('dk_prompt_executor', anonymous=False)
        self.base = SegbotController()
        self.arm = UR5eController()

        # Location mappings (AMCL frame)
        self.locations = {
            "loc_main": {"x": -13.1071120064, "y": -15.1124078236},      # Room 1 (Lab) - Task 1 & 2 center
            "loc_room2": {"x": 1.43808, "y": -4.881605},     # Room 2 (TA Area) - bottle 1
            "loc_room3": {"x": -0.26633, "y": -0.252465},    # Room 3 (Conference) - bottle 2, firewood
            "loc_knife": {"x": -6.9837571282, "y": -20.5025550065},  # Knife location (Task 2)
            "loc_egg": {"x": -13.1071120064, "y": -15.1124078236},       # Egg location (Task 2)
            "door_room1_to_room2": {"x": -6.52195, "y": -8.179987}  # Door between Room 1-2 (Task 3)
        }

    def create_pose(self, xy_coords, orientation=(0.0, 0.0, 0.0, 1.0)):
        """Create a pose dictionary from xy coordinates."""
        return {
            "x": xy_coords["x"],
            "y": xy_coords["y"],
            "z": 0.0,
            "ox": orientation[0],
            "oy": orientation[1],
            "oz": orientation[2],
            "ow": orientation[3]
        }

    def task_1_bottle_collection(self):
        """
        Task 1: Bring Empty Bottles
        Collect bottle1 from room2 and bottle2 from room3, deliver to room1
        """
        print("\n" + "="*70)
        print("TASK 1: BOTTLE COLLECTION")
        print("="*70)

        task_1_info = [
            {
                "task_name": "pickup_bottle1",
                "object_name": "bottle1",
                "room": "room2",
                "object_location_xy": [1.43808, -4.881605],
                "offset": [0.05, 0.05, 0.08],
                "robot_pickup_pose": self.create_pose(self.locations["loc_room2"]),
                "pickup_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_pickup": 0.08,
                "surface": "table_room2"
            },
            {
                "task_name": "place_bottle1_at_collection",
                "object_name": "bottle1",
                "room": "room1",
                "object_location_xy": [-13.0133, -15.1409],
                "offset": [0.05, 0.05, 0.08],
                "robot_place_pose": self.create_pose(self.locations["loc_main"]),
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.08,
                "surface": "collection_table"
            },
            {
                "task_name": "pickup_bottle2",
                "object_name": "bottle2",
                "room": "room3",
                "object_location_xy": [-0.26633, -0.252465],
                "offset": [0.05, 0.05, 0.08],
                "robot_pickup_pose": self.create_pose(self.locations["loc_room3"]),
                "pickup_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_pickup": 0.08,
                "surface": "table_room3"
            },
            {
                "task_name": "place_bottle2_at_collection",
                "object_name": "bottle2",
                "room": "room1",
                "object_location_xy": [-13.0133, -15.1409],
                "offset": [0.05, 0.05, 0.08],
                "robot_place_pose": self.create_pose(self.locations["loc_main"]),
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.08,
                "surface": "collection_table"
            }
        ]

        try:
            print("\n[TASK 1] Step 1: Navigate to Room 2 and pick bottle 1")
            self.arm.pickup(task_1_info[0])

            print("\n[TASK 1] Step 2: Navigate to Room 1 and place bottle 1")
            self.arm.place(task_1_info[1])

            print("\n[TASK 1] Step 3: Navigate to Room 3 and pick bottle 2")
            self.arm.pickup(task_1_info[2])

            print("\n[TASK 1] Step 4: Navigate to Room 1 and place bottle 2")
            self.arm.place(task_1_info[3])

            print("\n[SUCCESS] TASK 1 COMPLETED: Both bottles collected and placed in Room 1")
            return True

        except Exception as e:
            print("\n[ERROR] TASK 1 FAILED: {}".format(e))
            return False

    def cut_egg_two_stage(self):
        """
        Egg Cutting Only - Two Stage Cut
        Assumes gripper is already holding the knife closed.
        Halves the egg using a two-stage cutting approach.

        Egg location: Room 1 (Lab) at center of table (-13.0133, -15.1124078236)
        """
        print("\n" + "="*70)
        print("EGG CUTTING SEQUENCE - TWO STAGE")
        print("="*70)

        egg_info = {
            "object_name": "egg",
            "object_target_xy": [-13.1071120064, -15.1124078236],
            "offset": [0.00, 0.00, 0.09],
            "robot_pickup_pose": {
                "x": -13.1071120064, "y": -15.1124078236, "z": 0.0,
                "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0
            },
            "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
            "Z_offset_for_place": 0.09,
            "X_offset_for_place": 0,
            "Y_offset_for_place": 0
        }

        try:
            import time
            from geometry_msgs.msg import Pose

            # Knife pickup functionality commented out
            # print("\n[TASK 2] Step 1: Navigate to knife location and pick knife")
            # self.arm.pickup(task_2_info[0])
            #
            # print("\n[TASK 2] Waiting 60 seconds for robot to return to starting position...")
            # import time
            # time.sleep(60.0)
            # print("[TASK 2] Robot returned to starting position, proceeding with egg cutting")

            print("\n[EGG CUTTING] Step 0: Ensure gripper is CLOSED and holding knife")
            self.arm.gripper.close()
            print("[EGG CUTTING] Gripper CLOSED - knife secured")
            time.sleep(0.5)

            print("\n[EGG CUTTING] Step 1: Navigate to egg location using arm marker detection")
            print("[EGG CUTTING] Gripper stays CLOSED throughout")

            # Use pickup logic but without opening gripper
            # First, establish initial pose
            success = self.arm._move_to_target_by_joint([-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603])
            self.arm.move_group.stop()
            print("[EGG CUTTING] At initial pose")
            time.sleep(0.5)

            # Move to observation position
            success = self.arm._move_to_target_by_joint([0, -0.9, 0.5, -1.57, -1.57, 0])
            self.arm.move_group.stop()
            print("[EGG CUTTING] At observation position")
            time.sleep(0.5)

            # Get marker position
            position, orientation = self.arm._get_object_position()
            print("[EGG CUTTING] Marker detected at: {}".format(position))
            time.sleep(0.5)

            # Move to above egg
            offset = egg_info["offset"]
            current_pose = self.arm.move_group.get_current_pose().pose
            target_pose = Pose()
            target_pose.position.x = -position[1] - offset[0]
            target_pose.position.y = position[0] - offset[1]
            target_pose.position.z = 1.1  # Above table
            target_pose.orientation = current_pose.orientation

            self.arm._move_to_target_by_pose(target_pose)
            self.arm.move_group.stop()
            print("[EGG CUTTING] Positioned above egg")
            print("[EGG CUTTING] Gripper CLOSED - knife secured")
            time.sleep(0.5)

            print("\n[EGG CUTTING] Step 2: First cut - lowering knife (~2cm depth)")
            current_pose = self.arm.move_group.get_current_pose().pose
            target_pose = current_pose
            target_pose.position.z = current_pose.position.z - 0.02
            self.arm._move_to_target_by_pose(target_pose)
            print("[EGG CUTTING] First stage cut - knife in contact")
            print("[EGG CUTTING] Gripper CLOSED - holding knife")
            time.sleep(0.5)

            print("\n[EGG CUTTING] Step 3: Retracting knife between cuts")
            current_pose = self.arm.move_group.get_current_pose().pose
            target_pose = current_pose
            target_pose.position.z = current_pose.position.z + 0.02
            self.arm._move_to_target_by_pose(target_pose)
            print("[EGG CUTTING] Knife retracted - pausing between cuts")
            print("[EGG CUTTING] Gripper CLOSED - knife still held")
            time.sleep(0.5)

            print("\n[EGG CUTTING] Step 4: Second cut - lowering knife deeper (~5cm total)")
            current_pose = self.arm.move_group.get_current_pose().pose
            target_pose = current_pose
            target_pose.position.z = current_pose.position.z - 0.05
            self.arm._move_to_target_by_pose(target_pose)
            print("[EGG CUTTING] Second stage cut - cutting through egg")
            print("[EGG CUTTING] Gripper CLOSED - complete cut")
            time.sleep(1.0)

            print("\n[EGG CUTTING] Step 5: Final retraction")
            current_pose = self.arm.move_group.get_current_pose().pose
            target_pose = current_pose
            target_pose.position.z = current_pose.position.z + 0.05
            self.arm._move_to_target_by_pose(target_pose)
            print("[EGG CUTTING] Knife fully retracted")
            print("[EGG CUTTING] Gripper CLOSED - ready for next action")
            time.sleep(0.5)

            print("\n[SUCCESS] EGG HALVING COMPLETED")
            print("[SUCCESS] Gripper remained CLOSED throughout")
            print("="*70)
            return True

        except Exception as e:
            print("\n[ERROR] Egg cutting failed: {}".format(e))
            import traceback
            traceback.print_exc()
            return False

    def task_3_firewood_storage(self):
        """
        Task 3: Firewood Storage
        Room 2 -> Room 3 (120 sec), Pick firewood (60 sec), Room 3 -> Room 1 door

        Note: Wait logic has been moved to eval_real_robot.py
        DKPrompt now only executes motion primitives without hardcoded waits.
        """
        print("\n" + "="*70)
        print("TASK 3: FIREWOOD STORAGE")
        print("="*70)

        task_3_info = [
            {
                "object_name": "firewood",
                "object_target_xy": [-0.26633, -0.252465],
                "offset": [0.10, 0.05, 0.15],
                "robot_pickup_pose": {
                    "x": -0.26633, "y": -0.252465, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.15,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0
            }
        ]

        try:
            print("\n[TASK 3] Step 1: Navigate from Room 2 to Room 3")
            print("   (Wait handled by eval_real_robot.py: 120 seconds)")
            # Navigation logic handled by eval_real_robot.py

            print("\n[TASK 3] Step 2: Pick firewood")
            print("   (Wait handled by eval_real_robot.py: 60 seconds)")
            self.arm.pickup(task_3_info[0])

            print("\n[TASK 3] Step 3: Navigate from Room 3 to Room 1 door")
            print("   Location: [-6.52195, -8.179987]")
            print("   (Wait logic and door handling in eval_real_robot.py)")

            print("\n[SUCCESS] TASK 3 COMPLETED")
            return True

        except Exception as e:
            print("\n[ERROR] TASK 3 FAILED: {}".format(e))
            return False

    def run_task(self, task_number):
        """Execute a specific task."""
        if task_number == 1:
            return self.task_1_bottle_collection()
        elif task_number == 2:
            return self.cut_egg_two_stage()
        elif task_number == 3:
            return self.task_3_firewood_storage()
        else:
            print("[ERROR] Unknown task: {}".format(task_number))
            return False

    def run_all_tasks(self):
        """Execute all three tasks in sequence."""
        print("\n" + "="*70)
        print("DKPrompt MULTI-TASK EXECUTION")
        print("="*70)

        results = []

        # Task 1
        result_1 = self.task_1_bottle_collection()
        results.append(("Task 1: Bottle Collection", result_1))

        # Task 2
        result_2 = self.task_2_halve_egg()
        results.append(("Task 2: Halve an Egg", result_2))

        # Task 3
        result_3 = self.task_3_firewood_storage()
        results.append(("Task 3: Firewood Storage", result_3))

        # Print summary
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        for task_name, result in results:
            status = "[PASSED]" if result else "[FAILED]"
            print("{}: {}".format(task_name, status))
        print("="*70 + "\n")

        return all(r for _, r in results)


if __name__ == '__main__':
    try:
        executor = DKPromptExecutor()

        # Option 1: Run a single task
        task_number = 2  # Run Task 2: Halve an Egg
        executor.run_task(task_number)

        # Option 2: Run all tasks
        # executor.run_all_tasks()

    except rospy.ROSInterruptException:
        rospy.loginfo("DKPrompt executor terminated")
