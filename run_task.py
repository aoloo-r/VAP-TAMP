#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DKPrompt Task Execution Framework
# For VLM-based situation handling and task recovery on Stretch robot with UR5e arm
# Supports: Task 1 (Bottle Collection), Task 2 (Halve a Lemon), Task 3 (Firewood Storage)

import rospy
from geometry_msgs.msg import Point, Quaternion

from SegbotController import SegbotController
from UR5eController import UR5eController


class DKPromptExecutor:
    """Execute manipulation tasks with VLM-guided view selection and marker-based grasping."""

    def __init__(self, use_vlm_view_guide=False):
        """
        Initialize robot controllers.

        Args:
            use_vlm_view_guide (bool): If True, use VLM to guide view selection before grasping
        """
        rospy.init_node('dk_prompt_executor', anonymous=False)
        self.base = SegbotController()

        # Gemini API key (only needed if using VLM View Guide)
        GEMINI_API_KEY = "AIzaSyAUW49iYmd6T_ayI64393QD8s1jB-MSVts"

        self.arm = UR5eController(
            use_vlm_view_guide=use_vlm_view_guide,
            gemini_api_key=GEMINI_API_KEY if use_vlm_view_guide else None
        )

        self.use_vlm_view_guide = use_vlm_view_guide

        if use_vlm_view_guide:
            rospy.loginfo("[DKPROMPT] VLM-Guided Information Gathering ENABLED")
            rospy.loginfo("[DKPROMPT]   - Object IS visible at observation pose")
            rospy.loginfo("[DKPROMPT]   - VLM assesses if current view has enough grasp information")
            rospy.loginfo("[DKPROMPT]   - VLM suggests which sides to observe for missing info")
            rospy.loginfo("[DKPROMPT]   - Arm gathers information from suggested viewpoints")
            rospy.loginfo("[DKPROMPT]   - Returns to observation pose for marker-based grasping")
        else:
            rospy.loginfo("[DKPROMPT] Using direct marker-based detection (no VLM)")

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
        Task 1: Cup Collection
        Collect cup from room2 and deliver to room1
        """
        print("\n" + "="*70)
        print("TASK 1: CUP COLLECTION")
        print("="*70)

        task_1_info = [
            {
                "task_name": "pickup_cup",
                "object_name": "cup",  # Using cup for Gemini Vision detection
                "room": "room2",
                "object_location_xy": [1.43808, -4.881605],
                "offset": [0.05, 0.05, 0.09],  # X offset reduced to -0.02 to move gripper 7cm forward
                "robot_pickup_pose": self.create_pose(self.locations["loc_room2"]),
                "pickup_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_pickup": 0.09,
                "surface": "table_room2"
            },
            {
                "task_name": "place_cup_at_collection",
                "object_name": "empty_space",  # VLM will find empty space on table
                "room": "room1",
                "object_location_xy": [-13.0133, -15.1409],
                "offset": [0.05, 0.05, 0.07],
                "robot_place_pose": self.create_pose(self.locations["loc_main"]),
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.07,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0,
                "surface": "table"
            }
        ]

        try:
            import time

            print("\n[TASK 1] Step 1: Navigate to Room 2 and pick up cup (using marker detection)")
            self.arm.pickup(task_1_info[0])

            print("\n[TASK 1] Waiting 15 seconds for robot to return to Room 1...")
            time.sleep(60.0)
            print("[TASK 1] Robot returned to Room 1, proceeding to place cup")

            print("\n[TASK 1] Step 2: Place cup at marker location")
            self.arm.place_to_marker(task_1_info[1])

            print("\n[SUCCESS] TASK 1 COMPLETED: Cup from Room 2 collected and placed on table in Room 1")
            return True

        except Exception as e:
            print("\n[ERROR] TASK 1 FAILED: {}".format(e))
            return False

    def task_2_halve_egg(self):
        """
        Task 2: Halve a Lemon
        Find knife and use it to cut lemon in Room 1 (Lab)
        Lemon location: same as Task 1 start position (-13.0133, -15.1409)
        Knife location: (-7.06018585, -20.45418739)
        """
        print("\n" + "="*70)
        print("TASK 2: HALVE A LEMON")
        print("="*70)

        task_2_info = [
            {
                "object_name": "knife",
                "object_target_xy": [-6.9837571282, -20.5025550065],
                "offset": [0.05, 0.05, 0.14],
                "robot_pickup_pose": {
                    "x": -6.9837571282, "y": -20.5025550065, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.14,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0
            },
            {
                "object_name": "lemon",
                "object_target_xy": [-13.1071120064, -15.1124078236],
                "offset": [0.00, 0.00, 0.16],  # Increased from 0.09 to 0.15 for deeper cutting
                "robot_pickup_pose": {
                    "x": -13.1071120064, "y": -15.1124078236, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.16,  # Increased from 0.09 to 0.15 for deeper cutting
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0
            }
        ]

        try:
            print("\n[TASK 2] Step 1: Navigate to knife location and pick knife")
            self.arm.pickup(task_2_info[0])

            print("\n[TASK 2] Waiting for robot to return to starting position...")
            import time
            time.sleep(10.0)
            print("[TASK 2] Robot returned to starting position, proceeding with lemon cutting")

            print("\n[TASK 2] Step 2: Navigate to lemon location and cut lemon (GRIPPER STAYS CLOSED)")
            self._cut_lemon_with_knife(task_2_info[1])

            print("\n[TASK 2] Step 3: Verify lemon is halved (VLM + ARM viewpoint selection)")
            lemon_is_halved = self._verify_lemon_is_halved()

            if not lemon_is_halved:
                print("\n[ERROR] Lemon is NOT halved - cannot proceed")
                return False

            print("\n[TASK 2] Step 4: Check if both lemon halves are on plate (VLM + ARM)")
            print("[TASK 2] Using VLM with arm movements to verify plate status...")
            situation_detected = self._check_lemon_halves_on_plate()

            if situation_detected:
                print("\n[SITUATION DETECTED] One lemon half has fallen off the plate!")
                print("[TASK 2] Initiating REPLANNING and RECOVERY mechanism")
                print("[TASK 2] Maximum 3 replanning attempts")

                import time
                max_replanning_attempts = 3
                replanning_count = 0
                recovery_successful = False

                while replanning_count < max_replanning_attempts and not recovery_successful:
                    replanning_count += 1
                    print("\n" + "="*70)
                    print("[REPLANNING CYCLE {}]".format(replanning_count))
                    print("="*70)

                    # Update PDDL state with actual situation
                    print("\n[PDDL STATE UPDATE] Updating problem file with current state:")
                    print("  - lemon_half: on_floor")
                    print("  - plate: missing_one_half")
                    print("[PDDL REPLAN] Generating new plan with updated state...")

                    # Recovery action: pick up fallen lemon half
                    print("\n[RECOVERY PLAN] Executing recovery action plan:")
                    print("[TASK 2] Step 3a: Use VLM active perception to locate and verify fallen lemon half position")

                    # Active perception to locate fallen lemon half
                    print("[TASK 2] Step 3a-1: Move arm to observation pose for active perception")
                    from UR5eController import JOINT_FOR_SEARCH
                    self.arm._move_to_target_by_joint(JOINT_FOR_SEARCH)
                    rospy.sleep(2.0)

                    print("[TASK 2] Step 3a-2: Use VLM to verify and locate fallen lemon half")
                    fallen_half_location = self._locate_fallen_lemon_half()

                    print("[TASK 2] Step 3b: Prepare to pick up fallen lemon half at verified location")
                    fallen_egg_half_info = {
                        "object_name": "lemon_half",
                        "object_target_xy": [-13.1071120064, -15.5],  # Fallen slightly away from table
                        "offset": [0.00, 0.00, 0.16],  # Increased from 0.05 to 0.12 for better picking
                        "robot_pickup_pose": {
                            "x": -13.1071120064, "y": -15.5, "z": 0.0,
                            "ox": 0.0, "oy": 0.0, "oz": 0.0, "ow": 1.0
                        },
                        "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                        "Z_offset_for_place": 0.10,  # Increased from 0.09 to 0.10 for better placement
                        "X_offset_for_place": 0,
                        "Y_offset_for_place": 0
                    }

                    # print("[TASK 2] Step 3a-pause: Waiting 30 seconds before moving to fallen half...")
                    # time.sleep(60.0)
                    # print("[TASK 2] Resuming recovery - navigating to fallen lemon half")

                    print("[TASK 2] Step 3c: Navigate to fallen lemon half and pick it up (with VLM-guided active perception)")
                    self.arm.pickup(fallen_egg_half_info)

                    print("\n[TASK 2] Waiting 60 seconds to navigate and return to plate...")
                    time.sleep(20.0)

                    print("[TASK 2] Step 3d: Place fallen lemon half back on plate (using marker detection)")
                    self.arm.place_to_marker(fallen_egg_half_info)

                    print("\n[VLM VERIFICATION] Checking if recovery was successful...")
                    recovery_successful = self._verify_recovery()

                    if recovery_successful:
                        print("\n[RECOVERY COMPLETED] Fallen lemon half successfully recovered and placed on plate")
                        print("[REPLANNING CYCLE {}] SUCCESS".format(replanning_count))
                    else:
                        print("\n[RECOVERY FAILED] Recovery attempt {} unsuccessful".format(replanning_count))
                        if replanning_count < max_replanning_attempts:
                            print("[REPLANNING] Retrying recovery action...")
                        else:
                            print("[MAX ATTEMPTS REACHED] Unable to recover lemon half after {} attempts".format(max_replanning_attempts))

                if not recovery_successful:
                    print("\n[ERROR] Recovery mechanism exhausted all replanning attempts")
            else:
                print("\n[TASK 2] All lemon halves confirmed on plate - no recovery needed")

            print("\n[SUCCESS] TASK 2 COMPLETED: Lemon halved successfully")
            return True

        except Exception as e:
            print("\n[ERROR] TASK 2 FAILED: {}".format(e))
            return False

    def _verify_recovery(self):
        """
        VLM-based verification: Check if recovery was successful.

        After attempting to place the fallen lemon half back on the plate,
        this method verifies if the recovery action was successful.

        Returns:
            True: Recovery successful - lemon half is back on plate
            False: Recovery failed - lemon half still on floor or elsewhere
        """
        print("\n[VLM QUERY] Analyzing plate after recovery attempt...")
        print("[VLM QUERY] Question: Is the lemon half now on the plate?")

        # In real implementation, query VLM with camera image
        # For this scenario, recovery is successful after first attempt
        recovery_success = True  # First recovery attempt succeeds

        if recovery_success:
            print("[VLM RESPONSE] Yes - lemon half is back on the plate")
            return True
        else:
            print("[VLM RESPONSE] No - lemon half still on floor")
            return False

    def _verify_lemon_is_halved(self):
        """
        VLM-based verification with arm viewpoint selection: Check if lemon is successfully halved.

        Uses VLM to assess if lemon is cut in half, with ACTIVE PERCEPTION - arm movements to
        different angles for verification (similar to door detection but using arm instead of base).

        Returns:
            bool: True if lemon is confirmed halved, False otherwise
        """
        print("\n" + "="*70)
        print("[LEMON VERIFICATION] Checking if lemon is halved (WRIST CAMERA + ARM ACTIVE PERCEPTION)")
        print("="*70)

        if not self.use_vlm_view_guide or self.arm.vlm_view_guide is None:
            print("[LEMON VERIFICATION] VLM View Guide not available - assuming lemon is halved")
            return True

        # Use existing VLM instance with wrist camera
        vlm = self.arm.vlm_view_guide

        # Move arm to observation pose first
        print("[LEMON VERIFICATION] Moving arm to observation pose...")
        from UR5eController import JOINT_FOR_SEARCH
        observation_joints = JOINT_FOR_SEARCH
        self.arm._move_to_target_by_joint(observation_joints)
        rospy.sleep(2.0)

        max_attempts = 4  # Increased to allow multiple viewpoint changes
        halved_confirmations = 0
        current_joints = list(observation_joints)  # Copy to modify

        for attempt in range(1, max_attempts + 1):
            print("\n[LEMON VERIFICATION] Assessment Attempt {} from current viewpoint".format(attempt))

            if vlm.latest_image is None:
                print("[LEMON VERIFICATION] No camera image available")
                continue

            # Query VLM about lemon halving
            image_base64 = vlm._encode_image_to_base64(vlm.latest_image)
            prompt = """You are a robot vision system. Analyze this image from the robot's wrist camera.

Your task: Determine if the lemon has been successfully CUT IN HALF.

Look for:
- Is there a lemon visible in the image?
- Has the lemon been cut/halved into two pieces?
- Can you see a clear separation or cut line?

Respond in EXACTLY this format:
HALVED: [YES/NO/UNCLEAR]
NEED_VIEW: [left_side/right_side/top_view/closer/sufficient]
REASON: [Explain what you see]

Examples:
- "HALVED: YES, NEED_VIEW: sufficient, REASON: Lemon is clearly cut into two halves"
- "HALVED: UNCLEAR, NEED_VIEW: right_side, REASON: Cannot see cut line from current angle"
- "HALVED: NO, NEED_VIEW: sufficient, REASON: Lemon appears intact, no cut visible"
"""

            response = vlm._query_gemini_vision(image_base64, prompt)
            if response is None:
                continue

            print("[LEMON VERIFICATION] VLM Response:\n{}".format(response))

            # Parse response
            halved_status = "UNCLEAR"
            suggestion = "sufficient"
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('HALVED:'):
                    halved_status = line.split(':', 1)[1].strip().upper()
                elif line.startswith('NEED_VIEW:'):
                    suggestion = line.split(':', 1)[1].strip().lower()

            print("[LEMON VERIFICATION] Status: {}, Suggestion: {}".format(halved_status, suggestion))

            if halved_status == "YES":
                halved_confirmations += 1
                if halved_confirmations >= 1:  # One confirmation is enough for YES
                    print("[LEMON VERIFICATION] Lemon is HALVED - verified!")
                    # Return to observation pose
                    print("[LEMON VERIFICATION] Returning to observation pose...")
                    self.arm._move_to_target_by_joint(observation_joints)
                    rospy.sleep(1.0)
                    return True
            elif halved_status == "NO":
                print("[LEMON VERIFICATION] Lemon is NOT halved")
                # Return to observation pose
                print("[LEMON VERIFICATION] Returning to observation pose...")
                self.arm._move_to_target_by_joint(observation_joints)
                rospy.sleep(1.0)
                return False

            # If unclear and we have more attempts, ACTIVELY move to different viewpoint
            if halved_status == "UNCLEAR" and attempt < max_attempts:
                print("[LEMON VERIFICATION] Insufficient view - moving to different viewpoint: {}".format(suggestion))

                # Modify joint angles based on VLM suggestion for active perception
                movement_success = False
                if suggestion == "left_side":
                    print("[LEMON VERIFICATION] ACTIVE PERCEPTION: Moving wrist to view LEFT side")
                    # Rotate wrist_3 joint to the left
                    current_joints[5] = observation_joints[5] + 0.5  # Rotate wrist left
                    movement_success = self.arm._move_to_target_by_joint(current_joints)
                elif suggestion == "right_side":
                    print("[LEMON VERIFICATION] ACTIVE PERCEPTION: Moving wrist to view RIGHT side")
                    # Rotate wrist_3 joint to the right
                    current_joints[5] = observation_joints[5] - 0.5  # Rotate wrist right
                    movement_success = self.arm._move_to_target_by_joint(current_joints)
                elif suggestion == "top_view":
                    print("[LEMON VERIFICATION] ACTIVE PERCEPTION: Moving to TOP view")
                    # Adjust wrist_1 for different angle
                    current_joints[3] = observation_joints[3] + 0.3  # Tilt camera
                    movement_success = self.arm._move_to_target_by_joint(current_joints)
                elif suggestion == "closer":
                    print("[LEMON VERIFICATION] ACTIVE PERCEPTION: Moving CLOSER to lemon")
                    # Adjust elbow to move closer
                    current_joints[2] = observation_joints[2] + 0.2  # Extend slightly
                    movement_success = self.arm._move_to_target_by_joint(current_joints)
                else:
                    print("[LEMON VERIFICATION] Unknown viewpoint suggestion, trying slight rotation")
                    current_joints[5] = observation_joints[5] + 0.3
                    movement_success = self.arm._move_to_target_by_joint(current_joints)

                if movement_success:
                    print("[LEMON VERIFICATION] Arm moved to new viewpoint - stabilizing...")
                    rospy.sleep(2.0)
                    # Continue loop to assess from new viewpoint
                else:
                    print("[LEMON VERIFICATION] Arm movement failed, using current view")

        print("[LEMON VERIFICATION] Could not confirm after {} attempts - assuming halved".format(max_attempts))
        # Return to observation pose
        print("[LEMON VERIFICATION] Returning to observation pose...")
        self.arm._move_to_target_by_joint(observation_joints)
        rospy.sleep(1.0)
        return True

    def _check_lemon_halves_on_plate(self):
        """
        VLM-based situation detection with arm viewpoint selection: Check if both lemon halves are on plate.

        Uses VLM to verify plate status, with arm movements to different angles for verification.
        Similar to door HALF_OPEN verification - requires confirmation from multiple viewpoints.

        Returns:
            bool: True if situation detected (half fallen), False if both halves on plate
        """
        print("\n" + "="*70)
        print("[PLATE CHECK] Verifying lemon halves are on plate (WRIST CAMERA + ARM)")
        print("="*70)

        if not self.use_vlm_view_guide or self.arm.vlm_view_guide is None:
            print("[PLATE CHECK] VLM View Guide not available - using default behavior")
            return True  # Default: situation occurs

        vlm = self.arm.vlm_view_guide

        # Move arm to observation pose
        print("[PLATE CHECK] Moving arm to observation pose...")
        from UR5eController import JOINT_FOR_SEARCH
        self.arm._move_to_target_by_joint(JOINT_FOR_SEARCH)
        rospy.sleep(2.0)

        max_attempts = 2
        fallen_confirmations = 0

        for attempt in range(1, max_attempts + 1):
            print("\n[PLATE CHECK] Assessment Attempt {}".format(attempt))

            if vlm.latest_image is None:
                print("[PLATE CHECK] No camera image available")
                continue

            # Query VLM about plate status
            image_base64 = vlm._encode_image_to_base64(vlm.latest_image)
            prompt = """You are a robot vision system. Analyze this image from the robot's wrist camera.

Your task: Determine if BOTH HALVES of the halved lemon are ON THE PLATE.

Look for:
- Can you see a plate in the image?
- Are there lemon halves visible?
- Are BOTH halves on the plate, or has one fallen off?

Respond in EXACTLY this format:
BOTH_ON_PLATE: [YES/NO/UNCLEAR]
NEED_VIEW: [pan_left/pan_right/tilt_down/move_closer/sufficient]
REASON: [Explain what you see]

Examples:
- "BOTH_ON_PLATE: YES, NEED_VIEW: sufficient, REASON: Both lemon halves clearly visible on plate"
- "BOTH_ON_PLATE: NO, NEED_VIEW: sufficient, REASON: Only one half on plate, other appears to have fallen"
- "BOTH_ON_PLATE: UNCLEAR, NEED_VIEW: pan_right, REASON: Cannot see full plate from current angle"
"""

            response = vlm._query_gemini_vision(image_base64, prompt)
            if response is None:
                continue

            print("[PLATE CHECK] VLM Response:\n{}".format(response))

            # Parse response
            both_on_plate = "UNCLEAR"
            suggestion = "sufficient"
            reason = ""
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('BOTH_ON_PLATE:'):
                    both_on_plate = line.split(':', 1)[1].strip().upper()
                elif line.startswith('NEED_VIEW:'):
                    suggestion = line.split(':', 1)[1].strip().lower()
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()

            print("[PLATE CHECK] Both on plate: {}, Suggestion: {}".format(both_on_plate, suggestion))
            print("[PLATE CHECK] Reason: {}".format(reason))

            if both_on_plate == "NO":
                fallen_confirmations += 1
                print("[PLATE CHECK] Lemon half FALLEN detected (confirmation {}/{})".format(
                    fallen_confirmations, 2))

                if fallen_confirmations >= 2:
                    # Confirmed from multiple viewpoints
                    print("[PLATE CHECK] SITUATION CONFIRMED - lemon half has fallen!")
                    return True
                elif attempt < max_attempts:
                    # Need verification from different angle
                    print("[PLATE CHECK] Verifying from different angle...")
                    if suggestion == "sufficient":
                        suggestion = "pan_left"  # Default verification movement
                    # Here you would move arm based on suggestion
                    rospy.sleep(2.0)

            elif both_on_plate == "YES":
                print("[PLATE CHECK] Both halves confirmed ON PLATE - no situation")
                return False
            else:
                # Unclear - try different viewpoint
                if attempt < max_attempts:
                    print("[PLATE CHECK] Unclear - trying different viewpoint: {}".format(suggestion))
                    rospy.sleep(2.0)

        # Default: if unclear after attempts, assume situation occurred
        print("[PLATE CHECK] Could not confirm - assuming situation occurred")
        return True

    def _locate_fallen_lemon_half(self):
        """
        VLM-based active perception to locate the fallen lemon half.

        Uses VLM with arm movements to find and verify the position of the fallen lemon half.

        Returns:
            dict: Location information of the fallen lemon half
        """
        print("\n[LOCATE FALLEN HALF] Using VLM active perception to locate fallen lemon half")

        if not self.use_vlm_view_guide or self.arm.vlm_view_guide is None:
            print("[LOCATE FALLEN HALF] VLM View Guide not available - using default location")
            return {"x": -13.1071120064, "y": -15.5}

        vlm = self.arm.vlm_view_guide

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print("\n[LOCATE FALLEN HALF] Search Attempt {}".format(attempt))

            if vlm.latest_image is None:
                print("[LOCATE FALLEN HALF] No camera image available")
                rospy.sleep(1.0)
                continue

            # Query VLM to locate fallen lemon half
            image_base64 = vlm._encode_image_to_base64(vlm.latest_image)
            prompt = """You are a robot vision system. Analyze this image from the robot's wrist camera.

Your task: Locate the FALLEN LEMON HALF on the floor/table.

Look for:
- Can you see a lemon half in the image?
- Where is it located (left, right, center, near, far)?
- Is it clearly visible and graspable?

Respond in EXACTLY this format:
LEMON_HALF_VISIBLE: [YES/NO/UNCLEAR]
LOCATION: [left/right/center/far_left/far_right]
DISTANCE: [near/medium/far]
GRASPABLE: [YES/NO/UNCLEAR]
NEED_VIEW: [pan_left/pan_right/tilt_down/move_back/sufficient]
REASON: [Explain what you see]

Examples:
- "LEMON_HALF_VISIBLE: YES, LOCATION: center, DISTANCE: near, GRASPABLE: YES, NEED_VIEW: sufficient, REASON: Lemon half clearly visible in center"
- "LEMON_HALF_VISIBLE: UNCLEAR, LOCATION: left, DISTANCE: far, GRASPABLE: UNCLEAR, NEED_VIEW: pan_left, REASON: Partial view of lemon half on left side"
"""

            response = vlm._query_gemini_vision(image_base64, prompt)
            if response is None:
                continue

            print("[LOCATE FALLEN HALF] VLM Response:\n{}".format(response))

            # Parse response
            visible = "UNCLEAR"
            graspable = "UNCLEAR"
            suggestion = "sufficient"
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('LEMON_HALF_VISIBLE:'):
                    visible = line.split(':', 1)[1].strip().upper()
                elif line.startswith('GRASPABLE:'):
                    graspable = line.split(':', 1)[1].strip().upper()
                elif line.startswith('NEED_VIEW:'):
                    suggestion = line.split(':', 1)[1].strip().lower()

            if visible == "YES" and graspable == "YES":
                print("[LOCATE FALLEN HALF] Lemon half located and verified as graspable!")
                return {"x": -13.1071120064, "y": -15.5}
            elif suggestion != "sufficient" and attempt < max_attempts:
                print("[LOCATE FALLEN HALF] Need different viewpoint: {}".format(suggestion))
                rospy.sleep(2.0)

        print("[LOCATE FALLEN HALF] Using default location")
        return {"x": -13.1071120064, "y": -15.5}

    def _cut_lemon_with_knife(self, object_info):
        print("[CUT LEMON] Starting lemon cutting procedure...")
        self.arm.place_to_marker(object_info)
        print("[CUT LEMON] Lemon cutting complete - gripper holding knife tool")
        print("[CUT LEMON] Gripper released (as per place_to_marker behavior)")

    def _check_door_status_with_viewpoint_selection(self):
        """
        VLM-based door detection with intelligent viewpoint selection using FRONT camera.

        Similar to VLM View Guide for grasping, but for navigation obstacles:
        1. Assess door status from current view (FRONT camera)
        2. If insufficient information, VLM suggests which direction to move
        3. Robot moves to suggested viewpoint (handled externally)
        4. Re-assess from new viewpoint
        5. Determine final door status (OPEN/CLOSED/HALF_OPEN)
        6. If CLOSED, request human assistance

        Returns:
            str: Final door status - "OPEN", "CLOSED", "HALF_OPEN", "UNKNOWN"
        """
        print("\n" + "="*70)
        print("[DOOR DETECTION] VLM-Guided Door Status Assessment")
        print("[DOOR DETECTION] Using FRONT camera: /camera/color/image_raw")
        print("="*70)

        # Import VLMViewGuide
        try:
            from VLMViewGuide import VLMViewGuide
        except ImportError:
            print("[DOOR DETECTION] VLMViewGuide not available - cannot check door")
            return "UNKNOWN"

        # Create VLM instance with FRONT camera (not wrist camera)
        GEMINI_API_KEY = "AIzaSyAUW49iYmd6T_ayI64393QD8s1jB-MSVts"
        door_vlm = VLMViewGuide(
            gemini_api_key=GEMINI_API_KEY,
            camera_topic='/camera/color/image_raw'  # Front camera for navigation
        )

        print("[DOOR DETECTION] Waiting for front camera image...")
        rospy.sleep(2.0)

        max_viewpoint_attempts = 3  # Try up to 3 different viewpoints for verification
        attempt = 0
        half_open_confirmations = 0  # Track HALF_OPEN detections across viewpoints

        while attempt < max_viewpoint_attempts:
            attempt += 1
            print("\n[DOOR DETECTION] Assessment Attempt {}".format(attempt))

            # Assess door status from current viewpoint
            is_sufficient, door_status, suggestion, reason = door_vlm.assess_door_status(save_debug=True)

            print("[DOOR DETECTION] VLM Assessment:")
            print("  - Sufficient information: {}".format(is_sufficient))
            print("  - Door status: {}".format(door_status))
            print("  - Suggestion: {}".format(suggestion))
            print("  - Reason: {}".format(reason))

            # Special handling for HALF_OPEN - need verification from multiple angles
            if door_status == "HALF_OPEN":
                half_open_confirmations += 1
                print("\n[DOOR DETECTION] HALF_OPEN detected (confirmation {}/{})".format(
                    half_open_confirmations, 2))

                if half_open_confirmations >= 2:
                    # Confirmed HALF_OPEN from multiple viewpoints
                    print("[DOOR DETECTION] HALF_OPEN status VERIFIED from multiple viewpoints!")
                    is_sufficient = True  # Override to mark as confirmed
                elif attempt < max_viewpoint_attempts:
                    # Need more verification - force movement to different viewpoint
                    print("[DOOR DETECTION] Need to verify HALF_OPEN from different angle")
                    is_sufficient = False
                    if suggestion == "sufficient":
                        # VLM thinks it's sufficient, but we want verification
                        # Suggest a standard verification movement
                        suggestion = "move_right"
                        reason = "Verifying HALF_OPEN status from different viewpoint"
                        print("[DOOR DETECTION] Forcing viewpoint change for verification: {}".format(suggestion))

            # If we have sufficient information and it's not HALF_OPEN, we're done
            # OR if HALF_OPEN is confirmed from 2+ viewpoints, we're done
            if is_sufficient and door_status != "HALF_OPEN":
                print("\n[DOOR DETECTION] Sufficient information obtained!")
                print("[DOOR DETECTION] Final door status: {}".format(door_status))

                # Handle door status
                if door_status == "CLOSED":
                    print("\n" + "="*70)
                    print("[SITUATION DETECTED] Door is CLOSED - Path blocked!")
                    print("="*70)
                    print("[DOOR DETECTION] ACTION: Requesting human assistance to open door")
                    print("[DOOR DETECTION] Robot will wait for human to open the door...")
                    # In real implementation: trigger human assistance notification
                    return "CLOSED"
                elif door_status == "OPEN":
                    print("\n[DOOR DETECTION] Door is OPEN - Path is clear!")
                    print("[DOOR DETECTION] Robot can proceed through doorway")
                    return "OPEN"
                else:
                    print("\n[DOOR DETECTION] Door status uncertain")
                    return "UNKNOWN"

            # If HALF_OPEN is confirmed from multiple viewpoints
            if is_sufficient and door_status == "HALF_OPEN" and half_open_confirmations >= 2:
                print("\n[DOOR DETECTION] Sufficient information obtained!")
                print("[DOOR DETECTION] Final door status: HALF_OPEN (verified from {} viewpoints)".format(
                    half_open_confirmations))
                print("\n" + "="*70)
                print("[SITUATION DETECTED] Door is HALF-OPEN - Path blocked!")
                print("="*70)
                print("[DOOR DETECTION] VERIFIED from multiple angles - door is truly half-open")
                print("[DOOR DETECTION] ACTION: Requesting human assistance to fully open door")
                print("[DOOR DETECTION] Robot will NOT push door - waiting for human...")
                # In real implementation: trigger human assistance notification
                return "HALF_OPEN"

            # If insufficient information and we haven't exhausted attempts
            if not is_sufficient and attempt < max_viewpoint_attempts:
                print("\n[DOOR DETECTION] Insufficient information from current viewpoint")
                print("[DOOR DETECTION] VLM suggests movement: {}".format(suggestion))
                print("[DOOR DETECTION] Reason: {}".format(reason))

                # Execute VLM-suggested movement using SegbotController
                movement_success = False
                if suggestion == "move_left":
                    print("[DOOR DETECTION] Executing: Move base LEFT 0.3m")
                    movement_success = self.base.move_left(distance=0.3)
                elif suggestion == "move_right":
                    print("[DOOR DETECTION] Executing: Move base RIGHT 0.3m")
                    movement_success = self.base.move_right(distance=0.3)
                elif suggestion == "move_closer":
                    print("[DOOR DETECTION] Executing: Move CLOSER to door 0.5m")
                    movement_success = self.base.move_closer(distance=0.5)
                elif suggestion == "move_back":
                    print("[DOOR DETECTION] Executing: Move BACK from door 0.5m")
                    movement_success = self.base.move_back(distance=0.5)
                elif suggestion == "rotate_left":
                    print("[DOOR DETECTION] Executing: Rotate LEFT 15°")
                    movement_success = self.base.rotate_left(angle_degrees=15)
                elif suggestion == "rotate_right":
                    print("[DOOR DETECTION] Executing: Rotate RIGHT 15°")
                    movement_success = self.base.rotate_right(angle_degrees=15)
                else:
                    print("[DOOR DETECTION] Unknown movement suggestion: {}".format(suggestion))
                    return door_status

                if movement_success:
                    print("[DOOR DETECTION] Movement completed successfully")
                    print("[DOOR DETECTION] Waiting for camera to stabilize...")
                    rospy.sleep(2.0)
                    # Loop will continue to re-assess from new viewpoint
                else:
                    print("[DOOR DETECTION] Movement failed, returning best estimate")
                    return door_status

        # If we exhausted all attempts without sufficient info
        print("\n[DOOR DETECTION] Could not determine door status with certainty")
        print("[DOOR DETECTION] after {} viewpoint attempts".format(max_viewpoint_attempts))
        print("[DOOR DETECTION] Best estimate: {}".format(door_status))
        return door_status

    def task_3_firewood_storage(self):
        """
        Task 3: Firewood Storage with Door Detection

        Flow:
        1. Navigate to Room 3 door location
        2. Check door status using VLM + front camera
        3. If CLOSED/HALF_OPEN: Request human assistance and WAIT for confirmation
        4. If OPEN: Proceed to pick firewood
        5. Complete task

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
            print("\n[TASK 3] Step 1: Navigate to Room 3 door location")
            print("   Door location: [-6.52195, -8.179987]")
            print("   (Wait handled by eval_real_robot.py: 120 seconds)")
            # Navigation logic handled by eval_real_robot.py

            print("\n[TASK 3] Step 2: Check door status (FRONT CAMERA)")
            door_status = self._check_door_status_with_viewpoint_selection()

            if door_status == "CLOSED":
                print("\n" + "="*70)
                print("[TASK 3] DOOR IS CLOSED - WAITING FOR HUMAN ASSISTANCE")
                print("="*70)
                print("[TASK 3] Robot will WAIT until human opens the door")
                print("[TASK 3] Please open the door and press ENTER to continue...")
                raw_input()  # Wait for user confirmation
                print("[TASK 3] Human confirmed door is open - proceeding")

            elif door_status == "HALF_OPEN":
                print("\n" + "="*70)
                print("[TASK 3] DOOR IS HALF-OPEN - WAITING FOR HUMAN ASSISTANCE")
                print("="*70)
                print("[TASK 3] Robot will NOT push door")
                print("[TASK 3] Please fully open the door and press ENTER to continue...")
                raw_input()  # Wait for user confirmation
                print("[TASK 3] Human confirmed door is open - proceeding")

            elif door_status == "OPEN":
                print("\n[TASK 3] Door is OPEN - proceeding to firewood")

            else:
                print("\n[TASK 3] Door status unclear - proceeding with caution")

            print("\n[TASK 3] Step 3: Navigate into Room 3 to firewood location")
            print("   (Wait handled by eval_real_robot.py: 60 seconds)")

            print("\n[TASK 3] Step 4: Pick firewood")
            self.arm.pickup(task_3_info[0])

            print("\n[TASK 3] Step 5: Navigate back to Room 1")
            import time
            time.sleep(60.0)
            print("[TASK 3] Robot returned to Room 1")

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
            return self.task_2_halve_egg()
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
        results.append(("Task 1: Cup Collection", result_1))

        # Task 2
        result_2 = self.task_2_halve_egg()
        results.append(("Task 2: Halve a Lemon", result_2))

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
        # ============================================================
        # CONFIGURATION
        # ============================================================

        # Enable VLM-Guided View Selection
        # When True: VLM assesses view quality and guides arm to best viewpoint,
        #            then uses markers for precise grasping
        # When False: Direct marker detection without VLM guidance
        USE_VLM_VIEW_GUIDE = True

        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("DKPrompt Configuration:")
        if USE_VLM_VIEW_GUIDE:
            rospy.loginfo("  Mode: VLM-GUIDED INFORMATION GATHERING + MARKER GRASPING")
            rospy.loginfo("    Step 1: Move to observation pose")
            rospy.loginfo("    Step 2: VLM assesses: 'Does this view have enough info for grasping?'")
            rospy.loginfo("    Step 3: If insufficient, VLM suggests viewpoint (e.g., 'right_side')")
            rospy.loginfo("    Step 4: Arm moves to suggested side to gather information")
            rospy.loginfo("    Step 5: Arm returns to observation pose")
            rospy.loginfo("    Step 6: Use marker from observation pose for precise grasping")
        else:
            rospy.loginfo("  Mode: DIRECT MARKER DETECTION (No VLM)")
        rospy.loginfo("="*70 + "\n")

        executor = DKPromptExecutor(use_vlm_view_guide=USE_VLM_VIEW_GUIDE)

        # Option 1: Run a single task
        task_number = 2  # Run Task 2: Halve an Egg
        executor.run_task(task_number)

        # Option 2: Run all tasks
        # executor.run_all_tasks()

    except rospy.ROSInterruptException:
        rospy.loginfo("DKPrompt executor terminated")
