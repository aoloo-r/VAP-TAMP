#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM View Guide - Uses VLM to assess if current view provides sufficient
information for grasping, and suggests additional viewpoints to gather info.
"""

import rospy
import cv2
import numpy as np
import base64
import requests
import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VLMViewGuide(object):
    """
    Uses VLM to determine if additional viewpoints are needed for grasping.
    Arm moves to suggested viewpoint, gathers information, then returns to
    original observation pose for marker-based grasping.
    """

    def __init__(self, gemini_api_key, camera_topic='/usb_cam/image_raw'):
        """
        Initialize VLM View Guide.

        Args:
            gemini_api_key (str): Google Gemini API key
            camera_topic (str): ROS camera topic to subscribe to
                               Default: '/usb_cam/image_raw' (wrist camera for manipulation)
                               For door detection: '/camera/color/image_raw' (front camera)
        """
        self.api_key = gemini_api_key
        self.bridge = CvBridge()
        self.latest_image = None
        self.model_name = "gemini-2.0-flash-exp"

        # Subscribe to specified camera topic
        # Default: wrist camera for manipulation tasks
        # Can specify front camera for navigation/door detection
        self.camera_topic = camera_topic
        self.image_sub = rospy.Subscriber(
            self.camera_topic,
            Image,
            self._image_callback,
            queue_size=1
        )

        rospy.loginfo("[VLM_VIEW_GUIDE] Initialized with model: {}".format(self.model_name))
        rospy.loginfo("[VLM_VIEW_GUIDE] Subscribed to camera: {}".format(self.camera_topic))
        rospy.loginfo("[VLM_VIEW_GUIDE] Waiting for camera images...")
        rospy.sleep(2.0)  # Wait for first image

    def _image_callback(self, msg):
        """Callback to receive camera images."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to convert image: {}".format(e))

    def _encode_image_to_base64(self, cv_image):
        """
        Encode OpenCV image to base64 string.

        Args:
            cv_image: OpenCV image (numpy array)

        Returns:
            str: Base64 encoded image
        """
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def _query_gemini_vision(self, image_base64, prompt):
        """
        Query Gemini Vision API with an image and prompt.

        Args:
            image_base64 (str): Base64 encoded image
            prompt (str): Text prompt for Gemini

        Returns:
            str: Gemini's response text
        """
        # Use Gemini 2.0 Flash model
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
            self.model_name, self.api_key)

        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 1024,
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            else:
                rospy.logerr("[VLM_VIEW_GUIDE] No candidates in Gemini response")
                return None

        except requests.exceptions.RequestException as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Gemini API request failed: {}".format(e))
            rospy.logerr("[VLM_VIEW_GUIDE] Response content: {}".format(
                response.text if 'response' in locals() else 'No response'))
            return None
        except (KeyError, IndexError) as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to parse Gemini response: {}".format(e))
            return None

    def assess_grasp_information(self, object_name, save_debug=False):
        """
        Ask VLM if current view provides sufficient information for grasping.

        Args:
            object_name (str): Name of object to grasp (e.g., "cup", "bottle")
            save_debug (bool): Save debug images if True

        Returns:
            tuple: (is_sufficient (bool), suggestion (str), reason (str))
                   - is_sufficient: True if current view provides enough info
                   - suggestion: "left_side", "right_side", "front_side", "back_side", "top_side", "sufficient"
                   - reason: Explanation from VLM
        """
        if self.latest_image is None:
            rospy.logerr("[VLM_VIEW_GUIDE] No camera image available")
            return False, "none", "No image available"

        rospy.loginfo("[VLM_VIEW_GUIDE] Assessing grasp information for '{}'...".format(object_name))

        # Save debug image if requested
        if save_debug:
            debug_path = "/tmp/vlm_grasp_assess_{}.jpg".format(rospy.Time.now().to_nsec())
            cv2.imwrite(debug_path, self.latest_image)
            rospy.loginfo("[VLM_VIEW_GUIDE] Saved debug image: {}".format(debug_path))

        # Encode image
        image_base64 = self._encode_image_to_base64(self.latest_image)

        # Create prompt for grasp information assessment
        prompt = """You are a robot vision system determining if the current view provides SUFFICIENT INFORMATION FOR GRASPING a {object_name}.

IMPORTANT: The {object_name} IS visible in this image. The question is whether you have enough information about its graspable features to execute a successful grasp.

For a successful grasp, you need to understand:
1. The object's 3D shape and structure
2. Where to grasp (handle, body, edges, etc.)
3. The object's orientation
4. Potential grasp points and their accessibility

Current view: You are looking at the {object_name} from the FRONT.

Analyze this image and determine if you have enough information to grasp:
- For a cup/mug: Can you see if there's a handle? Do you know which side the handle is on?
- For a bottle: Can you see the cap orientation and body shape clearly?
- For a box/object: Can you identify good grasp points?

If you CANNOT determine the best grasp approach from this front view alone, you need additional viewpoints.

Respond in EXACTLY this format:
SUFFICIENT: [YES/NO]
NEED_VIEW: [front_side/sufficient]
REASON: [explain what grasp information is missing OR why current view is sufficient]

Guidelines:
- If the object IS visible but you need more information about its features (like handle location, orientation, etc.), say NEED_VIEW: front_side
- If you can already determine the best grasp approach, say NEED_VIEW: sufficient

Example responses:
- "SUFFICIENT: NO, NEED_VIEW: front_side, REASON: Cup is visible but cannot determine if there's a handle or which side it's on"
- "SUFFICIENT: YES, NEED_VIEW: sufficient, REASON: Clear view of bottle, can see body and top, sufficient for grasping"
- "SUFFICIENT: NO, NEED_VIEW: front_side, REASON: Cannot see the full shape of the object, need additional angles"
""".format(object_name=object_name)

        # Query VLM
        response = self._query_gemini_vision(image_base64, prompt)

        if response is None:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to get VLM response")
            return False, "none", "VLM query failed"

        rospy.loginfo("[VLM_VIEW_GUIDE] VLM Response:\n{}".format(response))

        # Parse response
        try:
            is_sufficient = False
            suggestion = "sufficient"
            reason = ""

            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('SUFFICIENT:'):
                    answer = line.split(':', 1)[1].strip().upper()
                    is_sufficient = (answer == 'YES')
                elif line.startswith('NEED_VIEW:'):
                    suggestion = line.split(':', 1)[1].strip().lower()
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()

            rospy.loginfo("[VLM_VIEW_GUIDE] Parsed - Sufficient: {}, Need View: {}, Reason: {}".format(
                is_sufficient, suggestion, reason))

            return is_sufficient, suggestion, reason

        except Exception as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to parse VLM response: {}".format(e))
            return False, "none", "Parse error"

    def _ask_for_movement_suggestion(self, object_name, initial_reason):
        """
        Ask VLM what specific camera movement would provide better grasp information.

        Args:
            object_name (str): Name of object
            initial_reason (str): The reason from first assessment explaining what info is missing

        Returns:
            dict: Movement suggestion with type and description
                  e.g., {"type": "pan_left", "amount": "medium", "reason": "..."}
        """
        if self.latest_image is None:
            rospy.logerr("[VLM_VIEW_GUIDE] No camera image available")
            return {"type": "pan_left", "amount": "medium", "reason": "Default exploration"}

        rospy.loginfo("[VLM_VIEW_GUIDE] Asking VLM for intelligent movement suggestion...")

        # Encode current image
        image_base64 = self._encode_image_to_base64(self.latest_image)

        # Create prompt for intelligent active perception
        prompt = """You are a robot vision system with active perception capabilities. Analyze this image CAREFULLY.

PREVIOUS ASSESSMENT:
- The {object_name} IS visible in this view
- Missing information: {initial_reason}

Your task: Decide which camera movement will give you the BEST information for grasping.

CRITICAL: Analyze the image for VISUAL CLUES before deciding:

1. OBJECT POSITION in frame:
   - If object is on LEFT side of image → PAN_LEFT to center it and see it better
   - If object is on RIGHT side of image → PAN_RIGHT to center it and see it better
   - If object is centered → Look for other clues

2. SHADOWS and LIGHTING:
   - Dark shadow on left side of object → Handle/feature might be on LEFT → PAN_LEFT
   - Dark shadow on right side of object → Handle/feature might be on RIGHT → PAN_RIGHT
   - No clear shadow → Choose based on object asymmetry

3. OBJECT ASYMMETRY:
   - Object leans/tilts left → PAN_LEFT to see that side
   - Object leans/tilts right → PAN_RIGHT to see that side
   - Round/symmetric → Look at position in frame

4. SIZE/DISTANCE:
   - Object very small in frame → MOVE_CLOSER first
   - Object very large/close → MOVE_BACK slightly
   - Good size → Pan to sides

5. NEED TOP VIEW:
   - Need to see opening/top → TILT_DOWN

AVAILABLE MOVEMENTS:
- PAN_LEFT: See object's left side
- PAN_RIGHT: See object's right side
- MOVE_CLOSER: Get better detail
- MOVE_BACK: Get wider view
- TILT_DOWN: See top of object
- TILT_UP: See bottom of object

AMOUNTS: small (subtle adjustment), medium (standard), large (exploration)

DO NOT default to right! Choose based on ACTUAL visual clues you see.

Respond in EXACTLY this format:
MOVEMENT: [PAN_LEFT/PAN_RIGHT/MOVE_CLOSER/MOVE_BACK/TILT_DOWN/TILT_UP]
AMOUNT: [small/medium/large]
REASON: [Describe the VISUAL CLUE you observed and why this movement addresses it]

Examples:
- "MOVEMENT: PAN_LEFT, AMOUNT: medium, REASON: Object positioned on left side of frame, panning left will center it and reveal left features"
- "MOVEMENT: PAN_RIGHT, AMOUNT: small, REASON: Shadow visible on right edge suggests handle protrusion, small pan will confirm"
- "MOVEMENT: MOVE_CLOSER, AMOUNT: medium, REASON: Cup appears small and far, closer view needed to see handle details"
- "MOVEMENT: PAN_LEFT, AMOUNT: medium, REASON: Cup body tilts slightly left, suggests handle may be on that side"
""".format(object_name=object_name, initial_reason=initial_reason)

        # Query VLM
        response = self._query_gemini_vision(image_base64, prompt)

        if response is None:
            rospy.logwarn("[VLM_VIEW_GUIDE] Failed to get VLM response, using default movement")
            return {"type": "pan_left", "amount": "medium", "reason": "Default exploration"}

        rospy.loginfo("[VLM_VIEW_GUIDE] VLM Movement Suggestion:\n{}".format(response))

        # Parse response
        try:
            movement = {"type": "pan_left", "amount": "medium", "reason": ""}

            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('MOVEMENT:'):
                    movement["type"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('AMOUNT:'):
                    movement["amount"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('REASON:'):
                    movement["reason"] = line.split(':', 1)[1].strip()

            rospy.loginfo("[VLM_VIEW_GUIDE] Parsed movement: {} ({}) - {}".format(
                movement["type"], movement["amount"], movement["reason"]))
            return movement

        except Exception as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to parse VLM response: {}".format(e))
            return {"type": "pan_left", "amount": "medium", "reason": "Parse error, using default"}

    def gather_additional_information(self, object_name, arm_controller, observation_joints):
        """
        Actively explore viewpoints until sufficient information is gathered for grasping,
        then return to observation pose for marker-based grasping.

        Uses wrist2, wrist1, and elbow joints for active perception.
        Loops multiple times, gathering information from different viewpoints.

        Args:
            object_name (str): Object to assess
            arm_controller: UR5eController instance with arm movement methods
            observation_joints: Joint configuration for observation pose (to return to)

        Returns:
            tuple: (success (bool), viewpoints_used (list))
        """
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("[VLM_VIEW_GUIDE] ACTIVE PERCEPTION MODE: Exploring viewpoints for '{}'".format(object_name))
        rospy.loginfo("[VLM_VIEW_GUIDE] Using wrist2, wrist1, and elbow for camera movements")
        rospy.loginfo("="*70)

        # Movement amounts in radians
        # JOINT_FOR_SEARCH (observation pose) = [0, -0.9, 0.5, -1.57, -1.57, 0]
        # Joint indices: [0=shoulder_pan, 1=shoulder_lift, 2=elbow, 3=wrist_1, 4=wrist_2, 5=wrist_3]
        # Using wrist2 (index 4), wrist1 (index 3), and elbow (index 2) for active perception
        MOVEMENT_DELTAS = {
            # Pan movements: primarily wrist2 with some wrist1 and elbow support
            "pan_left": {
                "small": [0, 0, 0.1, 0.1, 0.3, 0],    # elbow, wrist1, wrist2
                "medium": [0, 0, 0.2, 0.2, 0.5, 0],
                "large": [0, 0, 0.3, 0.3, 0.7, 0]
            },
            "pan_right": {
                "small": [0, 0, -0.1, -0.1, -0.3, 0],
                "medium": [0, 0, -0.2, -0.2, -0.5, 0],
                "large": [0, 0, -0.3, -0.3, -0.7, 0]
            },
            # Move closer/back: using elbow and wrist1
            "move_closer": {
                "small": [0, -0.05, 0.1, 0.05, 0, 0],
                "medium": [0, -0.1, 0.2, 0.1, 0, 0]
            },
            "move_back": {
                "small": [0, 0.05, -0.1, -0.05, 0, 0],
                "medium": [0, 0.1, -0.2, -0.1, 0, 0]
            },
            # Tilt movements: using wrist1 and elbow
            "tilt_down": {
                "small": [0, 0, -0.1, 0.15, 0, 0],
                "medium": [0, 0, -0.15, 0.25, 0, 0]
            },
            "tilt_up": {
                "small": [0, 0, 0.1, -0.15, 0, 0],
                "medium": [0, 0, 0.15, -0.25, 0, 0]
            },
        }

        movements_performed = []

        # PHASE 1: VLM suggests initial direction
        rospy.loginfo("\n" + "-"*70)
        rospy.loginfo("[VLM_VIEW_GUIDE] PHASE 1: Initial Assessment")
        rospy.loginfo("-"*70)

        # Wait for camera to stabilize
        rospy.sleep(2.0)

        # Assess current viewpoint
        is_sufficient, need_view, reason = self.assess_grasp_information(
            object_name, save_debug=True)

        rospy.loginfo("[VLM_VIEW_GUIDE] Initial assessment:")
        rospy.loginfo("  - Sufficient: {}".format(is_sufficient))
        rospy.loginfo("  - Need view: {}".format(need_view))
        rospy.loginfo("  - Reason: {}".format(reason))

        if not is_sufficient:
            # Ask VLM for intelligent movement suggestion
            rospy.loginfo("\n[VLM_VIEW_GUIDE] Asking VLM for best viewpoint direction...")
            movement = self._ask_for_movement_suggestion(object_name, reason)

            rospy.loginfo("[VLM_VIEW_GUIDE] VLM suggests: {} direction".format(
                movement["type"].replace("_", " ")))
            rospy.loginfo("[VLM_VIEW_GUIDE] Reasoning: {}".format(movement["reason"]))

            # Execute base/shoulder movement to VLM-suggested direction
            if movement["type"] in MOVEMENT_DELTAS:
                rospy.loginfo("\n[VLM_VIEW_GUIDE] Moving base/shoulder to {} position...".format(
                    movement["type"].replace("_", " ")))

                # Get current joints
                current_joints = arm_controller.move_group.get_current_joint_values()
                target_joints = list(current_joints)

                # Apply movement (use medium for base movement)
                step_amount = "medium"
                delta_list = MOVEMENT_DELTAS[movement["type"]][step_amount]
                for i in range(len(delta_list)):
                    target_joints[i] += delta_list[i]

                # Execute movement
                success = arm_controller._move_to_target_by_joint(target_joints)

                if success:
                    movements_performed.append({
                        "type": movement["type"],
                        "amount": step_amount,
                        "reason": movement["reason"]
                    })
                    rospy.loginfo("[VLM_VIEW_GUIDE] Base/shoulder movement completed")
                    rospy.sleep(2.0)
                else:
                    rospy.logwarn("[VLM_VIEW_GUIDE] Base movement failed")

            # PHASE 2: Active wrist exploration sequence
            rospy.loginfo("\n" + "-"*70)
            rospy.loginfo("[VLM_VIEW_GUIDE] PHASE 2: Active Wrist Exploration (4 views)")
            rospy.loginfo("-"*70)
            rospy.loginfo("[VLM_VIEW_GUIDE] Wrist will actively search around object...")

            # Define wrist exploration sequence using wrist_2 primarily
            # This creates a natural "looking around" behavior
            wrist_exploration_sequence = [
                ("pan_left", "small", "Wrist scanning left side"),
                ("tilt_down", "small", "Wrist looking down from above"),
                ("pan_right", "small", "Wrist scanning right side"),
                ("tilt_up", "small", "Wrist looking up from below")
            ]

            # Execute all 4 wrist movements in sequence
            for idx, (wrist_move_type, wrist_amount, wrist_reason) in enumerate(wrist_exploration_sequence):
                rospy.loginfo("\n[VLM_VIEW_GUIDE] Wrist view {}/4: {}".format(idx + 1, wrist_reason))

                if wrist_move_type in MOVEMENT_DELTAS:
                    # Get current joints
                    current_joints = arm_controller.move_group.get_current_joint_values()
                    target_joints = list(current_joints)

                    # Apply wrist movement
                    wrist_delta = MOVEMENT_DELTAS[wrist_move_type][wrist_amount]
                    for i in range(len(wrist_delta)):
                        target_joints[i] += wrist_delta[i]

                    # Execute wrist movement
                    success = arm_controller._move_to_target_by_joint(target_joints)

                    if success:
                        movements_performed.append({
                            "type": wrist_move_type,
                            "amount": wrist_amount,
                            "reason": wrist_reason
                        })
                        rospy.loginfo("[VLM_VIEW_GUIDE] Wrist movement completed")
                        rospy.sleep(1.5)  # Brief pause between wrist movements
                    else:
                        rospy.logwarn("[VLM_VIEW_GUIDE] Wrist movement failed")
        else:
            rospy.loginfo("[VLM_VIEW_GUIDE] Sufficient information from initial view, skipping exploration")

        # After active exploration, return to observation pose for marker-based grasping
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("[VLM_VIEW_GUIDE] ACTIVE PERCEPTION COMPLETE")
        rospy.loginfo("="*70)
        rospy.loginfo("[VLM_VIEW_GUIDE] Total movements performed: {}".format(len(movements_performed)))
        rospy.loginfo("[VLM_VIEW_GUIDE] Exploration pattern: VLM-suggested position + 4 wrist views")
        rospy.loginfo("[VLM_VIEW_GUIDE] Returning to observation pose for marker-based grasping...")

        # Return to observation pose
        success = arm_controller._move_to_target_by_joint(observation_joints)

        if success:
            rospy.loginfo("[VLM_VIEW_GUIDE] Successfully returned to observation pose")
            rospy.sleep(2.0)  # Wait for camera to stabilize
        else:
            rospy.logwarn("[VLM_VIEW_GUIDE] Failed to return to observation pose")

        rospy.loginfo("[VLM_VIEW_GUIDE] Ready for marker-based grasping from observation pose")

        return True, movements_performed

    def assess_door_status(self, save_debug=False):
        """
        Ask VLM if current view provides sufficient information to determine door status.

        This is similar to assess_grasp_information() but for navigation obstacles.
        Typically used with FRONT camera (/camera/color/image_raw) not wrist camera.

        Args:
            save_debug (bool): Save debug images if True

        Returns:
            tuple: (is_sufficient (bool), door_status (str), suggestion (str), reason (str))
                   - is_sufficient: True if current view provides enough info
                   - door_status: "OPEN", "CLOSED", "HALF_OPEN", "UNKNOWN"
                   - suggestion: "move_left", "move_right", "move_closer", "move_back", "sufficient"
                   - reason: Explanation from VLM
        """
        if self.latest_image is None:
            rospy.logerr("[VLM_VIEW_GUIDE] No camera image available")
            return False, "UNKNOWN", "none", "No image available"

        rospy.loginfo("[VLM_VIEW_GUIDE] Assessing door status from current view...")

        # Save debug image if requested
        if save_debug:
            debug_path = "/tmp/vlm_door_assess_{}.jpg".format(rospy.Time.now().to_nsec())
            cv2.imwrite(debug_path, self.latest_image)
            rospy.loginfo("[VLM_VIEW_GUIDE] Saved debug image: {}".format(debug_path))

        # Encode image
        image_base64 = self._encode_image_to_base64(self.latest_image)

        # Create prompt for door status assessment
        prompt = """You are a robot navigation system analyzing the view from the robot's FRONT camera.

Your task: Determine if you have SUFFICIENT INFORMATION to know if a door ahead is OPEN, CLOSED, or HALF-OPEN.

IMPORTANT:
- You may or may not see a door in this image
- The question is whether you have enough information to make a confident determination

Analyze this image carefully:

1. Is there a door visible in the current view?
2. If YES, can you confidently determine if it's:
   - OPEN (fully open, passage is clear)
   - CLOSED (fully closed, blocking passage)
   - HALF_OPEN (partially open)
3. If you CANNOT make a confident determination, what additional viewpoint would help?

VIEWPOINT SUGGESTIONS (if current view is insufficient):
- move_left: Robot should move left to see the door from a different angle
- move_right: Robot should move right to see the door from a different angle
- move_closer: Need to get closer to see door details better
- move_back: Need to get farther to see the full door
- rotate_left: Need to rotate left to see if there's a door
- rotate_right: Need to rotate right to see if there's a door

Respond in EXACTLY this format:
SUFFICIENT: [YES/NO]
DOOR_STATUS: [OPEN/CLOSED/HALF_OPEN/UNKNOWN]
NEED_VIEW: [move_left/move_right/move_closer/move_back/rotate_left/rotate_right/sufficient]
REASON: [Explain what you see and why you need (or don't need) a different viewpoint]

Examples:
- "SUFFICIENT: YES, DOOR_STATUS: OPEN, NEED_VIEW: sufficient, REASON: Door is clearly visible and fully open, passage is clear"
- "SUFFICIENT: YES, DOOR_STATUS: CLOSED, NEED_VIEW: sufficient, REASON: Door is clearly visible and fully closed, blocking the passage"
- "SUFFICIENT: NO, DOOR_STATUS: UNKNOWN, NEED_VIEW: move_right, REASON: Can only see door edge from current angle, need to move right to see if it's open or closed"
- "SUFFICIENT: NO, DOOR_STATUS: UNKNOWN, NEED_VIEW: move_closer, REASON: Door is visible but too far to determine if it's fully closed or slightly ajar"
- "SUFFICIENT: YES, DOOR_STATUS: HALF_OPEN, NEED_VIEW: sufficient, REASON: Door is clearly partially open, can see gap between door and frame"
"""

        # Query VLM
        response = self._query_gemini_vision(image_base64, prompt)

        if response is None:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to get VLM response")
            return False, "UNKNOWN", "none", "VLM query failed"

        rospy.loginfo("[VLM_VIEW_GUIDE] VLM Response:\n{}".format(response))

        # Parse response
        try:
            is_sufficient = False
            door_status = "UNKNOWN"
            suggestion = "sufficient"
            reason = ""

            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('SUFFICIENT:'):
                    answer = line.split(':', 1)[1].strip().upper()
                    is_sufficient = (answer == 'YES')
                elif line.startswith('DOOR_STATUS:'):
                    door_status = line.split(':', 1)[1].strip().upper()
                elif line.startswith('NEED_VIEW:'):
                    suggestion = line.split(':', 1)[1].strip().lower()
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()

            rospy.loginfo("[VLM_VIEW_GUIDE] Parsed - Sufficient: {}, Door: {}, Need View: {}, Reason: {}".format(
                is_sufficient, door_status, suggestion, reason))

            return is_sufficient, door_status, suggestion, reason

        except Exception as e:
            rospy.logerr("[VLM_VIEW_GUIDE] Failed to parse VLM response: {}".format(e))
            return False, "UNKNOWN", "none", "Parse error"
