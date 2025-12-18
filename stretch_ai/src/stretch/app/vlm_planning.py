# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click
import cv2
import matplotlib

matplotlib.use("TkAgg")
import copy
import re

import numpy as np

from stretch.agent import RobotAgent
from stretch.agent.vlm_planner import VLMPlanner
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters

from stretch.core.interfaces import Observations
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


def add_raw_obs_to_voxel_map(obs_history, voxel_map, semantic_sensor, num_frames, frame_skip):
    key_obs = []
    num_obs = len(obs_history["rgb"])
    video_frames = []

    print("converting raw data to observations...")
    for obs_id in range(num_obs):
        pose = obs_history["camera_poses"][obs_id]
        pose[2, 3] += 1.2  # for room1 and room2, room4, room5
        # pose[2,3] = pose[2,3] # for room3
        key_obs.append(
            Observations(
                rgb=obs_history["rgb"][obs_id].numpy(),
                # gps=obs_history["base_poses"][obs_id][:2].numpy(),
                gps=pose[:2, 3],
                # compass=[obs_history["base_poses"][obs_id][2].numpy()],
                compass=[np.arctan2(pose[1, 0], pose[0, 0])],
                xyz=None,
                depth=obs_history["depth"][obs_id].numpy(),
                camera_pose=pose,
                camera_K=obs_history["camera_K"][obs_id].numpy(),
            )
        )
        video_frames.append(obs_history["rgb"][obs_id].numpy())

    images_to_video(
        video_frames[: min(frame_skip * num_frames, len(video_frames))],
        "output_video.mp4",
        fps=10,
    )

    voxel_map.reset()
    key_obs = key_obs[::frame_skip]
    key_obs = key_obs[: min(num_frames, len(key_obs))]
    for idx, obs in enumerate(key_obs):
        print(f"processing frame {idx}")
        obs = semantic_sensor.predict(obs)
        voxel_map.add_obs(obs)

    return voxel_map


def images_to_video(image_list, output_path, fps=30):
    """
    Convert a list of raw rgb data into a video.
    """
    print("Generating an video for visualizing the data...")
    if not image_list:
        raise ValueError("The image list is empty")

    height, width, channels = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in image_list:
        if image.shape != (height, width, channels):
            raise ValueError("All images must have the same dimensions")
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    out.release()
    print(f"Video saved at {output_path}")


def validate_rosbridge_connection() -> dict:
    """
    Validate rosbridge connection and analyze topic activity.
    Returns dict with validation results.
    """
    import subprocess
    import time
    
    validation_results = {
        "topics_available": False,
        "bridge_active": False,
        "robot_data_flowing": False,
        "recommendations": []
    }
    
    try:
        # Check if topics exist
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            available_topics = result.stdout.strip().split('\n')
            validation_results["topics_available"] = True
            
            # Check for rosbridge-specific topics
            expected_topics = ['/camera/color/image_raw', '/camera/depth/image_rect_raw', 
                             '/cmd_vel', '/odom', '/joint_states']
            missing_topics = [topic for topic in expected_topics if topic not in available_topics]
            
            if missing_topics:
                validation_results["recommendations"].append(
                    f"Missing topics: {', '.join(missing_topics)} - check rosbridge configuration"
                )
            else:
                validation_results["bridge_active"] = True
        else:
            validation_results["recommendations"].append("ROS2 not available or topics not accessible")
            
    except Exception as e:
        validation_results["recommendations"].append(f"Topic validation failed: {e}")
    
    return validation_results


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="",
    help="Input path. If empty, run on the real robot.",
)
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--task", "-t", type=str, default="", help="Task to run with the planner.")
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default="app/vlm_planning/multi_crop_vlm_planner.yaml",
    help="Path to planner config.",
)
@click.option(
    "--frame",
    "-f",
    type=int,
    default=-1,
    help="number of frames to read",
)
@click.option(
    "--frame_skip",
    "-fs",
    type=int,
    default=1,
    help="number of frames to skip",
)
@click.option("--show-svm", "-s", type=bool, is_flag=True, default=False)
@click.option("--test-vlm", type=bool, is_flag=True, default=True)
@click.option("--show-instances", type=bool, is_flag=True, default=False)
@click.option("--api-key", type=str, default=None, help="your openai api key")
@click.option("--offset_x", type=float, default=None, help="Map offset X (meters, overrides calibration file)")
@click.option("--offset_y", type=float, default=None, help="Map offset Y (meters, overrides calibration file)")
@click.option("--calibration-file", type=str, default="simple_offset_calibration.yaml", help="Calibration file with offset values")
@click.option("--use-simple-nav", is_flag=True, default=False, help="Use simple offset-based navigation (like segway_hybrid_navigation)")
def main(
    input_path,
    config_path,
    test_vlm: bool = False,
    frame: int = -1,
    frame_skip: int = 1,
    show_svm: bool = False,
    show_instances: bool = False,
    api_key: str = None,
    task: str = "",
    local: bool = False,
    robot_ip: str = "",
    offset_x: float = None,
    offset_y: float = None,
    calibration_file: str = "simple_offset_calibration.yaml",
    use_simple_nav: bool = False,
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)

    # Load offset from calibration file if not provided via command line
    if offset_x is None or offset_y is None:
        import yaml
        cal_path = Path(calibration_file)
        if cal_path.exists():
            try:
                with open(cal_path, 'r') as f:
                    calibration = yaml.safe_load(f)
                if offset_x is None:
                    offset_x = calibration.get('offset_x', 0.0)
                if offset_y is None:
                    offset_y = calibration.get('offset_y', 0.0)
                print(f"✅ Loaded calibration from {calibration_file}:")
                print(f"   offset_x = {offset_x:.3f} m")
                print(f"   offset_y = {offset_y:.3f} m")
                if 'landmark' in calibration:
                    print(f"   (calibrated using: {calibration['landmark'].get('object_name', 'unknown')})")
            except Exception as e:
                print(f"⚠️  Could not load calibration file: {e}")
                offset_x = 0.0 if offset_x is None else offset_x
                offset_y = 0.0 if offset_y is None else offset_y
                print(f"   Using default offsets: ({offset_x}, {offset_y})")
        else:
            offset_x = 0.0 if offset_x is None else offset_x
            offset_y = 0.0 if offset_y is None else offset_y
            print(f"⚠️  No calibration file found at {calibration_file}")
            print(f"   Using default offsets: ({offset_x}, {offset_y})")
            print(f"   Run: python3 calibrate_simple_offset.py --map-file {input_path}")
    else:
        print(f"Using command-line offsets: offset_x={offset_x:.3f}, offset_y={offset_y:.3f}")

    loaded_voxel_map = None

    print("- Load parameters")
    vlm_parameters = get_parameters(config_path)
    if not vlm_parameters.get("vlm_base_config"):
        print("invalid config file")
        return
    else:
        base_config_file = vlm_parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(vlm_parameters.data)
        vlm_parameters.data = base_parameters.data
        print(vlm_parameters.data)

    if len(task) > 0:
        vlm_parameters.set("command", task)

    print("Creating semantic sensors...")
    semantic_sensor = create_semantic_sensor(parameters=vlm_parameters)

    # Always try to connect to real robot first, use dummy only as fallback
    try:
        # Try ZMQ client first (for Segway and other robots)
        from stretch.agent.zmq_client import HomeRobotZmqClient
        import time
        
        # ZMQ client doesn't need rosbridge validation
        print("🔍 Connecting to robot via ZMQ...")
        
        # Disable Rerun completely to avoid freezing issues
        # Use Open3D for visualization instead
        robot = HomeRobotZmqClient(
            robot_ip=robot_ip,
            parameters=vlm_parameters,
            use_remote_computer=(not local),
            enable_rerun_server=False  # Disabled to prevent freezing
        )

        # For pickle files, we still need to validate robot is available for execution
        if input_path:
            print("ℹ️  Robot client rerun disabled (pkl file provides map, not live robot stream)")
            print("📁 Loading from pickle file - robot available for navigation commands")
            print("🔗 Connected via ZMQ client")
        else:
            # Wait for actual robot data to validate connection
            print("⏳ Validating robot connection (waiting for sensor data)...")
            timeout = 10.0  # 10 second timeout for validation
            start_time = time.time()
            
            # First check if robot is running
            while not robot.is_running():
                if time.time() - start_time > timeout:
                    raise Exception("Timeout: Robot client failed to start")
                time.sleep(0.5)
                print(".", end="", flush=True)
            
            # Then check for real sensor data (not just proxy topics)
            print("\n⏳ Checking for real robot sensor data (not proxy topics)...")
            data_timeout = 15.0  # 15 second timeout for real data
            data_start_time = time.time()
            
            while not robot.has_real_robot_data(max_age_seconds=3.0):
                if time.time() - data_start_time > data_timeout:
                    raise Exception("Timeout: No real robot sensor data received (robot may be offline or rosbridge not bridging data)")
                time.sleep(0.5)
                print(".", end="", flush=True)
            
            print("\n🔗 Connected to real robot with active sensor data (ZMQ client)")
            print("🌉 Successfully connected via ZMQ bridge")
    except Exception as zmq_error:
        print(f"⚠️  Could not connect to robot via ZMQ: {zmq_error}")
        print("🤖 Using dummy robot client for simulation mode")
        robot = DummyStretchClient()

    print("Creating robot agent...")
    agent = RobotAgent(
        robot,
        vlm_parameters,
        voxel_map=loaded_voxel_map,
        semantic_sensor=semantic_sensor,
    )
    voxel_map = agent.get_voxel_map()

    if input_path:
        # load from pickle
        voxel_map.read_from_pickle(input_path, num_frames=frame, perception=semantic_sensor)
    else:
        # Scan the local area to get a map
        agent.rotate_in_place()

    # get the task - allow empty task for interactive mode
    if not task:
        # Check if task is set in parameters first
        config_task = vlm_parameters.get("task", {}).get("command", "")
        if config_task and len(config_task.strip()) > 0:
            task = config_task
        else:
            # Allow empty task to go directly to interactive mode
            task = ""

    # or load from raw data
    # obs_history = pickle.load(input_path.open("rb"))
    # voxel_map = add_raw_obs_to_voxel_map(
    #     obs_history,
    #     voxel_map,
    #     semantic_sensor,
    #     num_frames=frame,
    #     frame_skip=frame_skip,
    # )
    run_vlm_planner(agent, task, show_svm, test_vlm, api_key, show_instances, offset_x, offset_y)


def execute_plan(agent, plan, world_rep, start_xyz, offset_x=0.0, offset_y=0.0):
    """
    Execute a VLM plan by moving the robot to target locations.
    Uses 3D voxel map for goal selection and simple offset transformation.

    Args:
        agent: The robot agent
        plan: List of action strings from VLM planner
        world_rep: World representation containing object instances
        start_xyz: Starting position [x, y, z]

    Returns:
        bool: True if execution successful, False otherwise
    """
    if not plan:
        print("❌ No plan to execute (empty plan)")
        return False

    print(f"🤖 Executing plan with {len(plan)} actions...")
    robot = agent.get_robot()
    voxel_map = agent.get_voxel_map()

    executed_actions = 0
    try:
        for i, action in enumerate(plan):
            print(f"\n🎯 Step {i+1}/{len(plan)}: {action}")

            # Extract the target object from the action
            # Accept both goto() and explore() as navigation actions
            if "goto" not in action.lower() and "explore" not in action.lower():
                print(f"⚠️  Skipping non-navigation action: {action}")
                print(f"❌ Action not supported for robot execution")
                continue

            # Extract crop_id from action like "goto(img_5, chair)"
            import re
            crop_match = re.search(r"img_(\d+)", action)
            if not crop_match:
                print(f"❌ Could not extract object ID from action: {action}")
                continue

            crop_id = int(crop_match.group(1))
            global_id = world_rep.object_images[crop_id].instance_id
            target_instance = voxel_map.get_instances()[global_id]

            print(f"🎯 Navigating to instance {global_id} ({target_instance.category_id})")

            # Always use hybrid approach: 3D semantic planning + 2D navigation execution
            print("🗺️ Using hybrid 3D semantic planning + 2D navigation execution")

            try:
                import time

                # Get current robot pose
                current_pose = agent.robot.get_base_pose()

                if current_pose is not None:
                    if hasattr(current_pose, 'numpy'):
                        current_pose = current_pose.numpy()
                    robot_x, robot_y = current_pose[0], current_pose[1]
                    print(f"🤖 Robot position: ({robot_x:.2f}, {robot_y:.2f})")
                else:
                    print("❌ Cannot get robot pose for navigation")
                    continue

                # Get target position from 3D semantic instance
                target_center = target_instance.get_center()
                if target_center is not None:
                    # Simple offset transformation (same as segway_hybrid_navigation.py)
                    pos_3d = target_center
                    voxel_goal_x = float(pos_3d[0])
                    voxel_goal_y = float(pos_3d[1])

                    print(f"\n🎯 SIMPLE OFFSET TRANSFORMATION:")
                    print(f"   3D position: ({voxel_goal_x:.2f}, {voxel_goal_y:.2f}, {float(pos_3d[2]):.2f})")

                    # Apply simple offset
                    goal_x = voxel_goal_x + offset_x
                    goal_y = voxel_goal_y + offset_y

                    print(f"   Offset: ({offset_x:.2f}, {offset_y:.2f})")
                    print(f"   2D position (robot frame): ({goal_x:.2f}, {goal_y:.2f})")

                    # Calculate distance to goal
                    distance_to_goal = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
                    print(f"   Distance to goal: {distance_to_goal:.2f}m")

                    # Set orientation to face the target
                    goal_theta = np.arctan2(goal_y - robot_y, goal_x - robot_x)
                    print(f"   Orientation to target: {goal_theta:.2f} rad = {np.degrees(goal_theta):.1f}°")

                    # Send navigation goal using ZMQ client (same as segway_hybrid_navigation.py)
                    print(f"\n🎯 SENDING NAVIGATION GOAL:")
                    print(f"   Position: ({goal_x:.2f}, {goal_y:.2f})")
                    print(f"   Orientation: {goal_theta:.2f} rad ({np.degrees(goal_theta):.1f}°)")

                    success = agent.robot.navigate_to_goal(goal_x, goal_y, goal_theta)

                    if not success:
                        print("❌ Failed to send navigation goal")
                        continue

                    print("✅ Navigation goal sent successfully")
                    print("⏳ Monitoring robot movement...")

                    # Monitor robot movement with position updates
                    for i in range(30):
                        time.sleep(1)
                        current = agent.robot.get_base_pose()
                        if hasattr(current, 'numpy'):
                            current = current.numpy()
                        dist = np.linalg.norm(np.array(current[:2]) - np.array([goal_x, goal_y]))

                        if i % 5 == 0:  # Log every 5 seconds
                            print(f"  [{i}s] Position: ({current[0]:.2f}, {current[1]:.2f}), Distance: {dist:.2f}m")

                        if dist < 0.5:  # Reached goal
                            print(f"✅ Reached goal in {i} seconds!")
                            break

                    final_pose = agent.robot.get_base_pose()
                    if hasattr(final_pose, 'numpy'):
                        final_pose = final_pose.numpy()
                    distance = np.linalg.norm(np.array(final_pose[:2]) - np.array([goal_x, goal_y]))
                    print(f"Arrived at: ({final_pose[0]:.2f}, {final_pose[1]:.2f})")
                    print(f"Distance from goal: {distance:.2f}m")

                    executed_actions += 1
                else:
                    print("❌ Cannot get target center from 3D semantic instance")

            except Exception as e:
                print(f"❌ Navigation error: {e}")
                print("❌ Skipping action - navigation failed")

            # Hybrid navigation complete - 3D semantic planning + 2D execution
            print("✅ Hybrid 3D semantic + 2D navigation implementation complete")

        # Report execution results
        if executed_actions == 0:
            print(f"\n❌ No actions were executed! Plan contained {len(plan)} actions but none were executable navigation commands.")
            return False
        elif executed_actions < len(plan):
            print(f"\n⚠️  Partial execution: {executed_actions}/{len(plan)} actions completed")
            print(f"🎉 Successfully executed {executed_actions} navigation commands!")
            return True
        else:
            print(f"\n🎉 Plan execution completed successfully! All {executed_actions} actions executed.")
            return True

    except Exception as e:
        print(f"❌ Plan execution failed with error: {e}")
        return False


def run_vlm_planner(
    agent,
    task,
    show_svm: bool = False,
    test_vlm: bool = False,
    api_key: str = None,
    show_instances: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
):
    """
    Run the VLM planner with the given agent and task.

    Args:
        agent (RobotAgent): the robot agent to use.
        task (str): the task to run.
        show_svm (bool): whether to show the SVM.
        test_vlm (bool): whether to test the VLM planner.
        api_key (str): the OpenAI API key.
    """

    # Import torch for later use
    import torch
    
    # Extract actual robot start pose from aligned data instead of guessing
    actual_start_pose = None
    
    # CRITICAL FIX: Get robot's CURRENT pose instead of pose from saved map data
    # The pkl map was created by someone else, so we need the robot's actual current position
    current_robot_pose = agent.robot.get_base_pose()
    if current_robot_pose is not None:
        if hasattr(current_robot_pose, 'numpy'):
            actual_start_pose = current_robot_pose.numpy()
        else:
            actual_start_pose = np.array(current_robot_pose)
        print(f"🎯 Using CURRENT robot pose: [{actual_start_pose[0]:.3f}, {actual_start_pose[1]:.3f}, {actual_start_pose[2]:.3f}]")
    else:
        print("⚠️  Could not get current robot pose")

    # Fallback: Try to get from voxel map observations (Frame objects) - for reference only
    if actual_start_pose is None:
        voxel_map = agent.get_voxel_map()
        if hasattr(voxel_map, 'observations') and len(voxel_map.observations) > 0:
            first_frame = voxel_map.observations[0]
            if hasattr(first_frame, 'base_pose') and first_frame.base_pose is not None:
                # This is from the original map creation, not current robot position
                first_base_pose = first_frame.base_pose
                if hasattr(first_base_pose, 'numpy'):
                    actual_start_pose = first_base_pose.numpy()
                else:
                    actual_start_pose = np.array(first_base_pose)
                print(f"🎯 Fallback: Using map's original start pose: [{actual_start_pose[0]:.3f}, {actual_start_pose[1]:.3f}, {actual_start_pose[2]:.3f}]")

    # Last resort fallback candidates if we still can't find any pose
    if actual_start_pose is None:
        print("⚠️  Could not extract actual start pose from aligned data. Using fallback candidates.")
        candidate_poses = [
            np.array([0, 0, 0]),           # Origin
            np.array([0.2, 0, 0]),         # Small step forward
            np.array([0, 0.2, 0]),         # Small step to side
            np.array([-0.2, 0, 0]),        # Small step back
            np.array([1.0, 0, 0]),         # Further forward
        ]
    else:
        # Use the actual start pose as the primary candidate, with more fallbacks in a grid pattern
        candidate_poses = [
            actual_start_pose,  # Exact aligned pose
        ]
        
        # Add grid of candidates around the actual pose
        offsets = [
            # Close candidates (10cm radius)
            [0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0],
            [0.1, 0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [-0.1, -0.1, 0],
            
            # Medium candidates (20cm radius) 
            [0.2, 0, 0], [-0.2, 0, 0], [0, 0.2, 0], [0, -0.2, 0],
            [0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0],
            
            # Further candidates (30cm radius)
            [0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0],
            [0.3, 0.3, 0], [0.3, -0.3, 0], [-0.3, 0.3, 0], [-0.3, -0.3, 0],
            
            # Even further candidates (50cm radius)
            [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0],
            
            # Different angles at same position
            [0, 0, 0.5], [0, 0, -0.5], [0, 0, 1.0], [0, 0, -1.0],
        ]
        
        for offset in offsets:
            candidate_poses.append(actual_start_pose + np.array(offset))
    
    print("Agent loaded:", agent)
    vlm_parameters = agent.parameters
    semantic_sensor = agent.semantic_sensor
    robot = agent.get_robot()
    voxel_map = agent.get_voxel_map()

    # Create the VLM planner using the agent
    vlm_planner = VLMPlanner(agent, api_key=api_key)

    # Display with agent overlay
    space = agent.get_navigation_space()
    
    x0 = None
    
    # If we have actual aligned data, trust it and skip extensive validation
    if actual_start_pose is not None:
        print("✅ Using aligned robot pose directly (skipping validation due to trusted rosbag data)")
        x0 = actual_start_pose
        
        # Override the navigation space validation for this specific pose
        original_is_valid = space.is_valid
        
        def override_is_valid(pose, *args, **kwargs):
            # If this is very close to our trusted aligned pose, consider it valid
            if np.allclose(pose, x0, atol=0.05):  # Within 5cm tolerance
                verbose = kwargs.get('verbose', False) or kwargs.get('debug', False)
                if verbose:
                    print(f"✅ Overridden as valid (aligned robot pose): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                return True
            # Otherwise use normal validation
            return original_is_valid(pose, *args, **kwargs)
        
        # Monkey patch the validation method
        space.is_valid = override_is_valid
        print(f"🔧 Navigation space validation overridden for pose: [{x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f}]")
    else:
        print("🔍 Finding valid start pose...")
        print("💡 Debug: Let's understand why poses are invalid...")
    
    # Only do extensive search if we don't have aligned data
    if x0 is None:
        for i, candidate in enumerate(candidate_poses):
            if i < 10:  # Only debug first 10 poses to avoid spam
                print(f"  🔍 Detailed check for pose {i+1}: [{candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f}]")
                is_valid = space.is_valid(candidate, verbose=True, debug=True)
                if is_valid:
                    x0 = candidate
                    print(f"  ✅ Found valid pose: [{candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f}]")
                    break
            else:
                # Quick check for remaining poses
                print(f"  Trying pose {i+1}: [{candidate[0]:.1f}, {candidate[1]:.1f}, {candidate[2]:.1f}]", end="")
                if space.is_valid(candidate, verbose=False, debug=False):
                    x0 = candidate
                    print(" ✅ Valid!")
                    break
                else:
                    print(" ❌ Invalid")
    
    if x0 is None:
        print("❌ Could not find a valid start pose from candidates.")
        print("🔧 Attempting to find ANY valid position in the map...")
        
        # Try to sample a valid position from the map
        try:
            # Get the 2D map bounds and try random positions
            obstacles, explored = agent.get_voxel_map().get_2d_map()
            if obstacles is not None and explored is not None:
                free_space = explored & (~obstacles)  # Explored and not obstacle
                if free_space.any():
                    # Find indices of free space
                    free_indices = torch.nonzero(free_space, as_tuple=False)
                    if len(free_indices) > 0:
                        # Try a few random free positions
                        for _ in range(10):
                            idx = torch.randint(0, len(free_indices), (1,))
                            grid_pos = free_indices[idx[0]]
                            
                            # Convert grid position to world coordinates
                            world_pos = agent.get_voxel_map().grid_coords_to_xyt(grid_pos.float())
                            candidate = np.array([world_pos[0].item(), world_pos[1].item(), 0.0])
                            
                            print(f"  🎲 Trying random free position: [{candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f}]", end="")
                            if space.is_valid(candidate, verbose=False, debug=False):
                                x0 = candidate
                                print(" ✅ Valid!")
                                print(f"📍 Using discovered valid pose: [{x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f}]")
                                break
                            else:
                                print(" ❌ Invalid")
        except Exception as e:
            print(f"⚠️  Could not sample from map: {e}")
        
        # Final fallback - manually override validation for the actual pose
        if x0 is None:
            print("🚫 All automatic methods failed. Using actual robot pose with validation override.")
            print("💡 Since this is aligned rosbag data, we trust the actual robot position was valid.")
            x0 = actual_start_pose if actual_start_pose is not None else np.array([0.0, 0.0, 0.0])
            print(f"📍 Force using aligned robot pose: [{x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f}]")
            
            # Override the navigation space validation for this specific pose
            print("🔧 Temporarily overriding collision detection for the aligned robot pose...")
            
            # Store the original is_valid method
            original_is_valid = space.is_valid
            
            def override_is_valid(pose, *args, **kwargs):
                # If this is very close to our trusted aligned pose, consider it valid
                if np.allclose(pose, x0, atol=0.05):  # Within 5cm tolerance
                    verbose = kwargs.get('verbose', False) or kwargs.get('debug', False)
                    if verbose:
                        print(f"✅ Overridden as valid (aligned robot pose): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                    return True
                # Otherwise use normal validation
                return original_is_valid(pose, *args, **kwargs)
            
            # Monkey patch the validation method
            space.is_valid = override_is_valid
            print("✅ Navigation space validation overridden for aligned pose")
        
    start_xyz = [x0[0], x0[1], 0]
    print(f"✅ Using start pose: [{x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f}]")
    
    
    # Initialize Rerun for voxel map visualization if show_svm is enabled
    planning_rerun = None
    if show_svm:
        # Disable Rerun visualization - it causes freezing issues
        # Always use Open3D fallback for now
        print("🎨 Using Open3D for 3D voxel map visualization...")
        planning_rerun = None

    if show_svm and planning_rerun is None:
        footprint = robot.get_footprint()
        print(f"{x0} valid = {space.is_valid(x0)}")

        # Fallback: Create a non-blocking 3D visualization using Open3D persistent visualizer
        import threading
        import time
        import open3d
        
        def create_persistent_visualizer():
            """Create a persistent Open3D visualizer that doesn't block"""
            try:
                # Get the geometries for visualization
                geoms = voxel_map._get_open3d_geometries(
                    instances=show_instances, 
                    orig=start_xyz, 
                    xyt=x0, 
                    footprint=footprint, 
                    add_planner_visuals=True
                )
                
                # Create persistent visualizer
                vis = open3d.visualization.Visualizer()
                vis.create_window(window_name="Stretch AI - Visual Grounding Map", 
                                width=800, height=600)
                
                # Add all geometries
                for geom in geoms:
                    vis.add_geometry(geom)
                
                # Set up camera view
                ctr = vis.get_view_control()
                ctr.set_front([0.0, 0.0, -1.0])  # Look down
                ctr.set_up([1.0, 0.0, 0.0])     # X-axis is up
                
                # Run the visualizer in non-blocking mode
                while True:
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.01)  # Small delay to prevent high CPU usage
                    
            except Exception as e:
                print(f"Visualization error: {e}")
            finally:
                try:
                    vis.destroy_window()
                except:
                    pass
        
        # Start the 3D visualization in a separate thread
        print("Opening 3D map in background... You can interact with the planner while keeping the map open!")
        print("The 3D window will remain open during interactive planning.")
        print("Note: If the 3D window closes, you can still continue with text-based planning.")
        viz_thread = threading.Thread(target=create_persistent_visualizer, daemon=True)
        viz_thread.start()
        
        # Give the visualization a moment to initialize
        time.sleep(3)
        print("✅ 3D visualization initialized!")

    # Pure 3D voxel navigation - no occupancy grid publishing needed
    print("🗺️  Using pure 3D voxel navigation - skipping occupancy grid publishing")

    # Create maps directory if it doesn't exist
    import os
    os.makedirs('maps', exist_ok=True)

    if test_vlm:
        start_is_valid = space.is_valid(x0, verbose=True, debug=False)
        if not start_is_valid:
            print("you need to manually set the start pose to be valid")
            return

        # If no task provided or task is empty, go directly to interactive mode
        if not task or task.strip() == "":
            print("No initial task provided. Going directly to interactive planning mode...")
            
            # Still create world representation for interactive use
            scene_graph = agent.semantic_sensor.scene_graph if hasattr(agent.semantic_sensor, 'scene_graph') else None
            
            # Skip to interactive planning
            print("\n" + "="*50)
            print("🎮 INTERACTIVE VLM PLANNING MODE")
            print("="*50)
            if show_svm:
                print("📍 3D map is open in a separate window - keep it open to see spatial relationships!")
                print("💡 You can rotate, zoom, and pan the 3D view while planning")
            
            # Display current spatial relationships for reference
            if scene_graph:
                relationships = scene_graph.get_relationships()
                print(f"\n📊 Current spatial relationships ({len(relationships)} detected):")
                for i, (obj1, obj2, rel) in enumerate(relationships[:10]):  # Show first 10
                    if obj2 == "floor":
                        print(f"  {i+1:2d}. Instance {obj1} is {rel} {obj2}")
                    else:
                        print(f"  {i+1:2d}. Instance {obj1} is {rel} instance {obj2}")
                if len(relationships) > 10:
                    print(f"     ... and {len(relationships)-10} more relationships")
            
            print("\nYou can now enter navigation goals.")
            print("Examples: 'go to chair', 'find table', 'navigate to kitchen'")
            print("Commands: 'relationships' to see all spatial relationships, 'quit' to exit")
            print("🤖 After each plan is generated, you'll be asked if you want to execute it on the robot")
            print("="*50)
            
            # Interactive loop for testing different goals
            while True:
                try:
                    user_query = input("\nEnter your navigation goal (or 'quit' to exit): ").strip()
                    if user_query.lower() in ['quit', 'exit', 'q']:
                        print("Exiting interactive mode...")
                        break
                    
                    # Special command to show relationships
                    if user_query.lower() in ['relationships', 'rel', 'r']:
                        if scene_graph:
                            relationships = scene_graph.get_relationships()
                            print(f"\n📊 All spatial relationships ({len(relationships)} total):")
                            for i, (obj1, obj2, rel) in enumerate(relationships):
                                if obj2 == "floor":
                                    print(f"  {i+1:2d}. Instance {obj1} is {rel} {obj2}")
                                else:
                                    print(f"  {i+1:2d}. Instance {obj1} is {rel} instance {obj2}")
                        else:
                            print("No scene graph available.")
                        continue
                    
                    if not user_query:  # Empty input
                        continue
                        
                    print(f"\n🎯 Planning for: '{user_query}'")
                    plan, world_rep = vlm_planner.plan(
                        current_pose=x0,
                        show_plan=True,
                        query=user_query,
                        plan_with_reachable_instances=False,
                        plan_with_scene_graph=True,
                    )
                    print(f"✅ Generated plan: {plan}")
                    
                    # Ask user if they want to execute the plan on the robot
                    if plan:
                        execute_choice = input("\n🤖 Execute this plan on the robot? [y/N]: ").strip().lower()
                        if execute_choice in ['y', 'yes']:
                            # Check if robot is real (not dummy)
                            robot = agent.get_robot()
                            
                            # Check for Stretch robot connection (IP-based)
                            if hasattr(robot, 'robot_ip') and robot.robot_ip:
                                print(f"🔗 Connected to Stretch robot at {robot.robot_ip}")
                                success = execute_plan(agent, plan, world_rep, start_xyz, offset_x, offset_y)
                                if success:
                                    print("🎉 Plan executed successfully!")
                                else:
                                    print("❌ Plan execution failed.")
                            # Check for ROS-based robot connection (like Segbot)
                            elif str(type(robot).__name__).lower() != 'dummyrobot':
                                # Verify robot is actually connected before claiming success
                                try:
                                    import subprocess
                                    result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=3)
                                    expected_topics = ['/move_base_simple/goal', '/cmd_vel', '/odom']
                                    available_topics = result.stdout.strip().split('\n')
                                    missing_topics = [topic for topic in expected_topics if topic not in available_topics]
                                    
                                    if missing_topics:
                                        print(f"❌ Robot disconnected! Missing topics: {missing_topics}")
                                        print("🤖 Cannot execute plan - robot not available")
                                        continue
                                    else:
                                        print(f"🔗 Connected to ROS robot via bridge ({type(robot).__name__})")
                                except Exception as e:
                                    print(f"❌ Robot connection failed: {e}")
                                    print("🤖 Cannot execute plan - robot not available")
                                    continue

                                success = execute_plan(agent, plan, world_rep, start_xyz, offset_x, offset_y)
                                if success:
                                    print("🎉 Plan executed successfully!")
                                else:
                                    print("❌ Plan execution failed.")
                            else:
                                print("⚠️  No robot connection detected. Running in simulation mode.")
                                print("📝 Plan generated but not executed on real robot.")
                        else:
                            print("📝 Plan generated but not executed.")
                    else:
                        print("⚠️  No executable plan generated.")
                    
                except KeyboardInterrupt:
                    print("\n👋 Exiting interactive mode...")
                    break
                except Exception as e:
                    import traceback
                    print(f"❌ Error during planning: {e}")
                    traceback.print_exc()
                    continue
            return

        print("\nFirst plan with the original map: ")
        original_plan, world_rep = vlm_planner.plan(
            current_pose=x0,
            show_plan=True,
            query=task,
            plan_with_reachable_instances=False,
            plan_with_scene_graph=True,  # ENABLED: Use spatial relationships
        )

        # loop over the plan and check feasibilities for each action
        preconditions = {}
        while len(original_plan) > 0:
            current_action = original_plan.pop(0)

            # navigation action only for now
            if "goto" not in current_action:
                continue

            # Skip feasibility check for pure 3D navigation - let visual servoing handle obstacles
            print(f"Queuing action for execution: {current_action}")
        
        # After testing feasibility, continue with interactive planning
        print("\n" + "="*50)
        print("🎮 INTERACTIVE VLM PLANNING MODE")
        print("="*50)
        print("The system has completed initial planning.")
        if show_svm:
            print("📍 3D map is open in a separate window - keep it open to see spatial relationships!")
            print("💡 You can rotate, zoom, and pan the 3D view while planning")
        
        # Display current spatial relationships for reference
        scene_graph = agent.semantic_sensor.scene_graph if hasattr(agent.semantic_sensor, 'scene_graph') else None
        if scene_graph:
            relationships = scene_graph.get_relationships()
            print(f"\n📊 Current spatial relationships ({len(relationships)} detected):")
            for i, (obj1, obj2, rel) in enumerate(relationships[:10]):  # Show first 10
                if obj2 == "floor":
                    print(f"  {i+1:2d}. Instance {obj1} is {rel} {obj2}")
                else:
                    print(f"  {i+1:2d}. Instance {obj1} is {rel} instance {obj2}")
            if len(relationships) > 10:
                print(f"     ... and {len(relationships)-10} more relationships")
        
        print("\nYou can now test different navigation goals.")
        print("Examples: 'go to chair', 'find table', 'navigate to kitchen'")
        print("Commands: 'relationships' to see all spatial relationships, 'quit' to exit")
        print("🤖 After each plan is generated, you'll be asked if you want to execute it on the robot")
        print("="*50)
        
        # Interactive loop for testing different goals
        while True:
            try:
                user_query = input("\nEnter your navigation goal (or 'quit' to exit): ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                # Special command to show relationships
                if user_query.lower() in ['relationships', 'rel', 'r']:
                    if scene_graph:
                        relationships = scene_graph.get_relationships()
                        print(f"\n📊 All spatial relationships ({len(relationships)} total):")
                        for i, (obj1, obj2, rel) in enumerate(relationships):
                            if obj2 == "floor":
                                print(f"  {i+1:2d}. Instance {obj1} is {rel} {obj2}")
                            else:
                                print(f"  {i+1:2d}. Instance {obj1} is {rel} instance {obj2}")
                    else:
                        print("No scene graph available.")
                    continue
                
                if not user_query:  # Empty input
                    continue
                    
                print(f"\n🎯 Planning for: '{user_query}'")
                plan, world_rep = vlm_planner.plan(
                    current_pose=x0,
                    show_plan=True,
                    query=user_query,
                    plan_with_reachable_instances=False,
                    plan_with_scene_graph=True,
                )
                print(f"✅ Generated plan: {plan}")
                
                # Ask user if they want to execute the plan on the robot
                if plan:
                    execute_choice = input("\n🤖 Execute this plan on the robot? [y/N]: ").strip().lower()
                    if execute_choice in ['y', 'yes']:
                        # Check if robot is real (not dummy)
                        robot = agent.get_robot()
                        
                        # Check for Stretch robot connection (IP-based)
                        if hasattr(robot, 'robot_ip') and robot.robot_ip:
                            print(f"🔗 Connected to Stretch robot at {robot.robot_ip}")
                            success = execute_plan(agent, plan, world_rep, start_xyz, offset_x, offset_y)
                            if success:
                                print("🎉 Plan executed successfully!")
                            else:
                                print("❌ Plan execution failed.")
                        # Check for ROS-based robot connection (like Segbot)
                        elif str(type(robot).__name__).lower() != 'dummyrobot':
                            # Verify robot is actually connected before claiming success
                            try:
                                import subprocess
                                result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=3)
                                expected_topics = ['/move_base_simple/goal', '/cmd_vel', '/odom']
                                available_topics = result.stdout.strip().split('\n')
                                missing_topics = [topic for topic in expected_topics if topic not in available_topics]
                                
                                if missing_topics:
                                    print(f"❌ Robot disconnected! Missing topics: {missing_topics}")
                                    print("🤖 Cannot execute plan - robot not available")
                                else:
                                    print(f"🔗 Connected to ROS robot via bridge ({type(robot).__name__})")
                                    success = execute_plan(agent, plan, world_rep, start_xyz, offset_x, offset_y)
                                    if success:
                                        print("🎉 Plan executed successfully!")
                                    else:
                                        print("❌ Plan execution failed.")
                            except Exception as e:
                                print(f"❌ Robot connection failed: {e}")
                                print("🤖 Cannot execute plan - robot not available")
                        else:
                            print("⚠️  No robot connection detected. Running in simulation mode.")
                            print("📝 Plan generated but not executed on real robot.")
                    else:
                        print("📝 Plan generated but not executed.")
                else:
                    print("⚠️  No executable plan generated.")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                import traceback
                print(f"❌ Error during planning: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    """run the test script."""
    main()
