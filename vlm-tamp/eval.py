"""
Example script demo'ing robot primitive to solve a task
"""

import copy
import random
import os
import json
import time
import yaml
import numpy as np
import omnigibson as og
import matplotlib.pyplot as plt
import cv2
from omnigibson import object_states
from PIL import Image
from omnigibson.macros import gm
from omnigibson.utils.constants import CLASS_NAME_TO_CLASS_ID
from pddl_sim import pddlsim
from gpt4v import GPT4VAgent
from gemini import GeminiAgent

# from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
    RobotCopy,
    PlanningContext,
)
import transforms3d as tf
import quaternion as quat
from omnigibson.utils.object_state_utils import sample_cuboid_for_predicate
import omnigibson.utils.transform_utils as T
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    plan_arm_motion_ik,
    set_base_and_detect_collision,
    detect_robot_collision_in_sim,
)
from omnigibson.object_states import *
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.utils.bddl_utils import (
    OmniGibsonBDDLBackend,
    BDDLEntity,
    BEHAVIOR_ACTIVITIES,
    BDDLSampler,
)
from omnigibson.systems import (
    get_system,
    is_physical_particle_system,
    is_visual_particle_system,
)

IMPERCEIVABLE_PREDS = [
    "inroom",
    "insource",
    "inhand",
    "handempty",
    "cooked",
    "filled",
    "inside",
    "turnedon",
    "found",
    "filledsink",
    "hot",
]
GT_PREDS = ["handempty", "inhand", "filled", "inside"]
PREDICATE_SAMPLING_Z_OFFSET = 0.02
MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 1000
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 1000
PICK_OBJ_HEIGHT = 1.15
PLACE_ON_FLOOR_DIST = 1.5
FALL_ON_FLOOR_DIST = 1.5
NUM_TRIALS = 20
CHECK_PRECONDITION = False
CHECK_EFFECT = False
CHECK_IN_NL = False
MAX_NUM_ACTION = 50
MAX_TELEPORT_DIST = 2.5
MIN_TELEPORT_DIST = 1.0

# action probs
OTHER_ACTION_SUCCESS_PROB = 0.9
NAV_SUCCESS_PROB = 0.8
PICK_SUCCESS_PROB = 0.5
PLACE_SUCCESS_PROB = 0.8
OPEN_SUCCESS_PROB = 0.9
CLOSE_SUCCESS_PROB = 0.9
HALVE_SUCCESS_PROB = 0.5
PLACE_ON_FLOOR_SUCCESS_PROB = 0.8
# for action fill, grasp and place, there is a probability that the object will fall on the floor
FALL_ON_GROUND_PROB_IF_FAILED = 0.5

OPEN_FULLY = True
LOG_DIR = "datadump/"

VLM_PLANNING = True  # use VLM as the planner
is_oracle = False

vlm_agent = GPT4VAgent()
# vlm_agent = GeminiAgent()

# Active perception module (initialized if needed)
active_perception_module = None
USE_ACTIVE_PERCEPTION = False

# from claude3 import Claude3Agent
# vlm_agent = Claude3Agent()

# remember to reset them for new episode!!!
# obj_held = None
# filled = None
# inside_relationships = []
# onfloor_relationships = []


def initialize_active_perception(
    robot_ip,
    map_file,
    config_file="rosbridge_robot_config.yaml",
    calibration_file="simple_offset_calibration.yaml",
    max_exploration_attempts=3,
    viewpoint_distance=1.0
):
    """
    Initialize active perception module for real-robot experiments.

    Args:
        robot_ip: Robot IP address
        map_file: Path to voxel map (.pkl)
        config_file: Robot configuration file
        calibration_file: Map calibration file
        max_exploration_attempts: Maximum viewpoints to try
        viewpoint_distance: Distance from object for viewpoints

    Returns:
        ActivePerceptionModule instance
    """
    global active_perception_module, USE_ACTIVE_PERCEPTION

    try:
        from active_perception import ActivePerceptionModule

        active_perception_module = ActivePerceptionModule(
            robot_ip=robot_ip,
            map_file=map_file,
            config_file=config_file,
            calibration_file=calibration_file,
            max_exploration_attempts=max_exploration_attempts,
            viewpoint_distance=viewpoint_distance
        )

        USE_ACTIVE_PERCEPTION = True
        print("✅ Active Perception initialized for real-robot experiments")

        return active_perception_module

    except Exception as e:
        print(f"⚠️  Failed to initialize Active Perception: {e}")
        print("   Continuing without active perception capability")
        USE_ACTIVE_PERCEPTION = False
        return None


def get_class_name_from_class_id(target_class_id):
    for class_name, class_id in CLASS_NAME_TO_CLASS_ID.items():
        if class_id == target_class_id:
            return class_name
    return None


def yaw_to_quaternion(yaw):
    """
    Convert yaw (rotation around the vertical axis) to quaternion.

    Parameters:
    - yaw: Yaw angle in radians.

    Returns:
    - quaternion: A 4-element numpy array representing the quaternion.
    """
    # Calculate half angles
    yaw_half = 0.5 * yaw

    # Quaternion components
    w = np.cos(yaw_half)
    x = 0.0
    y = 0.0
    z = np.sin(yaw_half)

    # Construct quaternion
    quaternion = np.array([x, y, z, w])

    return quaternion


def sample_teleport_pose_near_object(ap, obj, pose_on_obj=None, **kwargs):
    with PlanningContext(ap.robot, ap.robot_copy, "simplified") as context:
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            # if pose_on_obj is None:
            pos_on_obj = ap._sample_position_on_aabb_side(obj)
            pose_on_obj = [pos_on_obj, np.array([0, 0, 0, 1])]

            distance = np.random.uniform(MIN_TELEPORT_DIST, MAX_TELEPORT_DIST)
            yaw = np.random.uniform(-np.pi, np.pi)
            avg_arm_workspace_range = np.mean(ap.robot.arm_workspace_range[ap.arm])
            pose_2d = np.array(
                [
                    pose_on_obj[0][0] + distance * np.cos(yaw),
                    pose_on_obj[0][1] + distance * np.sin(yaw),
                    yaw + np.pi - avg_arm_workspace_range,
                ]
            )
            # Check room

            obj_rooms = (
                obj.in_rooms
                if obj.in_rooms
                else [
                    ap.env.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])
                ]
            )

            if obj_rooms == [None]:
                print("object not in any room.")
                continue
            if (
                ap.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2])
                not in obj_rooms
            ):
                print("Candidate position is in the wrong room.")
                continue
            if set_base_and_detect_collision(
                context, ap._get_robot_pose_from_2d_pose(pose_2d)
            ):
                print("Candidate position failed collision test.")
                continue
            # import pdb; pdb.set_trace()
            return pose_2d
        print("Could not find valid position near object.")
        # raise ActionPrimitiveError(
        #     ActionPrimitiveError.Reason.SAMPLING_ERROR,
        #     "Could not find valid position near object.",
        #     {
        #         "target object": obj.name,
        #         "target pos": obj.get_position(),
        #         "pose on target": pose_on_obj,
        #     },
        # )
        return None


def _sample_pose_with_object_and_predicate(
    predicate, held_obj, target_obj, near_poses=None, near_poses_threshold=None
):
    """
    Returns a pose for the held object relative to the target object that satisfies the predicate

    Args:
        predicate (object_states.OnTop or object_states.Inside): Relation between held object and the target object
        held_obj (StatefulObject): Object held by the robot
        target_obj (StatefulObject): Object to sample a pose relative to
        near_poses (Iterable of arrays): Poses in the world frame to sample near
        near_poses_threshold (float): The distance threshold to check if the sampled pose is near the poses in near_poses

    Returns:
        2-tuple:
            - 3-array: (x,y,z) Position in the world frame
            - 4-array: (x,y,z,w) Quaternion orientation in the world frame
    """
    pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}

    for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE):
        _, _, bb_extents, _ = held_obj.get_base_aligned_bbox()
        sampling_results = sample_cuboid_for_predicate(
            pred_map[predicate], target_obj, bb_extents
        )
        if sampling_results[0][0] is None:
            continue
        sampled_bb_center = sampling_results[0][0] + np.array(
            [0, 0, PREDICATE_SAMPLING_Z_OFFSET]
        )
        sampled_bb_orn = sampling_results[0][2]

        # Get the object pose by subtracting the offset
        # sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn)) @ T.pose_inv(T.pose2mat((bb_center_in_base, [0, 0, 0, 1])))
        sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn))
        # Check that the pose is near one of the poses in the near_poses list if provided.
        # if near_poses:
        #     sampled_pos = np.array([sampled_obj_pose[0]])
        #     if not np.any(np.linalg.norm(near_poses - sampled_pos, axis=1) < near_poses_threshold):
        #         continue

        # Return the pose
        return T.mat2pose(sampled_obj_pose)

    # If we get here, sampling failed.
    print(
        "Could not find a position to put this object in the desired relation to the target object"
    )


def get_fpv_rgb():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["rgb"]


def get_tpv_rgb():
    return og.sim.viewer_camera.get_obs()["rgb"]


def get_seg_semantic():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["seg_semantic"]


def get_seg_instance():
    return robot.get_obs()["fetch:eyes_Camera_sensor"]["seg_instance"]


def rotate_x(initial_quaternion, angle_degrees):
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create a quaternion representing the rotation
    rotation_quaternion = np.quaternion(
        np.cos(angle_radians / 2), np.sin(angle_radians / 2), 0, 0
    )

    # Perform quaternion multiplication to apply the rotation
    new_quaternion = rotation_quaternion * quat.as_quat_array(initial_quaternion)

    return quat.as_float_array(new_quaternion)


def rotate_z(initial_quaternion, angle_degrees):
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    rotation_quaternion = np.quaternion(
        np.cos(angle_radians / 2), 0, 0, np.sin(angle_radians / 2)
    )

    # Perform quaternion multiplication to apply the rotation
    new_quaternion = rotation_quaternion * quat.as_quat_array(initial_quaternion)

    return quat.as_float_array(new_quaternion)


def inview(obj_name):
    obj = env.task.object_scope[obj_name].wrapped_obj
    seg = get_seg_instance()
    instances = np.unique(seg)
    for instance_id in instances:
        if obj == scene.objects[instance_id - 1]:
            print("object in view!!!!!")
            return True
    print("object not in view")
    return False


def inspect():
    try:
        while True:
            og.sim.step()
    except KeyboardInterrupt:
        print("Exiting ...")


def run_sim(step=20):
    for _ in range(step):
        og.sim.step()
        # Test: object in hand might be
    dummy_action = np.zeros((11))
    env.step(dummy_action)
    global sim_counter
    Image.fromarray(og.sim.viewer_camera.get_obs()["rgb"], "RGBA").save(
        os.path.join(debug_path, str(sim_counter) + ".png")
    )
    sim_counter += 1


def lookat(obj_name="can_of_soda_89"):
    obj = env.task.object_scope[obj_name].wrapped_obj
    target_obj_pose = obj.get_position_orientation()
    # ap = StarterSemanticActionPrimitives(env)
    head_q = ap._get_head_goal_q(target_obj_pose)
    head_action = ap.robot.get_joint_positions()
    for qid, cid in enumerate(ap.robot.camera_control_idx):
        head_action[cid] = -head_q[qid]

    robot.set_joint_positions(head_action)


def watch_robot():
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0, 0, 3.5]) + robot.get_position(),
        orientation=rotate_z(rotate_x(robot.get_orientation(), 90), -20),
    )


def goto(obj_name="can_of_soda_89", oracle=False):

    # preconditions
    # TODO:the object and robot should be in the same room?
    inview_obj = False
    while not inview_obj:
        # execution
        robot.tuck()
        run_sim()
        obj = env.task.object_scope[obj_name].wrapped_obj
        xyt = sample_teleport_pose_near_object(ap, obj)
        if xyt is None:
            print("there is no free space near the object -- action failed")
            return False
        pos = np.empty([3])
        pos[2] = robot_init_z
        pos[1] = xyt[1]
        pos[0] = xyt[0]
        orien = yaw_to_quaternion(xyt[2])
        robot.set_position_orientation(pos, orien)
        global obj_held
        if obj_held:
            print("there is an object in hand -- moving this object as well")
            obj_held.set_position([pos[0], pos[1], PICK_OBJ_HEIGHT])
        watch_robot()
        lookat(obj_name)
        robot.tuck()
        run_sim()

        if not oracle:  # there is a chance that object in hand will fall
            prob = random.random()
            if prob > NAV_SUCCESS_PROB:
                print("action failed randomly")
                fall_prob = random.random()
                if fall_prob < FALL_ON_GROUND_PROB_IF_FAILED and obj_held:
                    # object falls on ground
                    print("object falls on ground")
                    # fall(obj_name)
                    obj_in_hand_name = [
                        k
                        for k, v in env.task.object_scope.items()
                        if v.wrapped_obj == obj_held
                    ][0]
                    fall(obj_in_hand_name)
            break
        inview_obj = inview(obj_name)

    return True


def turnon(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            print("turn on object failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[obj].states[ToggledOn].set_value(True)
    run_sim()


def fill_sink(sink, oracle=False):
    if not oracle:
        if not inview(sink):
            print("filling sink failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[sink].states[ToggledOn].set_value(True)
    run_sim(step=40)
    env.task.object_scope[sink].states[ToggledOn].set_value(False)
    run_sim()


def fall(obj_name):
    obj = env.task.object_scope[obj_name].wrapped_obj
    obj.set_position_orientation(
        position=np.array(
            [
                np.random.uniform(-FALL_ON_FLOOR_DIST, FALL_ON_FLOOR_DIST),
                np.random.uniform(-FALL_ON_FLOOR_DIST, FALL_ON_FLOOR_DIST),
                0.5,
            ]
        )
        + robot.get_position(),
        orientation=robot.get_orientation(),
    )
    obj.enable_gravity()
    run_sim(50)

    global obj_held
    if obj == obj_held:
        obj_held = None
        global filled
        filled = None

    global inside_relationships
    for in_obj, recep in inside_relationships:
        if in_obj == obj_name:
            inside_relationships.remove((in_obj, recep))
            break

    global onfloor_relationships
    if obj_name not in onfloor_relationships:
        onfloor_relationships.append(obj_name)


def grasp(obj_name="can_of_soda_89", oracle=False):

    # preconditions

    # check if hand is empty
    global obj_held
    if not oracle:
        if obj_held:
            print(
                "there is already an object in hand -- unsatisfied hidden preconditions"
            )
            return
            # TODO: return False ?

        # check if object is in the field of view -- assuming vision-based grasping
        if not inview(obj_name):
            print(
                "grasp object failed because of the object not in the view -- unsatisfied hidden preconditions"
            )
            return

        # there is some probability that the grasp action will fail
        prob = random.random()
        if prob > PICK_SUCCESS_PROB:
            print("action failed randomly")
            fall_prob = random.random()
            if fall_prob < FALL_ON_GROUND_PROB_IF_FAILED:
                # object falls on ground
                print("object falls on ground")
                fall(obj_name)
            return

    obj = env.task.object_scope[obj_name].wrapped_obj
    # obj_held = env.task.object_scope['mug.n.04_1'].wrapped_obj
    robot_pose = robot.get_position()
    obj.set_position_orientation(
        position=[robot_pose[0], robot_pose[1], PICK_OBJ_HEIGHT],
        orientation=[0, 0, 0, 1],
    )
    # obj.set_orientation([0,0,0,1])
    obj_held = obj
    obj.keep_still()
    obj.disable_gravity()
    # global
    # if filled
    # system.remove_all_particles()
    run_sim(50)
    # manually remove this if it exists in our relationship tracker
    global inside_relationships
    for in_obj, recep in inside_relationships:
        if in_obj == obj_name:
            inside_relationships.remove((in_obj, recep))
            break
    global onfloor_relationships
    if obj_name in onfloor_relationships:
        onfloor_relationships.remove(obj_name)
    return


def fill(container, sink, liquid="water", oracle=False):
    global filled
    if not oracle:
        if not obj_held:
            print(
                "you have to hold a container to fill a liquid -- unsatisfied hidden preconditions"
            )
            return
        if filled:
            print(
                "there are already something in the container -- unsatisfied hidden preconditions"
            )
            return
        if not inview(sink):
            print(
                "filling container near source failed -- unsatisfied hidden preconditions"
            )
            return
        prob = random.random()
        if prob > OTHER_ACTION_SUCCESS_PROB:
            print("action failed randomly")
            fall_prob = random.random()
            if fall_prob < FALL_ON_GROUND_PROB_IF_FAILED:
                # object falls on ground
                print("object falls on ground")
                fall(container)
            return
    # system = get_system(liquid)

    container_obj = obj_held
    place_with_predicate(container, sink, OnTop, oracle=True)
    while not container_obj.states[Filled].get_value(get_system(liquid)):
        assert container_obj.states[Filled].set_value(get_system(liquid), True)
        run_sim()
    # assert container_obj.states[Filled].get_value(get_system(liquid))
    # obj_held.states[Filled].get_value(get_system('water'))
    run_sim()
    # import pdb; pdb.set_trace()
    # TODO: something wrong here
    # assert container_obj.states[Filled].get_value(system)
    # import pdb; pdb.set_trace()
    filled = liquid
    # container_obj.states[Filled].get_value(system)
    get_system(liquid).remove_all_particles()
    grasp(container, oracle=True)


def openit(obj, oracle=False):

    # check if object is in the field of view -- effects
    if not oracle:
        if not inview(obj):
            print("opening object failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > OPEN_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[obj].states[Open].set_value(True, fully=OPEN_FULLY)
    run_sim()


def closeit(obj, oracle=False):
    if not oracle:
        if not inview(obj):
            print("closing object failed -- unsatisfied hidden preconditions")
            return
        prob = random.random()
        if prob > CLOSE_SUCCESS_PROB:
            print("action failed randomly")
            return
    env.task.object_scope[obj].states[Open].set_value(False)
    run_sim()


def cut_into_half(knife_name, obj_name, oracle=False):
    if not oracle:
        if not inview(obj_name):
            print("cutting object failed -- unsatisfied hidden preconditions")
            return
        if not obj_held:
            print("you are not holding a knife")
            return
        prob = random.random()
        if prob > HALVE_SUCCESS_PROB:
            print("action failed randomly")
            fall_prob = random.random()
            if fall_prob < FALL_ON_GROUND_PROB_IF_FAILED:
                # object falls on ground
                print("object falls on ground")
                fall(knife_name)
            return
    obj = env.task.object_scope[obj_name]
    knife = env.task.object_scope[knife_name]
    knife.enable_gravity()
    knife.keep_still()
    knife.set_position_orientation(
        position=obj.get_position() + np.array([-0.15, 0, 0.2]),
        orientation=T.euler2quat([-np.pi / 2, 0, 0]),
    )
    run_sim(100)
    grasp(knife_name, oracle=True)


def place_on_floor(obj, oracle=False):
    obj_in_hand = env.task.object_scope[obj].wrapped_obj
    global obj_held
    if not oracle:
        if obj_in_hand != obj_held:
            print(
                "You need to be grasping the object first to place it somewhere. -- unsatified hidden preconditions"
            )
            return
        prob = random.random()
        if prob > PLACE_ON_FLOOR_SUCCESS_PROB:
            print("action failed randomly")
            return
    obj_in_hand.set_position_orientation(
        position=np.array([0, PLACE_ON_FLOOR_DIST, 0.5]) + robot.get_position(),
        orientation=robot.get_orientation(),
    )
    obj_in_hand.enable_gravity()
    run_sim(50)
    lookat(obj)

    global filled
    if filled:
        while not obj_held.states[Filled].get_value(get_system(filled)):
            assert obj_held.states[Filled].set_value(get_system(filled), True)
            run_sim()
        # assert obj_held.states[Filled].get_value(filled)
        filled = None
    obj_held = None
    run_sim(50)

    global onfloor_relationships
    if obj not in onfloor_relationships:
        onfloor_relationships.append(obj)


def place_with_predicate(
    obj_in_hand_name="can_of_soda_89",
    obj_name="trash_can_85",
    predicate=OnTop,
    oracle=False,
):
    """
    Yields action for the robot to navigate to the object if needed, then to place it

    Args:
        obj (StatefulObject): Object for robot to place the object in its hand on
        predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

    Returns:
        np.array or None: Action array for one step for the robot to place or None if place completed
    """
    # Update the tracking to track the object.
    # self._tracking_object = obj
    obj = env.task.object_scope[obj_name].wrapped_obj
    # ap = StarterSemanticActionPrimitives(env)
    obj_in_hand = env.task.object_scope[obj_in_hand_name].wrapped_obj
    # TODO: verify it here
    global obj_held
    if not oracle:
        if obj_in_hand != obj_held:
            print(
                "You need to be grasping the object first to place it somewhere. -- unsatified hidden preconditions"
            )
            return
        if not inview(obj_name):
            print("placing object failed -- unsatisfied hidden preconditions")
            return
        if (
            predicate == Inside
            and not env.task.object_scope[obj_name].states[Open].get_value()
        ):
            print(
                "placing object failed because the receptacle is closed-- unsatisfied hidden preconditions"
            )
            return
        prob = random.random()
        if prob > PLACE_SUCCESS_PROB:
            print("action failed randomly")
            fall_prob = random.random()
            if fall_prob < FALL_ON_GROUND_PROB_IF_FAILED:
                # object falls on ground
                print("object falls on ground")
                fall(obj_in_hand_name)
            return
    # Sample location to place object
    # while (
    #     env.task.object_scope["mug.n.04_1"]
    #     .states[Inside]
    #     .get_value(env.task.object_scope["microwave.n.02_1"])
    # ):
    obj_pose = _sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
    obj_in_hand.set_position_orientation(obj_pose[0], obj_pose[1])
    # TODO: this line manually set the object relationships -- which should not happen
    obj_in_hand.enable_gravity()
    run_sim(100)
    lookat(obj_in_hand_name)

    if predicate == Inside:
        # manually add this into our inside relationship tracker due to issues with the simulator
        global inside_relationships
        if (obj_in_hand_name, obj_name) not in inside_relationships:
            inside_relationships.append((obj_in_hand_name, obj_name))

    global filled
    if filled:
        while not obj_held.states[Filled].get_value(get_system(filled)):
            assert obj_held.states[Filled].set_value(get_system(filled), True)
            run_sim()
        # assert obj_held.states[Filled].get_value(filled)
        filled = None
    obj_held = None


def format_action_params(action):
    params = action[1:]
    formatted_params = [p.replace("-", ".") for p in params]

    # some exceptions in naming convention
    # for p_id, p in enumerate(formatted_params):
    #     if p == "hard.boiled_egg.n.01_1":
    #         formatted_params[p_id] = "hard-boiled_egg.n.01_1"

    return [p.replace("__", "-") for p in formatted_params]


def translate_fact_to_question(fact):
    neg_suffix = " "
    if fact[0] == "not":
        # neg_suffix = " not "
        fact = fact[1:]
    predicate = fact[0]

    if predicate in ["inside", "inroom", "filled", "ontop"]:
        return f"Is {fact[1].split('-')[0]}{neg_suffix}{predicate} {fact[2].split('-')[0]}?"
    elif predicate in ["insource", "inhand", "inview"]:
        return f"Is {fact[2].split('-')[0]}{neg_suffix}{predicate} {fact[1].split('-')[0]}?"
    elif predicate in [
        "handempty",
        "closed",
        "turnedon",
        "cooked",
        "halved",
        "onfloor",
    ]:
        return f"Is {fact[1].split('-')[0]}{neg_suffix}{predicate}?"
    else:
        raise RuntimeError


def update_states_by_fact(states, fact):
    if fact[0] == "not":
        formatted_fact = f"({fact[1]}"
        for param in fact[2:]:
            formatted_fact += f" {param}"
        formatted_fact += ")"
        if formatted_fact not in states:
            states.append(formatted_fact)
            print(f"adding {formatted_fact}")
    else:
        formatted_fact = f"({fact[0]}"
        for param in fact[1:]:
            formatted_fact += f" {param}"
        formatted_fact += ")"
        if formatted_fact in states:
            states.remove(formatted_fact)
            print(f"removing {formatted_fact}")

    return states


def write_states_into_problem(states, previous_problem):
    prob = open(previous_problem).readlines()
    init_line = prob.index("    (:init\n")
    end_init_line = prob.index("    (:goal\n")
    before_init = prob[: init_line + 1]
    after_init = prob[end_init_line - 2 :]
    for state_id, state in enumerate(states):
        states[state_id] = f"        {state}\n"
    new_problem = before_init + states + after_init
    new_problem_name = "updated_problem.pddl"
    with open(new_problem_name, "w") as f:
        f.write("".join(new_problem))
    return new_problem_name


def check_gt_facts(facts):
    match_res = []
    for fact in facts:
        if any(s for s in fact if "handempty" in s):
            match = "yes" if not obj_held else "no"
        elif any(s for s in fact if "inhand" in s):
            match = (
                "yes" if obj_held else "no"
            )  # TODO: check if the specific object is in hand
        elif any(s for s in fact if "filled" in s):
            container = fact[1].replace("-", ".").replace("__", "-")
            if (
                obj_held
                and env.task.object_scope[container].wrapped_obj == obj_held
                and filled
            ):
                match = "yes"
            elif (
                env.task.object_scope[container]
                .states[Filled]
                .get_value(get_system(fact[2].split("-")[0]))
            ):
                match = "yes"
            else:
                match = "no"
        elif any(s for s in fact if "inside" in s):
            match = (
                "yes"
                if (
                    fact[-2].replace("-", ".").replace("__", "-"),
                    fact[-1].replace("-", ".").replace("__", "-"),
                )
                in inside_relationships
                else "no"
            )
        else:
            NotImplementedError
        match_res.append(match)
    return match_res


def is_uncertain_vlm_response(response):
    """Check if VLM response indicates uncertainty."""
    response_lower = response.lower()
    uncertainty_markers = [
        "uncertain", "unclear", "cannot tell", "can't tell", "not sure",
        "unsure", "hard to tell", "difficult to determine",
        "partially visible", "occluded", "blocked", "maybe", "possibly"
    ]
    return any(marker in response_lower for marker in uncertainty_markers)


def check_states_and_update_problem(
    int_states,
    effs,
    pres,
    previous_problem,
    prev_states,
    cur_action=None,
    next_action=None,
):
    states = int_states.copy()
    unmatched_pres = []
    unmatched_effs = []

    is_state_updated_by_eff = False
    is_state_updated_by_pre = False

    facts_nl = []
    valid_facts = []
    facts = effs + pres
    pre_start_idx = 0
    pre_start_idx_gt = 0
    gt_facts = []

    for fact_id, fact in enumerate(facts):
        if fact[0] in IMPERCEIVABLE_PREDS or fact[1] in IMPERCEIVABLE_PREDS:
            continue

        if not CHECK_IN_NL:
            fact_nl = translate_fact_to_question(fact)
        else:
            if fact_id < len(effs):
                if fact[0] == "not":
                    fact_nl = f"Was the action {cur_action[:-1]} failed?"
                else:
                    fact_nl = f"Was the action {cur_action[:-1]} successful?"
            else:
                if fact[0] == "not":
                    fact_nl = f"Is it impossible to {next_action[:-1]}?"
                else:
                    fact_nl = f"Is it possible to {next_action[:-1]}?"
        facts_nl.append(fact_nl)
        valid_facts.append(fact)

        if fact_id < len(effs):
            pre_start_idx += 1

    print(f"Questions to VLMs: ")
    print(f"current action effects: {facts_nl[:pre_start_idx]}")
    print(f"next action preconditions: {facts_nl[pre_start_idx:]}")

    if len(facts_nl) > 0:
        is_match_results = vlm_agent.ask(";".join(facts_nl), get_fpv_rgb())
    else:
        is_match_results = []

    # Active perception: retry uncertain responses with exploration
    if USE_ACTIVE_PERCEPTION and active_perception_module is not None:
        for idx, is_match in enumerate(is_match_results):
            if is_uncertain_vlm_response(is_match):
                print(f"\n🔍 [ACTIVE PERCEPTION] Uncertain response detected")
                print(f"   Question: {facts_nl[idx]}")
                print(f"   Response: {is_match}")

                try:
                    # Use active perception to explore
                    new_answer, new_rgb, success = active_perception_module.explore_for_better_view(
                        predicate=valid_facts[idx],
                        vlm_agent=vlm_agent,
                        question=facts_nl[idx]
                    )

                    if success:
                        print(f"   ✅ After exploration: {new_answer}")
                        is_match_results[idx] = new_answer
                    else:
                        print(f"   ❌ Exploration failed, using original response")
                except Exception as e:
                    print(f"   ⚠️  Exploration error: {e}")

    for idx, is_match in enumerate(is_match_results):
        if idx < len(valid_facts) and (
            (("no" in is_match) and valid_facts[idx][0] != "not")
            or (("yes" in is_match) and valid_facts[idx][0] == "not")
        ):
            if idx < pre_start_idx:
                unmatched_effs.append(valid_facts[idx])
                is_state_updated_by_eff = True
            else:
                unmatched_pres.append(valid_facts[idx])
                is_state_updated_by_pre = True

    # for some predicates like handempty, inhand etc, we use ground truth
    for fact_id, fact in enumerate(facts):
        if fact[0] in GT_PREDS or fact[1] in GT_PREDS:
            gt_facts.append(fact)
            if fact_id < len(effs):
                pre_start_idx_gt += 1
    gt_fact_results = check_gt_facts(gt_facts)
    for idx, is_match in enumerate(gt_fact_results):
        if idx < len(gt_facts) and (
            (("no" in is_match) and gt_facts[idx][0] != "not")
            or (("yes" in is_match) and gt_facts[idx][0] == "not")
        ):
            if idx < pre_start_idx_gt:
                if gt_facts[idx][0] == "inhand":
                    # grasp something failed, the agent will assume it falls on the floor
                    unmatched_effs.append(
                        ["not", "ontop", gt_facts[idx][2], "floor-n-01_1"]
                    )
                    # and whatever it is filled is gone
                    unmatched_effs.append(["filled", gt_facts[idx][2], "water-n-06_1"])
                if gt_facts[idx][1] == "inhand":
                    # place something failed, and not in hand, we assume it's on the floor
                    unmatched_effs.append(
                        ["not", "ontop", gt_facts[idx][3], "floor-n-01_1"]
                    )
                    # and whatever it is filled is gone
                    unmatched_effs.append(["filled", gt_facts[idx][3], "water-n-06_1"])
                unmatched_effs.append(gt_facts[idx])
                is_state_updated_by_eff = True
            else:
                # deal with situations like object falling during navigation
                if gt_facts[idx][0] == "inhand":
                    # if not in hand before place and object, then we believe it's falling on the floor
                    unmatched_pres.append(
                        ["not", "ontop", gt_facts[idx][2], "floor-n-01_1"]
                    )
                    # and whatever it is filled is gone
                    unmatched_pres.append(["filled", gt_facts[idx][2], "water-n-06_1"])
                unmatched_pres.append(gt_facts[idx])
                is_state_updated_by_pre = True

    if (not is_state_updated_by_pre) and (not is_state_updated_by_eff):
        print("All facts match -- current states remain unchanged")
    else:
        print(f"Here is a list of unmatched facts according to VLM's response --")
        print(f"unmatched effs: {unmatched_effs}")
        print(f"unmatched pres: {unmatched_pres}")

    if CHECK_EFFECT and is_state_updated_by_eff:
        print("updating problem with previous states based on effects mismatch")
        for unmatched_fact in unmatched_effs:
            print(f"Previous states: {states}")
            states = update_states_by_fact(states, unmatched_fact)
            print(f"Updated states: {states}")
        updated_problem_file = write_states_into_problem(states, previous_problem)
    elif CHECK_PRECONDITION and is_state_updated_by_pre:
        print("updating states based on preconditions mismatch")
        for unmatched_fact in unmatched_pres:
            print(f"Previous states: {states}")
            states = update_states_by_fact(states, unmatched_fact)
            print(f"Updated states: {states}")
        updated_problem_file = write_states_into_problem(states, previous_problem)
    else:
        updated_problem_file = write_states_into_problem(states, previous_problem)

    return (
        (is_state_updated_by_eff or is_state_updated_by_pre),
        updated_problem_file,
        {
            "fpv": get_fpv_rgb(),
            "tpv": get_tpv_rgb(),
            "eff_queries": facts_nl[:pre_start_idx],
            "pre_queries": facts_nl[pre_start_idx:],
            "eff_ans": is_match_results[:pre_start_idx],
            "pre_ans": is_match_results[pre_start_idx:],
            "eff_mismatch": is_state_updated_by_eff,
            "pre_mismatch": is_state_updated_by_pre,
        },
    )


def log_writer(message, log_file):
    with open(log_file, "a+") as f:
        f.write(message)


# Load the config
# config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
config["scene"]["load_task_relevant_only"] = True
config["scene"]["not_load_object_categories"] = ["ceilings"]

#########################################################################################

# BOIL WATER IN THE MICROWAVE
# a_name = "boil_water_in_the_microwave"
# config["scene"]["scene_model"] = "Beechwood_0_int"

# a_name = "bringing_water"
# config["scene"]["scene_model"] = "house_single_floor"

# a_name = "cook_a_frozen_pie"
# config["scene"]["scene_model"] = "Beechwood_0_int"

# a_name = "halve_an_egg"
# config["scene"]["scene_model"] = "Benevolence_1_int"

a_name = "store_firewood"
config["scene"]["scene_model"] = "Ihlen_0_int"

#########################################################################################

config["task"] = {
    "type": "BehaviorTask",
    "activity_name": a_name,
    "activity_definition_id": 0,
    "activity_instance_id": 0,
    "predefined_problem": None,
    "online_object_sampling": False,
}

# initialize temp dump dir for visualization
import shutil
import os

debug_path = "datadump/third_person"
shutil.rmtree(debug_path, ignore_errors=True)
os.makedirs(debug_path, exist_ok=True)

init_problem_file = f"domains/{a_name}/problem.pddl"

exp_results = {"num_success": 0, "num_failed": 0}

trial_counter = 0
os.makedirs(LOG_DIR, exist_ok=True)
run_dir = os.path.join(LOG_DIR, str(time.time()))
os.makedirs(run_dir, exist_ok=False)

domain_file = f"domains/{a_name}/domain.pddl"
planner = pddlsim(domain_file)

# Load the environment
env = og.Environment(configs=config)
while trial_counter < NUM_TRIALS:

    og.sim.stop()
    env.load()
    scene = env.scene
    robot = env.robots[0]

    robot_init_z = robot.get_position()[2]
    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()
    og.sim.set_lighting_mode("Default")
    # run_sim_inf()
    ap = StarterSemanticActionPrimitives(env)

    trial_dir = os.path.join(run_dir, str(trial_counter))
    os.makedirs(trial_dir, exist_ok=False)
    # env.reset()
    watch_robot()

    obj_held = None
    filled = None
    inside_relationships = []
    onfloor_relationships = []

    sim_counter = 0
    run_sim()
    action_counter = 0

    if a_name == "boil_water_in_the_microwave":
        # TODO: the episode initialization has some bug -- hard code something for now
        env.task.object_scope["mug.n.04_1"].set_position_orientation(
            position=[-7.62, -3.76, 0.66], orientation=[0, 0, 0, 1]
        )
        inside_relationships.append(("mug.n.04_1", "cabinet.n.01_1"))

    if a_name == "cook_a_frozen_pie":
        env.task.object_scope["oven.n.01_1"].disable_gravity()
        run_sim()
        env.task.object_scope["oven.n.01_1"].states[Open].set_value(False)
        run_sim()
        env.task.object_scope["oven.n.01_1"].states[Open].set_value(False)
        inside_relationships.append(("pie.n.01_1", "electric_refrigerator.n.01_1"))

    if a_name == "store_brownies":
        height = -1.8
        brownie = env.task.object_scope["brownie.n.03_1"]
        brownie.set_position(position=np.array([0, 0, height]) + brownie.get_position())
        brownie = env.task.object_scope["brownie.n.03_2"]
        brownie.set_position(position=np.array([0, 0, height]) + brownie.get_position())
        tray = env.task.object_scope["tray.n.01_1"]
        tray.set_position(position=np.array([0, 0, height]) + tray.get_position())
        tupperware = env.task.object_scope["tupperware.n.01_1"]
        tupperware.set_position(
            position=np.array([0, 0, height]) + tupperware.get_position()
        )
    if a_name == "bringing_water":
        PICK_OBJ_HEIGHT = 2.0
    if a_name == "store_firewood":
        env.task.object_scope["wooden_stick.n.01_1"] = env.task.object_scope[
            "firewood.n.01_1"
        ]
        env.task.object_scope["wooden_stick.n.01_2"] = env.task.object_scope[
            "firewood.n.01_2"
        ]
        env.task.object_scope["wooden_stick.n.01_3"] = env.task.object_scope[
            "firewood.n.01_3"
        ]
        PICK_OBJ_HEIGHT = 2.8

    problem_file = init_problem_file
    terminate = False
    invalid_epi = False

    while not terminate:
        # planning
        if VLM_PLANNING:
            plan = vlm_agent.plan(problem_file, domain_file)
        else:
            plan = planner.plan(problem_file)
        print(f"Planning -- {plan}")
        if not plan:
            break

        # begin execution
        intermediate_states = planner.get_intermediate_states(
            problem_file, "pddl_output.txt"
        )

        for action_step, action in enumerate(plan):

            if not intermediate_states:
                terminate = True
                break

            if action_step == 0 and CHECK_PRECONDITION:
                current_states = intermediate_states[action_step]
                preconditions = planner.get_preconditions_by_action(plan[action_step])
                unmatch, problem_file, obs_log = check_states_and_update_problem(
                    current_states,
                    [],
                    preconditions,
                    problem_file,
                    None,
                    cur_action=None,
                    next_action=plan[action_step],
                )
                if unmatch:
                    break

            primitive = action[0]
            action_params = format_action_params(action)
            print(f"Executing action: {action}")
            if primitive == "find":
                # if not goto(action_params[1], oracle=is_oracle):
                #     # there is something wrong with the skill, not the agent's fault
                #     invalid_epi = True
                goto(action_params[1], oracle=is_oracle)
            elif primitive == "openit":
                openit(action_params[1], oracle=is_oracle)
            elif primitive in ["grasp", "graspin", "graspon"]:
                grasp(action_params[1], oracle=is_oracle)
            elif primitive == "fillsink":
                fill_sink(action_params[1], oracle=is_oracle)
            elif primitive == "fill":
                fill(action_params[1], action_params[2], oracle=is_oracle)
            elif primitive == "placein":
                place_with_predicate(
                    action_params[1], action_params[2], Inside, oracle=is_oracle
                )
            elif primitive == "place_on_floor":
                place_on_floor(action_params[1], oracle=is_oracle)
            elif primitive == "closeit":
                closeit(action_params[1], oracle=is_oracle)
            elif primitive == "microwave_water":
                turnon(action_params[1], oracle=is_oracle)
            elif primitive == "heat_food_with_oven":
                turnon(action_params[1], oracle=is_oracle)
            elif primitive == "cut_into_half":
                cut_into_half(action_params[1], action_params[2], oracle=is_oracle)
            elif primitive == "placeon":
                place_with_predicate(
                    action_params[1], action_params[2], OnTop, oracle=is_oracle
                )
            else:
                raise RuntimeError

            action_counter += 1

            if action_counter > MAX_NUM_ACTION:
                terminate = True
                break

            if CHECK_EFFECT or CHECK_PRECONDITION:
                current_states = intermediate_states[action_step + 1]
                # current action effect
                effects = planner.get_effects_by_states(
                    intermediate_states[action_step], current_states
                )
                # next action precondition
                if len(plan) > action_step + 1:
                    preconditions = planner.get_preconditions_by_action(
                        plan[action_step + 1]
                    )
                else:
                    preconditions = []

                unmatch, problem_file, obs_log = check_states_and_update_problem(
                    current_states,
                    effects,
                    preconditions,
                    problem_file,
                    intermediate_states[action_step],
                    cur_action=plan[action_step],
                    next_action=(
                        plan[action_step + 1] if len(plan) > action_step + 1 else None
                    ),
                )

                # visualization
                from matplotlib import gridspec

                fig = plt.figure(figsize=(20, 10))
                fig.suptitle(f"Step: {action_counter-1}", fontsize=20)
                spec = gridspec.GridSpec(
                    ncols=2, nrows=1, width_ratios=[1, 2], wspace=0.5
                )

                ax0 = fig.add_subplot(spec[0])
                ax0.imshow(cv2.resize(obs_log["fpv"], (512, 512)))
                ax0.set_title("First Person View", fontsize=15)
                label = f"Current: {action}"
                for query_idx, query in enumerate(obs_log["eff_queries"]):
                    label += f"\nEffect {query_idx+1}: {query}"
                    label += f"\nAnswer: {obs_log['eff_ans'][query_idx]}"
                label += f"\nMismatch: {obs_log['eff_mismatch']}"
                ax0.set_xlabel(label, fontsize=15)

                ax1 = fig.add_subplot(spec[1])
                ax1.imshow(cv2.resize(obs_log["tpv"], (1024, 512)))
                ax1.set_title("Third Person View", fontsize=15)
                if len(plan) > action_step + 1:
                    label = f"Next: {plan[action_step + 1]}"
                    for query_idx, query in enumerate(obs_log["pre_queries"]):
                        label += f"\nPrecondition {query_idx+1}: {query}"
                        label += f"\nAnswer: {obs_log['pre_ans'][query_idx]}"
                    label += f"\nMismatch: {obs_log['pre_mismatch']}"
                    ax1.set_xlabel(label, fontsize=15)
                plt.savefig(os.path.join(trial_dir, f"{action_counter-1}.png"))

                if unmatch:
                    break

            if action_step == len(plan) - 1 or invalid_epi:
                terminate = True

    # check success condition
    if not invalid_epi:
        run_sim(200)
        if a_name == "boil_water_in_the_microwave":
            if (
                env.task.object_scope["mug.n.04_1"]
                .states[Filled]
                .get_value(get_system("water"))
                and ("mug.n.04_1", "microwave.n.02_1") in inside_relationships
                and env.task.object_scope["microwave.n.02_1"]
                .states[ToggledOn]
                .get_value()
            ):
                print("success")
                exp_results["num_success"] += 1
            else:
                print("failed")
                exp_results["num_failed"] += 1
        elif a_name == "cook_a_frozen_pie":
            if (
                "pie.n.01_1",
                "oven.n.01_1",
            ) in inside_relationships and env.task.object_scope["oven.n.01_1"].states[
                ToggledOn
            ].get_value():
                print("success")
                exp_results["num_success"] += 1
            else:
                print("failed")
                exp_results["num_failed"] += 1
            # env.task.object_scope["pie.n.01_1"].states[Inside].get_value(env.task.object_scope["oven.n.01_1"])
        elif a_name == "halve_an_egg":

            if (
                env.task.object_scope["half__hard-boiled_egg.n.01_1"].exists
                and env.task.object_scope["half__hard-boiled_egg.n.01_2"].exists
            ):
                print("success")
                exp_results["num_success"] += 1
            else:
                print("failed")
                exp_results["num_failed"] += 1
        elif a_name == "bringing_water":
            if (
                0 < env.task.object_scope["water_bottle.n.01_1"].get_position()[2] < 0.1
                and 0
                < env.task.object_scope["water_bottle.n.01_2"].get_position()[2]
                < 0.1
                and "water_bottle.n.01_1" in onfloor_relationships
                and "water_bottle.n.01_2" in onfloor_relationships
            ):
                print("success")
                exp_results["num_success"] += 1
            else:
                print("failed")
                exp_results["num_failed"] += 1
        elif a_name == "store_firewood":
            if (
                env.task.object_scope["wooden_stick.n.01_2"].get_position()[2] > 0.3
                and env.task.object_scope["wooden_stick.n.01_3"].get_position()[2] > 0.3
                and not obj_held
            ):
                print("success")
                exp_results["num_success"] += 1
            else:
                print("failed")
                exp_results["num_failed"] += 1
        else:
            raise NotImplementedError

        with open("exp_results.json", "w") as f:
            json.dump(exp_results, f)
        trial_counter += 1

print("=" * 30)
print("SUMMARY:")
print(f"Number of success: {exp_results['num_success']}")
print(f"Number of failure: {exp_results['num_failed']}")
print("=" * 30)

env.close()
