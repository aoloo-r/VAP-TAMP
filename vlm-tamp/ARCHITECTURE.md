# VLM-TAMP + Stretch AI + Segway Robot Architecture

## Overview

This system integrates DKPrompt (VLM-TAMP) with a Segway robot using Stretch AI framework for navigation and manipulation.

## System Components

### 1. Robot Hardware
- **Base**: Segway RMP with differential drive
- **Arm**: UR5e robotic arm (6-DOF)
- **Sensors**:
  - SICK TIM LIDAR (2D laser scanner)
  - RealSense D435 camera (RGB-D)
  - IMU (inertial measurement)
  - Wheel encoders

### 2. Software Stack

#### Navigation & Localization (ROS1 on Robot)
- **AMCL**: Adaptive Monte Carlo Localization
  - Uses SICK LIDAR + 2D occupancy grid map
  - Publishes `/amcl_pose` (robot position in map frame)
  - Handles drift correction and localization

- **move_base**: Navigation stack
  - Receives goals via `/move_base_simple/goal`
  - Uses global planner (Navfn) + local planner (EBand)
  - Handles obstacle avoidance with costmaps

#### 3D Semantic Mapping (Record3D + Stretch AI)
- **Record3D**: iPhone-based 3D mapping
  - Creates dense RGB-D voxel map
  - Saved as `.pkl` files
  - Used for semantic understanding

- **Stretch AI Voxel Map**:
  - Loads Record3D map
  - Runs semantic segmentation (DINOv2 + SIGLIP)
  - Provides object instances with categories
  - Coordinate frame: 3D map frame (different from AMCL map)

#### Manipulation (UR5e via ROS2)
- **ROS2 Control**: Joint trajectory control
- **MoveIt**: Motion planning (available but not primary)
- **Topics**:
  - `/joint_states` - Current joint positions
  - `/scaled_pos_joint_traj_controller/follow_joint_trajectory` - Arm control

#### ROS1-ROS2 Bridge
- **Purpose**: Bridge robot (ROS1) with client (ROS2)
- **Script**: `./scripts/run_segway_bridge.sh`
- **Key bridged topics**:
  - `/amcl_pose` (ROS1 → ROS2)
  - `/camera/*` (ROS1 → ROS2)
  - `/scan` (ROS1 → ROS2)
  - `/move_base_simple/goal` (ROS2 → ROS1)

#### ZMQ Server (Stretch AI Communication)
- **File**: `scripts/segway_adapter/home_zmq_server.py`
- **Purpose**: Convert ROS2 topics to ZMQ messages
- **Ports**:
  - 4401: Observations (RGB, depth, camera pose)
  - 4402: Actions (navigation goals)
  - 4403: State (robot pose, joint states)
  - 4404: Servo (camera data)

#### Active Perception Module
- **File**: `vlm-tamp/active_perception.py`
- **Purpose**: Explore for better views when VLM is uncertain
- **Features**:
  - Detects VLM uncertainty
  - Samples viewpoints around objects
  - Navigates to new viewpoints
  - Retries perception
  - Returns to original position

## Coordinate Frames

### Frame Relationships

```
map (AMCL 2D map)
 ├─ base_footprint (robot base, from AMCL)
 │   └─ camera_color_optical_frame (camera)
 │
 └─ [offset_x, offset_y] → Record3D voxel map frame
```

### Key Frames
1. **`map`**: AMCL's 2D map frame (global reference)
2. **`odom`**: Wheel odometry frame (drifts over time)
3. **`base_footprint`**: Robot base frame
4. **Record3D frame**: 3D voxel map frame (independent)

### Coordinate Transformations

**AMCL provides**: `map → odom` transform (corrects drift)

**Calibration provides**: Record3D → map transform
- Stored in: `simple_offset_calibration.yaml`
- Parameters: `offset_x`, `offset_y`
- Usage: When navigating to objects from voxel map

**Formula**:
```python
# Object in Record3D voxel map
obj_x_voxel, obj_y_voxel = instance.get_center()

# Convert to AMCL map frame for navigation
nav_x = obj_x_voxel + offset_x
nav_y = obj_y_voxel + offset_y

# Send to move_base (expects map frame)
robot.navigate_to_goal(nav_x, nav_y, theta)
```

## Data Flow

### Navigation Command Flow
```
VLM-TAMP (find object "cup")
  ↓
Active Perception Module
  → Query voxel map for "cup" instance
  → Get center: (x_voxel, y_voxel)
  → Apply calibration: (x_map, y_map) = (x_voxel + offset_x, y_voxel + offset_y)
  ↓
ZMQ Client (navigate_to_goal)
  → Send {"navigation_goal": [x_map, y_map, theta]} via ZMQ
  ↓
ZMQ Server (home_zmq_server.py)
  → Publish PoseStamped to /move_base_simple/goal
  ↓
ROS1-ROS2 Bridge
  → Bridge to ROS1
  ↓
move_base (ROS1)
  → Plan path using AMCL pose and costmaps
  → Execute motion
```

### Pose Reporting Flow
```
AMCL (ROS1)
  → Publishes /amcl_pose (PoseWithCovarianceStamped)
  ↓
ROS1-ROS2 Bridge
  → Bridge to ROS2
  ↓
ZMQ Server
  → Subscribe to /amcl_pose
  → Extract [x, y, theta] in map frame
  → Publish via ZMQ on port 4403
  ↓
ZMQ Client (get_base_pose)
  → Returns robot position in map frame
  → Used by navigation planner
```

## Critical Design Decisions

### ✅ Using AMCL Pose (not odometry)
**Why**:
- AMCL corrects for wheel slip and drift
- Provides accurate position in map frame
- Consistent with move_base expectations

**Implementation**:
```python
# ZMQ Server (home_zmq_server.py)
def amcl_pose_cb(self, msg: PoseStamped):
    # Extract AMCL pose in map frame
    self.latest_amcl_pose = np.array([pos.x, pos.y, yaw])

def lookup_poses(self):
    # Use AMCL pose (map frame) instead of odom
    if self.latest_amcl_pose is not None:
        base_pose_xyt = self.latest_amcl_pose  # PRIMARY
    else:
        base_pose_xyt = self.latest_odom_pose  # FALLBACK
```

### ✅ Calibration: Record3D → AMCL Map
**Why**:
- Record3D and AMCL use different coordinate frames
- Need to align 3D semantic map with 2D navigation map

**How to calibrate**:
```bash
# Use the calibration tool
python /home/aoloo/code/stretch_ai/pick_coordinates_with_odom.py

# Steps:
# 1. Click landmarks in Record3D 2D map
# 2. Input robot's AMCL pose at each landmark
#    (Get from: ros2 topic echo /amcl_pose)
# 3. Tool computes offset_x, offset_y
# 4. Saves to: simple_offset_calibration.yaml
```

### ✅ No Initial Pose Needed
**Why**:
- AMCL handles localization automatically
- Robot starts with particle filter spread
- Converges to correct position as it moves

**Exception**:
If AMCL is lost, use:
```bash
# Set initial pose in RViz or via topic
ros2 topic pub /initialpose geometry_msgs/PoseWithCovarianceStamped ...
```

## Running the System

### 1. Start Robot Bridge
```bash
cd /home/aoloo/code/stretch_ai
./scripts/run_segway_bridge.sh
```

### 2. Test Active Perception
```bash
cd /home/aoloo/code/vlm-tamp
python test_active_perception_predicates.py
```

### 3. Run Full PDDL Task
```bash
cd /home/aoloo/code/vlm-tamp
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/bringing_water/domain.pddl \
    --problem domains/bringing_water/problem.pddl \
    --api-key YOUR_GEMINI_KEY
```

## Troubleshooting

### Robot goes to wrong location
**Problem**: Calibration is incorrect
**Solution**:
1. Check calibration file values are reasonable
2. Re-run calibration with more landmarks (3-5 points)
3. Verify AMCL is localized (check `/amcl_pose` in RViz)

### AMCL not publishing pose
**Problem**: AMCL lost or not started
**Solution**:
1. Check LIDAR is working: `ros2 topic echo /scan`
2. Check map is loaded: `ros2 topic echo /map --once`
3. Set initial pose in RViz
4. Drive robot to help AMCL converge

### Navigation fails
**Problem**: move_base can't plan path
**Solution**:
1. Check costmaps in RViz
2. Verify goal is reachable
3. Check for obstacles blocking path
4. Reduce goal tolerance in move_base params

### VLM always uncertain
**Problem**: Poor camera view or object not visible
**Solution**:
1. Active perception will explore automatically
2. Check camera is working: `ros2 topic echo /camera/color/image_raw`
3. Verify object is in voxel map
4. Adjust viewpoint_distance parameter

## File Locations

### Configuration
- Robot config: `/home/aoloo/code/stretch_ai/src/stretch/config/rosbridge_robot_config.yaml`
- Calibration: `/home/aoloo/code/stretch_ai/simple_offset_calibration.yaml`

### Code
- Active perception: `/home/aoloo/code/vlm-tamp/active_perception.py`
- Real robot executor: `/home/aoloo/code/vlm-tamp/eval_real_robot.py`
- ZMQ server: `/home/aoloo/code/stretch_ai/scripts/segway_adapter/home_zmq_server.py`
- UR5e client: `/home/aoloo/code/stretch_ai/src/stretch/core/ur5e_robot.py`

### Maps
- Voxel maps: `/home/aoloo/code/stretch_ai/scripts/visual_grounding_benchmark/*.pkl`
- 2D map: Served by map_server on robot

## Future Improvements

1. **Dynamic calibration**: Auto-align Record3D map with AMCL map using feature matching
2. **Continuous mapping**: Update voxel map in real-time from RealSense camera
3. **Better failure recovery**: Handle AMCL relocalization failures
4. **Gripper integration**: Add actual gripper control for UR5e
5. **Visual servoing**: Use camera feedback for precise manipulation
