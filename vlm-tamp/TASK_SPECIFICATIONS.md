# VLM-TAMP Real Robot Task Specifications

This document details the three experimental tasks, mapping high-level objectives to implementation details.

---

## Task 1: Bring in Empty Bottles

### Objective
Bring two water bottles - one from the TA Area (Room 2) and one from the Conference Room (Room 3) - and place both on the lab table in Room 1 (Lab).

### Environment Setup
- **Room 1 (Lab)**: Target room; placement table at center
- **Room 2 (TA Area)**: First bottle on table at center
- **Room 3 (Conference Room)**: Second bottle on conference table closest to door

### Location Annotation Requirements

When running `annotate_locations.py`, click on these positions:

1. **loc_main** (Room 1 - Lab): Click on/near the lab table at center of room
2. **loc_room2** (Room 2 - TA Area): Click on/near the TA table at center
3. **loc_room3** (Room 3 - Conference): Click on/near the conference table near door

### Process Sequence

#### 1. Starting Position
- Robot begins in Room 1 (Lab), facing main corridor

#### 2. Navigate to Room 2 (TA Area)
```
Action: navigate(robot1, loc_main, loc_room2, room2)
```
- Follow navigation goal from 3D map object position
- Pass through open door from Room 1 to Room 2
- Upon reaching goal coordinates, activate object detection

#### 3. Object Detection & Pickup (Room 2)
```
Action: pick(robot1, bottle1, loc_room2, table_room2)
```

**Implementation Details**:
- Activate RGB-D vision system (Detic semantic segmentation)
- Search for object with class "bottle" in semantic map
- If bottle not visible → **Active Perception**:
  - VLM queries current camera view: "Is there a bottle visible?"
  - If "uncertain" or "no" → explore viewpoints around mapped bottle position
  - Generate 8 viewpoints in circle around object (1.0m radius)
  - Navigate to each viewpoint, query VLM until "yes" or max attempts (3)
- Once detected, adjust robot base to align gripper with bottle
- Extend manipulator and grasp bottle
- Verify grasp (gripper closed successfully)

**Failure Recovery**:
- Object not detected after active perception → rotate in 30° increments (up to 360°), retry detection twice
- Grasp fails → reposition manipulator, reattempt (max 3 retries)
- Navigation fails → clear costmap, trigger local replanning (max 3 retries)

#### 4. Return to Room 1 (Lab)
```
Action: navigate(robot1, loc_room2, loc_main, room1)
```
- Navigate back through same path
- Dynamic obstacle avoidance active

#### 5. Place First Bottle
```
Action: place(robot1, bottle1, loc_main, collection_table)
```
- Navigate to placement table coordinates
- Lower manipulator to table height
- Open gripper to release bottle
- Retract manipulator

#### 6. Navigate to Room 3 (Conference Room)
```
Action: navigate(robot1, loc_main, loc_room3, room3)
```
- Move from Room 1 to Room 3 using navigation goal from 3D map

#### 7. Detect and Pickup Second Bottle (Room 3)
```
Action: pick(robot1, bottle2, loc_room3, table_room3)
```
- Same detection and active perception procedure as step 3
- Pickup from conference table closest to door

#### 8. Return and Place Second Bottle
```
Action: navigate(robot1, loc_room3, loc_main, room1)
Action: place(robot1, bottle2, loc_main, collection_table)
```
- Navigate back to Room 1
- Place second bottle next to first one on same table

### Success Criteria
- Both bottles successfully placed on lab table in Room 1
- No collisions during navigation
- Successful grasps maintained throughout transport

---

## Task 2: Halve an Egg

### Objective
In Room 1 (Lab), use a toy knife to cut a toy egg placed on the table into two halves.

### Environment Setup
- **Knife location**: On board to the left of the table
- **Egg location**: On top of table at center of lab table
- **All in Room 1** (single room task)

### Location Annotation Requirements

When running `annotate_locations.py`:

1. **loc_kitchen** (Room 1 - Lab): Click on/near the lab table where egg is located

### Process Sequence

#### 1. Starting Position
- Robot starts in Room 1 facing the table

#### 2. Locate Knife
```
Action: navigate(robot1, loc_start, loc_kitchen, kitchen)
```
- Get knife position from 3D voxel map (semantic class "knife")
- Navigate to knife's mapped coordinates

#### 3. Grasp Knife
```
Action: pick-tool(robot1, knife, loc_kitchen, counter)
```

**Implementation Details**:
- Activate object detection for "knife" class
- If not visible → **Active Perception** around counter area
- Align manipulator with knife handle position
- Approach with appropriate wrist orientation for downward slicing motion
- Close gripper and verify grasp stability (force feedback)

**Failure Recovery**:
- Knife detection fails → rotate in place, rescan workspace (3 retries max)
- Grasp fails → adjust approach pose, retry up to 2 times

#### 4. Move to Table and Cut Egg
```
Action: cut-egg(robot1, knife, egg, loc_kitchen, table)
```

**Implementation Details**:
- Navigate to egg's position on cutting board
- Detect egg using semantic segmentation (class "egg")
- If egg not visible → **Active Perception**
- Position manipulator above egg (aligned with cutting trajectory)
- Execute controlled linear downward cutting motion:
  - Defined trajectory: ~5 cm depth
  - Slow velocity for precision (5 cm/s)
  - Pause briefly at mid-point
  - Retract knife to complete slice
- Monitor force feedback for excessive resistance

**Failure Recovery**:
- Egg position drifts → re-localize egg, adjust target coordinates
- Excessive resistance during cutting → stop motion, retract, retry with reduced force
- Cutting trajectory fails → reset manipulator, retry from starting position

#### 5. Return Knife (Optional)
```
Action: place-tool(robot1, knife, loc_kitchen, counter)
```
- Navigate back to original knife location
- Place knife back on board

### Success Criteria
- Egg successfully cut into two visible halves
- Knife properly grasped and manipulated throughout
- No damage to robot or environment

---

## Task 3: Store Firewood

### Objective
Retrieve two wooden sticks from Conference Room (Room 3) and place them on lab table in Room 1.

### Environment Setup
- **Room 1 (Lab)**: Target area for storing sticks (same table as Task 1)
- **Room 2 (TA Area)**: Intermediate room between Lab and Conference Room
- **Room 3 (Conference Room)**: Two wooden sticks on table near left wall
- **Door between Room 2 and Room 3**: Initially closed

### Location Annotation Requirements

When running `annotate_locations.py`:

1. **loc_main** (Room 1 - Lab): Lab table at center
2. **loc_storage** (Room 3 - Conference): Conference table near left wall

### Process Sequence

#### 1. Starting Position
- Robot begins in Room 2 (TA Area)

#### 2. Navigate to Room 3 Door
```
Action: navigate(robot1, loc_main, loc_storage_door, storage_room)
```
- Approach closed door separating Room 2 and Room 3

#### 3. Request Door Opening
```
Action: request-door-open(robot1, storage_door, room2, storage_room, loc_storage_door)
```

**Implementation Details**:
- Detect door closure (lidar scan shows obstacle at expected doorway)
- Make output request: "Please open the door."
  - Display text on terminal
  - Optional: Play audio request if TTS available
- Wait for human assistance or access confirmation
- Timeout: 30 seconds before retry
- Monitor door state (check for path clearance)

**Failure Recovery**:
- Door remains closed for >30s → repeat request once
- After 2nd timeout (60s total) → abort task or wait indefinitely (configurable)

#### 4. Pass Through Door to Room 3
```
Action: pass-through-door(robot1, loc_storage_door, loc_storage, storage_door, room2, storage_room)
```
- Once door is open, navigate through doorway
- Extra caution near doorway (slow velocity, obstacle checking)
- Enter Room 3 to object position from 3D map

#### 5. Detect and Pickup First Stick
```
Action: pick(robot1, stick1, loc_storage, floor_storage)
```

**Implementation Details**:
- Use object detection for "stick" or "wood" class
- If not visible → **Active Perception**:
  - Rotate 30° increments up to 360° scan
  - Query VLM: "Is there a wooden stick visible?"
  - Explore viewpoints around mapped stick position
- Grasp first stick with appropriate gripper orientation

**Failure Recovery**:
- Object detection fails → rotate with increasing angles (30° → 360°), retry
- Grasp fails → retry with adjusted orientation (max 3 attempts)

#### 6. Transport First Stick to Room 1
```
Action: pass-through-door(robot1, loc_storage, loc_main, storage_door, storage_room, main_room)
Action: navigate(robot1, loc_storage, loc_main, main_room)
Action: place(robot1, stick1, loc_main, main_table)
```
- Navigate back through door to Room 1
- Place stick on lab table

#### 7. Return for Second Stick
```
Action: navigate(robot1, loc_main, loc_storage, storage_room)
Action: pick(robot1, stick2, loc_storage, floor_storage)
```
- Navigate back to Room 3
- Detect and grasp second stick

#### 8. Transport Second Stick to Room 1
```
Action: navigate(robot1, loc_storage, loc_main, main_room)
Action: place(robot1, stick2, loc_main, main_table)
```
- Navigate back to Room 1
- Place second stick next to first one

### Success Criteria
- Both wooden sticks successfully placed on lab table
- Door interaction handled correctly (human assistance requested)
- Safe navigation through doorway

---

## Common Implementation Details

### Active Perception Module

Used across all tasks when VLM is uncertain about object visibility:

**Trigger Conditions**:
- VLM responds "uncertain" to visibility query
- Object not detected by semantic segmentation
- Grasp fails due to poor view

**Procedure**:
1. Get object position from 3D voxel map (x, y, z)
2. Generate N viewpoints in circle around object (default: 8 viewpoints, 1.0m radius)
3. For each viewpoint:
   - Navigate to viewpoint
   - Capture RGB image
   - Query VLM with image: "Is there a [object] visible?"
   - If "yes" → return success, stay at this viewpoint
   - If "no" or "uncertain" → try next viewpoint
4. If all viewpoints exhausted → return failure
5. If successful → return to original position (optional)

**Parameters** (configurable):
- `max_exploration_attempts`: 3
- `viewpoint_distance`: 1.0m
- `num_viewpoints`: 8

### Failure Recovery Strategies

**Navigation Failures**:
- Clear local costmap
- Trigger replanning with updated obstacles
- Retry up to 3 times
- If all retries fail → mark action as failed, stop execution

**Object Detection Failures**:
- Active perception (described above)
- Rotate in place (30° increments, up to 360°)
- Retry detection 2-3 times
- If all fail → mark action as failed

**Grasp Failures**:
- Reposition manipulator (adjust approach angle)
- Retry with different gripper orientation
- Max 3 attempts
- If all fail → mark action as failed

### Statistics Collected

For each experiment run, collect:
- `total_actions`: Total PDDL actions executed
- `successful_actions`: Actions that succeeded
- `failed_actions`: Actions that failed after retries
- `retried_actions`: Number of retry attempts
- `vlm_queries`: Total VLM API calls
- `uncertain_responses`: Times VLM was uncertain
- `active_perception_triggers`: Times active perception was triggered
- `active_perception_successes`: Times active perception found object
- `navigation_failures`: Navigation action failures
- `grasp_failures`: Grasp action failures
- `object_not_found_failures`: Object detection failures

### Coordinate Frame Considerations

**Critical**: All coordinates must be in the **AMCL map frame** (not the Record3D frame)

1. **Location Mapping** (`location_mapping.yaml`):
   - User clicks on occupancy grid (already in AMCL frame)
   - Coordinates saved directly in AMCL frame
   - No transformation needed

2. **Object Positions** (from voxel map):
   - Voxel map built from Record3D data
   - Requires calibration transform: `simple_offset_calibration.yaml`
   - Transform applied: `(x_amcl, y_amcl) = (x_record3d + offset_x, y_record3d + offset_y)`

3. **Robot Position** (from AMCL):
   - Subscribe to `/amcl_pose` topic
   - Already in map frame
   - Used for navigation commands to `move_base`

---

## Execution Checklist

### Before Running Experiments:

- [ ] Segway robot powered on and initialized
- [ ] UR5e arm initialized and in home position
- [ ] AMCL localization running (`ros2 topic echo /amcl_pose` shows valid pose)
- [ ] Move_base navigation running
- [ ] ZMQ bridge running: `./scripts/run_segway_bridge.sh`
- [ ] Record3D voxel map captured (`.pkl` file)
- [ ] Map calibration completed (`simple_offset_calibration.yaml`)
- [ ] Location mapping created (`location_mapping.yaml`) ✨ **NEW**
- [ ] Objects placed in environment according to task requirements
- [ ] Doors opened/closed as required for each task
- [ ] Human available for Task 3 door opening

### Running Single Task:

```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/bottle_collection/domain.pddl \
    --problem domains/bottle_collection/problem.pddl \
    --api-key YOUR_GEMINI_API_KEY \
    --location-map location_mapping.yaml \
    --output bottle_collection_results.json
```

### Running All Tasks:

```bash
python run_experiments.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --api-key YOUR_GEMINI_API_KEY \
    --location-map location_mapping.yaml
```

---

## Expected Performance

Based on task complexity:

| Task | Expected Duration | Expected Success Rate | Active Perception Triggers |
|------|-------------------|----------------------|----------------------------|
| Task 1: Bottles | 10-15 min | 80-90% | 2-4 times |
| Task 2: Egg Halving | 8-12 min | 70-85% | 1-3 times |
| Task 3: Firewood | 12-18 min | 75-90% | 2-4 times |

Common failure modes:
- Object not found after active perception (5-10%)
- Grasp failure due to object slip (5-10%)
- Navigation failure due to dynamic obstacles (3-5%)
- Door not opened in time for Task 3 (5-10%)

---

## Safety Considerations

1. **Emergency Stop**: E-stop button accessible at all times
2. **Workspace**: Clear paths for robot navigation
3. **Obstacles**: Remove unexpected obstacles from planned paths
4. **Door Task**: Human must stay near door for Task 3
5. **Cutting Task**: Ensure egg is stable, knife is safe toy knife
6. **Monitoring**: Actively watch robot during manipulation tasks
7. **Velocity Limits**: Reduced velocity near doors, obstacles, and during manipulation

