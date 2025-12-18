# Real Robot Experiments Guide

This guide explains how to run the three experimental tasks on the Segway robot with UR5e arm.

## Prerequisites

1. **Robot Setup**
   - Segway robot powered on
   - UR5e arm initialized
   - AMCL localization running
   - All sensors active (SICK LIDAR, RealSense camera)

2. **Environment Setup**
   - Record3D voxel map captured (`.pkl` file)
   - Map calibration completed (`simple_offset_calibration.yaml`)
   - Objects placed in environment as per task requirements

3. **Software Setup**
   - Stretch AI bridge running: `./scripts/run_segway_bridge.sh`
   - Gemini API key available

## Task Descriptions

### Task 1: Bottle Collection
**Objective**: Collect 2 empty bottles from different rooms and place them in a collection area in the main room.

**Setup Requirements**:
- 2 empty bottles placed in different rooms (e.g., room2, room3)
- Designated collection table in main room
- All rooms accessible via navigation

**PDDL Files**:
- Domain: `domains/bottle_collection/domain.pddl`
- Problem: `domains/bottle_collection/problem.pddl`

**Expected Actions**:
1. Navigate to room2
2. Pick bottle1
3. Navigate to main room
4. Place bottle1 on collection table
5. Navigate to room3
6. Pick bottle2
7. Navigate to main room
8. Place bottle2 on collection table

**Failure Recovery**:
- Object not found: Active perception explores for better view
- Navigation failure: Retries up to 3 times
- Grasp failure: Retries with adjusted approach

---

### Task 2: Egg Halving
**Objective**: Locate a knife and use it to cut a hard-boiled egg into two halves.

**Setup Requirements**:
- 1 hard-boiled egg on kitchen table
- 1 knife on counter (must be sharp and clean)
- All items within reachable space

**PDDL Files**:
- Domain: `domains/egg_halving/domain.pddl`
- Problem: `domains/egg_halving/problem.pddl`

**Expected Actions**:
1. Navigate to kitchen
2. Pick up knife from counter
3. Cut egg on table (manipulation task)
4. Place knife back on counter (optional cleanup)

**Failure Recovery**:
- Tool not found: Active perception searches
- Cutting failure: Retries cutting motion
- Manipulation error: Adjusts arm trajectory

---

### Task 3: Firewood Storage
**Objective**: Collect 2 wooden sticks from storage room and place them on a table in the main room, handling a closed door.

**Setup Requirements**:
- 2 wooden sticks in storage room
- Door between main room and storage room (initially closed)
- Storage table in main room
- Human available to open door when requested

**PDDL Files**:
- Domain: `domains/firewood_storage/domain.pddl`
- Problem: `domains/firewood_storage/problem.pddl`

**Expected Actions**:
1. Navigate to storage room door
2. Request door to be opened (wait for human)
3. Pass through door to storage room
4. Pick stick1
5. Navigate back through door to main room
6. Place stick1 on storage table
7. Navigate back to storage room
8. Pick stick2
9. Navigate to main room
10. Place stick2 on storage table

**Failure Recovery**:
- Door not opened: Waits indefinitely or times out
- Navigation through door: Retries if blocked
- Stick not found: Active perception explores

---

## Running Experiments

### Option 1: Run All Experiments Sequentially

```bash
cd /home/aoloo/code/vlm-tamp

python run_experiments.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/your/map.pkl \
    --api-key YOUR_GEMINI_API_KEY
```

This will run all 3 tasks in sequence and save results to `experiment_results/`.

### Option 2: Run Individual Tasks

**Task 1: Bottle Collection**
```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/bottle_collection/domain.pddl \
    --problem domains/bottle_collection/problem.pddl \
    --api-key YOUR_GEMINI_API_KEY \
    --output bottle_collection_results.json
```

**Task 2: Egg Halving**
```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/egg_halving/domain.pddl \
    --problem domains/egg_halving/problem.pddl \
    --api-key YOUR_GEMINI_API_KEY \
    --output egg_halving_results.json
```

**Task 3: Firewood Storage**
```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/firewood_storage/domain.pddl \
    --problem domains/firewood_storage/problem.pddl \
    --api-key YOUR_GEMINI_API_KEY \
    --output firewood_storage_results.json
```

### Option 3: Resume After Interruption

If experiments are interrupted, resume from a specific task:

```bash
python run_experiments.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --api-key YOUR_GEMINI_API_KEY \
    --start-from 1  # 0=Task 1, 1=Task 2, 2=Task 3
```

## Experiment Results

Results are saved in JSON format with the following structure:

```json
{
  "robot_ip": "172.20.10.3",
  "map_file": "/path/to/map.pkl",
  "timestamp": "20250115_143022",
  "experiments": [
    {
      "experiment": "Task 1: Bottle Collection",
      "description": "Collect 2 empty bottles from different rooms",
      "domain": "domains/bottle_collection/domain.pddl",
      "problem": "domains/bottle_collection/problem.pddl",
      "start_time": "2025-01-15T14:30:22",
      "end_time": "2025-01-15T14:45:18",
      "duration_seconds": 896.3,
      "statistics": {
        "total_actions": 8,
        "successful_actions": 8,
        "failed_actions": 0,
        "retried_actions": 1,
        "vlm_queries": 12,
        "uncertain_responses": 2,
        "active_perception_triggers": 2,
        "active_perception_successes": 2,
        "navigation_failures": 0,
        "grasp_failures": 0,
        "object_not_found_failures": 0
      }
    }
  ]
}
```

## Monitoring Progress

During execution, the system prints:
- Current action being executed
- VLM responses (yes/no/uncertain)
- Active perception triggers
- Retry attempts
- Navigation status
- Grasp success/failure

Example output:
```
======================================================================
Executing: pick([robot1, bottle1, loc_room2, table_room2])
======================================================================
Picking bottle

[PERCEPTION] Querying VLM: Is there a bottle visible?
[VLM] Response: uncertain - the view is partially occluded

⚠️ VLM uncertain, triggering active perception...
🔍 ACTIVE EXPLORATION for predicate: ['item-at', 'bottle1', 'loc_room2']
   Target object: bottle
   Object at: (3.45, -2.10)
   Generated 8 viewpoints

   Viewpoint 1/3: (4.45, -2.10, 0.0°)
      🧠 Querying VLM with new view...
      VLM response: yes
      ✅ Got confident answer: yes

   🔙 Returning to original position...
   ✅ Returned to original position

✅ Action succeeded
```

## Safety Considerations

1. **Emergency Stop**: Keep E-stop button accessible at all times
2. **Workspace**: Ensure clear path for robot navigation
3. **Obstacles**: Remove unexpected obstacles from planned path
4. **Door Task**: Stay near door to open it when requested
5. **Cutting Task**: Ensure egg is stable and knife is safe to use
6. **Monitoring**: Watch robot during manipulation tasks

## Troubleshooting

### Robot doesn't move
- Check AMCL localization: `ros2 topic echo /amcl_pose`
- Verify move_base is running: `ros2 topic list | grep move_base`
- Check for navigation errors in bridge output

### Object not found
- Verify object is in voxel map
- Check semantic segmentation labels
- Adjust active perception parameters

### Grasp failures
- Check gripper functionality
- Verify object is reachable
- Adjust approach distance

### VLM always uncertain
- Check camera feed quality
- Verify lighting conditions
- Increase exploration attempts

## Expected Performance

Based on preliminary tests:
- **Success Rate**: 70-90% per task
- **Duration**: 10-20 minutes per task
- **Active Perception**: Triggered 1-3 times per task
- **Retries**: 0-2 retries per task

## Data Collection

The system automatically logs:
- Action execution times
- VLM query/response pairs
- Active perception trajectories
- Failure modes and recovery attempts
- Final task success/failure status

This data is valuable for analyzing:
- Most common failure modes
- Effectiveness of active perception
- Impact of retry strategies
- Overall system robustness
