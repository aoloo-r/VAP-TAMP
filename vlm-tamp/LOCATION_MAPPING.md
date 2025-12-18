# Location Mapping Guide

Since the voxel map contains objects but not room labels, you need to manually annotate locations on the map for navigation.

## Step 1: Annotate Locations on Your Map

Run the interactive annotation tool:

```bash
cd /home/aoloo/code/vlm-tamp

python annotate_locations.py \
    --map-file /home/aoloo/code/stretch_ai/scripts/visual_grounding_benchmark/sample9.pkl \
    --output location_mapping.yaml
```

This will open a window showing your occupancy grid map.

### Controls:
- **Left click**: Add a location marker (you'll be prompted to enter a name)
- **Right click**: Remove the nearest marker
- **Press 'q'**: Save and quit

### Naming Convention:
Use the same location names as in your PDDL problem file. For example, if your PDDL has:
```pddl
loc_main loc_room2 loc_room3 - location
```

Then click on the map and name your locations:
- `loc_main` - Click where the main room/collection area is
- `loc_room2` - Click where room 2 is (where bottle1 will be)
- `loc_room3` - Click where room 3 is (where bottle2 will be)

### Tips:
- Click in the center of each room/area
- Make sure locations are reachable by the robot (not in walls!)
- The coordinates are in the map frame (same as AMCL)

## Step 2: Run Your Experiment with Location Mapping

Now run your PDDL task with the location mapping:

```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /home/aoloo/code/stretch_ai/scripts/visual_grounding_benchmark/sample9.pkl \
    --domain domains/bottle_collection/domain.pddl \
    --problem domains/bottle_collection/problem.pddl \
    --api-key YOUR_GEMINI_API_KEY \
    --location-map location_mapping.yaml \
    --output bottle_collection_results.json
```

## Example Location Mapping File

The annotation tool creates a YAML file like this:

```yaml
map_file: /path/to/map.pkl
resolution: 0.05
origin: [0.0, 0.0, 0.0]
locations:
  loc_main:
    x: 1.5
    y: 0.5
  loc_room2:
    x: 3.2
    y: -1.8
  loc_room3:
    x: -2.1
    y: 2.4
```

## Verifying Your Locations

After creating the mapping, you can verify the coordinates make sense by checking:

1. **View in RViz** (if you have it running):
   - Locations should be in free space, not obstacles
   - Should match where you actually want the robot to go

2. **Check with robot's current position**:
   ```bash
   # In separate terminal, check AMCL pose
   ros2 topic echo /amcl_pose
   ```
   The location coordinates should be in the same frame as the AMCL pose.

## Troubleshooting

### "Cannot navigate - no location mapping or landmark found"
- Make sure you created the `location_mapping.yaml` file
- Make sure you passed `--location-map location_mapping.yaml` to eval_real_robot.py
- Check that location names in YAML match location names in PDDL problem

### Robot navigates to wrong place
- Re-run annotation tool and adjust the location markers
- Check calibration between Record3D map and AMCL map

### Annotation tool crashes
- Make sure you have matplotlib and PyYAML installed:
  ```bash
  pip install matplotlib pyyaml
  ```
