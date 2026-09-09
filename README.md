# VAP-TAMP

Vision-language task and motion planning on a real mobile manipulator.

This repository combines the **DKPrompt VLM-TAMP planner** with the **Stretch AI** perception and navigation stack, and deploys the result on a Segway mobile base with a UR5e arm. A vision-language model grounds PDDL predicates against a 3D semantic voxel map, a classical planner produces the task plan, and the robot executes it with active perception when the model is uncertain.

| Component | Directory | What it does |
| --- | --- | --- |
| VLM-TAMP planner | [`vlm-tamp/`](vlm-tamp/) | DKPrompt planning loop, PDDL domains, Fast Downward, VAL, real-robot and simulation evaluation |
| Robot stack | [`stretch_ai/`](stretch_ai/) | Semantic voxel mapping, object detection, navigation, ROS 2 bridge for the Segway |
| Manipulation | [`DKPrompt.py`](DKPrompt.py), [`VLMViewGuide.py`](VLMViewGuide.py) | Marker-based grasping with VLM-guided view selection on the UR5e |
| Scene graphs | [`generate_scene_graph.py`](generate_scene_graph.py) | Export a scene graph JSON from a saved voxel map |

---

## Quick start: run on the real robot

### 1. Start the Segway bridge

This connects the ROS 2 system to the Segway robot via the ROS bridge. Run it on the robot computer and leave it running.

```bash
./stretch_ai/scripts/run_segway_bridge.sh
```

### 2. Run the evaluation script

From the `vlm-tamp/` directory, on the workstation:

```bash
cd vlm-tamp
python eval_real_robot.py \
  --robot-ip <ROBOT_IP> \
  --map-file <path/to/map.pkl> \
  --domain domains/bottle_collection/domain.pddl \
  --problem domains/bottle_collection/problem.pddl \
  --api-key <GEMINI_API_KEY> \
  --location-map location_mapping.yaml \
  --config rosbridge_robot_config.yaml \
  --calibration simple_offset_calibration.yaml \
  --output bottle_collection_results.json
```

Swap the domain and problem files to run a different task. Available domains live in [`vlm-tamp/domains/`](vlm-tamp/domains/): bottle collection, bringing water, egg halving, fruit pickup, firewood storage, and more.

Results are written to the JSON file given by `--output`.

---

## Working offline from a saved map

Most of the pipeline runs without a robot once you have a `.pkl` voxel map.

**Plan with a VLM over the map**, visualize instances and the voxel grid, and optionally send the plan to the robot:

```bash
cd stretch_ai
python -m stretch.app.vlm_planning -i <path/to/map.pkl> --show-instances --show-svm -f 20 -fs 3
```

No robot IP is needed in this mode. Add `--robot_ip <ROBOT_IP>` to execute the plan on the robot. The full option list and the three run modes are documented in [`stretch_ai/docs/apps.md`](stretch_ai/docs/apps.md#vlm-planning).

**Export a scene graph** from the same map:

```bash
python3 generate_scene_graph.py -i <path/to/map.pkl> -o scene_graph.json
```

**Annotate room and location labels** for navigation, which the voxel map does not carry on its own. See [`vlm-tamp/LOCATION_MAPPING.md`](vlm-tamp/LOCATION_MAPPING.md).

---

## Documentation

| Document | Covers |
| --- | --- |
| [`vlm-tamp/README.md`](vlm-tamp/README.md) | Installing the planner, building Fast Downward and VAL, simulation mode |
| [`vlm-tamp/ARCHITECTURE.md`](vlm-tamp/ARCHITECTURE.md) | System design, coordinate frames, data flow |
| [`vlm-tamp/EXPERIMENTS.md`](vlm-tamp/EXPERIMENTS.md) | Running the three experimental tasks on the Segway |
| [`vlm-tamp/TASK_SPECIFICATIONS.md`](vlm-tamp/TASK_SPECIFICATIONS.md) | Task definitions and their PDDL encodings |
| [`vlm-tamp/LOCATION_MAPPING.md`](vlm-tamp/LOCATION_MAPPING.md) | Annotating semantic locations on a map |
| [`stretch_ai/README.md`](stretch_ai/README.md) | Stretch AI setup, hardware requirements, Docker |
| [`stretch_ai/docs/apps.md`](stretch_ai/docs/apps.md) | Every Stretch AI app, including mapping, `read_map`, and [VLM planning](stretch_ai/docs/apps.md#vlm-planning) |

---

## Notes

- The ROS bridge must be running before `eval_real_robot.py` is started.
- `--api-key` takes a Gemini API key. Copy [`.env.example`](.env.example) to `.env` if you prefer to keep keys out of the shell history.
- The first run of any perception app downloads the Detic and CLIP checkpoints and takes a few minutes.
- Saved maps and instance crops used in the experiments live under [`maps/`](maps/) and [`instances/`](instances/).
