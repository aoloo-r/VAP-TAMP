# VAP-TAMP

**Vision-language Active Perception for Task and Motion Planning** on a real mobile manipulator.

VAP-TAMP plans and executes long-horizon household tasks on a Segway mobile base with a UR5e arm. A vision-language model grounds the symbolic state of a PDDL problem against a 3D semantic voxel map of the environment, a classical planner produces the task plan, and the robot executes it. When the model is uncertain about a predicate, the system actively moves to a better viewpoint before committing to a plan.

The system is organized in three layers:

| Layer | Where it lives | Responsibility |
| --- | --- | --- |
| Task planning | [`vlm-tamp/`](vlm-tamp/) | VLM grounding of PDDL predicates, Fast Downward planning, VAL plan validation, active perception, and the real-robot and simulation evaluation entry points |
| Perception and navigation | [`stretch_ai/`](stretch_ai/) | Semantic voxel mapping, open-vocabulary object detection, motion planning, and the ROS 2 bridge to the Segway |
| Manipulation | [`DKPrompt.py`](DKPrompt.py), [`VLMViewGuide.py`](VLMViewGuide.py) | Marker-based grasping on the UR5e with VLM-guided view selection |

[`generate_scene_graph.py`](generate_scene_graph.py) exports a scene graph JSON from any saved voxel map and is shared by all three layers.

---

## Quick start: run on the real robot

### 1. Start the robot bridge

This connects the ROS 2 system to the Segway robot via the ROS bridge. Run it on the robot computer and leave it running.

```bash
./stretch_ai/scripts/run_segway_bridge.sh
```

### 2. Run a task

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

Most of VAP-TAMP runs without a robot once you have a `.pkl` voxel map.

**Plan with the VLM over the map**, visualize detected instances and the voxel grid, and optionally send the plan to the robot:

```bash
cd stretch_ai
python -m stretch.app.vlm_planning -i <path/to/map.pkl> --show-instances --show-svm -f 20 -fs 3
```

No robot IP is needed in this mode. Add `--robot_ip <ROBOT_IP>` to execute the plan on the robot. The full option list and the three run modes are documented in the [VLM planning section of the app reference](stretch_ai/docs/apps.md#vlm-planning).

**Export a scene graph** from the same map:

```bash
python3 generate_scene_graph.py -i <path/to/map.pkl> -o scene_graph.json
```

**Annotate room and location labels** for navigation, which the voxel map does not carry on its own. See the [location mapping guide](vlm-tamp/LOCATION_MAPPING.md).

---

## Documentation

| Document | Covers |
| --- | --- |
| [Planning layer setup](vlm-tamp/README.md) | Installing dependencies, building Fast Downward and VAL, simulation mode |
| [Architecture](vlm-tamp/ARCHITECTURE.md) | System design, coordinate frames, data flow between layers |
| [Experiments](vlm-tamp/EXPERIMENTS.md) | Running the three experimental tasks on the robot |
| [Task specifications](vlm-tamp/TASK_SPECIFICATIONS.md) | Task definitions and their PDDL encodings |
| [Location mapping](vlm-tamp/LOCATION_MAPPING.md) | Annotating semantic locations on a map |
| [Perception and navigation setup](stretch_ai/README.md) | Hardware requirements, workstation install, Docker |
| [App reference](stretch_ai/docs/apps.md) | Every command-line app, including mapping, `read_map`, and [VLM planning](stretch_ai/docs/apps.md#vlm-planning) |

---

## Notes

- The robot bridge must be running before `eval_real_robot.py` is started.
- `--api-key` takes a Gemini API key. Copy [`.env.example`](.env.example) to `.env` if you prefer to keep keys out of the shell history.
- The first run of any perception app downloads the Detic and CLIP checkpoints and takes a few minutes.
- Saved maps and instance crops used in the experiments live under [`maps/`](maps/) and [`instances/`](instances/).
