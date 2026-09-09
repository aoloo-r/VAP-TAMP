# VAP-TAMP: Planning Layer

This directory holds the task-planning layer of **VAP-TAMP**, a Vision-language Active Perception system for Task and Motion Planning. It grounds PDDL predicates with a vision-language model, plans with Fast Downward, validates with VAL, and drives the perception and navigation layer in [`../stretch_ai/`](../stretch_ai/) to execute on a real robot (Segway + UR5e) or in the OmniGibson simulator. See the [top-level README](../README.md) for the whole system.

## Features

- **VLM-grounded task planning**: PDDL planning with vision-language grounding of the symbolic state
- **Real Robot Support**: Integration with Segway mobile base and UR5e arm
- **Active Perception**: Autonomous viewpoint exploration when VLM is uncertain
- **Hybrid Mapping**: Combines 2D AMCL navigation with 3D semantic voxel maps
- **PDDL Planning**: Classical planning with VLM-guided grounding

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design, coordinate frames, and data flow.

## Installation

### Prerequisites
- Python 3.8+
- ROS 2 (Humble or later)
- CUDA-capable GPU (for perception models)

### 1. Clone Repository
```bash
git clone https://github.com/aoloo-r/VAP-TAMP.git
cd VAP-TAMP/vlm-tamp
git submodule update --init --recursive
```

### 2. Install OmniGibson (for simulation)
This project uses the OmniGibson simulator and Behavior-1k benchmark. Follow their [instructions](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) to install. We suggest installing OmniGibson from source.

### 3. Install Dependencies
```bash
# Create conda environment
conda env create -f env.yml
conda activate omnigibson

# Install additional dependencies
pip install -r requirements.txt
```

### 4. Build Planning Tools
```bash
# Build Fast Downward planner
cd downward
./build.py
cd ..

# Build VAL validator
cd VAL
make
cd ..
```

## Usage

### Simulation Mode

Run evaluation in OmniGibson simulator:
```bash
python eval.py
```

With active perception:
```bash
python eval_with_active_perception.py
```

### Real Robot Mode

#### 1. Start Robot Bridge
On the robot computer:
```bash
cd ../stretch_ai
./scripts/run_segway_bridge.sh
```

#### 2. Run Task Execution
```bash
python eval_real_robot.py \
    --robot-ip 172.20.10.3 \
    --map-file /path/to/map.pkl \
    --domain domains/bringing_water/domain.pddl \
    --problem domains/bringing_water/problem.pddl \
    --api-key YOUR_GEMINI_KEY
```

## Project Structure

```
.
├── domains/              # PDDL domain and problem files
├── src/                  # Core planning implementation
├── ../stretch_ai/       # Perception and navigation layer (sibling directory)
├── active_perception.py # Active perception module
├── eval.py             # Simulation evaluation
├── eval_real_robot.py  # Real robot evaluation
├── downward/           # Fast Downward planner (submodule)
├── VAL/                # Plan validator (submodule)
└── ARCHITECTURE.md     # System architecture documentation
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [TASK_SPECIFICATIONS.md](TASK_SPECIFICATIONS.md) - Task definitions and specifications
- [EXPERIMENTS.md](EXPERIMENTS.md) - Experimental setup and results
- [LOCATION_MAPPING.md](LOCATION_MAPPING.md) - Semantic location mapping

## Supported Tasks

- Bringing water/beverages
- Object retrieval and delivery
- Bottle collection
- Egg halving (with manipulation)
- Multi-room navigation tasks

## Key Contributions

- **Active Perception Module**: Automatically explores when VLM confidence is low
- **Real Robot Integration**: Seamless integration with Segway and UR5e hardware
- **Coordinate Frame Calibration**: Aligns semantic 3D maps with 2D navigation maps
- **Hybrid Navigation**: Combines AMCL localization with semantic voxel mapping

## Citation

If you use this work, please cite:
```bibtex
@article{vaptamp2026,
  title={VAP-TAMP: Vision-language Active Perception for Task and Motion Planning},
  author={Your Name and Collaborators},
  year={2026}
}
```

## Acknowledgments

VAP-TAMP builds upon:
- The VLM-TAMP and DKPrompt line of work on vision-language task and motion planning
- Stretch AI by Hello Robot, from which the perception and navigation layer is derived
- OmniGibson simulator
- Fast Downward planner
- VAL plan validator

## License

This project is a collaborative work combining multiple frameworks. See individual component licenses in their respective directories.
