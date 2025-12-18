# DKPrompt: VLM-TAMP for Real-World Robotics

An integrated system for Vision-Language Model based Task and Motion Planning (VLM-TAMP) with real robot execution capabilities. This project combines VLM-TAMP planning with Stretch AI framework and supports both simulation (OmniGibson) and real robot deployment (Segway + UR5e).

## Features

- **DKPrompt VLM-TAMP**: Enhanced VLM-TAMP framework with active perception
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
git clone https://github.com/aoloo-r/DKPrompt.git
cd DKPrompt
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
cd stretch_ai
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
├── src/                  # Core VLM-TAMP implementation
├── stretch_ai/          # Robot control and perception
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
@article{dkprompt2024,
  title={DKPrompt: VLM-TAMP for Real-World Robotics},
  author={Your Name and Collaborators},
  year={2024}
}
```

## Acknowledgments

This project builds upon:
- VLM-TAMP framework
- Stretch AI by Hello Robot
- OmniGibson simulator
- Fast Downward planner
- VAL plan validator

## License

This project is a collaborative work combining multiple frameworks. See individual component licenses in their respective directories.
