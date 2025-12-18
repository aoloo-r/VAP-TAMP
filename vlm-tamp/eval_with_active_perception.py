#!/usr/bin/env python3
"""
DKPrompt eval.py with Stretch AI Active Perception

This is a minimal wrapper around the original eval.py that adds active perception
when VLM is uncertain. It reuses all simulation logic and adds real-robot experiments.

Usage:
    python eval_with_active_perception.py \
        --robot-ip 172.20.10.3 \
        --map-file ../stretch_ai/maps/room_map.pkl \
        --domain domains/bringing_water/domain.pddl \
        --problem domains/bringing_water/problem.pddl \
        --api-key YOUR_GEMINI_KEY
"""

import argparse
from active_perception import ActivePerceptionModule

# For now, just test if active_perception module loads correctly
# Full integration would modify eval.py's check_states_and_update_problem

def main():
    parser = argparse.ArgumentParser(description="DKPrompt with Active Perception")
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--map-file", required=True, help="Path to voxel map (.pkl)")
    parser.add_argument("--domain", required=True, help="PDDL domain file")
    parser.add_argument("--problem", required=True, help="PDDL problem file")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--config", default="rosbridge_robot_config.yaml", help="Robot config")
    parser.add_argument("--calibration", default="simple_offset_calibration.yaml", help="Calibration file")

    args = parser.parse_args()

    print("="*70)
    print("DKPrompt with Stretch AI Active Perception")
    print("="*70)
    print(f"Robot IP: {args.robot_ip}")
    print(f"Map: {args.map_file}")
    print(f"Domain: {args.domain}")
    print(f"Problem: {args.problem}")
    print("="*70)

    # Initialize active perception module
    print("\n🔧 Initializing Active Perception Module...")
    try:
        active_perception = ActivePerceptionModule(
            robot_ip=args.robot_ip,
            map_file=args.map_file,
            config_file=args.config,
            calibration_file=args.calibration,
            max_exploration_attempts=3,
            viewpoint_distance=1.0
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ Active Perception Module initialized successfully!")
    print("Ready for active perception experiments!")


if __name__ == "__main__":
    main()
