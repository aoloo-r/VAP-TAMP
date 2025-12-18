#!/usr/bin/env python3
"""
Run all three experimental tasks for VLM-TAMP real robot evaluation.

Tasks:
1. Bottle Collection - Collect 2 bottles from different rooms
2. Egg Halving - Use knife to cut hard-boiled egg
3. Firewood Storage - Collect sticks, handle closed door

Usage:
    python run_experiments.py --robot-ip 172.20.10.3 --map-file /path/to/map.pkl --api-key YOUR_KEY
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from active_perception import ActivePerceptionModule
from gemini_api import GeminiAPIAgent
from eval_real_robot import RealRobotPDDLExecutor


class ExperimentRunner:
    """Run multiple experimental tasks and collect results."""

    def __init__(
        self,
        robot_ip: str,
        map_file: str,
        api_key: str,
        config_file: str = "rosbridge_robot_config.yaml",
        calibration_file: str = "simple_offset_calibration.yaml",
        output_dir: str = "experiment_results"
    ):
        """
        Initialize experiment runner.

        Args:
            robot_ip: Robot IP address
            map_file: Path to voxel map
            api_key: Gemini API key
            config_file: Robot configuration
            calibration_file: Map calibration file
            output_dir: Directory to save results
        """
        self.robot_ip = robot_ip
        self.map_file = map_file
        self.api_key = api_key
        self.config_file = config_file
        self.calibration_file = calibration_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize modules (reused across experiments)
        print("🔧 Initializing modules...")
        self.active_perception = ActivePerceptionModule(
            robot_ip=robot_ip,
            map_file=map_file,
            config_file=config_file,
            calibration_file=calibration_file,
            max_exploration_attempts=3,
            viewpoint_distance=1.0,
            use_ur5e=True
        )

        self.vlm = GeminiAPIAgent(api_key=api_key)

        # Experiment definitions
        self.experiments = [
            {
                "name": "Task 1: Bottle Collection",
                "domain": "domains/bottle_collection/domain.pddl",
                "problem": "domains/bottle_collection/problem.pddl",
                "description": "Collect 2 empty bottles from different rooms"
            },
            {
                "name": "Task 2: Egg Halving",
                "domain": "domains/egg_halving/domain.pddl",
                "problem": "domains/egg_halving/problem.pddl",
                "description": "Use knife to halve a hard-boiled egg"
            },
            {
                "name": "Task 3: Firewood Storage",
                "domain": "domains/firewood_storage/domain.pddl",
                "problem": "domains/firewood_storage/problem.pddl",
                "description": "Collect wooden sticks, handle closed door"
            }
        ]

        self.results = []

    def run_experiment(self, experiment: dict) -> dict:
        """
        Run a single experiment.

        Args:
            experiment: Experiment configuration dict

        Returns:
            Results dictionary
        """
        print(f"\n{'='*80}")
        print(f"STARTING: {experiment['name']}")
        print(f"{'='*80}")
        print(f"Description: {experiment['description']}")
        print(f"Domain: {experiment['domain']}")
        print(f"Problem: {experiment['problem']}")
        print(f"{'='*80}\n")

        # Create executor
        executor = RealRobotPDDLExecutor(
            active_perception=self.active_perception,
            vlm_agent=self.vlm,
            domain_file=experiment['domain'],
            problem_file=experiment['problem'],
            verbose=True
        )

        # Record start time
        start_time = time.time()

        # Run task
        stats = executor.run()

        # Record end time
        end_time = time.time()
        duration = end_time - start_time

        # Compile results
        result = {
            "experiment": experiment['name'],
            "description": experiment['description'],
            "domain": experiment['domain'],
            "problem": experiment['problem'],
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "statistics": stats
        }

        return result

    def run_all(self, start_from: int = 0) -> list:
        """
        Run all experiments.

        Args:
            start_from: Index of experiment to start from (for resuming)

        Returns:
            List of result dictionaries
        """
        print(f"\n{'#'*80}")
        print(f"# VLM-TAMP REAL ROBOT EXPERIMENTS")
        print(f"#")
        print(f"# Total experiments: {len(self.experiments)}")
        print(f"# Robot IP: {self.robot_ip}")
        print(f"# Map file: {self.map_file}")
        print(f"{'#'*80}\n")

        for i in range(start_from, len(self.experiments)):
            experiment = self.experiments[i]

            try:
                result = self.run_experiment(experiment)
                self.results.append(result)

                # Save intermediate results
                self._save_results()

                print(f"\n✅ Experiment {i+1}/{len(self.experiments)} completed")

                # Wait between experiments
                if i < len(self.experiments) - 1:
                    print(f"\n⏸️  Waiting 5 seconds before next experiment...")
                    time.sleep(5)

            except KeyboardInterrupt:
                print(f"\n\n⚠️  Interrupted by user at experiment {i+1}")
                self._save_results()
                print(f"Results saved. Resume with --start-from {i}")
                break

            except Exception as e:
                print(f"\n❌ Experiment {i+1} failed with exception: {e}")
                import traceback
                traceback.print_exc()

                # Record failure
                result = {
                    "experiment": experiment['name'],
                    "description": experiment['description'],
                    "domain": experiment['domain'],
                    "problem": experiment['problem'],
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                self.results.append(result)
                self._save_results()

                # Ask if should continue
                response = input("\nContinue to next experiment? (y/n): ").strip().lower()
                if response != 'y':
                    break

        # Final summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"experiment_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "robot_ip": self.robot_ip,
                "map_file": self.map_file,
                "timestamp": timestamp,
                "experiments": self.results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    def _print_summary(self):
        """Print summary of all experiments."""
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT SUMMARY")
        print(f"{'#'*80}\n")

        for i, result in enumerate(self.results):
            print(f"{i+1}. {result['experiment']}")

            if 'error' in result:
                print(f"   Status: ❌ FAILED")
                print(f"   Error: {result['error']}")
            elif 'statistics' in result:
                stats = result['statistics']
                success_rate = (stats['successful_actions'] / stats['total_actions'] * 100
                               if stats['total_actions'] > 0 else 0)
                print(f"   Status: ✅ COMPLETED")
                print(f"   Duration: {result.get('duration_seconds', 0):.1f}s")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Actions: {stats['successful_actions']}/{stats['total_actions']}")
                print(f"   Retries: {stats.get('retried_actions', 0)}")

            print()


def main():
    parser = argparse.ArgumentParser(description="Run VLM-TAMP Real Robot Experiments")
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--map-file", required=True, help="Path to voxel map (.pkl)")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--config", default="rosbridge_robot_config.yaml", help="Robot config")
    parser.add_argument("--calibration", default="simple_offset_calibration.yaml", help="Calibration file")
    parser.add_argument("--output-dir", default="experiment_results", help="Output directory")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from experiment index")

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(
        robot_ip=args.robot_ip,
        map_file=args.map_file,
        api_key=args.api_key,
        config_file=args.config,
        calibration_file=args.calibration,
        output_dir=args.output_dir
    )

    # Run all experiments
    runner.run_all(start_from=args.start_from)

    print("\n✅ All experiments complete!")


if __name__ == "__main__":
    main()
