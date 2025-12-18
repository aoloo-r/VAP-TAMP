#!/usr/bin/env python3
"""Test active perception on individual PDDL predicates"""
from active_perception import ActivePerceptionModule
from gemini_api import GeminiAPIAgent
import sys

# Initialize
active_perception = ActivePerceptionModule(
    robot_ip="172.20.10.3",
    map_file="/home/aoloo/code/stretch_ai/scripts/visual_grounding_benchmark/sample9.pkl",
    config_file="rosbridge_robot_config.yaml",
    use_ur5e=True  # Use UR5e + Segway robot
)

vlm = GeminiAPIAgent(api_key="AIzaSyAUW49iYmd6T_ayI64393QD8s1jB-MSVts")

# Test predicates
test_cases = [
    {
        "predicate": ["ontop", "bottle", "table"],
        "question": "Is there a bottle on top of the table in this view?"
    },
    {
        "predicate": ["inside", "cup", "cabinet"],
        "question": "Is there a cup inside the cabinet in this view?"
    },
    {
        "predicate": ["closed", "door"],
        "question": "Is the door closed in this view?"
    }
]

for i, test in enumerate(test_cases):
    print(f"\n{'='*70}")
    print(f"Test {i+1}: {test['predicate']}")
    print(f"{'='*70}")

    answer, rgb, success = active_perception.explore_for_better_view(
        predicate=test["predicate"],
        vlm_agent=vlm,
        question=test["question"]
    )

    print(f"\nResult: {answer}")
    print(f"Success: {success}")