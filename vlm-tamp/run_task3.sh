#!/bin/bash
# Task 3: Firewood Storage Execution

python eval_real_robot.py \
  --robot-ip 172.20.10.4 \
  --map-file /home/aoloo/code/stretch_ai/scripts/visual_grounding_benchmark/map4.pkl \
  --domain domains/firewood_storage/domain.pddl \
  --problem domains/firewood_storage/problem.pddl \
  --api-key "AIzaSyAUW49iYmd6T_ayI64393QD8s1jB-MSVts" \
  --location-map location_mapping.yaml \
  --config rosbridge_robot_config.yaml \
  --calibration simple_offset_calibration.yaml \
  --output firewood_storage_results.json \
  --verbose
