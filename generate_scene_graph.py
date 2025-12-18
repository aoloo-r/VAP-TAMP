#!/usr/bin/env python3
"""
Generate Scene Graph from Voxel Map and Save as JSON

Usage:
    python3 generate_scene_graph.py -i map4.pkl -o scene_graph.json

Author: DKPrompt Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Add stretch_ai to path
sys.path.insert(0, str(Path(__file__).parent / "stretch_ai" / "src"))

from stretch.agent.robot_agent import RobotAgent
from stretch.mapping.scene_graph import SceneGraph
from stretch.mapping.voxel import SparseVoxelMap
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


def load_voxel_map_with_instances(pkl_path: str, config_path: str = None, num_frames: int = -1) -> SparseVoxelMap:
    """
    Load voxel map from pkl file using the proper stretch_ai approach.
    This follows the exact pattern from calibrate_simple_offset.py.
    """
    print(f"[INFO] Loading voxel map from: {pkl_path}")

    # Load parameters
    if not config_path:
        config_path = "rosbridge_robot_config.yaml"

    print(f"[INFO] Loading configuration from {config_path}...")
    parameters = get_parameters(config_path)

    # Merge with base config if specified
    if parameters.get("vlm_base_config"):
        base_config_file = parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(parameters.data)
        parameters.data = base_parameters.data

    # Create semantic sensor for instance segmentation
    print("[INFO] Initializing semantic sensor...")
    semantic_sensor = create_semantic_sensor(parameters=parameters)

    # Create agent with dummy robot
    print("[INFO] Creating robot agent...")
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(dummy_robot, parameters, semantic_sensor=semantic_sensor)
    voxel_map = agent.get_voxel_map()

    # Set matplotlib to non-GUI backend to avoid display issues
    import matplotlib
    matplotlib.use('Agg')

    # Load from pickle using the proper method that handles Record3D data
    print(f"[INFO] Loading 3D map from {pkl_path}...")
    print("[INFO] This may take a few minutes depending on the number of frames...")
    voxel_map.read_from_pickle(str(Path(pkl_path)), num_frames=num_frames, perception=semantic_sensor)

    num_instances = len(voxel_map.get_instances())
    print(f"[SUCCESS] Loaded map with {num_instances} instances")
    return voxel_map, semantic_sensor


def extract_scene_graph(voxel_map: SparseVoxelMap,
                       config_path: str = None) -> SceneGraph:
    """Extract scene graph from voxel map."""
    print("[INFO] Extracting scene graph...")

    # Load parameters
    if config_path:
        parameters = get_parameters(config_path)
    else:
        parameters = get_parameters("default_planner.yaml")

    # Get instances
    instances = voxel_map.get_instances()

    if not instances:
        print("[WARNING] No instances found in voxel map!")
        return None

    # Create scene graph
    scene_graph = SceneGraph(parameters, instances)
    relationships = scene_graph.get_relationships(debug=False)

    print(f"[INFO] Found {len(relationships)} spatial relationships")

    return scene_graph


def scene_graph_to_json(scene_graph: SceneGraph,
                        voxel_map: SparseVoxelMap,
                        semantic_sensor=None) -> Dict:
    """
    Convert scene graph to structured JSON format.

    Returns:
        {
            "instances": [...],
            "relationships": [...],
            "statistics": {...}
        }
    """
    instances_data = []
    instances = voxel_map.get_instances()

    # Extract instance information
    print("[INFO] Extracting instance metadata...")
    for instance in instances:
        try:
            # Get center position
            if hasattr(instance, 'point_cloud') and instance.point_cloud is not None:
                center = torch.mean(instance.point_cloud, axis=0).cpu().numpy()
            else:
                center = instance.get_center()

            # Get bounding box if available
            bbox = None
            if hasattr(instance, 'bounds'):
                bbox = {
                    "min": instance.bounds[0].tolist() if hasattr(instance.bounds[0], 'tolist') else list(instance.bounds[0]),
                    "max": instance.bounds[1].tolist() if hasattr(instance.bounds[1], 'tolist') else list(instance.bounds[1])
                }

            # Get category
            category = "unknown"
            if hasattr(instance, 'category_id'):
                if semantic_sensor:
                    try:
                        category = semantic_sensor.get_class_name_for_id(instance.category_id)
                    except:
                        category = f"category_{instance.category_id}"
                else:
                    category = f"category_{instance.category_id}"

            instance_data = {
                "instance_id": int(instance.global_id),
                "category": category,
                "position": {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2])
                },
                "bounding_box": bbox,
                "point_count": len(instance.point_cloud) if hasattr(instance, 'point_cloud') and instance.point_cloud is not None else 0
            }

            instances_data.append(instance_data)

        except Exception as e:
            print(f"[WARNING] Error processing instance {instance.global_id}: {e}")
            continue

    # Extract relationships
    print("[INFO] Formatting relationships...")
    relationships_data = []
    for rel in scene_graph.relationships:
        instance_a_id, instance_b_id, relationship_type = rel

        # Get instance names/categories
        instance_a_name = f"instance_{instance_a_id}"
        instance_b_name = f"instance_{instance_b_id}" if instance_b_id != "floor" else "floor"

        # Find instance metadata
        instance_a_meta = next((inst for inst in instances_data if inst["instance_id"] == instance_a_id), None)
        instance_b_meta = next((inst for inst in instances_data if inst["instance_id"] == instance_b_id), None) if instance_b_id != "floor" else None

        relationship_data = {
            "subject": {
                "id": int(instance_a_id),
                "name": instance_a_name,
                "category": instance_a_meta["category"] if instance_a_meta else "unknown",
                "position": instance_a_meta["position"] if instance_a_meta else None
            },
            "predicate": relationship_type,
            "object": {
                "id": instance_b_id if instance_b_id != "floor" else "floor",
                "name": instance_b_name,
                "category": instance_b_meta["category"] if instance_b_meta else "floor",
                "position": instance_b_meta["position"] if instance_b_meta else {"x": 0, "y": 0, "z": 0}
            },
            "text": f"{instance_a_name} is {relationship_type} {instance_b_name}"
        }

        relationships_data.append(relationship_data)

    # Compute statistics
    relationship_types = {}
    for rel in relationships_data:
        rel_type = rel["predicate"]
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

    statistics = {
        "total_instances": len(instances_data),
        "total_relationships": len(relationships_data),
        "relationship_distribution": relationship_types,
        "average_relationships_per_instance": len(relationships_data) / len(instances_data) if instances_data else 0
    }

    # Build final JSON structure
    scene_graph_json = {
        "metadata": {
            "description": "Scene graph extracted from 3D semantic voxel map",
            "format_version": "1.0",
            "coordinate_frame": "voxel_map"
        },
        "instances": instances_data,
        "relationships": relationships_data,
        "statistics": statistics
    }

    return scene_graph_json


def generate_vlm_format(scene_graph: SceneGraph, voxel_map: SparseVoxelMap) -> str:
    """
    Generate VLM-ready scene description text.

    Example: "img_0 is on img_3; img_1 is near img_0; img_2 is on floor;"
    """
    instances = voxel_map.get_instances()
    instance_to_name = {inst.global_id: f"img_{idx}" for idx, inst in enumerate(instances)}

    vlm_text = "Scene descriptions: "
    for rel in scene_graph.relationships:
        instance_a, instance_b, relationship = rel
        name_a = instance_to_name.get(instance_a, f"instance_{instance_a}")
        name_b = instance_to_name.get(instance_b, "floor") if instance_b != "floor" else "floor"
        vlm_text += f"{name_a} is {relationship} {name_b}; "

    return vlm_text


def save_scene_graph_json(scene_graph_json: Dict, output_path: str):
    """Save scene graph to JSON file."""
    print(f"[INFO] Saving scene graph to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(scene_graph_json, f, indent=2)

    print(f"[SUCCESS] Scene graph saved to {output_path}")


def visualize_scene_graph(scene_graph_json: Dict):
    """Print a human-readable summary of the scene graph."""
    print("\n" + "="*70)
    print("SCENE GRAPH SUMMARY")
    print("="*70)

    # Print statistics
    stats = scene_graph_json["statistics"]
    print(f"\n📊 Statistics:")
    print(f"   Total Instances: {stats['total_instances']}")
    print(f"   Total Relationships: {stats['total_relationships']}")
    print(f"   Avg Relationships per Instance: {stats['average_relationships_per_instance']:.2f}")

    print(f"\n📈 Relationship Distribution:")
    for rel_type, count in stats['relationship_distribution'].items():
        print(f"   {rel_type}: {count}")

    # Print instances
    print(f"\n🔷 Instances:")
    for inst in scene_graph_json["instances"][:10]:  # Show first 10
        pos = inst["position"]
        print(f"   Instance {inst['instance_id']}: {inst['category']} at ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")
    if len(scene_graph_json["instances"]) > 10:
        print(f"   ... and {len(scene_graph_json['instances']) - 10} more")

    # Print relationships
    print(f"\n🔗 Relationships:")
    for rel in scene_graph_json["relationships"][:15]:  # Show first 15
        print(f"   {rel['text']}")
    if len(scene_graph_json["relationships"]) > 15:
        print(f"   ... and {len(scene_graph_json['relationships']) - 15} more")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene graph from voxel map and save as JSON"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input voxel map pkl file (e.g., map4.pkl)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output JSON file (default: <input_name>_scene_graph.json)"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config file (default: uses vlm_planning config)"
    )
    parser.add_argument(
        "-f", "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to process from pkl file (-1 for all frames)"
    )
    parser.add_argument(
        "--vlm-format",
        action="store_true",
        help="Also save VLM-ready text format"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print scene graph summary to console"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_scene_graph.json")

    try:
        # Load voxel map with instance segmentation
        voxel_map, semantic_sensor = load_voxel_map_with_instances(
            args.input,
            args.config,
            num_frames=args.num_frames
        )

        # Extract scene graph
        scene_graph = extract_scene_graph(voxel_map, args.config)

        if scene_graph is None:
            print("[ERROR] Failed to extract scene graph (no instances)")
            return 1

        # Convert to JSON
        scene_graph_json = scene_graph_to_json(scene_graph, voxel_map, semantic_sensor)

        # Save JSON
        save_scene_graph_json(scene_graph_json, args.output)

        # Optionally save VLM format
        if args.vlm_format:
            vlm_text = generate_vlm_format(scene_graph, voxel_map)
            vlm_output_path = args.output.replace(".json", "_vlm.txt")
            with open(vlm_output_path, 'w') as f:
                f.write(vlm_text)
            print(f"[SUCCESS] VLM format saved to {vlm_output_path}")

        # Optionally visualize
        if args.visualize:
            visualize_scene_graph(scene_graph_json)

        print("\n✅ Scene graph generation complete!")
        return 0

    except Exception as e:
        print(f"\n❌ [ERROR] Failed to generate scene graph: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
