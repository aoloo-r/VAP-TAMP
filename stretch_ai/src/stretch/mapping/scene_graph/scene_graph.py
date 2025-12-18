# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from stretch.core.parameters import Parameters
from stretch.mapping.instance import Instance
from stretch.utils.memory import get_path_to_debug


class SceneGraph:
    """Compute a very simple scene graph. Use it to extract relationships between instances."""

    def __init__(self, parameters: Parameters, instances: List[Instance]):
        self.parameters = parameters
        self.instances = instances
        self.relationships: List[Tuple[int, int, str]] = []
        self.update(instances)

    def update(self, instances):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        self.relationships: List[Tuple[int, int, str]] = []
        self.instances = instances
        
        if not instances:
            return
            
        for idx_a, ins_a in enumerate(instances):
            for idx_b, ins_b in enumerate(instances):
                if idx_a == idx_b:
                    continue
                    
                # Skip invalid instances
                if not hasattr(ins_a, 'global_id') or not hasattr(ins_b, 'global_id'):
                    continue
                    
                # Skip if we've already processed this pair
                pair_key = (min(ins_a.global_id, ins_b.global_id), max(ins_a.global_id, ins_b.global_id))
                if hasattr(self, '_processed_pairs'):
                    if pair_key in self._processed_pairs:
                        continue
                else:
                    self._processed_pairs = set()
                self._processed_pairs.add(pair_key)
                
                try:
                    # Determine the SINGLE most important relationship between these objects
                    primary_relationship = None
                    
                    # RESTRICTED RELATIONSHIP SET: only "on", "near"
                    # Note: "on_the_floor" is handled separately below
                    
                    # 1. Physical support relationships ("on") - highest priority
                    if self.on(ins_a.global_id, ins_b.global_id):
                        primary_relationship = (ins_a.global_id, ins_b.global_id, "on")
                    elif self.on(ins_b.global_id, ins_a.global_id):
                        primary_relationship = (ins_b.global_id, ins_a.global_id, "on")
                        
                    # 2. General proximity ("near") - only if no "on" relationship exists  
                    elif self.near(ins_a.global_id, ins_b.global_id):
                        id1, id2 = (ins_a.global_id, ins_b.global_id) if ins_a.global_id < ins_b.global_id else (ins_b.global_id, ins_a.global_id)
                        primary_relationship = (id1, id2, "near")
                    
                    # Add the primary relationship if found and not already exists
                    if primary_relationship and primary_relationship not in self.relationships:
                        self.relationships.append(primary_relationship)

                except Exception as e:
                    print(f"Warning: Error processing relationship between {ins_a.global_id} and {ins_b.global_id}: {e}")
                    continue
                    
            # Add "on floor" relationship for if something is on the floor
            try:
                if self.on_floor(ins_a.global_id):
                    self.relationships.append((ins_a.global_id, "floor", "on"))
            except Exception as e:
                print(f"Warning: Error processing floor relationship for {ins_a.global_id}: {e}")

    def get_matching_relations(
        self,
        id0: Optional[Union[int, str]],
        id1: Optional[Union[int, str]],
        relation: Optional[str],
    ) -> List[Tuple[int, int, str]]:
        """Get all relationships between two instances.

        Args:
            id0: The first instance id
            id1: The second instance id
            relation: The relationship between the two instances

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        if isinstance(id1, Instance):
            id1 = id1.global_id
        if isinstance(id0, Instance):
            id0 = id0.global_id
        return [
            rel
            for rel in self.relationships
            if (id0 is None or rel[0] == id0)
            and (id1 is None or rel[1] == id1)
            and (rel[2] == relation or relation is None)
        ]

    def get_ins_center_pos(self, global_id: int):
        """Get the center of an instance based on point cloud"""
        # Find instance by global_id
        for instance in self.instances:
            if instance.global_id == global_id:
                if hasattr(instance, 'point_cloud') and instance.point_cloud is not None:
                    return torch.mean(instance.point_cloud, axis=0)
                else:
                    # Fallback to instance center if point cloud not available
                    return torch.tensor(instance.get_center())
        raise ValueError(f"Instance with global_id {global_id} not found")

    def get_instance_image(self, global_id: int) -> np.ndarray:
        """Get a viewable image from tensorized instances"""
        # Find instance by global_id
        for instance in self.instances:
            if instance.global_id == global_id:
                return (
                    (
                        instance.get_best_view().cropped_image
                        * instance.get_best_view().mask
                        / 255.0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        raise ValueError(f"Instance with global_id {global_id} not found")

    def get_relationships(self, debug: bool = False) -> List[Tuple[int, int, str]]:
        """Return the relationships between instances.

        Args:
            debug: If True, show the relationships in a matplotlib window

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        # show symbolic relationships
        if debug:
            for global_id_a, global_id_b, rel in self.relationships:
                print(global_id_a, global_id_b, rel)

                try:
                    if global_id_b == "floor":
                        img_a = self.get_instance_image(global_id_a)
                        img_b = np.zeros_like(img_a)
                    else:
                        img_a = self.get_instance_image(global_id_a)
                        img_b = self.get_instance_image(global_id_b)
                except Exception as e:
                    print(f"Warning: Could not get images for relationship {global_id_a} {rel} {global_id_b}: {e}")
                    continue

                import matplotlib

                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt

                plt.subplot(1, 2, 1)
                plt.imshow(img_a)
                plt.title("Instance A is " + rel)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(img_b)
                plt.title("Instance B")
                plt.axis("off")
                # plt.show()
                plt.savefig(get_path_to_debug(f"scene_graph_{global_id_a}_{global_id_b}_{rel}.png"))

        # Return the detected relationships in list form
        return self.relationships

    def near(self, ins_a, ins_b):
        dist = torch.pairwise_distance(
            self.get_ins_center_pos(ins_a), self.get_ins_center_pos(ins_b)
        ).item()
        if dist < self.parameters["scene_graph"]["max_near_distance"]:
            return True
        return False

    def on(self, ins_a, ins_b):
        """On is defined as near and above, within some tolerance"""
        if self.near(ins_a, ins_b):
            z_dist = self.get_ins_center_pos(ins_a)[2] - self.get_ins_center_pos(ins_b)[2]
            if (
                z_dist < self.parameters["scene_graph"]["max_on_height"]
                and z_dist > self.parameters["scene_graph"]["min_on_height"]
            ):
                return True
        return False

    def on_floor(self, ins_a):
        """Check if an instance is on the floor"""
        pos = self.get_ins_center_pos(ins_a)
        if pos[2] < self.parameters["scene_graph"]["max_on_height"] and pos[2] > 0:
            return True
        return False

    def under(self, ins_a, ins_b):
        """Check if instance A is underneath instance B"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        # Check horizontal distance
        horiz_dist = torch.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2).item()
        if horiz_dist > self.parameters["scene_graph"]["max_under_distance"]:
            return False
            
        # Check if A is below B by enough height
        z_diff = pos_b[2] - pos_a[2]
        return z_diff >= self.parameters["scene_graph"]["min_under_height"]

    def beside(self, ins_a, ins_b):
        """Check if instances are beside each other (similar height, close horizontally)"""
        if self.near(ins_a, ins_b):  # Must be near first
            pos_a = self.get_ins_center_pos(ins_a)
            pos_b = self.get_ins_center_pos(ins_b)
            
            # Check horizontal distance
            horiz_dist = torch.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2).item()
            if horiz_dist > self.parameters["scene_graph"]["max_beside_distance"]:
                return False
                
            # Check height difference
            height_diff = abs(pos_a[2] - pos_b[2]).item()
            return height_diff <= self.parameters["scene_graph"]["max_beside_height_diff"]
        return False

    def above(self, ins_a, ins_b):
        """Check if instance A is above instance B (more flexible than 'on')"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        # Check horizontal distance
        horiz_dist = torch.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2).item()
        if horiz_dist > self.parameters["scene_graph"]["max_above_distance"]:
            return False
            
        # Check if A is above B by enough height
        z_diff = pos_a[2] - pos_b[2]
        return z_diff >= self.parameters["scene_graph"]["min_above_height"]

    def inside(self, ins_a, ins_b):
        """Check if instance A is inside instance B (containment)"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        # Simple containment check - A is very close to B and might be contained
        dist = torch.pairwise_distance(pos_a, pos_b).item()
        return dist < self.parameters["scene_graph"]["max_inside_distance"]

    def next_to(self, ins_a, ins_b):
        """Check if instances are next to each other (more specific than near)"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        dist = torch.pairwise_distance(pos_a, pos_b).item()
        return dist < self.parameters["scene_graph"]["max_next_to_distance"]

    def behind(self, ins_a, ins_b):
        """Check if instance A is behind instance B (based on Y coordinate)"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        # Check if they're within reasonable distance
        dist = torch.pairwise_distance(pos_a, pos_b).item()
        if dist < self.parameters["scene_graph"]["min_behind_distance"] or dist > self.parameters["scene_graph"]["max_behind_distance"]:
            return False
            
        # Simple behind check - A has a larger Y coordinate than B (further back)
        # This assumes the robot typically faces toward negative Y direction
        return pos_a[1] > pos_b[1]

    def against(self, ins_a, wall_or_furniture_id):
        """Check if instance A is against a wall or large furniture"""
        # This would require more sophisticated geometry analysis
        # For now, we'll implement a simple version
        pos_a = self.get_ins_center_pos(ins_a)
        
        # Check if object is very close to a wall (simplified)
        # In a more complete implementation, this would check against actual wall geometry
        # For now, we'll use proximity to large static objects
        try:
            pos_wall = self.get_ins_center_pos(wall_or_furniture_id)
            dist = torch.pairwise_distance(pos_a, pos_wall).item()
            return dist < self.parameters["scene_graph"]["max_against_distance"]
        except:
            return False

    def is_large_furniture(self, instance):
        """Check if an instance is large furniture that things can be against"""
        # Define large furniture types that objects can be "against"
        large_furniture_types = {
            'cabinet', 'wardrobe', 'chest_of_drawers', 'bookshelf', 'book_shelf',
            'dresser', 'wall', 'door', 'desk', 'table', 'counter', 'sink', 
            'stove', 'refrigerator', 'washer_dryer', 'couch', 'sofa'
        }
        
        # Get the category name if available
        if hasattr(instance, 'category_id'):
            # This would need access to the semantic sensor to convert ID to name
            # For now, we'll use a simple heuristic based on the instance size or ID
            return True  # Simplified - assume larger objects can have things against them
        return False

    def get_semantic_relationship(self, ins_a, ins_b):
        """Get semantic relationships based on object types"""
        # Define semantic relationship rules based on object combinations
        semantic_rules = {
            # Kitchen relationships
            ('cup', 'table'): 'served_on',
            ('plate', 'table'): 'served_on', 
            ('bowl', 'counter'): 'placed_on',
            ('knife', 'cutting_board'): 'used_with',
            ('spoon', 'bowl'): 'used_with',
            
            # Office relationships
            ('laptop', 'desk'): 'placed_on',
            ('mouse_pad', 'desk'): 'placed_on',
            ('book', 'shelf'): 'stored_on',
            ('pen', 'desk'): 'placed_on',
            
            # Living room relationships  
            ('cushion', 'couch'): 'placed_on',
            ('remote', 'table'): 'placed_on',
            ('lamp', 'table'): 'positioned_on',
            
            # Storage relationships
            ('box', 'shelf'): 'stored_on',
            ('bag', 'chair'): 'resting_on',
            
            # Bathroom relationships
            ('towel', 'rack'): 'hanging_on',
            ('soap', 'sink'): 'near',
        }
        
        # This is a simplified version - in practice you'd need access to 
        # the semantic sensor to get category names
        # For now, we'll return None and let spatial relationships dominate
        return None

    def get_contextual_relationship(self, ins_a, ins_b):
        """Get contextual relationships that consider the broader scene"""
        pos_a = self.get_ins_center_pos(ins_a)
        pos_b = self.get_ins_center_pos(ins_b)
        
        # Example: if one object is much smaller and positioned precisely on another
        # it might be "displayed_on" rather than just "on"
        if self.on(ins_a, ins_b):
            # Check if it's a small object on a larger surface
            dist_threshold = self.parameters["scene_graph"]["max_on_height"]
            z_diff = pos_a[2] - pos_b[2]
            
            if 0.02 < z_diff < dist_threshold:  # Very precisely placed
                return "displayed_on"
                
        return None
