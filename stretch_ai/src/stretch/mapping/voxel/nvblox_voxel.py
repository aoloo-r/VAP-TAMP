#!/usr/bin/env python3
# nvblox_voxel.py - nvblox-integrated voxel implementation for Stretch AI

import numpy as np
import torch
import open3d as o3d
import rclpy
from typing import List, Optional, Tuple, Dict, Any
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# Optional nvblox_msgs import - fallback gracefully if not available
try:
    from nvblox_msgs.msg import Mesh
    NVBLOX_MSGS_AVAILABLE = True
except ImportError:
    NVBLOX_MSGS_AVAILABLE = False

from stretch.mapping.voxel.voxel import SparseVoxelMap
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.utils.logger import Logger

logger = Logger(__name__)


class NvbloxSparseVoxelMap(SparseVoxelMap):
    """
    nvblox-integrated voxel map that combines:
    - nvblox's high-quality 3D reconstruction (via ROS2)
    - Stretch AI's semantic understanding and navigation planning
    """
    
    def __init__(self, *args, ros_node=None, executor=None, **kwargs):
        # Initialize ROS2 attributes first to avoid AttributeError in __del__
        self._ros_node = ros_node  # Use provided node if available
        self._executor = executor  # Shared executor for proper spinning
        self._pc_sub = None
        self._mesh_sub = None
        self._owns_node = ros_node is None  # Track if we created the node
        
        super().__init__(*args, **kwargs)
        
        # nvblox integration settings
        self.use_nvblox_geometry = True
        self.nvblox_pointcloud_topic = "/nvblox_node/static_esdf_pointcloud"
        self.nvblox_mesh_topic = "/nvblox_node/mesh"
        
        # Cached nvblox data
        self.nvblox_pointcloud = None
        self.nvblox_mesh = None
        self.nvblox_last_update = None
        
        logger.info("NvbloxSparseVoxelMap initialized - will use nvblox geometry + Stretch AI semantics")
        logger.info(f"Using {'provided' if ros_node else 'new'} ROS2 node")
    
    def _init_ros_subscribers(self):
        """Initialize ROS2 subscribers for nvblox data."""
        logger.info("[DEBUG] _init_ros_subscribers called")
        
        if self._pc_sub is not None:
            logger.info("[DEBUG] ROS subscribers already exist, skipping")
            return
            
        if self._ros_node is None:
            logger.info("[DEBUG] No ROS node provided, creating new one")
            try:
                self._ros_node = rclpy.create_node('nvblox_voxel_bridge')
                self._owns_node = True
                logger.info("[DEBUG] Successfully created new ROS2 node")
                
                # Add to shared executor if available
                if self._executor is not None:
                    logger.info("[DEBUG] Adding new node to shared executor")
                    self._executor.add_node(self._ros_node)
                else:
                    logger.warning("[DEBUG] No shared executor provided - callbacks may not fire consistently")
                    
            except Exception as e:
                logger.error(f"[DEBUG] Failed to create nvblox ROS node: {e}")
                self.use_nvblox_geometry = False
                return
        else:
            logger.info("[DEBUG] Using provided ROS2 node")
        
        try:
            logger.info(f"[DEBUG] Creating subscription to {self.nvblox_pointcloud_topic}")
            
            # DEBUG: List available topics to check for mismatches
            try:
                import subprocess
                result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=3)
                available_topics = result.stdout.strip().split('\n')
                nvblox_topics = [t for t in available_topics if 'nvblox' in t]
                logger.info(f"[DEBUG] Available nvblox topics: {nvblox_topics}")
                if self.nvblox_pointcloud_topic not in available_topics:
                    logger.error(f"[DEBUG] TARGET TOPIC {self.nvblox_pointcloud_topic} NOT FOUND!")
                else:
                    logger.info(f"[DEBUG] TARGET TOPIC {self.nvblox_pointcloud_topic} CONFIRMED AVAILABLE")
            except Exception as e:
                logger.warning(f"[DEBUG] Could not list topics: {e}")
            
            # Subscribe to nvblox pointcloud
            self._pc_sub = self._ros_node.create_subscription(
                PointCloud2, 
                self.nvblox_pointcloud_topic,
                self._nvblox_pointcloud_callback,
                10
            )
            logger.info(f"[DEBUG] Successfully created subscription to {self.nvblox_pointcloud_topic}")
            logger.info(f"[DEBUG] Subscription object: {self._pc_sub}")
            
            # Subscribe to nvblox mesh (optional, for advanced features)
            if NVBLOX_MSGS_AVAILABLE:
                logger.info("[DEBUG] nvblox_msgs available, subscribing to mesh")
                try:
                    self._mesh_sub = self._ros_node.create_subscription(
                        Mesh,
                        self.nvblox_mesh_topic, 
                        self._nvblox_mesh_callback,
                        10
                    )
                    logger.info("[DEBUG] Successfully subscribed to nvblox mesh")
                except Exception as e:
                    logger.warning(f"[DEBUG] Could not subscribe to nvblox mesh: {e}")
            else:
                logger.info("[DEBUG] nvblox_msgs not available, skipping mesh subscription")
                    
        except Exception as e:
            logger.error(f"[DEBUG] Failed to initialize nvblox ROS subscribers: {e}")
            self.use_nvblox_geometry = False
    
    def _nvblox_pointcloud_callback(self, msg: PointCloud2):
        """Handle nvblox pointcloud updates."""
        print(f"🔥 NVBLOX CALLBACK TRIGGERED: {msg.width * msg.height} points")
        logger.info(f"NVBLOX CALLBACK: Received pointcloud with {msg.width * msg.height} points")
        try:
            # Convert ROS PointCloud2 to numpy
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) > 0:
                self.nvblox_pointcloud = np.array(points_list, dtype=np.float32)
                self.nvblox_last_update = self._ros_node.get_clock().now()
                print(f"✅ NVBLOX: Successfully stored {len(points_list)} points")
                logger.info(f"NVBLOX CALLBACK: Successfully processed {len(points_list)} points")
            else:
                logger.warning("NVBLOX CALLBACK: Received empty pointcloud")
            
        except Exception as e:
            logger.error(f"Error processing nvblox pointcloud: {e}")
    
    def _nvblox_mesh_callback(self, msg):
        """Handle nvblox mesh updates."""
        try:
            # Store mesh data for advanced visualization
            self.nvblox_mesh = msg
            logger.debug("Updated nvblox mesh data")
        except Exception as e:
            logger.warning(f"Error processing nvblox mesh: {e}")
    
    def add_obs(self, obs: Observations, **kwargs):
        """
        Hybrid approach: Use nvblox for geometry, Stretch AI for semantics.
        """
        # Initialize ROS subscribers on first use
        logger.info(f"[DEBUG] add_obs called - _ros_node exists: {self._ros_node is not None}")
        logger.info(f"[DEBUG] add_obs called - use_nvblox_geometry: {self.use_nvblox_geometry}")
        logger.info(f"[DEBUG] add_obs called - _pc_sub exists: {self._pc_sub is not None}")
        logger.info(f"[DEBUG] add_obs called - nvblox_pointcloud data: {self.nvblox_pointcloud is not None}")
        
        if self._ros_node is not None and self.use_nvblox_geometry and self._pc_sub is None:
            logger.info("[DEBUG] Initializing ROS subscribers for nvblox...")
            self._init_ros_subscribers()
            logger.info(f"[DEBUG] After init - _pc_sub created: {self._pc_sub is not None}")
            logger.info(f"[DEBUG] After init - use_nvblox_geometry: {self.use_nvblox_geometry}")
        
        # Note: No manual spinning needed - the shared executor handles this automatically
        # The MultiThreadedExecutor in the robot client is continuously spinning all nodes
        
        if self.use_nvblox_geometry and self.nvblox_pointcloud is not None:
            # Use nvblox geometry + Stretch AI semantics
            logger.info("[SUCCESS] Using nvblox geometry + Stretch AI semantics")
            self._add_obs_with_nvblox(obs, **kwargs)
        else:
            # Fallback to standard Stretch AI processing
            nvblox_reason = "not enabled" if not self.use_nvblox_geometry else "no data received"
            logger.info(f"[FALLBACK] Using standard Stretch AI geometry processing (nvblox {nvblox_reason})")
            super().add_obs(obs, **kwargs)
    
    def _add_obs_with_nvblox(self, obs: Observations, **kwargs):
        """
        Add observation using nvblox geometry with Stretch AI semantic processing.
        This version MODIFIES the map by accumulating nvblox geometry with proper pose transforms.
        """
        # 1. Process semantics with Stretch AI (but skip its geometry reconstruction)
        kwargs_no_geometry = kwargs.copy()
        kwargs_no_geometry['process_geometry'] = False
        super().add_obs(obs, **kwargs_no_geometry)
        
        # 2. Extract pose information for coordinate transformation
        camera_pose = self.fix_type(obs.camera_pose) if obs.camera_pose is not None else None
        base_pose = torch.from_numpy(np.array([obs.gps[0], obs.gps[1], obs.compass[0]])).float() if obs.gps is not None and obs.compass is not None else None
        
        # 3. NEW: Integrate the latest nvblox pointcloud into our persistent storage with proper pose
        if self.nvblox_pointcloud is not None:
            print(f"🔄 INTEGRATING {len(self.nvblox_pointcloud)} nvblox points into voxel map")
            print(f"📍 Robot pose - GPS: {obs.gps if obs.gps is not None else 'None'}, Compass: {obs.compass if obs.compass is not None else 'None'}")
            self._integrate_nvblox_pointcloud_with_pose(camera_pose, base_pose)
        else:
            print("❌ No nvblox pointcloud data available for integration")
    
    def _integrate_nvblox_pointcloud_with_pose(self, camera_pose, base_pose):
        """
        Integrate nvblox pointcloud into the voxel grid structure with proper pose transformation.
        """
        try:
            # Convert nvblox points to torch tensors
            nvblox_points_torch = torch.from_numpy(self.nvblox_pointcloud).float()
            print(f"🎯 Original nvblox points shape: {nvblox_points_torch.shape}")
            
            # Transform nvblox points from camera frame to world frame
            if camera_pose is not None:
                print(f"🔄 Applying camera pose transformation...")
                # nvblox points are typically in camera frame, transform to world frame
                # Add homogeneous coordinate (w=1) for transformation
                ones = torch.ones(nvblox_points_torch.shape[0], 1)
                points_homogeneous = torch.cat([nvblox_points_torch, ones], dim=1)  # [N, 4]
                
                # Apply camera pose transformation
                transformed_points = (camera_pose @ points_homogeneous.T).T  # [N, 4]
                nvblox_points_torch = transformed_points[:, :3]  # Take only XYZ
                print(f"✅ Applied camera pose transformation")
            else:
                print("⚠️ No camera pose available - using points as-is")
            
            # Additional base pose offset if available
            if base_pose is not None:
                print(f"🔄 Applying base pose offset: {base_pose}")
                # Add base position offset (x, y, yaw rotation)
                x_offset, y_offset, yaw = base_pose[0], base_pose[1], base_pose[2]
                
                # Apply 2D rotation around Z axis for yaw
                cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
                rotation_2d = torch.tensor([[cos_yaw, -sin_yaw, 0],
                                          [sin_yaw,  cos_yaw, 0],
                                          [0,        0,       1]], dtype=torch.float32)
                
                # Apply rotation and translation
                nvblox_points_torch = (rotation_2d @ nvblox_points_torch.T).T
                nvblox_points_torch[:, 0] += x_offset  # X offset
                nvblox_points_torch[:, 1] += y_offset  # Y offset
                print(f"✅ Applied base pose transformation")
            else:
                print("⚠️ No base pose available - skipping base transform")
            
            print(f"🎯 Final transformed points range:")
            print(f"   X: [{nvblox_points_torch[:, 0].min():.3f}, {nvblox_points_torch[:, 0].max():.3f}]")
            print(f"   Y: [{nvblox_points_torch[:, 1].min():.3f}, {nvblox_points_torch[:, 1].max():.3f}]")
            print(f"   Z: [{nvblox_points_torch[:, 2].min():.3f}, {nvblox_points_torch[:, 2].max():.3f}]")
            
            # Create a distinct color for nvblox points for easy visualization
            nvblox_rgb = torch.full((len(nvblox_points_torch), 3), 128, dtype=torch.uint8)
            nvblox_rgb[:, 1] = 200  # Make it grayish-green
            
            # Add to the persistent voxel point cloud storage
            self.voxel_pcd.add(
                nvblox_points_torch,
                rgb=nvblox_rgb,
                features=None,  # No features for pure geometry
                weights=None,
                min_weight_per_voxel=1
            )
            
            logger.debug(f"Integrated {len(nvblox_points_torch)} transformed nvblox points into voxel grid")
            
        except Exception as e:
            logger.error(f"Error integrating nvblox pointcloud with pose: {e}")
            import traceback
            traceback.print_exc()
    
    def _integrate_nvblox_pointcloud(self):
        """
        Legacy method - now calls the pose-aware version with None poses.
        """
        self._integrate_nvblox_pointcloud_with_pose(None, None)
    
    def _get_open3d_geometries(
        self,
        instances: bool = True,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        xyt: Optional[np.ndarray] = None,
        footprint = None,
        add_planner_visuals: bool = True,
        **kwargs
    ):
        """
        Override to provide high-quality geometry visualization from the
        ACCUMULATED voxel point cloud.
        """
        geoms = []

        # Always use the accumulated point cloud from the voxel grid,
        # which now contains integrated data from nvblox.
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        
        if points is not None and len(points) > 0:
            from stretch.utils.point_cloud import numpy_to_pcd
            pcd = numpy_to_pcd(points.detach().cpu().numpy(), (rgb / 255.0).detach().cpu().numpy())
            geoms.append(pcd)
        
        # Add semantic instances (Stretch AI's strength)
        if instances:
            self._get_instances_open3d(geoms)
        
        return geoms
    
    def _convert_nvblox_mesh_to_open3d(self, nvblox_mesh):
        """Convert nvblox mesh message to Open3D geometry."""
        try:
            import open3d as o3d
            
            # Extract vertices from the mesh message
            vertices = []
            for i in range(0, len(nvblox_mesh.vertices), 3):
                vertices.append([
                    nvblox_mesh.vertices[i],
                    nvblox_mesh.vertices[i+1], 
                    nvblox_mesh.vertices[i+2]
                ])
            
            # Extract triangle indices
            triangles = []
            for i in range(0, len(nvblox_mesh.triangles), 3):
                triangles.append([
                    nvblox_mesh.triangles[i],
                    nvblox_mesh.triangles[i+1],
                    nvblox_mesh.triangles[i+2]
                ])
            
            if len(vertices) > 0 and len(triangles) > 0:
                # Create Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                
                # Paint with a uniform color
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
                mesh.compute_vertex_normals()
                
                return mesh
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error converting nvblox mesh to Open3D: {e}")
            return None
    
    def get_2d_map(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 2D navigation map from nvblox data.
        
        Returns:
            obstacles: 2D tensor of obstacle locations
            explored: 2D tensor of explored areas
        """
        if self.use_nvblox_geometry and self.nvblox_pointcloud is not None:
            return self._generate_2d_map_from_nvblox()
        else:
            return super().get_2d_map()
    
    def _generate_2d_map_from_nvblox(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 2D navigation map from nvblox 3D data.
        """
        try:
            # Project nvblox 3D points to 2D grid
            points_2d = self.nvblox_pointcloud[:, :2]  # X, Y coordinates
            
            # Convert to grid coordinates
            grid_coords = (points_2d / self.grid_resolution + self.grid.grid_origin[:2].cpu().numpy()).astype(int)
            
            # Create obstacle and explored maps
            obstacles = torch.zeros(self.grid.map_size_in_voxels[:2], dtype=torch.bool)
            explored = torch.zeros(self.grid.map_size_in_voxels[:2], dtype=torch.bool)
            
            # Mark cells based on point heights
            for i, (x, y) in enumerate(grid_coords):
                if 0 <= x < self.grid.map_size_in_voxels[0] and 0 <= y < self.grid.map_size_in_voxels[1]:
                    z = self.nvblox_pointcloud[i, 2]
                    explored[x, y] = True
                    
                    # Consider points above certain height as obstacles
                    if z > self.obs_min_height and z < self.obs_max_height:
                        obstacles[x, y] = True
            
            return obstacles, explored
            
        except Exception as e:
            logger.error(f"Error generating 2D map from nvblox: {e}")
            return super().get_2d_map()
    
    def cleanup(self):
        """Clean up ROS resources."""
        if self._ros_node is not None and self._owns_node:
            try:
                self._ros_node.destroy_node()
                logger.info("nvblox ROS2 node cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up ROS node: {e}")
        elif self._ros_node is not None:
            logger.info("ROS2 node cleanup skipped (using shared node)")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Factory function to create nvblox-integrated voxel map
def create_nvblox_voxel_map(
    parameters: Parameters,
    encoder,
    voxel_size: float = 0.05,
    use_instance_memory: bool = True,
    ros_node=None,
    executor=None,
    **kwargs
) -> NvbloxSparseVoxelMap:
    """
    Factory function to create an nvblox-integrated voxel map.
    
    Usage:
        # In your mapping script:
        from stretch.mapping.voxel.nvblox_voxel import create_nvblox_voxel_map
        
        voxel_map = create_nvblox_voxel_map(
            parameters,
            encoder,
            voxel_size=params["voxel_size"],
            use_instance_memory=use_semantic,
        )
    """
    return NvbloxSparseVoxelMap(
        resolution=voxel_size,
        local_radius=parameters["local_radius"],
        grid_resolution=parameters["voxel_size"],
        obs_min_height=parameters["obs_min_height"],
        obs_max_height=parameters["obs_max_height"],
        min_depth=parameters["min_depth"],
        max_depth=parameters["max_depth"],
        add_local_radius_every_step=parameters["add_local_every_step"],
        min_points_per_voxel=parameters["min_points_per_voxel"],
        pad_obstacles=parameters["pad_obstacles"],
        add_local_radius_points=parameters.get("add_local_radius_points", True),
        remove_visited_from_obstacles=parameters.get("remove_visited_from_obstacles", False),
        obs_min_density=parameters["obs_min_density"],
        use_instance_memory=use_instance_memory,
        encoder=encoder,
        ros_node=ros_node,
        executor=executor,
        **kwargs
    )