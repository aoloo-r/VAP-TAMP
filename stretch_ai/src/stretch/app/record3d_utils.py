#!/usr/bin/env python3
"""
Utility functions for Record3D data processing and device discovery.
"""

import socket
import struct
import time
import numpy as np
import cv2
import requests
from typing import List, Dict, Optional, Tuple, Any
import json
import asyncio
from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
import threading

from stretch.utils.logger import Logger

logger = Logger(__name__)


class Record3DDeviceDiscovery(ServiceListener):
    """Discover Record3D devices on the local network using mDNS/Bonjour."""
    
    def __init__(self):
        self.devices = {}
        self.discovery_complete = threading.Event()
        
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a Record3D service is discovered."""
        info = zc.get_service_info(type_, name)
        if info:
            device_info = {
                'name': name,
                'ip': socket.inet_ntoa(info.addresses[0]),
                'port': info.port,
                'properties': {k.decode(): v.decode() if isinstance(v, bytes) else v 
                             for k, v in info.properties.items()},
                'server': info.server
            }
            self.devices[name] = device_info
            logger.info(f"Discovered Record3D device: {device_info}")
            
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a Record3D service is removed."""
        if name in self.devices:
            logger.info(f"Record3D device removed: {name}")
            del self.devices[name]
            
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a Record3D service is updated."""
        self.add_service(zc, type_, name)


def discover_record3d_devices(timeout: float = 5.0) -> Dict[str, Dict[str, Any]]:
    """Discover Record3D devices on the local network.
    
    Args:
        timeout: Discovery timeout in seconds
        
    Returns:
        Dictionary of discovered devices
    """
    logger.info("Discovering Record3D devices on local network...")
    
    zeroconf = Zeroconf()
    listener = Record3DDeviceDiscovery()
    
    # Look for Record3D HTTP services
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    
    # Wait for discovery
    time.sleep(timeout)
    
    # Filter for actual Record3D devices (they typically have "Record3D" in the name)
    record3d_devices = {
        name: info for name, info in listener.devices.items()
        if 'record3d' in name.lower() or 'record3d' in info.get('server', '').lower()
    }
    
    zeroconf.close()
    
    if record3d_devices:
        logger.info(f"Found {len(record3d_devices)} Record3D device(s)")
        for name, info in record3d_devices.items():
            logger.info(f"  - {name}: {info['ip']}:{info['port']}")
    else:
        logger.info("No Record3D devices found")
        
    return record3d_devices


def test_record3d_connection(ip: str, port: int = 80, timeout: float = 5.0) -> Dict[str, Any]:
    """Test connection to a Record3D device.
    
    Args:
        ip: Device IP address
        port: Device port
        timeout: Connection timeout
        
    Returns:
        Connection test results
    """
    base_url = f"http://{ip}:{port}"
    result = {
        'ip': ip,
        'port': port,
        'reachable': False,
        'metadata_available': False,
        'streaming_available': False,
        'error': None,
        'metadata': None
    }
    
    try:
        # Test basic connectivity
        response = requests.get(f"{base_url}/", timeout=timeout)
        result['reachable'] = True
        
        # Test metadata endpoint
        try:
            metadata_response = requests.get(f"{base_url}/metadata", timeout=timeout)
            if metadata_response.status_code == 200:
                result['metadata_available'] = True
                result['metadata'] = metadata_response.json()
        except Exception as e:
            logger.debug(f"Metadata test failed: {e}")
            
        # Test streaming availability (getOffer endpoint)
        try:
            offer_response = requests.get(f"{base_url}/getOffer", timeout=timeout)
            if offer_response.status_code == 200:
                result['streaming_available'] = True
            elif offer_response.status_code == 403:
                result['streaming_available'] = True  # Available but someone else connected
                result['error'] = "Device already has an active connection"
        except Exception as e:
            logger.debug(f"Streaming test failed: {e}")
            
    except Exception as e:
        result['error'] = str(e)
        logger.debug(f"Connection test failed for {ip}:{port} - {e}")
        
    return result


def validate_camera_intrinsics(intrinsics: np.ndarray) -> bool:
    """Validate camera intrinsics matrix.
    
    Args:
        intrinsics: 3x3 camera intrinsics matrix
        
    Returns:
        True if valid, False otherwise
    """
    if intrinsics is None:
        return False
        
    if intrinsics.shape != (3, 3):
        return False
        
    # Check for reasonable values
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Focal lengths should be positive
    if fx <= 0 or fy <= 0:
        return False
        
    # Principal point should be reasonable (not at origin)
    if cx <= 0 or cy <= 0:
        return False
        
    return True


def filter_depth_image(depth: np.ndarray, min_depth: float = 0.01, max_depth: float = 3.0) -> np.ndarray:
    """Filter depth image to remove invalid values.
    
    Args:
        depth: Input depth image
        min_depth: Minimum valid depth value
        max_depth: Maximum valid depth value
        
    Returns:
        Filtered depth image
    """
    depth_filtered = depth.copy()
    
    # Remove invalid depths
    mask = (depth_filtered < min_depth) | (depth_filtered > max_depth) | np.isnan(depth_filtered)
    depth_filtered[mask] = 0.0
    
    return depth_filtered


def apply_bilateral_filter_to_depth(depth: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter to smooth depth image while preserving edges.
    
    Args:
        depth: Input depth image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered depth image
    """
    # Convert to uint16 for bilateral filtering
    depth_uint16 = (depth * 1000).astype(np.uint16)
    
    # Apply bilateral filter
    filtered_uint16 = cv2.bilateralFilter(depth_uint16, d, sigma_color, sigma_space)
    
    # Convert back to float32
    filtered_depth = filtered_uint16.astype(np.float32) / 1000.0
    
    return filtered_depth


def create_depth_visualization(depth: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Create a colored visualization of depth data.
    
    Args:
        depth: Input depth image (0-3 meters)
        colormap: OpenCV colormap to use
        
    Returns:
        Colored depth visualization (RGB)
    """
    # Normalize depth to 0-255 range
    depth_norm = np.clip((depth / 3.0) * 255, 0, 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_norm, colormap)
    
    # Convert BGR to RGB
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_rgb


def estimate_camera_pose_from_imu(acceleration: np.ndarray, gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
    """Estimate camera orientation from IMU acceleration data.
    
    Args:
        acceleration: 3D acceleration vector
        gravity: Expected gravity vector
        
    Returns:
        4x4 transformation matrix representing camera pose
    """
    # Normalize acceleration
    acc_norm = acceleration / np.linalg.norm(acceleration)
    gravity_norm = gravity / np.linalg.norm(gravity)
    
    # Compute rotation to align with gravity
    # This is a simplified approach - for better results, use full IMU data
    v = np.cross(acc_norm, gravity_norm)
    s = np.linalg.norm(v)
    c = np.dot(acc_norm, gravity_norm)
    
    if s == 0:
        # Already aligned
        rotation = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # Create 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation
    
    return pose


def calculate_point_cloud_stats(xyz: np.ndarray) -> Dict[str, Any]:
    """Calculate statistics for a point cloud.
    
    Args:
        xyz: Point cloud array (N x 3)
        
    Returns:
        Dictionary with point cloud statistics
    """
    if xyz.size == 0:
        return {"num_points": 0, "error": "Empty point cloud"}
    
    stats = {
        "num_points": xyz.shape[0],
        "bounds": {
            "min": xyz.min(axis=0).tolist(),
            "max": xyz.max(axis=0).tolist(),
            "range": (xyz.max(axis=0) - xyz.min(axis=0)).tolist()
        },
        "centroid": xyz.mean(axis=0).tolist(),
        "std": xyz.std(axis=0).tolist(),
    }
    
    # Calculate density (points per cubic meter)
    volume = np.prod(stats["bounds"]["range"])
    if volume > 0:
        stats["density"] = stats["num_points"] / volume
    else:
        stats["density"] = 0
        
    return stats


def save_rgbd_frame(rgb: np.ndarray, depth: np.ndarray, filename_prefix: str, timestamp: Optional[float] = None):
    """Save RGB and depth frames to disk.
    
    Args:
        rgb: RGB image array
        depth: Depth image array  
        filename_prefix: Prefix for saved files
        timestamp: Optional timestamp for filename
    """
    if timestamp is None:
        timestamp = time.time()
        
    timestamp_str = f"{timestamp:.3f}"
    
    # Save RGB
    rgb_filename = f"{filename_prefix}_rgb_{timestamp_str}.png"
    cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Save depth as 16-bit PNG (millimeters)
    depth_filename = f"{filename_prefix}_depth_{timestamp_str}.png"
    depth_mm = (depth * 1000).astype(np.uint16)
    cv2.imwrite(depth_filename, depth_mm)
    
    # Save metadata
    metadata_filename = f"{filename_prefix}_metadata_{timestamp_str}.json"
    metadata = {
        "timestamp": timestamp,
        "rgb_filename": rgb_filename,
        "depth_filename": depth_filename,
        "rgb_shape": rgb.shape,
        "depth_shape": depth.shape,
        "depth_scale": 0.001  # mm to m conversion factor
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved RGBD frame: {filename_prefix}_{timestamp_str}")


def load_rgbd_frame(metadata_filename: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load RGB and depth frames from disk.
    
    Args:
        metadata_filename: Path to metadata JSON file
        
    Returns:
        Tuple of (rgb, depth, metadata)
    """
    with open(metadata_filename, 'r') as f:
        metadata = json.load(f)
        
    # Load RGB
    rgb_bgr = cv2.imread(metadata['rgb_filename'])
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    # Load depth
    depth_mm = cv2.imread(metadata['depth_filename'], cv2.IMREAD_UNCHANGED)
    depth = depth_mm.astype(np.float32) * metadata['depth_scale']
    
    return rgb, depth, metadata


async def benchmark_streaming_performance(client, duration: float = 10.0) -> Dict[str, Any]:
    """Benchmark Record3D streaming performance.
    
    Args:
        client: Record3DClient instance
        duration: Test duration in seconds
        
    Returns:
        Performance statistics
    """
    frame_times = []
    frame_count = 0
    data_received = 0
    
    def frame_callback(rgb, depth, intrinsics):
        nonlocal frame_count, data_received
        frame_times.append(time.time())
        frame_count += 1
        data_received += rgb.nbytes + depth.nbytes
    
    # Set up callback
    original_callback = client.frame_callback
    client.set_frame_callback(frame_callback)
    
    # Run benchmark
    start_time = time.time()
    await asyncio.sleep(duration)
    end_time = time.time()
    
    # Restore original callback
    client.set_frame_callback(original_callback)
    
    # Calculate statistics
    actual_duration = end_time - start_time
    fps = frame_count / actual_duration if actual_duration > 0 else 0
    data_rate = data_received / actual_duration / (1024 * 1024) if actual_duration > 0 else 0  # MB/s
    
    # Frame timing analysis
    if len(frame_times) > 1:
        frame_intervals = np.diff(frame_times)
        avg_interval = np.mean(frame_intervals)
        std_interval = np.std(frame_intervals)
        min_interval = np.min(frame_intervals)
        max_interval = np.max(frame_intervals)
    else:
        avg_interval = std_interval = min_interval = max_interval = 0
    
    stats = {
        "duration": actual_duration,
        "frame_count": frame_count,
        "fps": fps,
        "data_received_mb": data_received / (1024 * 1024),
        "data_rate_mbps": data_rate,
        "frame_timing": {
            "avg_interval": avg_interval,
            "std_interval": std_interval,
            "min_interval": min_interval,
            "max_interval": max_interval,
        }
    }
    
    return stats


def get_device_ip() -> str:
    """Prompt user for device IP address."""
    while True:
        ip = input("Enter your Record3D device IP address: ").strip()
        if ip:
            # Basic IP validation
            parts = ip.split('.')
            if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
                return ip
            else:
                print("Invalid IP address format. Please enter a valid IP (e.g., 192.168.1.100)")
        else:
            print("Please enter an IP address.")


def connect_to_record3d(ip: str = None, port: int = 8080) -> Dict[str, Any]:
    """Connect to Record3D device with user-provided IP.
    
    Args:
        ip: Device IP address (if None, will prompt user)
        port: Device port
        
    Returns:
        Connection results and device info
    """
    if ip is None:
        ip = get_device_ip()
        
    # Try common Record3D ports
    common_ports = [port, 8080, 80, 8081, 8000] if port != 8080 else [8080, 80, 8081, 8000]
    
    result = None
    for test_port in common_ports:
        print(f"Testing connection to Record3D device at {ip}:{test_port}...")
        result = test_record3d_connection(ip, test_port, timeout=5.0)
        
        if result['reachable']:
            port = test_port  # Use the working port
            break
        else:
            print(f"   Port {test_port}: {result.get('error', 'Connection failed')}")
    
    if not result or not result['reachable']:
        print(f"\n❌ Could not connect to {ip} on any common ports")
        print("💡 Double-check:")
        print("   1. Record3D app is OPEN and RUNNING")
        print("   2. WiFi streaming is ENABLED in the app")
        print("   3. App shows 'Waiting for connection' or similar")
        print("   4. Both devices are on the SAME WiFi network")
        print("   5. IP address is correct")
        print("\n📱 In Record3D app:")
        print("   - Look for a WiFi/Streaming toggle or button")
        print("   - Check if there's a 'Start WiFi streaming' option")
        print("   - The app should show the port number it's using")
        return result or {'reachable': False, 'error': 'No working port found'}
    
    if result['reachable']:
        print(f"✅ Successfully connected to {ip}:{port}")
        
        if result['metadata_available'] and result['metadata']:
            print("📷 Camera metadata available:")
            metadata = result['metadata']
            if 'K' in metadata:
                K = np.array(metadata['K']).reshape(3, 3)
                print(f"   Intrinsics matrix:\n{K}")
            for key, value in metadata.items():
                if key != 'K':  # Already printed above
                    print(f"   {key}: {value}")
        else:
            print("⚠️  Camera metadata not available")
            
        if result['streaming_available']:
            print("🎥 Streaming available - ready to connect!")
        else:
            error_msg = result.get('error', 'Unknown streaming issue')
            print(f"❌ Streaming not available: {error_msg}")
            
    else:
        error_msg = result.get('error', 'Unknown connection error')
        print(f"❌ Failed to connect to {ip}:{port}: {error_msg}")
        print("💡 Make sure:")
        print("   - Record3D app is running on your device")
        print("   - WiFi streaming is enabled in the app") 
        print("   - Both devices are on the same WiFi network")
        print("   - The IP address is correct")
        
    return result


def main():
    """Simplified utility for connecting to Record3D device."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Record3D connection utility")
    parser.add_argument("--ip", help="Record3D device IP address")
    parser.add_argument("--port", type=int, default=8080, help="Device port (default: 8080)")
    parser.add_argument("--discover", action="store_true", help="Try device discovery (often fails)")
    args = parser.parse_args()
    
    if args.discover:
        print("Attempting device discovery (this often fails on some networks)...")
        devices = discover_record3d_devices()
        if devices:
            print(f"Found {len(devices)} Record3D device(s):")
            for name, info in devices.items():
                print(f"  {name}: {info['ip']}:{info['port']}")
        else:
            print("No devices found via discovery. Try connecting directly with IP address.")
            
    # Always try direct connection
    print("\n" + "="*50)
    print("DIRECT CONNECTION")
    print("="*50)
    
    result = connect_to_record3d(args.ip, args.port)
    
    if result['reachable'] and result['streaming_available']:
        print(f"\n🎉 Ready to stream! Use this IP for your scripts: {result['ip']}")
        print(f"Example command:")
        print(f"  python record3d_client.py {result['ip']} --show-frames")
        print(f"  python record3d_voxel_mapping.py {result['ip']} --visualize")
    else:
        print(f"\n🔧 Connection issues detected. Please check your setup.")


if __name__ == "__main__":
    main()