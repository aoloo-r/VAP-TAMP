#!/usr/bin/env python3
"""
Record3D WiFi streaming client for Stretch AI voxel mapping.

This script connects to a Record3D iOS app via WiFi and streams RGBD data
for real-time 3D mapping and navigation.
"""

import asyncio
import json
import time
from typing import Optional, Tuple
import numpy as np
import cv2
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRelay
import logging

logger = logging.getLogger(__name__)


class Record3DClient:
    """Client for connecting to Record3D iOS app via WiFi streaming."""
    
    def __init__(self, device_ip: str, port: int = 80):
        """Initialize Record3D client.
        
        Args:
            device_ip: IP address of the iOS device running Record3D
            port: Port number (default 8080)
        """
        self.device_ip = device_ip
        self.port = port
        self.base_url = f"http://{device_ip}:{port}"
        self.pc = None
        self.relay = MediaRelay()
        
        # Stream data
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None
        self.is_streaming = False
        
        # Callbacks
        self.frame_callback = None
        
    async def connect(self) -> bool:
        """Establish WebRTC connection to Record3D device."""
        try:
            # Get WebRTC offer from device
            offer_response = requests.get(f"{self.base_url}/getOffer", timeout=10)
            if offer_response.status_code == 403:
                logger.error("Device already has an active connection")
                return False
            elif offer_response.status_code != 200:
                logger.error(f"Failed to get offer: {offer_response.status_code}")
                return False
                
            offer_data = offer_response.json()
            
            # Get camera metadata
            try:
                metadata_response = requests.get(f"{self.base_url}/metadata", timeout=10)
                if metadata_response.status_code == 200:
                    metadata = metadata_response.json()
                    self.camera_intrinsics = np.array(metadata.get('K', [])).reshape(3, 3) if 'K' in metadata else None
                    logger.info(f"Retrieved camera intrinsics: {self.camera_intrinsics}")
            except Exception as e:
                logger.warning(f"Could not get camera metadata: {e}")
                
            # Create peer connection
            self.pc = RTCPeerConnection(RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            ))
            
            # Set up event handlers
            self.pc.on("track", self._on_track)
            self.pc.on("connectionstatechange", self._on_connection_state_change)
            
            # Set remote description from offer
            offer_sdp = RTCSessionDescription(
                sdp=offer_data["sdp"],
                type=offer_data["type"]
            )
            await self.pc.setRemoteDescription(offer_sdp)
            
            # Set up ICE candidate handler to send answer when gathering is complete
            self._answer_sent = False
            
            def on_icecandidate(candidate):
                if candidate is None and not self._answer_sent:
                    # ICE gathering complete, send answer
                    self._answer_sent = True
                    answer_data = {
                        "type": "answer",
                        "data": self.pc.localDescription.sdp
                    }
                    
                    try:
                        answer_response = requests.post(
                            f"{self.base_url}/answer",
                            json=answer_data,
                            headers={'Content-Type': 'application/json'},
                            timeout=10
                        )
                        
                        if answer_response.status_code != 200:
                            logger.error(f"Failed to send answer: {answer_response.status_code}")
                        else:
                            logger.info("Answer sent successfully!")
                            
                    except Exception as e:
                        logger.error(f"Error sending answer: {e}")
            
            self.pc.on("icecandidate", on_icecandidate)
            
            # Create and set local description
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            # Wait a moment for ICE gathering to potentially complete
            await asyncio.sleep(2)
                
            logger.info("WebRTC connection established successfully")
            self.is_streaming = True
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
            
    def _on_track(self, track):
        """Handle incoming WebRTC track (video stream)."""
        logger.info(f"Received track: {track.kind}")
        
        if track.kind == "video":
            # Process video frames
            asyncio.create_task(self._process_video_track(track))
            
    def _on_connection_state_change(self):
        """Handle WebRTC connection state changes."""
        if self.pc:
            logger.info(f"Connection state changed to: {self.pc.connectionState}")
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                self.is_streaming = False
                
    async def _process_video_track(self, track):
        """Process incoming video frames and extract RGB + depth."""
        try:
            while self.is_streaming:
                frame = await track.recv()
                
                # Convert to numpy array
                img = frame.to_ndarray(format="rgb24")
                
                if img is not None:
                    # Record3D WiFi format: The entire frame contains RGB with HSV-encoded depth
                    # We need to decode both RGB and depth from the same image
                    rgb_frame, depth_frame = self._decode_rgbd_from_stream(img)
                    
                    # Update latest frames
                    self.latest_rgb = rgb_frame
                    self.latest_depth = depth_frame
                    
                    # Call user callback if set
                    if self.frame_callback:
                        self.frame_callback(rgb_frame, depth_frame, self.camera_intrinsics)
                        
        except Exception as e:
            logger.error(f"Error processing video track: {e}")
            self.is_streaming = False
            
    def _decode_rgbd_from_stream(self, rgb_array):
        """Decode RGBD data from Record3D WiFi stream
        
        Record3D encodes depth in HSV color space within the RGB stream
        depth_in_meters = 3.0 * hue_component
        
        Args:
            rgb_array: RGB frame from Record3D WiFi stream
            
        Returns:
            Tuple of (rgb_frame, depth_frame) where depth is in meters
        """
        # Convert RGB to HSV for depth decoding
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        
        # Extract hue component (normalized to 0-1)
        hue = hsv[:, :, 0].astype(np.float32) / 180.0  # OpenCV hue is 0-180, normalize to 0-1
        
        # Convert hue to depth in meters (0-3 meters range)
        depth_meters = 3.0 * hue
        
        # For RGB, use the original frame (contains encoded depth but still has color info)
        rgb_clean = rgb_array.copy()
        
        return rgb_clean, depth_meters
        
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the latest RGB and depth frames.
        
        Returns:
            Tuple of (rgb_frame, depth_frame, camera_intrinsics)
        """
        return self.latest_rgb, self.latest_depth, self.camera_intrinsics
        
    def set_frame_callback(self, callback):
        """Set callback function for new frames.
        
        Args:
            callback: Function that takes (rgb, depth, intrinsics) as arguments
        """
        self.frame_callback = callback
        
    async def disconnect(self):
        """Disconnect from Record3D device."""
        self.is_streaming = False
        if self.pc:
            await self.pc.close()
            self.pc = None
        logger.info("Disconnected from Record3D device")


async def main():
    """Demo usage of Record3D client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Record3D WiFi streaming client")
    parser.add_argument("device_ip", help="IP address of iOS device running Record3D")
    parser.add_argument("--port", type=int, default=80, help="Port number (default: 80)")
    parser.add_argument("--show-frames", action="store_true", help="Display RGB and depth frames")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = Record3DClient(args.device_ip, args.port)
    
    def frame_callback(rgb, depth, intrinsics):
        """Handle new frames."""
        print(f"Received frame: RGB {rgb.shape}, Depth {depth.shape}")
        if args.show_frames:
            # Show RGB
            cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Show depth (normalized for visualization)
            depth_vis = (depth / 3.0 * 255).astype(np.uint8)
            cv2.imshow("Depth", depth_vis)
            
            cv2.waitKey(1)
    
    client.set_frame_callback(frame_callback)
    
    # Connect and stream
    try:
        print(f"Connecting to Record3D device at {args.device_ip}:{args.port}")
        if await client.connect():
            print("Connected! Streaming frames... Press Ctrl+C to stop.")
            while client.is_streaming:
                await asyncio.sleep(0.1)
        else:
            print("Failed to connect")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await client.disconnect()
        if args.show_frames:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())