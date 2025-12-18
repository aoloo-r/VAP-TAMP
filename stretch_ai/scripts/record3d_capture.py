import asyncio
import json
import pickle
import time
import zmq
import cv2
import numpy as np
import requests
from pathlib import Path
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Record3DWebRTCClient:
    def __init__(self, device_ip, use_zmq=False, zmq_port=5555):
        self.device_ip = device_ip
        self.device_url = f"http://{device_ip}"
        self.pc = RTCPeerConnection()
        self.relay = MediaRelay()
        self.use_zmq = use_zmq
        
        # ZMQ Setup (optional)
        if self.use_zmq:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{zmq_port}")
            print(f"ZMQ Publisher running on tcp://*:{zmq_port}")
        else:
            self.context = None
            self.socket = None
            print("Direct data access mode (no ZMQ)")
        
        self.frame_id = 0
        self.rgb_frame = None
        self.depth_frame = None
        self.pose = None
        self.intrinsics = None
        
        # Data buffer for direct access
        self.frame_buffer = []
        self.max_buffer_size = 100
        self.data_callback = None
        
        # Shared data file for other processes to access
        self.shared_data_file = Path("/tmp/record3d_latest_frame.pkl")
        self.enable_shared_data = True
        
        # Set up peer connection event handlers
        self.pc.on("track", self.on_track)
        self.pc.on("connectionstatechange", self.on_connection_state_change)
        self.pc.on("datachannel", self.on_datachannel)
        
        # Pose tracking
        self.current_pose = np.eye(4, dtype=np.float32)
        self.pose_data_channel = None
        self.last_pose_timestamp = 0
        
        print("Configured to receive ARKit poses from Record3D data channel")
    
    def set_data_callback(self, callback):
        """Set a callback function to be called when new frame data is available
        
        Args:
            callback: Function that takes (rgb, depth, pose, intrinsics) as arguments
        """
        self.data_callback = callback
    
    def get_latest_frame(self):
        """Get the most recent frame data
        
        Returns:
            dict or None: Frame data with keys: rgb, depth, pose, intrinsics
        """
        if self.rgb_frame is not None and self.depth_frame is not None:
            return {
                'rgb': self.rgb_frame.copy(),
                'depth': self.depth_frame.copy(), 
                'pose': self.pose.copy() if self.pose is not None else np.eye(4, dtype=np.float32),
                'intrinsics': self.intrinsics.copy() if self.intrinsics is not None else None
            }
        return None
    
    def get_buffered_frames(self):
        """Get all buffered frames and clear the buffer
        
        Returns:
            list: List of frame dictionaries
        """
        frames = self.frame_buffer.copy()
        self.frame_buffer.clear()
        return frames

    async def on_track(self, track):
        """Handle incoming video/audio tracks from Record3D"""
        logger.info(f"Track received: {track.kind}")
        
        if track.kind == "video":
            relay_track = self.relay.subscribe(track)
            logger.info("Video track subscribed")
            
            # Process video frames
            asyncio.create_task(self.process_video_frames(relay_track))

    async def process_video_frames(self, track):
        """Process incoming video frames from WebRTC stream"""
        try:
            while True:
                frame = await track.recv()
                
                # Convert WebRTC frame to numpy array
                rgb_array = frame.to_ndarray(format="rgb24")
                
                # Record3D WiFi stream contains depth encoded in HSV color space
                # Decode depth from the HSV-encoded RGB stream
                self.rgb_frame, self.depth_frame = self.decode_rgbd_from_stream(rgb_array)
                
                # Use iPhone's built-in ARKit pose (updated from data channel)
                self.pose = self.current_pose.copy()  # Make a copy to avoid race conditions
                
                # Publish the frame data
                await self.publish_frame()
                
                # Display the frame
                display_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.imshow("Record3D WebRTC Stream (Press 'q' to stop)", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error processing video frames: {e}")

    async def publish_frame(self):
        """Publish frame data (ZMQ or direct access)"""
        if self.rgb_frame is not None:
            data_packet = {
                'frame_id': self.frame_id,
                'rgb': self.rgb_frame,
                'depth': self.depth_frame,
                'pose': self.pose,
                'intrinsics': self.intrinsics  # Real intrinsics from Record3D
            }
            
            try:
                # ZMQ publishing (if enabled)
                if self.use_zmq and self.socket is not None:
                    serialized_data = pickle.dumps(data_packet)
                    self.socket.send_multipart([b"record3d_frames", serialized_data])
                
                # Direct data access - add to buffer
                if len(self.frame_buffer) >= self.max_buffer_size:
                    self.frame_buffer.pop(0)  # Remove oldest frame
                self.frame_buffer.append(data_packet.copy())
                
                # Call data callback if set
                if self.data_callback is not None:
                    self.data_callback(
                        self.rgb_frame,
                        self.depth_frame, 
                        self.pose,
                        self.intrinsics
                    )
                
                # Write to shared file for other processes using atomic write
                if self.enable_shared_data:
                    shared_data = {
                        'frame_id': self.frame_id,
                        'rgb': self.rgb_frame,
                        'depth': self.depth_frame,
                        'pose': self.pose,
                        'intrinsics': self.intrinsics,
                        'timestamp': time.time()
                    }
                    try:
                        # Use atomic write: write to temp file then rename
                        temp_file = self.shared_data_file.with_suffix('.tmp')
                        with open(temp_file, 'wb') as f:
                            pickle.dump(shared_data, f)
                        temp_file.rename(self.shared_data_file)  # Atomic on Unix systems
                    except Exception as e:
                        # Don't let file I/O errors break the main stream
                        if self.frame_id % 100 == 0:  # Only log occasionally
                            logger.warning(f"Could not write shared data: {e}")
                
                # Print progress every 30 frames
                if self.frame_id % 30 == 0:
                    mode_str = "ZMQ + Direct" if self.use_zmq else "Direct"
                    print(f"[{mode_str}] Processed {self.frame_id} frames - RGB: {self.rgb_frame.shape}, Depth: {self.depth_frame.shape}")
                    
                self.frame_id += 1
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()

    async def on_connection_state_change(self):
        """Handle connection state changes"""
        logger.info(f"Connection state: {self.pc.connectionState}")

    def on_datachannel(self, channel):
        """Handle incoming data channel for pose data"""
        logger.info(f"Data channel received: {channel.label}")
        
        if channel.label == "arkit_poses" or channel.label == "poses":
            self.pose_data_channel = channel
            channel.on("message", self.on_pose_message)
            logger.info("ARKit pose data channel connected")
    
    def on_pose_message(self, message):
        """Handle pose data from WebRTC data channel"""
        try:
            if isinstance(message, bytes):
                # Parse binary pose data
                pose_data = self.parse_pose_data(message)
                if pose_data is not None:
                    self.current_pose = pose_data
                    self.last_pose_timestamp = time.time()
            elif isinstance(message, str):
                # Parse JSON pose data
                pose_json = json.loads(message)
                pose_matrix = self.parse_pose_json(pose_json)
                if pose_matrix is not None:
                    self.current_pose = pose_matrix
                    self.last_pose_timestamp = time.time()
        except Exception as e:
            logger.warning(f"Error parsing pose data: {e}")
    
    def parse_pose_data(self, data: bytes) -> np.ndarray:
        """Parse binary pose data from Record3D
        
        Record3D sends 4x4 transformation matrices as binary data
        """
        try:
            # Record3D typically sends poses as 16 float32 values (4x4 matrix)
            if len(data) == 64:  # 16 * 4 bytes
                pose_flat = np.frombuffer(data, dtype=np.float32)
                pose_matrix = pose_flat.reshape(4, 4)
                return pose_matrix
            else:
                logger.warning(f"Unexpected pose data size: {len(data)} bytes")
                return None
        except Exception as e:
            logger.warning(f"Error parsing binary pose data: {e}")
            return None
    
    def parse_pose_json(self, pose_json: dict) -> np.ndarray:
        """Parse JSON pose data from Record3D"""
        try:
            # Look for transformation matrix in various formats
            if 'transform' in pose_json:
                transform = pose_json['transform']
                if isinstance(transform, list) and len(transform) == 16:
                    return np.array(transform, dtype=np.float32).reshape(4, 4)
            
            # Look for position + rotation format
            if 'position' in pose_json and 'rotation' in pose_json:
                pos = pose_json['position']
                rot = pose_json['rotation']
                
                # Create transformation matrix from position and rotation
                pose = np.eye(4, dtype=np.float32)
                pose[0, 3] = pos.get('x', 0)
                pose[1, 3] = pos.get('y', 0) 
                pose[2, 3] = pos.get('z', 0)
                
                # TODO: Handle rotation (quaternion or euler angles)
                # This would need to be implemented based on Record3D's actual format
                
                return pose
            
            return None
        except Exception as e:
            logger.warning(f"Error parsing JSON pose data: {e}")
            return None

    async def get_camera_intrinsics(self):
        """Get camera intrinsics from Record3D device metadata endpoint"""
        try:
            response = requests.get(f"{self.device_url}/metadata", timeout=10)
            if response.status_code == 200:
                metadata = response.json()
                logger.info("Got camera metadata from Record3D")
                return metadata
            else:
                logger.error(f"Failed to get metadata: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting metadata: {e}")
            return None

    async def get_webrtc_offer(self):
        """Get WebRTC offer from Record3D device"""
        try:
            response = requests.get(f"{self.device_url}/getOffer", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get offer: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting offer: {e}")
            return None

    async def send_webrtc_answer(self, answer):
        """Send WebRTC answer to Record3D device"""
        try:
            headers = {'Content-Type': 'application/json'}
            logger.info(f"Sending answer to {self.device_url}/answer")
            logger.info(f"Answer data: {answer}")
            
            response = requests.post(
                f"{self.device_url}/answer", 
                json=answer, 
                headers=headers,
                timeout=10
            )
            
            logger.info(f"Answer response status: {response.status_code}")
            logger.info(f"Answer response text: {response.text}")
            
            if response.status_code == 200:
                logger.info("Answer sent successfully")
                return True
            else:
                logger.error(f"Failed to send answer: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending answer: {e}")
            return False

    async def request_pose_data(self):
        """Request pose data from Record3D device"""
        try:
            # Try to enable pose streaming
            response = requests.post(f"{self.device_url}/enable_poses", json={"enable": True}, timeout=5)
            if response.status_code == 200:
                logger.info("Pose data streaming enabled")
                return True
            else:
                logger.warning(f"Could not enable pose streaming: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error requesting pose data: {e}")
            return False

    async def connect(self):
        """Establish WebRTC connection with Record3D device"""
        try:
            # Get camera intrinsics from metadata endpoint
            print("Getting camera intrinsics from Record3D...")
            metadata = await self.get_camera_intrinsics()
            if metadata and 'K' in metadata:
                # Convert intrinsics from flattened array to 3x3 matrix
                K_flat = metadata['K']
                # Record3D format: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
                # Convert to standard format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                self.intrinsics = np.array([
                    [K_flat[0], K_flat[1], K_flat[6]],  # [fx, 0, cx]
                    [K_flat[3], K_flat[4], K_flat[7]],  # [0, fy, cy]  
                    [K_flat[2], K_flat[5], K_flat[8]]   # [0, 0, 1]
                ], dtype=np.float32)
                logger.info(f"Camera intrinsics matrix:\n{self.intrinsics}")
            else:
                logger.warning("Could not get camera intrinsics from metadata")
            
            # Request pose data streaming
            print("Requesting ARKit pose data...")
            await self.request_pose_data()
                
            # Get the offer from Record3D
            print("Getting WebRTC offer from Record3D...")
            offer_data = await self.get_webrtc_offer()
            if not offer_data:
                return False

            # Set remote description from the offer
            offer = RTCSessionDescription(
                sdp=offer_data["sdp"], 
                type=offer_data["type"]
            )
            await self.pc.setRemoteDescription(offer)
            logger.info("Remote description set")

            # Create and set local answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            logger.info("Local description set")

            # Wait for ICE gathering to complete (like the JavaScript demo does)
            logger.info("Waiting for ICE gathering to complete...")
            while self.pc.iceGatheringState != "complete":
                await asyncio.sleep(0.1)
            
            logger.info("ICE gathering complete, sending answer...")

            # Send answer back to Record3D (matching the JavaScript format)
            answer_data = {
                "type": "answer",
                "data": self.pc.localDescription.sdp
            }
            
            success = await self.send_webrtc_answer(answer_data)
            if not success:
                return False

            print("WebRTC connection established! Waiting for video stream...")
            return True

        except Exception as e:
            logger.error(f"Error establishing WebRTC connection: {e}")
            return False

    async def run(self):
        """Main run loop"""
        try:
            # Wait for connection to be established
            while self.pc.connectionState != "connected":
                if self.pc.connectionState == "failed":
                    logger.error("WebRTC connection failed")
                    return
                await asyncio.sleep(0.1)

            logger.info("WebRTC connection established, receiving stream...")
            
            # Keep the connection alive
            while self.pc.connectionState == "connected":
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
        finally:
            await self.cleanup()

    def decode_rgbd_from_stream(self, rgb_array):
        """Decode RGBD data from Record3D WiFi stream
        
        Record3D encodes depth in HSV color space within the RGB stream
        depth_in_meters = 3.0 * hue_component
        """
        import cv2
        
        # Convert RGB to HSV for depth decoding
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        
        # Extract hue component (normalized to 0-1)
        hue = hsv[:, :, 0].astype(np.float32) / 180.0  # OpenCV hue is 0-180, normalize to 0-1
        
        # Convert hue to depth in meters (0-3 meters range)
        depth_meters = 3.0 * hue
        
        # Convert to millimeters and to uint16 format (standard depth format)
        depth_mm = (depth_meters * 1000.0).astype(np.uint16)
        
        # For RGB, we'll use the original frame (though it contains encoded depth)
        # In a production system, you might want to separate RGB and depth streams
        rgb_clean = rgb_array.copy()
        
        return rgb_clean, depth_mm

    async def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        await self.pc.close()
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()
        cv2.destroyAllWindows()

async def main():
    """Main function"""
    # Get device IP from user
    device_ip = input("Enter your iPhone's IP address (e.g., 149.125.149.216): ").strip()
    
    if not device_ip:
        print("Error: No IP address provided")
        return

    print(f"Attempting to connect to Record3D at {device_ip}")
    print("\nMake sure:")
    print("1. Your iPhone and computer are on the same WiFi network")
    print("2. Record3D app is open with WiFi Streaming enabled")
    print("3. WiFi Streaming Extension Pack is purchased and activated")
    print("4. Press the record button in the Record3D app")
    print()

    client = Record3DWebRTCClient(device_ip)
    
    # Establish WebRTC connection
    if await client.connect():
        # Run the main loop
        await client.run()
    else:
        print("Failed to establish WebRTC connection")
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")