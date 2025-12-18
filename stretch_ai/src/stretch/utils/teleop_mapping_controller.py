import threading
import time
import numpy as np
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient

class TeleopMappingController:
    def __init__(self, agent: RobotAgent, robot: HomeRobotZmqClient):
        self.agent = agent
        self.robot = robot
        self.running = True
        self.mapping_thread = None
        self.control_thread = None
        
    def start(self):
        """Start mapping and control threads"""
        self.mapping_thread = threading.Thread(target=self.mapping_loop)
        self.control_thread = threading.Thread(target=self.control_loop)
        
        self.mapping_thread.start()
        self.control_thread.start()
        
    def mapping_loop(self):
        """Continuous mapping loop"""
        while self.running and self.agent.is_running():
            # The agent will automatically update the map in real-time mode
            time.sleep(0.1)  # 10 Hz update rate
            
    def control_loop(self):
        """Handle control inputs from your Segway interface"""
        while self.running:
            # Get velocity commands from your Segway control interface
            # This is where you'll integrate with your BWI keyboard control
            linear_vel, angular_vel = self.get_velocity_commands()
            
            if linear_vel != 0 or angular_vel != 0:
                # Send velocity commands to robot
                self.robot.set_velocity(linear_vel, angular_vel)
            
            time.sleep(0.05)  # 20 Hz control rate
            
    def get_velocity_commands(self):
        """Override this with your Segway BWI control interface"""
        # Placeholder - integrate your keyboard control here
        return 0.0, 0.0
        
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.mapping_thread:
            self.mapping_thread.join()
        if self.control_thread:
            self.control_thread.join()