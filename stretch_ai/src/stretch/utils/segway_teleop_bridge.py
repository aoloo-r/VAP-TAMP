# segway_teleop_bridge.py
import rospy
from geometry_msgs.msg import Twist
from stretch.app.teleop_mapping_controller import TeleopMappingController

class SegwayTeleopBridge(TeleopMappingController):
    def __init__(self, agent, robot):
        super().__init__(agent, robot)
        # Subscribe to Segway velocity commands
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        self.current_linear = 0.0
        self.current_angular = 0.0
        
    def velocity_callback(self, msg):
        """ROS callback for velocity commands"""
        self.current_linear = msg.linear.x
        self.current_angular = msg.angular.z
        
    def get_velocity_commands(self):
        """Return current velocity commands from ROS"""
        return self.current_linear, self.current_angular