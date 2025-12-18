#!/usr/bin/env python
# import roslib;
# roslib.load_manifest('segbot_bu_moveit_config')

import rospy
from geometry_msgs.msg import Point, Quaternion

from SegbotController import SegbotController
from UR5eController import UR5eController


if __name__ == '__main__':
    try:
        robot_initial_position = {
                    "x": 14.9670456571, "y": 8.15893807461, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz":  0.794112346207, "ow": 0.607770994374
                }
        
        rospy.init_node('mobile_manipulator', anonymous=False)
        base = SegbotController()
        arm = UR5eController()


            
        """{
                "object_name": "teacup",
                "object_target_xy": [-0.5, -1.01],
                #"object_offset": [0.04, 0.0, 0.06],
                "object_offset": [0.07, 0.07, 0.06],
                "robot_pickup_pose": {
                    "x": 15.4609046839, "y": 10.9876245528, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.244161321616, "ow": 0.969734628147
                    #"x": 15.1487559078, "y": 11.23034191418, "z": 0.0,
                    #"ox": 0.0, "oy": 0.0, "oz": -0.124339115216, "ow": 0.99223978172
                },
                "robot_place_pose": {
                    #"x": 15.5, "y": 10.6, "z": 0.0,
                    #"ox": 0.0, "oy": 0.0, "oz": 0.959801630907, "ow": 0.280679228494
                    "x": 15.3374115672, "y": 10.9339690232, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.967187176624, "ow": 0.254064884183
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.01               

            },
            {
                "object_name": "fruit bowl",
                "object_target_xy": [0.33, -1.99],
                "object_offset": [0.07, 0.07, 0.13],
                "robot_pickup_pose": {
                    #"x": 14.5, "y": 11.8576272868, "z": 0.0,
                    #"ox": 0.0, "oy": 0.0, "oz": -0.809528947684, "ow": 0.587079962919
                    "x": 14.5820224393, "y": 12.1848702946, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.842002170556, "ow": 0.539474137265
                },
                "robot_place_pose": {
                    #"x": 14.0, "y": 9.46, "z": 0.0,
                    #"ox": 0.0, "oy": 0.0, "oz": -0.820342697364, "ow": 0.571872239998
                    "x": 14.1750586882, "y": 10.2263717045, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.828059312002, "ow": 0.560640504964
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.07                

            },
            """
            
        info= [ 
                {
                "object_name": "fridge",
                "object_target_xy": [-0.5, -1.01],
                "object_offset": [0.10, 0.07, 0.145],
                "robot_pickup_pose": {
                    "x": 13.7024206974, "y": 17.2435347303, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.731770999702, "ow": 0.681550587994
                },
                "robot_place_pose": {
                    "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
                },
                "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
                "Z_offset_for_place": 0.07,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0             
            },
            {
                "object_name": "plate",
                "object_target_xy": [-0.5, -1.01],
                "offset": [0.10, 0.07, 0.155],
                "robot_pickup_pose": {
                    "x": 13.9134634579, "y": 11.2350456748, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.557658607205, "ow": 0.830070405333                   
                },
                "robot_place_pose": {
                    "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
                },
                "place_joints": [0.070839, -0.78701, 1.41627, -2.19909, -1.57012, 0.060687],
                "Z_offset_for_place": 0.15,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0             
            },
            {
                "object_name": "cup",
                "object_target_xy": [-0.5, -1.01],
                "offset": [0.08, 0.04, 0.09],
                "robot_pickup_pose": {
                    "x": 13.9134634579, "y": 11.2350456748, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.557658607205, "ow": 0.830070405333                   
                },
                "robot_place_pose": {
                    "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
                },
                "place_joints": [0.07069, -0.86433, 1.37961, -2.08519, -1.56995, 0.060842],
                "Z_offset_for_place": 0.05,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0             
            },
            {
                "object_name": "med",
                "object_target_xy": [-0.5, -1.01],
                "offset": [0.03, 0.05, 0.11],
                "robot_pickup_pose": {
                    "x": 13.9134634579, "y": 11.2350456748, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.557658607205, "ow": 0.830070405333                   
                },
                "robot_place_pose": {
                    "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
                },
                "place_joints": [0.07069, -0.86433, 1.37961, -2.08519, -1.56995, 0.060842],
                "Z_offset_for_place": 0.07,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0             
            },
            {
                "object_name": "box",
                "object_target_xy": [-0.5, -1.01],
                "offset": [0.09, 0.01, 0.11],
                "robot_pickup_pose": {
                    "x": 13.9134634579, "y": 11.2350456748, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": 0.557658607205, "ow": 0.830070405333                   
                },
                "robot_place_pose": {
                    "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
                    "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
                },
                "place_joints": [0.07069, -0.86433, 1.37961, -2.08519, -1.56995, 0.060842],
                "Z_offset_for_place": 0.1,
                "X_offset_for_place": 0,
                "Y_offset_for_place": 0             
            },
            # {
            #     "object_name": "strawberry",
            #     "object_target_xy": [0.8, -0.28],
            #     "object_offset": [0.04, 0.03, 0.00],
            #     "robot_pickup_pose": {
            #         #"x": 13.8940597304, "y": 11.1612879544, "z": 0.0,
            #         #"ox": 0.0, "oy": 0.0, "oz": 0.567748891155, "ow": 0.82320179579
            #         "x": 13.9134634579, "y": 11.2350456748, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": 0.557658607205, "ow": 0.830070405333
            #     },
            #    "robot_place_pose": {
            #         "x": 14.4971860667, "y": 10.4828109431, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": -0.846710933116, "ow": 0.532053188827
            #     },
            #     "place_joints": [-0.00966, -0.48169, 0.54697, -1.67472, -1.56924, -0.019603],
            #     "Z_offset_for_place": 0.09,
            #     "X_offset_for_place": 0,
            #     "Y_offset_for_place": 0
            # },
            # {
            #     "object_name": "fork",
            #     "object_target_xy": [0.8, -0.28],
            #     "object_offset": [0.02, 0.0, 0.15],
            #     "robot_pickup_pose": {
            #         "x": 13.5835104639, "y": 9.35141625721, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": 0.980466346739, "ow": 0.196686915966
            #     },
            #     "robot_place_pose": {
            #         "x": 14.8291768597, "y": 8.9452086359, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": -0.248337862099, "ow": 0.968673477622
            #     },
            #     "place_joints": [-0.00968, -0.48178, 0.54697, -1.67470, -1.56924, -1.375010],
            #     "Z_offset_for_place": 0.09,
            #     #"X_offset_for_place": -0.10,
            #     "X_offset_for_place": 0.0,
            #     "Y_offset_for_place": 0.0
            # },
            # {
            #     "object_name": "knife",
            #     "object_target_xy": [0.8, -0.28],
            #     "object_offset": [0.02, 0.0, 0.16],
            #     "robot_pickup_pose": {
            #         "x": 14.3189980668, "y": 9.02423468491, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": 0.982866854101, "ow": 0.184316974554
            #     },
            #     "robot_place_pose": {
            #         "x": 15.7942009642, "y": 11.2211163298, "z": 0.0,
            #         "ox": 0.0, "oy": 0.0, "oz": 0.966382899495, "ow": 0.25710715969
            #     },
            #     "place_joints": [-0.00968, -0.48178, 0.54697, -1.67470, -1.56924, 1.375010],
            #     "Z_offset_for_place": 0.09,
            #     #"X_offset_for_place": 0.10,
            #     "X_offset_for_place": 0.0,
            #     "Y_offset_for_place": 0.0
            # }    
        ]
        

        # 1: plate, 2: cup, 3: medicine 4:box
        

        arm.pickup(info[4]) #pickup a plate
        #rospy.sleep(2)
        #arm.place(info[1])
        arm.place_to_marker(info[4])
        

    except rospy.ROSInterruptException:

        rospy.loginfo("nav node terminated")
