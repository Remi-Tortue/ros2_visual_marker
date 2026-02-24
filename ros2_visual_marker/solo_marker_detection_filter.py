#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from . import marker_dictionarys, rot2quat, dist_quat, quat2rot


#


class MarkerDetectionFilter(Node):
    def __init__(self) -> None:
        super().__init__('solo_visual_marker_detection_filter')

        self._window_size = 20  # You can adjust the window size
        self._pose_history = []
        self.__max_translation_jump = 0.3  # meters, adjust as needed
        self.__max_rotation_jump = 0.2   # quaternion distance, adjust as needed
        
        input_pose = self.declare_parameter("input_pose", "/marker_detection", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        output_pose = self.declare_parameter("output_pose", "/marker_detection/filtered", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value

        ####
        self.__sub_marker_pose = self.create_subscription(PoseStamped, input_pose, self.__callback_marker_pose, 10)
        self.__pub_filtered_marker_pose = self.create_publisher(PoseStamped, output_pose, 10)

        self.get_logger().info("Marker detection node initialized")

    #


    def __callback_marker_pose(self, msg: PoseStamped):
        t = np.array([
            msg.pose.position.x, 
            msg.pose.position.y, 
            msg.pose.position.z
            ])
        quaternion = np.array([
            msg.pose.orientation.w, 
            msg.pose.orientation.x, 
            msg.pose.orientation.y, 
            msg.pose.orientation.z
            ])

        if len(self._pose_history) == self._window_size:
            # Filter out absurd values (values too far from current ones)
            last_t, last_quat = self._pose_history[-1]
            t_dist = np.linalg.norm(t - last_t)
            # quat_dist = dist_quat(quaternion, last_quat)
            # quat_dist = np.arccos(np.clip((np.trace(R @ quat2rot(last_quat).T) - 1)/2, -1, 1))
            quat_dist = np.linalg.norm(quaternion - last_quat)
            if t_dist > self.__max_translation_jump or quat_dist > self.__max_rotation_jump:
                random.shuffle(self._pose_history)
                self.get_logger().debug(f"Skipping outlier pose: translation jump {t_dist:.3f}, rotation jump {quat_dist:.3f}")
                return
            else :
                # Store the latest pose
                self._pose_history.append((t.copy(), quaternion.copy()))
                self._pose_history.pop(0)

            # Compute the average pose
            t_avg = np.mean([pose[0] for pose in self._pose_history], axis=0)
            quat_arr = np.array([pose[1] for pose in self._pose_history])
            quaternion_avg = np.mean(quat_arr, axis=0)
            quaternion_avg = quaternion_avg / np.linalg.norm(quaternion_avg)

            t = t_avg
            quaternion = quaternion_avg

            marker_pose = PoseStamped()
            marker_pose.header = msg.header

            marker_pose.pose.position.x = t[0]
            marker_pose.pose.position.y = t[1]
            marker_pose.pose.position.z = t[2]
            marker_pose.pose.orientation.x = quaternion[1]
            marker_pose.pose.orientation.y = quaternion[2]
            marker_pose.pose.orientation.z = quaternion[3]
            marker_pose.pose.orientation.w = quaternion[0]

            self.__pub_filtered_marker_pose.publish(marker_pose)

        else:
            # Store the latest pose
            self._pose_history = [(t.copy(), quaternion.copy())] * self._window_size
            # self._pose_history.append((t.copy(), quaternion.copy()))


    #
    

    def __enter__(self):
        while True:
            try:
                return self
            except RuntimeError as e:
                self.get_logger().warn(f"Error when initializing: {e}. Retrying...")


    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


#


def run_node(node: Node, args):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"User stopped {node.get_name()}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    with MarkerDetectionFilter() as node:
        run_node(node, args)

if __name__ == '__main__':

  main()