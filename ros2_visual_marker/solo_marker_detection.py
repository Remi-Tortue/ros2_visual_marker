#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from . import marker_dictionarys, rot2quat


#


class MarkerDetection(Node):
    def __init__(self) -> None:
        super().__init__('solo_visual_merker_detection')

        input_image = self.declare_parameter("input_image", "/camera/rgb/image_raw", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        input_camera_info = self.declare_parameter("input_camera_info", "/camera/rgb/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        output_pose = self.declare_parameter("output_pose", "/marker_detection", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value

        marker_dict = self.declare_parameter("marker_dict", "DICT_APRILTAG_36h11", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        self.__marker_length = self.declare_parameter("marker_length", 0.05, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.__marker_id = self.declare_parameter("marker_id", 0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value

        dictionary = cv.aruco.getPredefinedDictionary(marker_dictionarys[marker_dict])
        parameters =  cv.aruco.DetectorParameters()
        self.__detector = cv.aruco.ArucoDetector(dictionary, parameters)

        self.__cv_bridge = CvBridge()
        self.cam_info = {
                    'fx': 0.,
                    'fy': 0.,
                    'cx': 0.,
                    'cy': 0.,
                    'k1': 0.,
                    'k2': 0.,
                    'p1': 0.,
                    'p2': 0.,
                    'k3': 0.
                }

        self.__initialized = False

        ####
        self.__sub_image = self.create_subscription(Image, input_image, self.__callback_image, 10)
        self.__sub_camera_info = self.create_subscription(CameraInfo, input_camera_info, self.__callback_camera_info, 10)
        self.__pub_marker_pose = self.create_publisher(PoseStamped, output_pose, 10)
        self.__pub_annotated_image = self.create_publisher(Image, input_image+"/visual_marker", 10)

        self.get_logger().info("Marker detection node initialized")

    #


    def __callback_camera_info(self, msg: CameraInfo):
        self.cam_info = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'k1': msg.d[0],
            'k2': msg.d[1],
            'p1': msg.d[2],
            'p2': msg.d[3],
            'k3': msg.d[4]
        }
        self.__initialized = True


    def __callback_image(self, msg: Image):
        image = self.__cv_bridge.imgmsg_to_cv2(msg)
        (corners, ids, rejected) = self.__detector.detectMarkers(image)

        if not ids is None:
            for i, marker_id in enumerate(ids):
                if self.__marker_id == int(marker_id):
                    obj_points = np.array([
                            [-self.__marker_length/2,  self.__marker_length/2, 0],
                            [ self.__marker_length/2,  self.__marker_length/2, 0],
                            [ self.__marker_length/2, -self.__marker_length/2, 0],
                            [-self.__marker_length/2, -self.__marker_length/2, 0]
                        ], dtype=np.float32)
                        
                    img_points = corners[i].reshape(-1, 2)
                    cameraMatrix = np.array([
                        [self.cam_info['fx'], 0, self.cam_info['cx']],
                        [0, self.cam_info['fy'], self.cam_info['cy']],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    distCoeffs = np.array([self.cam_info['k1'], self.cam_info['k2'], self.cam_info['p1'], self.cam_info['p2'], self.cam_info['k3']], dtype=np.float32)
                    success, rvec, tvec = cv.solvePnP(
                        obj_points,
                        img_points,
                        cameraMatrix,
                        distCoeffs,
                        flags=cv.SOLVEPNP_IPPE_SQUARE
                        )
                    R, _ = cv.Rodrigues(rvec)
                    t = tvec.flatten()
                    quaternion = rot2quat(R)

                    marker_pose = PoseStamped()
                    marker_pose.header.frame_id = msg.header.frame_id
                    marker_pose.pose.position.x = t[0]
                    marker_pose.pose.position.y = t[1]
                    marker_pose.pose.position.z = t[2]
                    marker_pose.pose.orientation.x = quaternion[0]
                    marker_pose.pose.orientation.y = quaternion[1]
                    marker_pose.pose.orientation.z = quaternion[2]
                    marker_pose.pose.orientation.w = quaternion[3]

                    self.__pub_marker_pose.publish(marker_pose)

                    annotated_image = image.copy()
                    cv.aruco.drawDetectedMarkers(annotated_image, corners, ids)
                    annotated_image_msg = self.__cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                    annotated_image_msg.header = msg.header
                    self.__pub_annotated_image.publish(annotated_image_msg)


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
    with MarkerDetection() as node:
        run_node(node, args)

if __name__ == '__main__':

  main()