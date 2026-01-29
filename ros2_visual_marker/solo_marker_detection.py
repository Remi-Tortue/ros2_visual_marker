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


class MarkerDetection(Node):
    def __init__(self) -> None:
        super().__init__('solo_visual_merker_detection')

        input_image = self.declare_parameter("input_image", "/camera/rgb/image_raw", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        input_camera_info = self.declare_parameter("input_camera_info", "/camera/rgb/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        output_pose = self.declare_parameter("output_pose", "/marker_detection", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value
        self.__output_frame = self.declare_parameter("output_frame", "", ParameterDescriptor(type=ParameterType.PARAMETER_STRING)).get_parameter_value().string_value

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
                    if not success:
                        self.get_logger().warn("Could not solve PnP for marker id {}".format(self.__marker_id))
                        continue

                    R, _ = cv.Rodrigues(rvec)
                    t = tvec.flatten()
                    quaternion = rot2quat(R)

                    # Apply a simple moving average filter to smooth the pose estimates
                    if not hasattr(self, '_pose_history'):
                        self._window_size = 10  # You can adjust the window size
                        self._pose_history = [(t.copy(), quaternion.copy())] * self._window_size
                        self.__max_translation_jump = 0.3  # meters, adjust as needed
                        self.__max_rotation_jump = 0.2   # quaternion distance, adjust as needed
                        # self.__max_translation_jump = 10e3
                        # self.__max_rotation_jump = 10e3

                    # Filter out absurd values (values too far from current ones)
                    last_t, last_quat = self._pose_history[-1]
                    t_dist = np.linalg.norm(t - last_t)
                    # quat_dist = dist_quat(quaternion, last_quat)
                    # quat_dist = np.arccos(np.clip((np.trace(R @ quat2rot(last_quat).T) - 1)/2, -1, 1))
                    quat_dist = np.linalg.norm(quaternion - last_quat)
                    if t_dist > self.__max_translation_jump or quat_dist > self.__max_rotation_jump:
                        # Instead of skipping, add a mean of the last pose and the absurd value to the history
                        # t_mean = (t*9 + last_t) / 10
                        # quat_mean = (quaternion*9 + last_quat) / 10
                        # quat_mean = quat_mean / np.linalg.norm(quat_mean)
                        # self._pose_history.append((t_mean.copy(), quat_mean.copy()))
                        # self._pose_history.pop(0)
                        random.shuffle(self._pose_history)
                        # Skip this pose as it's an outlier
                        self.get_logger().debug(f"Skipping outlier pose: translation jump {t_dist:.3f}, rotation jump {quat_dist:.3f}")
                        continue
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
                    marker_pose.header.stamp = msg.header.stamp
                    # self.get_logger().info("stamp: {}".format(marker_pose.header.stamp))
                    if self.__output_frame != "":
                        marker_pose.header.frame_id = self.__output_frame
                    else:
                        marker_pose.header.frame_id = msg.header.frame_id
                    marker_pose.pose.position.x = t[0]
                    marker_pose.pose.position.y = t[1]
                    marker_pose.pose.position.z = t[2]
                    marker_pose.pose.orientation.x = quaternion[1]
                    marker_pose.pose.orientation.y = quaternion[2]
                    marker_pose.pose.orientation.z = quaternion[3]
                    marker_pose.pose.orientation.w = quaternion[0]

                    self.__pub_marker_pose.publish(marker_pose)

                    annotated_image = image.copy()
                    cv.aruco.drawDetectedMarkers(annotated_image, corners, ids)

                    # Draw marker axes
                    cv.drawFrameAxes(
                        annotated_image,
                        cameraMatrix,
                        distCoeffs,
                        rvec,
                        tvec,
                        self.__marker_length * 0.5
                    )

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