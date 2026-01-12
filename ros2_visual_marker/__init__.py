import cv2 as cv
import numpy as np

marker_dictionarys = {
    "DICT_4X4_50":        cv.aruco.DICT_4X4_50,
    "DICT_4X4_100":       cv.aruco.DICT_4X4_100,
    "DICT_4X4_250":       cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000":      cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50":        cv.aruco.DICT_5X5_50,
    "DICT_5X5_100":       cv.aruco.DICT_5X5_100,
    "DICT_5X5_250":       cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000":      cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50":        cv.aruco.DICT_6X6_50,
    "DICT_6X6_100":       cv.aruco.DICT_6X6_100,
    "DICT_6X6_250":       cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000":      cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50":        cv.aruco.DICT_7X7_50,
    "DICT_7X7_100":       cv.aruco.DICT_7X7_100,
    "DICT_7X7_250":       cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000":      cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL":cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

def rot2quat(R):
    """
    R       -- rotation matrix 3x3
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    R11, R12, R13 = R[0, 0], R[0, 1], R[0, 2]
    R21, R22, R23 = R[1, 0], R[1, 1], R[1, 2]
    R31, R32, R33 = R[2, 0], R[2, 1], R[2, 2]

    # Compute quaternion components
    q_w = 0.5 * np.sqrt(max(1 + R11 + R22 + R33, 0))
    q_x = 0.5 * np.sign(R32 - R23) * np.sqrt(max(1 + R11 - R22 - R33, 0))
    q_y = 0.5 * np.sign(R13 - R31) * np.sqrt(max(1 - R11 + R22 - R33, 0))
    q_z = 0.5 * np.sign(R21 - R12) * np.sqrt(max(1 - R11 - R22 + R33, 0))

    quaternion = np.array([q_w, q_x, q_y, q_z])

    return quaternion