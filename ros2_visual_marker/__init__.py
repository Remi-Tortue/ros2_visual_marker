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

def quat2rot(q):
    """
    q       -- quaternion with scalar as first element [qw qx qy qz]
    R       -- rotation matrix 3x3
    """
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    R11 = q_w * q_w + q_x * q_x - q_y * q_y - q_z * q_z
    R12 = 2 * (q_x * q_y - q_w * q_z)
    R13 = 2 * (q_x * q_z + q_w * q_y)
    R21 = 2 * (q_x * q_y + q_w * q_z)
    R22 = q_w * q_w - q_x * q_x + q_y * q_y - q_z * q_z
    R23 = 2 * (q_y * q_z - q_w * q_x)
    R31 = 2 * (q_x * q_z - q_w * q_y)
    R32 = 2 * (q_y * q_z + q_w * q_x)
    R33 = q_w * q_w - q_x * q_x - q_y * q_y + q_z * q_z

    R = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    return R

def norm_2(y):
    return np.sqrt(y.T@y)

def dist_quat(q1, q2):
    """Computes the scalar distance between two quaternions.
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    q1n = q1/ norm_2(q1)
    q2n = q2/ norm_2(q2)
    return 1 - abs(np.dot(q1n,q2n))


if __name__ == "__main__":
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([1, 1, 0, 0])
    dist = dist_quat(q1, q2)
    print(f'quat distance: {dist}')