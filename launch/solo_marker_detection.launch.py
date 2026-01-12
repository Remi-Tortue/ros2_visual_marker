import launch
import os
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

ROS_DISTRO_HUMBLE = "Humble"
ROS_DISTRO = os.environ.get("ROS_DISTRO")



def generate_launch_description():

    declared_arguments = []
    declared_arguments.append(DeclareLaunchArgument("input_image", default_value="/zed/zed_node/rgb/image_rect_color"))
    declared_arguments.append(DeclareLaunchArgument("input_camera_info", default_value="/zed/zed_node/depth/camera_info"))
    declared_arguments.append(DeclareLaunchArgument("output_pose", default_value="/marker_detection"))
    declared_arguments.append(DeclareLaunchArgument("marker_dict", default_value="DICT_7X7_50"))
    declared_arguments.append(DeclareLaunchArgument("marker_length", default_value="0.1"))
    declared_arguments.append(DeclareLaunchArgument("marker_id", default_value="0")) 


    return launch.LaunchDescription(declared_arguments+[
        Node(
            package='ros2_visual_marker',
            executable='solo_marker_detection',
            output='log',
            parameters=[
                {'input_image': LaunchConfiguration('input_image'),
                 'input_camera_info': LaunchConfiguration('input_camera_info'),
                 'output_pose': LaunchConfiguration('output_pose'),
                 'marker_dict': LaunchConfiguration('marker_dict'),
                 'marker_length': LaunchConfiguration('marker_length'),
                 'marker_id': LaunchConfiguration('marker_id'),
                 }
            ]
        ),
    ])