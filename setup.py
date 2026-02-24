from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'ros2_visual_marker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rporee',
    maintainer_email='remi.poree@laas.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'solo_marker_detection = ros2_visual_marker.solo_marker_detection:main',
            'solo_marker_detection_filter = ros2_visual_marker.solo_marker_detection_filter:main',
        ],
    },
)
