import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_model_path = os.path.join(
        get_package_share_directory('transformer_object_detector'), 'models', 'multimodal_transformer.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model_path),
        SetParameter(name='use_sim_time', value=True),
        Node(
            package='transformer_object_detector', executable='detector_node', output='screen',
            parameters=[{'model_path': LaunchConfiguration('model_path'),
                         'device': 'cuda',
                         'rgb_topic': '/camera/image_raw',
                         'depth_topic': '/camera/depth/image_raw',
                         'camera_info_topic': '/camera/camera_info',
                         'sync_slop': 0.3}]),
        Node(
            package='transformer_object_detector', executable='fusion_node', output='screen',
            parameters=[{'rgb_topic': '/camera/image_raw',
                         'depth_topic': '/camera/depth/image_raw',
                         'camera_info_topic': '/camera/camera_info',
                         'mask_topic': '/segmentation/mask',
                         'sync_slop': 0.3,
                         'stride': 2}]),
    ])
