# Adapté de rtabmap_ros/rtabmap_demos/launch/find_object_demo.launch.py
# Remplace le nœud find_object_2d (SIFT/SURF classique) par notre
# transformer_object_detector (segmentation RGB-D par Transformer multimodal).
#
# Requirements:
#
# Example:
#
#   SLAM:
#     $ ros2 launch transformer_object_detector transformer_find_object_demo.launch.py
#
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.actions import SetParameter
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    localization = LaunchConfiguration('localization')

    parameters={
          'frame_id':'base_footprint',
          'odom_frame_id':'odom',
          'odom_tf_linear_variance':0.001,
          'odom_tf_angular_variance':0.001,
          'subscribe_rgbd':True,
          'subscribe_scan':False,
          'approx_sync':True,
          'sync_queue_size': 10,
          'RGBD/NeighborLinkRefining': 'true',
          'Reg/Strategy':              '1',
          'Reg/Force3DoF':             'true',
    }

    remappings=[
         ('rgb/image',       '/camera/image_raw'),
         ('depth/image',     '/camera/depth/image_raw'),
         ('rgb/camera_info', '/camera/camera_info')]

    config_rviz = os.path.join(
        get_package_share_directory('rtabmap_demos'), 'config', 'demo_robot_mapping.rviz'
    )

    # Chemin par défaut vers le modèle exporté par ce package
    default_model_path = os.path.join(
        get_package_share_directory('transformer_object_detector'), 'models', 'multimodal_transformer.pt'
    )

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument('rtabmap_viz',  default_value='false',  description='Launch RTAB-Map UI (optional).'),
        DeclareLaunchArgument('rviz',         default_value='true',   description='Launch RVIZ (optional).'),
        DeclareLaunchArgument('localization', default_value='false',  description='Launch in localization mode.'),
        DeclareLaunchArgument('rviz_cfg', default_value=config_rviz,  description='Configuration path of rviz2.'),
        DeclareLaunchArgument('model_path', default_value=default_model_path,
                               description='Chemin vers multimodal_transformer.pt'),

        SetParameter(name='use_sim_time', value=True),

        # Nodes to launch

        # Uncompress images
       

        Node(
            package='rtabmap_sync', executable='rgbd_sync', output='screen',
            parameters=[parameters,
              {'approx_sync_max_interval': 0.02}],
            remappings=remappings),

        # SLAM mode:
        Node(
            condition=UnlessCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=['-d']),

        # Localization mode:
        Node(
            condition=IfCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters,
              {'Mem/IncrementalMemory':'False',
               'Mem/InitWMWithAllNodes':'True'}],
            remappings=remappings),

        # Visualization:
        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            parameters=[parameters],
            remappings=remappings),
        Node(
            package='rviz2', executable='rviz2', name="rviz2", output='screen',
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=[["-d"], [LaunchConfiguration("rviz_cfg")]]),

        # --- Remplacement de find_object_2d par notre détecteur Transformer ---
        Node(
            package='transformer_object_detector', executable='detector_node', output='screen',
            parameters=[{'model_path': LaunchConfiguration('model_path'),
                         'device': 'cuda',
                         'rgb_topic': 'rgb/image_rect_color',
                         'depth_topic': 'depth_registered/image_raw',
                         'camera_info_topic': 'depth_registered/camera_info',
                         'sync_slop': 0.3}],
            remappings=[
    ('rgb/image_rect_color', '/camera/image_raw'),
    ('depth_registered/image_raw', '/camera/depth/image_raw'),
    ('depth_registered/camera_info', '/camera/camera_info'),
]),
    ])
