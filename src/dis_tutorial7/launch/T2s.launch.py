from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_dis_tutorial7 = get_package_share_directory('dis_tutorial7')

    # Arguments
    arguments = [
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('world', default_value='bird_demo1', description='Simulation world'),
        DeclareLaunchArgument('model', default_value='standard', description='Turtlebot4 model'),
        DeclareLaunchArgument('map', default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'map.yaml']), description='Map YAML')
    ]

    # Step 0: Base simulation and nav2 (disable default RViz inside it)
    sim_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial7, 'launch', 'sim_turtlebot_nav.launch.py'])
        ),
        launch_arguments={'rviz': 'false'}.items()
    )

    arm_mover = TimerAction(
        period=1.0,
        actions=[Node(
            package='dis_tutorial7',
            executable='arm_mover_actions.py',
            name='arm_mover_node',
            output='screen'
        )]
    )

    # RViz2 node with your custom config
    rviz_config_path = PathJoinSubstitution([pkg_dis_tutorial3, 'rviz', 'exported_config.rviz'])
    rviz_node = TimerAction(
    period=5.0,
    actions=[Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )]
)

    return LaunchDescription(arguments + [
        sim_base,
        arm_mover,
        rviz_node
    ])
