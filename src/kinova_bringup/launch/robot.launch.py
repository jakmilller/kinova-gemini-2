from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# this launches the robot description, realSense, and custom controller
def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.77.10',
        description='IP address of the robot'
    )

    username_arg = DeclareLaunchArgument(
        'username',
        default_value='admin',
        description='Username for robot connection'
    )
    
    password_arg = DeclareLaunchArgument(
        'password',
        default_value='admin',
        description='Password for robot connection'
    )
    
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='gen3',
        description='Type/series of robot'
    )
    
    dof_arg = DeclareLaunchArgument(
        'dof',
        default_value='7',
        description="Robot's degrees of freedom"
    )
    
    gripper_arg = DeclareLaunchArgument(
        'gripper',
        default_value='robotiq_2f_140',
        description='Gripper type'
    )

    # Boolean feature toggles (pass e.g. rviz:=false voice:=true on the command line)
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Launch RViz'
    )

    # robot description direct from pdf
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("kortex_description"), "robots", "kinova.urdf.xacro"]
            ),
            " ",
            "robot_ip:=", LaunchConfiguration('robot_ip'),
            " ",
            "name:=kinova",
            " ",
            "arm:=", LaunchConfiguration('robot_type'),
            " ",
            "gripper:=", LaunchConfiguration('gripper'),
            " ",
            "dof:=", LaunchConfiguration('dof'),
            " ",
            "vision:=false",
            " ",
            "use_fake_hardware:=false",
        ]
    )
    
    robot_description = {"robot_description": robot_description_content}

    # publish transform and description
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

# camera transform, according to OnShape assembly (end effector origin to left depth camera origin)
    # realsense_transform = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     arguments=[
    #         '0.0175',  # x offset (offsets calculated from OnShape assembly)
    #         '0.0635',    # y offset
    #         '-0.004234',   # z offset
    #         '-1.5708',    # roll
    #         '-1.5708',    # pitch
    #         '0.0',    # yaw
    #         'end_effector_link',  # kinova camera frame
    #         'realsense_link'  # external RealSense frame
    #     ],
    #     name='realsense_transform'
    # )

# from easy_handeye2 calibration

# transform:
#   translation:
#     x: 0.02007401496991271
#     y: 0.06197335285291799
#     z: -0.030076264611610615
#   rotation:
#     x: -0.48994649383326205
#     y: -0.5168790844934305
#     z: -0.4667655095515981
#     w: 0.5243266198532772

# both OnShape and easy_handeye2 calibration was slightly off. Manually calibrated using scratch/aruco_camera_calibration.py
    realsense_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.010074',  # x offset 
            '0.075',    # y offset
            '-0.004234',   # z offset
            '-1.5708',    # roll
            '-1.5708',    # pitch
            '0.0',    # yaw
            'end_effector_link',  # kinova camera frame
            'realsense_link'  # external RealSense frame
        ],
        name='realsense_transform'
    )

    # TCP frame for visualization; z must match the controller's bare_tool_z
    tcp_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--z', '0.22',
                   '--frame-id', 'end_effector_link',
                   '--child-frame-id', 'tcp'],
        name='tcp_transform'
    )

    # RViz for visualization
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("kinova_bringup"), "config", "robot.rviz"]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file],
        output="screen",
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    realsense_launch = GroupAction(
        scoped=True,
        forwarding=False,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('realsense2_camera'),
                        'launch',
                        'rs_launch.py'
                    ])
                ]),
                launch_arguments={
                    'align_depth.enable': 'true',
                    'pointcloud.enable': 'true',
                    'camera_name': 'realsense',
                }.items()
            )
        ]
    )

    # for web UI
    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('rosbridge_server'),
                'launch',
                'rosbridge_websocket_launch.xml'
            ])
        ])
    )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        output='screen'
    )

    controller_node = Node(
        package='kortex_controller',
        executable='controller',
        name='kinova_controller',
        output='screen',
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip'),
            'username': LaunchConfiguration('username'),
            'password': LaunchConfiguration('password'),
        }]
    )

    return LaunchDescription([
        robot_ip_arg,
        username_arg,
        password_arg,
        robot_type_arg,
        dof_arg,
        gripper_arg,
        rviz_arg,
        controller_node,
        robot_state_publisher,
        realsense_transform,
        tcp_transform,
        realsense_launch,
        rosbridge_launch,
        web_video_server_node,
        rviz_node,
    ])

