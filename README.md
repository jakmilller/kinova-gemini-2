# kinova-gemini

Natural-language control of a Kinova Gen3 arm. A user types (or speaks, from a phone app) an instruction; a Gemini Live session sees the wrist camera, locates objects, and calls motion tools that drive the arm through a C++ Kortex controller.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Packages

| Package | Language | What it does |
|---|---|---|
| `src/ros2_interfaces` | msg/srv/action | Custom ROS 2 actions, service, and `RobotState` message |
| `src/kortex_controller` | C++ | Talks to the arm over the Kortex API and exposes it as ROS 2 actions |
| `src/gemini_agent` | Python | Gemini Live session, tools (`look`, `pick`, `move_to`, `attach`, ...), and perception (Gemini boxes + SAM2 masks + depth) |
| `src/app_interface` | Python | Phone-app transports (BLE, Bluetooth RFCOMM, TCP) that publish instructions |
| `src/kinova_bringup` | launch | Launch files for the robot stack and the full agent |
| `web/index.html` | HTML/JS | Browser UI: chat, model thoughts, tool calls, images the model saw, camera stream, robot state, stop button |

## Requirements

- ROS 2 with `realsense2_camera`, `kortex_description`, `robot_state_publisher`, `rosbridge_server`, `web_video_server`, `cv_bridge`, `message_filters`, `tf2_ros`
- Kinova Kortex C++ API, unpacked into `src/kortex_controller/kortex_api/` (`include/` and `lib/release/libKortexApiCpp.a`; not checked in)
- Python deps for the agent: `google-genai`, `python-dotenv`, `pyyaml`, `opencv-python`, `numpy`, `torch`, and `bleak` (BLE transport only)
- A local [SAM2](https://github.com/facebookresearch/sam2) checkout with a checkpoint (path set in `config.yaml` under `perception`)
- A CUDA GPU (for SAM2) and a Gemini API key

## Setup

```bash
# 1. API key
cp .env.example .env   # then put your key in .env

# 2. Build (from the repo root)
colcon build --symlink-install
source install/setup.bash

# optional - ensure entry points point to conda env
sed -i '1s|.*|#!/home/mcrr-lab/anaconda3/envs/kinova-gemini/bin/python3|' install/{gemini_agent/lib/gemini_agent,app_interface/lib/app_interface}/*
```

Edit `config.yaml` for your setup: model, enabled tools, SAM2 paths, camera topics, preset joint positions, and the `holster:` map of ArUco marker id to the physical tool in that slot.

## Running

Full stack (controller, camera, TF, rosbridge, and the Gemini agent):

```bash
ros2 launch kinova_bringup agent.launch.py robot_ip:=192.168.1.10
```

Then open `web/index.html` in a browser (it connects to rosbridge on `ws://localhost:9090`).

Launch arguments: `robot_ip`, `username`, `password` (default `192.168.1.10` / `admin` / `admin`), `rviz:=true`, and `app:=ble|bluetooth|network|none` to start a phone transport.

Robot stack only, without the agent:

```bash
ros2 launch kinova_bringup robot.launch.py
```

### Running a single tool by hand

`tool_cli` calls one agent tool without the model, which is useful for testing motion and perception:

```bash
ros2 run gemini_agent tool_cli --list
ros2 run gemini_agent tool_cli move_relative '{"dz": 0.03}'
ros2 run gemini_agent tool_cli look '{"objects": ["red cup"]}'
ros2 run gemini_agent tool_cli pick '{"x": 0.45, "y": 0.0, "z": 0.1, "width_mm": 60, "depth_mm": 70}'
ros2 run gemini_agent tool_cli attach '{"name": "fork"}'
```

Ctrl-C cancels the running robot goal.

### Stopping the arm

Publish to `/stop` (the web UI's stop button does this) to halt the arm immediately:

```bash
ros2 topic pub --once /stop std_msgs/msg/Empty
```
