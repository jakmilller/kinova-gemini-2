"""Tune the tool-holster pickup sequence, one step at a time.

Steps (run up to and including --step N):

    1  move to the 'holster' joint preset
    2  detect ArUco marker MARKER_ID; popup shows the marker center and the estimated
       attachment point (marker center + ATTACH_OFFSET_CAM)
    3  move the tool to the attachment point raised HOVER_Z in base_link z, keeping the
       current orientation
    4  move straight APPROACH_DIST along the tool axis
    5  open the gripper by GRIPPER_OPEN_BY
    6  move straight RETREAT_DIST along the tool axis
    7  move to the 'user' joint preset

    python scratch/holster_test.py --step 2     # holster pose + detect, nothing else
    python scratch/holster_test.py --step 4     # ... through the first move_straight
    python scratch/holster_test.py              # full sequence

All tunable numbers are in the TUNING block below.
"""

import argparse
import asyncio
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty

from aruco import depth_center, detect, fmt, to_base
from gemini_agent.camera import Camera
from gemini_agent.config import load_config
from gemini_agent.robot import Robot

# ---------------------------------------------------------------- TUNING ----
MARKER_ID = 1                  # DICT_4X4_50 id on the tool holder
START_PRESET = 'holster'       # step 1, preset name in config.yaml
END_PRESET = 'user'            # step 7

# attachment point = marker center + this offset, in meters, in the camera
# (physical_realsense_link, optical convention) frame:
#   +x = image right, +y = image down, +z = away from the camera (depth)
ATTACH_OFFSET_CAM = (0.0, 0.06, 0.0)
HOVER_Z = 0.03                 # step 3, m added to the attachment point's base_link z

APPROACH_DIST = 0.07           # step 4, m along the tool axis (+ = forward)
GRIPPER_OPEN_BY = 0.10         # step 5, subtracted from the current gripper position (0 open .. 1 closed)
RETREAT_DIST = -0.25           # step 6, m along the tool axis (- = backward)
# ----------------------------------------------------------------------------

PICS = '/home/mcrr-lab/kinova-gemini-2/scratch/pics'
NUM_STEPS = 7


def check(res, label):
    print(f'  {label}: {res.status} {res.message}'.rstrip())
    return res.ok


def project(p, K):
    """Camera-frame point -> integer pixel."""
    fx, fy, cx, cy = K
    return int(round(fx * p[0] / p[2] + cx)), int(round(fy * p[1] / p[2] + cy))


def show_detection(snap, corners, marker_cam, attach_cam, no_popup):
    """Draw marker + attachment point, save the image, and wait for a key. False = abort."""
    out = cv2.cvtColor(snap.rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(out, [corners.reshape(1, 4, 2)], np.array([[MARKER_ID]]))

    m_px = project(marker_cam, snap.K)
    a_px = project(attach_cam, snap.K)
    cv2.line(out, m_px, a_px, (255, 255, 255), 1)
    cv2.circle(out, m_px, 5, (0, 0, 255), -1)
    cv2.circle(out, a_px, 5, (0, 255, 0), -1)
    cv2.putText(out, 'marker', (m_px[0] + 8, m_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, 'attach', (a_px[0] + 8, a_px[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, 'any key: continue   q/esc: abort', (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    path = f'{PICS}/holster_{int(time.time())}.jpg'
    cv2.imwrite(path, out)
    print(f'  image: {path}')

    if no_popup:
        return True
    win = 'holster_test: detection'
    cv2.imshow(win, out)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return key not in (ord('q'), 27)


async def run(robot, camera, args):
    last = args.step

    print(f'[1] move to preset {START_PRESET!r}')
    if not check(await robot.move_to_preset(START_PRESET), 'move_to_joints'):
        return
    if last < 2:
        return

    print(f'[2] detect marker {MARKER_ID}')
    snap = await camera.fresh_snapshot()
    corners, found = detect(snap.rgb, MARKER_ID)
    if corners is None:
        print(f'  marker {MARKER_ID} not found (saw ids {found})')
        return
    marker_cam = depth_center(corners, snap)
    if marker_cam is None:
        print('  marker found but no valid depth on it')
        return
    attach_cam = marker_cam + np.array(ATTACH_OFFSET_CAM)
    marker_base = to_base(marker_cam, snap)
    attach_base = to_base(attach_cam, snap)
    print(f'  marker  cam {fmt(marker_cam)}  base {fmt(marker_base)}')
    print(f'  attach  cam {fmt(attach_cam)}  base {fmt(attach_base)}  (offset {fmt(ATTACH_OFFSET_CAM)})')
    if not show_detection(snap, corners, marker_cam, attach_cam, args.no_popup):
        print('  aborted')
        return
    if last < 3:
        return

    pose = robot.current_pose()
    target = attach_base + np.array([0.0, 0.0, HOVER_Z])
    print(f'[3] move to attachment point +{HOVER_Z * 100:.1f} cm z: {fmt(target)}  theta {fmt(pose[3:])}')
    print(f'  current tool {fmt(pose[:3])}  delta {fmt(target - pose[:3])}')
    if not check(await robot.move_to_pose(*target, *pose[3:]), 'move_to_pose'):
        return
    print(f'  reached {fmt(robot.current_pose()[:3])}')
    if last < 4:
        return

    print(f'[4] move straight {APPROACH_DIST:+.3f} m')
    if not check(await robot.move_straight(APPROACH_DIST), 'move_straight'):
        return
    if last < 5:
        return

    current = float(robot._state.gripper_position)
    target = min(1.0, max(0.0, current - GRIPPER_OPEN_BY))
    print(f'[5] open gripper by {GRIPPER_OPEN_BY:.2f}: {current:.3f} -> {target:.3f}')
    if not check(await robot.command_gripper(target), 'command_gripper'):
        return
    if last < 6:
        return

    print(f'[6] move straight {RETREAT_DIST:+.3f} m')
    if not check(await robot.move_straight(RETREAT_DIST), 'move_straight'):
        return
    if last < 7:
        return

    print(f'[7] move to preset {END_PRESET!r}')
    check(await robot.move_to_preset(END_PRESET), 'move_to_joints')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--step', type=int, default=NUM_STEPS, choices=range(1, NUM_STEPS + 1),
                   help='last step to run (default: all)')
    p.add_argument('--no-popup', action='store_true', help='save the detection image without showing it')
    args = p.parse_args()

    cfg = load_config()

    rclpy.init()
    node = Node('holster_test')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = None
    try:
        robot = Robot(node, cfg)
        camera = Camera(node, cfg)
        missing = robot.wait_for_servers(5.0)
        if missing:
            print(f'action servers not available: {missing}', file=sys.stderr)
            return
        if not robot.wait_for_state(2.0):
            print('no RobotState received - is the controller running?', file=sys.stderr)
            return
        asyncio.run(run(robot, camera, args))
    except KeyboardInterrupt:
        print('\ninterrupted - stopping arm')
        if robot is not None:
            robot._stop_pub.publish(Empty())
            time.sleep(0.2)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
