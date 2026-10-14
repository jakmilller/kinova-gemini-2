"""Tune the tool-holster pickup and return sequences, one step at a time.

Steps (run --start M through --step N, inclusive):

    1  move to the 'holster' joint preset
    2  detect ArUco marker MARKER_ID; popup shows the marker center and the estimated
       attachment point (marker center + ATTACH_OFFSET_CAM)
    3  move the tool to the attachment point raised HOVER_Z in base_link z, keeping the
       current orientation
    4  move straight APPROACH_DIST along the tool axis
    5  open the gripper by GRIPPER_OPEN_BY
    6  move straight RETREAT_DIST along the tool axis
    7  move to the 'user' joint preset
    8  move to the 'holster' joint preset (start of the return; tool in the gripper)
    9  detect ArUco marker MARKER_ID; popup shows the marker center and the estimated
       detach point (marker center + DETACH_OFFSET_CAM)
   10  move the tool to the detach point raised DETACH_HOVER_Z in base_link z, keeping
       the current orientation
   11  move straight DETACH_APPROACH_DIST along the tool axis
   12  close the gripper by GRIPPER_OPEN_BY (undoes step 5)
   13  move straight DETACH_RETREAT_DIST along the tool axis

    python scratch/holster_test.py --step 2              # holster pose + detect, nothing else
    python scratch/holster_test.py --step 4              # ... through the first move_straight
    python scratch/holster_test.py --step 7              # pickup only
    python scratch/holster_test.py --start 8             # return only (tool already held)
    python scratch/holster_test.py --start 8 --step 9    # return: holster, detect
    python scratch/holster_test.py                       # full sequence

--start must be a step that doesn't depend on an earlier detection (1, 2, 7-9).

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
START_PRESET = 'holster'       # steps 1 and 8, preset name in config.yaml
END_PRESET = 'user'            # step 7

# attachment point = marker center + this offset, in meters, in the camera
# (physical_realsense_link, optical convention) frame:
#   +x = image right, +y = image down, +z = away from the camera (depth)
ATTACH_OFFSET_CAM = (0.0, 0.065, 0.0)
HOVER_Z = 0.03                 # step 3, m added to the attachment point's base_link z

APPROACH_DIST = 0.08          # step 4, m along the tool axis (+ = forward)
GRIPPER_OPEN_BY = 0.10         # step 5, subtracted from the current gripper position (0 open .. 1 closed)
                               # step 12, added back
RETREAT_DIST = -0.15           # step 6, m along the tool axis (- = backward)

# return sequence, same conventions as above
DETACH_OFFSET_CAM = (0.0, 0.05, 0.0)   # step 9, detach point = marker center + this (camera frame)
DETACH_HOVER_Z = 0.1          # step 10, m added to the detach point's base_link z
DETACH_APPROACH_DIST = 0.12    # step 11, m along the tool axis (+ = forward)
DETACH_RETREAT_DIST = -0.10    # step 13, m along the tool axis (- = backward)
# ----------------------------------------------------------------------------

PICS = '/home/mcrr-lab/kinova-gemini-2/scratch/pics'
NUM_STEPS = 13
START_OK = (1, 2, 7, 8, 9)   # steps that don't need an earlier detection


def check(res, label):
    print(f'  {label}: {res.status} {res.message}'.rstrip())
    return res.ok


def project(p, K):
    """Camera-frame point -> integer pixel."""
    fx, fy, cx, cy = K
    return int(round(fx * p[0] / p[2] + cx)), int(round(fy * p[1] / p[2] + cy))


def show_detection(snap, corners, marker_cam, attach_cam, label, no_popup):
    """Draw marker + target point, save the image, and wait for a key. False = abort."""
    out = cv2.cvtColor(snap.rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(out, [corners.reshape(1, 4, 2)], np.array([[MARKER_ID]]))

    m_px = project(marker_cam, snap.K)
    a_px = project(attach_cam, snap.K)
    cv2.line(out, m_px, a_px, (255, 255, 255), 1)
    cv2.circle(out, m_px, 5, (0, 0, 255), -1)
    cv2.circle(out, a_px, 5, (0, 255, 0), -1)
    cv2.putText(out, 'marker', (m_px[0] + 8, m_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, label, (a_px[0] + 8, a_px[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, 'any key: continue   q/esc: abort', (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    path = f'{PICS}/holster_{label}_{int(time.time())}.jpg'
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


async def detect_target(camera, offset_cam, label, no_popup):
    """Detect the marker and return the target point (marker + offset) in base_link, or None."""
    snap = await camera.fresh_snapshot()
    corners, found = detect(snap.rgb, MARKER_ID)
    if corners is None:
        print(f'  marker {MARKER_ID} not found (saw ids {found})')
        return None
    marker_cam = depth_center(corners, snap)
    if marker_cam is None:
        print('  marker found but no valid depth on it')
        return None
    target_cam = marker_cam + np.array(offset_cam)
    marker_base = to_base(marker_cam, snap)
    target_base = to_base(target_cam, snap)
    print(f'  marker  cam {fmt(marker_cam)}  base {fmt(marker_base)}')
    print(f'  {label:<7} cam {fmt(target_cam)}  base {fmt(target_base)}  (offset {fmt(offset_cam)})')
    if not show_detection(snap, corners, marker_cam, target_cam, label, no_popup):
        print('  aborted')
        return None
    return target_base


async def hover_move(robot, point_base, hover_z, label):
    """move_to_pose to point_base raised hover_z in base_link z, keeping the current orientation."""
    pose = robot.current_pose()
    target = point_base + np.array([0.0, 0.0, hover_z])
    print(f'  {label} point +{hover_z * 100:.1f} cm z: {fmt(target)}  theta {fmt(pose[3:])}')
    print(f'  current tool {fmt(pose[:3])}  delta {fmt(target - pose[:3])}')
    if not check(await robot.move_to_pose(*target, *pose[3:]), 'move_to_pose'):
        return False
    print(f'  reached {fmt(robot.current_pose()[:3])}')
    return True


async def gripper_by(robot, delta):
    """Move the gripper by delta (0 open .. 1 closed), clamped."""
    current = float(robot._state.gripper_position)
    target = min(1.0, max(0.0, current + delta))
    print(f'  {current:.3f} -> {target:.3f}')
    return check(await robot.command_gripper(target), 'command_gripper')


async def run(robot, camera, args):
    found = {}   # base_link target points from steps 2 ('attach') and 9 ('detach')

    async def detect_into(key, offset, label):
        found[key] = await detect_target(camera, offset, label, args.no_popup)
        return found[key] is not None

    steps = {
        1: (f'move to preset {START_PRESET!r}',
            lambda: robot.move_to_preset(START_PRESET), 'move_to_joints'),
        2: (f'detect marker {MARKER_ID}',
            lambda: detect_into('attach', ATTACH_OFFSET_CAM, 'attach'), None),
        3: ('move to attachment point',
            lambda: hover_move(robot, found['attach'], HOVER_Z, 'attach'), None),
        4: (f'move straight {APPROACH_DIST:+.3f} m',
            lambda: robot.move_straight(APPROACH_DIST), 'move_straight'),
        5: (f'open gripper by {GRIPPER_OPEN_BY:.2f}',
            lambda: gripper_by(robot, -GRIPPER_OPEN_BY), None),
        6: (f'move straight {RETREAT_DIST:+.3f} m',
            lambda: robot.move_straight(RETREAT_DIST), 'move_straight'),
        7: (f'move to preset {END_PRESET!r}',
            lambda: robot.move_to_preset(END_PRESET), 'move_to_joints'),
        8: (f'move to preset {START_PRESET!r}',
            lambda: robot.move_to_preset(START_PRESET), 'move_to_joints'),
        9: (f'detect marker {MARKER_ID}',
             lambda: detect_into('detach', DETACH_OFFSET_CAM, 'detach'), None),
        10: ('move to detach point',
             lambda: hover_move(robot, found['detach'], DETACH_HOVER_Z, 'detach'), None),
        11: (f'move straight {DETACH_APPROACH_DIST:+.3f} m',
             lambda: robot.move_straight(DETACH_APPROACH_DIST), 'move_straight'),
        12: (f'close gripper by {GRIPPER_OPEN_BY:.2f}',
             lambda: gripper_by(robot, GRIPPER_OPEN_BY), None),
        13: (f'move straight {DETACH_RETREAT_DIST:+.3f} m',
             lambda: robot.move_straight(DETACH_RETREAT_DIST), 'move_straight'),
    }

    for n in range(args.start, args.step + 1):
        desc, fn, action = steps[n]
        print(f'[{n}] {desc}')
        res = await fn()
        ok = check(res, action) if action else res
        if not ok:
            return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=int, default=1, choices=START_OK,
                   help='first step to run (default: 1)')
    p.add_argument('--step', type=int, default=NUM_STEPS, choices=range(1, NUM_STEPS + 1),
                   help='last step to run (default: all)')
    p.add_argument('--no-popup', action='store_true', help='save the detection image without showing it')
    args = p.parse_args()
    if args.step < args.start:
        p.error('--step must be >= --start')

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
