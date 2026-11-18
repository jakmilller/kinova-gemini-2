"""Tune the drink (cup) attach sequence, one step at a time.

Steps (run --start M through --step N, inclusive):

    1  (move to the CUP_PRESET joint preset if it is set in config.yaml, then) detect
       ArUco marker MARKER_ID and estimate its full pose with solvePnP; popup shows the
       marker axes and the approach point
    2  close the gripper to GRIPPER_CLOSE
    3  move to the approach point (marker center + APPROACH_OFFSET_MARKER, in the marker
       frame) with the tool z pointing into the marker face (z axes point at each other),
       keeping the roll the arm already has from the cup preset
    4  move straight APPROACH_DIST along the tool axis
    5  open the gripper by GRIPPER_OPEN_BY
    6  move up LIFT_Z in base_link z, keeping the current orientation

    python scratch/drink_test.py --step 1      # detect only (check the popup)
    python scratch/drink_test.py --step 3      # ... through the aligned approach point
    python scratch/drink_test.py               # full sequence
    python scratch/drink_test.py --start 4     # resume after step 3 (no detection needed)

--start must be a step that doesn't depend on an earlier detection (anything but 3).

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
from scipy.spatial.transform import Rotation
from std_msgs.msg import Empty

from aruco import depth_center, detect, fmt, to_base
from gemini_agent.camera import Camera
from gemini_agent.config import load_config
from gemini_agent.robot import Robot

# ---------------------------------------------------------------- TUNING ----
MARKER_ID = 4                  # DICT_4X4_50 id on the cup
MARKER_SIZE = 0.045            # marker edge length, m (black border to black border)
CUP_PRESET = 'cup'             # step 1, preset the cup is visible from; skipped if None
                               # or empty in config.yaml (detect from the current pose)
TOOL_LENGTH = 0.0              # TCP length set before moving (0 = bare gripper tip)

# approach point = marker center + this offset, in meters, in the MARKER frame:
#   +x = marker right, +y = marker up (as printed), +z = out of the marker face
# so z is the standoff in front of the face and x/y slide along the face.
APPROACH_OFFSET_MARKER = (0.01, 0.045, 0.01)

APPROACH_DIST = 0.05           # step 4, m along the tool axis (+ = forward, into the marker)
GRIPPER_CLOSE = 0.97           # step 2, absolute gripper position (0 open .. 1 closed)
GRIPPER_OPEN_BY = 0.25         # step 5, subtracted from the current gripper position
LIFT_Z = 0.15                  # step 6, m added to base_link z
# ----------------------------------------------------------------------------

PICS = '/home/mcrr-lab/kinova-gemini-2/scratch/pics'
NUM_STEPS = 6
START_OK = (1, 2, 4, 5, 6)   # steps that don't need an earlier detection


def check(res, label):
    print(f'  {label}: {res.status} {res.message}'.rstrip())
    return res.ok


def K_matrix(snap):
    fx, fy, cx, cy = snap.K
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def project(p, K):
    """Camera-frame point -> integer pixel."""
    uv = K @ (np.asarray(p) / p[2])
    return int(round(uv[0])), int(round(uv[1]))


def marker_rotation(corners, snap):
    """Marker orientation in the camera frame (3x3, columns = marker x/y/z) via solvePnP,
    with +z guaranteed to point out of the face toward the camera. -> (R, rvec, tvec) or None."""
    h = MARKER_SIZE / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    K = K_matrix(snap)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    if R_cam_marker[2, 2] > 0:   # normal points away from the camera: flip about marker x
        R_cam_marker = R_cam_marker @ np.diag([1.0, -1.0, -1.0])
    return R_cam_marker, rvec, tvec


def show_detection(snap, corners, rvec, tvec, target_cam, no_popup):
    """Draw marker axes + approach point, save the image, wait for a key. False = abort."""
    out = cv2.cvtColor(snap.rgb, cv2.COLOR_RGB2BGR)
    K = K_matrix(snap)
    cv2.aruco.drawDetectedMarkers(out, [corners.reshape(1, 4, 2)], np.array([[MARKER_ID]]))
    cv2.drawFrameAxes(out, K, None, rvec, tvec, MARKER_SIZE)

    m_px = tuple(int(round(c)) for c in corners.mean(axis=0))
    if target_cam[2] > 0:
        a_px = project(target_cam, K)
        cv2.line(out, m_px, a_px, (255, 255, 255), 1)
        cv2.circle(out, a_px, 5, (0, 255, 0), -1)
        cv2.putText(out, 'approach', (a_px[0] + 8, a_px[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, 'any key: continue   q/esc: abort', (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    path = f'{PICS}/drink_detect_{int(time.time())}.jpg'
    cv2.imwrite(path, out)
    print(f'  image: {path}')

    if no_popup:
        return True
    win = 'drink_test: detection'
    cv2.imshow(win, out)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return key not in (ord('q'), 27)


async def detect_marker(robot, camera, cfg, no_popup):
    """Optionally park at CUP_PRESET, then -> (marker center (3,), R_base_marker 3x3) in
    base_link, or None."""
    if CUP_PRESET and (cfg.get('presets') or {}).get(CUP_PRESET):
        print(f'  moving to preset {CUP_PRESET!r}')
        if not check(await robot.move_to_preset(CUP_PRESET), 'move_to_joints'):
            return None
    else:
        print(f'  preset {CUP_PRESET!r} not set - detecting from the current pose')

    snap = await camera.fresh_snapshot()
    corners, found = detect(snap.rgb, MARKER_ID)
    if corners is None:
        print(f'  marker {MARKER_ID} not found (saw ids {found})')
        return None
    center_cam = depth_center(corners, snap)
    if center_cam is None:
        print('  marker found but no valid depth on it')
        return None
    pnp = marker_rotation(corners, snap)
    if pnp is None:
        print('  solvePnP failed')
        return None
    R_cam_marker, rvec, tvec = pnp

    R_base_cam = snap.T_base_proj[:3, :3]
    R_base_marker = R_base_cam @ R_cam_marker
    center_base = to_base(center_cam, snap)
    target_cam = center_cam + R_cam_marker @ np.array(APPROACH_OFFSET_MARKER)

    normal = R_base_marker[:, 2]
    tilt = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
    print(f'  marker  cam {fmt(center_cam)}  base {fmt(center_base)}')
    print(f'  pnp center cam {fmt(tvec.flatten())}  '
          f'(depth - pnp range {np.linalg.norm(center_cam) - np.linalg.norm(tvec):+.4f} m)')
    print(f'  marker normal (base) {fmt(normal)}  ({tilt:.1f} deg from base +z)')
    if not show_detection(snap, corners, rvec, tvec, target_cam, no_popup):
        print('  aborted')
        return None
    return center_base, R_base_marker


def aligned_pose(robot, center_base, R_base_marker):
    """Approach point + Euler angles (Kortex extrinsic XYZ, deg): the current orientation,
    tilted by the smallest rotation that points tool z at -marker z. Roll is left alone."""
    pose = robot.current_pose()
    R_cur = Rotation.from_euler('xyz', pose[3:], degrees=True)
    z_cur, z_tgt = R_cur.as_matrix()[:, 2], -R_base_marker[:, 2]
    axis = np.cross(z_cur, z_tgt)
    angle = np.arctan2(np.linalg.norm(axis), np.dot(z_cur, z_tgt))
    tilt = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-9) * angle)
    position = center_base + R_base_marker @ np.array(APPROACH_OFFSET_MARKER)
    theta = (tilt * R_cur).as_euler('xyz', degrees=True)
    return pose, position, theta


async def approach(robot, found):
    center_base, R_base_marker = found
    pose, position, theta = aligned_pose(robot, center_base, R_base_marker)
    print(f'  current tool {fmt(pose[:3])}  theta {fmt(pose[3:])}')
    print(f'  target  tool {fmt(position)}  theta {fmt(theta)}')
    print(f'  delta {fmt(position - np.array(pose[:3]))}')
    if not check(await robot.move_to_pose(*position, *theta), 'move_to_pose'):
        return False
    reached = robot.current_pose()
    z_now = Rotation.from_euler('xyz', reached[3:], degrees=True).as_matrix()[:, 2]
    err = np.degrees(np.arccos(np.clip(np.dot(z_now, -R_base_marker[:, 2]), -1, 1)))
    print(f'  reached {fmt(reached[:3])}  tool z vs -marker z: {err:.1f} deg')
    return True


async def gripper_by(robot, delta):
    """Move the gripper by delta (0 open .. 1 closed), clamped."""
    current = float(robot.gripper_position())
    target = min(1.0, max(0.0, current + delta))
    print(f'  {current:.3f} -> {target:.3f}')
    return check(await robot.command_gripper(target), 'command_gripper')


async def lift(robot, dz):
    pose = robot.current_pose()
    target = np.array(pose[:3]) + np.array([0.0, 0.0, dz])
    print(f'  {fmt(pose[:3])} -> {fmt(target)}')
    return check(await robot.move_to_pose(*target, *pose[3:]), 'move_to_pose')


async def run(robot, camera, cfg, args):
    found = {}

    async def detect_into():
        found['marker'] = await detect_marker(robot, camera, cfg, args.no_popup)
        return found['marker'] is not None

    steps = {
        1: (f'detect marker {MARKER_ID}', detect_into, None),
        2: (f'close gripper to {GRIPPER_CLOSE:.2f}',
            lambda: robot.command_gripper(GRIPPER_CLOSE), 'command_gripper'),
        3: ('move to aligned approach point', lambda: approach(robot, found['marker']), None),
        4: (f'move straight {APPROACH_DIST:+.3f} m',
            lambda: robot.move_straight(APPROACH_DIST), 'move_straight'),
        5: (f'open gripper by {GRIPPER_OPEN_BY:.2f}',
            lambda: gripper_by(robot, -GRIPPER_OPEN_BY), None),
        6: (f'move up {LIFT_Z:+.3f} m', lambda: lift(robot, LIFT_Z), None),
    }

    print(f'set tool length {TOOL_LENGTH:.3f} m')
    if not check(await robot.set_tool_length(TOOL_LENGTH), 'set_tool_length'):
        return

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
    node = Node('drink_test')
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
        asyncio.run(run(robot, camera, cfg, args))
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
