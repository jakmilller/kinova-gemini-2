"""Hover the gripper over an ArUco marker to check the camera's static transform.

Detects the marker in the wrist RealSense image, projects its center into base_link the
same way perception/localize.py does (optical-convention point -> T_base_proj), then
moves the tool to hover above it with move_to_pose, keeping the current orientation.
If the static transform is right, the gripper tip ends up directly over the marker center.

    python scratch/aruco.py                       # detect, confirm, move 10 cm above
    python scratch/aruco.py --hover 0.05 --yes    # 5 cm above, no prompt
    python scratch/aruco.py --size 0.05           # also report a solvePnP estimate (marker edge, m)
    python scratch/aruco.py --dry-run             # detect and report only

After the move it re-detects the marker from the new viewpoint. With a correct transform
both base-frame estimates agree; a large difference means the transform is off.
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

from gemini_agent.camera import Camera
from gemini_agent.config import load_config
from gemini_agent.robot import Robot


def detect(rgb, marker_id):
    """Pixel corners (4x2, TL/TR/BR/BL) of `marker_id`, or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, []
    found = [int(i) for i in ids.flatten()]
    for c, i in zip(corners, found):
        if i == marker_id:
            return c.reshape(4, 2), found
    return None, found


def depth_center(corners, snap):
    """Marker center in the projection frame from the depth image (what `look` does)."""
    mask = np.zeros(snap.depth_m.shape, np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 1)
    z = snap.depth_m[mask.astype(bool)]
    z = z[z > 0.0]
    if len(z) == 0:
        return None
    u, v = corners.mean(axis=0)
    fx, fy, cx, cy = snap.K
    d = float(np.median(z))
    return np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])


def pnp_center(corners, snap, size):
    """Marker center in the projection frame from solvePnP (independent of depth)."""
    h = size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    fx, fy, cx, cy = snap.K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (tvec.flatten(), rvec, K) if ok else None


def to_base(p, snap):
    return (snap.T_base_proj @ np.append(p, 1.0))[:3]


def fmt(v):
    return '[' + ', '.join(f'{x:+.4f}' for x in v) + ']'


async def measure(camera, args, label):
    """Snapshot, detect, and return the marker center in base_link (depth-based)."""
    snap = await camera.fresh_snapshot()
    corners, found = detect(snap.rgb, args.id)

    out = cv2.cvtColor(snap.rgb, cv2.COLOR_RGB2BGR)
    if corners is None:
        path = f'/home/mcrr-lab/kinova-gemini-2/scratch/pics/aruco_{label}_{int(time.time())}.jpg'
        cv2.imwrite(path, out)
        print(f'[{label}] marker {args.id} not found (saw ids {found}); image: {path}')
        return None

    cam = depth_center(corners, snap)
    if cam is None:
        print(f'[{label}] marker found but no valid depth on it')
        return None
    base = to_base(cam, snap)

    cv2.aruco.drawDetectedMarkers(out, [corners.reshape(1, 4, 2)], np.array([[args.id]]))
    print(f'[{label}] camera frame (depth): {fmt(cam)}')
    print(f'[{label}] base_link    (depth): {fmt(base)}')

    if args.size:
        pnp = pnp_center(corners, snap, args.size)
        if pnp is not None:
            t, rvec, K = pnp
            print(f'[{label}] base_link    (pnp):   {fmt(to_base(t, snap))}  '
                  f'(depth vs pnp range: {np.linalg.norm(cam) - np.linalg.norm(t):+.4f} m)')
            cv2.drawFrameAxes(out, K, None, rvec, t, args.size * 0.75)

    path = f'/home/mcrr-lab/kinova-gemini-2/scratch/pics/aruco_{label}_{int(time.time())}.jpg'
    cv2.imwrite(path, out)
    print(f'[{label}] image: {path}')
    return base


async def run(robot, camera, args):
    before = await measure(camera, args, 'before')
    if before is None or args.dry_run:
        return

    pose = robot.current_pose()
    if pose is None:
        print('no RobotState - is the controller running?')
        return

    target = (before[0], before[1], before[2] + args.hover)
    print(f'\ncurrent tool: {fmt(pose[:3])}  theta {fmt(pose[3:])}')
    print(f'target tool:  {fmt(target)}  ({args.hover * 100:.0f} cm above marker, same orientation)')
    if not args.yes and input('move? [y/N] ').strip().lower() != 'y':
        return

    res = await robot.move_to_pose(*target, *pose[3:])
    print(f'move_to_pose: {res.status} {res.message}')
    if not res.ok:
        return

    reached = robot.current_pose()
    print(f'reached tool: {fmt(reached[:3])}')
    print('-> check by eye/ruler that the gripper tip is centered over the marker')

    after = await measure(camera, args, 'after')
    if after is not None:
        d = after - before
        print(f'\nre-measure delta (after - before): {fmt(d)}  |d| = {np.linalg.norm(d) * 1000:.1f} mm')
        print('a consistent transform gives a small delta; a large one points to a rotation '
              'or offset error in the static transform')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--id', type=int, default=1, help='marker id (DICT_4X4_50)')
    p.add_argument('--hover', type=float, default=0.10, help='height above marker, m')
    p.add_argument('--size', type=float, default=None, help='marker edge length, m (enables solvePnP)')
    p.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    p.add_argument('--dry-run', action='store_true', help='detect only, do not move')
    args = p.parse_args()

    cfg = load_config()

    rclpy.init()
    node = Node('aruco_check')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        robot = Robot(node, cfg)
        camera = Camera(node, cfg)
        if 'move_to_pose' in robot.wait_for_servers(5.0):
            print('move_to_pose action server not available', file=sys.stderr)
            return
        if not robot.wait_for_state(2.0):
            print('no RobotState received - is the controller running?', file=sys.stderr)
            return
        asyncio.run(run(robot, camera, args))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
