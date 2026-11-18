"""Click an ArUco marker in the live wrist-camera view; the TCP moves to its center.

Tests the current camera static transform end to end. The marker center is found exactly
the way the skills find points (median aligned depth over the marker, deprojected with the
color intrinsics, then T_base_proj from TF), and the TCP is sent there with move_to_pose,
keeping its current orientation. If the transform is right, the TCP lands on the marker
center. Measure the miss by eye or ruler.

    python scratch/aruco_camera_calibration.py                # TCP goes HOVER m above the center
    python scratch/aruco_camera_calibration.py --hover 0      # TCP goes to the center itself
    python scratch/aruco_camera_calibration.py --hover 0.05

In the window:
    left click a marker   move the TCP to it
    h                     return to the pose the script started in (to click the next marker)
    s                     stop the arm
    q / esc               stop the arm and quit

Every move is appended to scratch/pics/aruco_calibration_log.csv.
"""

import argparse
import asyncio
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty

from gemini_agent.camera import Camera
from gemini_agent.config import load_config
from gemini_agent.robot import Robot

HOVER = 0.005   # m above the marker center (base_link z) the TCP is sent to; --hover overrides

WINDOW = 'aruco camera calibration'
LOG = '/home/mcrr-lab/kinova-gemini-2/scratch/pics/aruco_calibration_log.csv'

_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())


def detect_all(rgb):
    """-> list of (id, 4x2 pixel corners) for every marker in view."""
    corners, ids, _ = _detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for c, i in zip(corners, ids.flatten())]


def marker_at(markers, u, v):
    """The marker whose outline contains pixel (u, v), or None."""
    for mid, corners in markers:
        if cv2.pointPolygonTest(corners.astype(np.float32), (float(u), float(v)), False) >= 0:
            return mid, corners
    return None


def center_cam(corners, snap):
    """Marker center in the projection frame, median depth over the marker.
    Same math as gemini_agent.perception.aruco.center_cam (copied to avoid importing the
    perception package, which loads SAM2)."""
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


def fmt(v):
    return '[' + ', '.join(f'{x:+.4f}' for x in v) + ']'


def log_row(row):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['time', 'marker_id', 'hover_m', 'cam_x', 'cam_y', 'cam_z',
                        'marker_base_x', 'marker_base_y', 'marker_base_z',
                        'target_x', 'target_y', 'target_z',
                        'reached_x', 'reached_y', 'reached_z', 'status'])
        w.writerow(row)


class App:
    def __init__(self, robot, camera, hover):
        self.robot = robot
        self.camera = camera
        self.hover = hover
        self.click = None          # (u, v) waiting to be handled
        self.task = None           # running move, if any
        self.status = 'click a marker'
        self.home = robot.current_pose()

    def on_mouse(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (u, v)

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    async def go_to_marker(self, u, v):
        # a fresh, synchronized color/depth/TF snapshot taken after the arm has settled
        snap = await self.camera.fresh_snapshot()
        hit = marker_at(detect_all(snap.rgb), u, v)
        if hit is None:
            self.status = 'no marker under the click'
            print(f'click ({u}, {v}): no marker there')
            return
        mid, corners = hit

        cam = center_cam(corners, snap)
        if cam is None:
            self.status = f'marker {mid}: no depth on it'
            print(f'marker {mid}: no valid depth on it')
            return
        marker_base = (snap.T_base_proj @ np.append(cam, 1.0))[:3]
        target = marker_base + np.array([0.0, 0.0, self.hover])

        pose = self.robot.current_pose()
        print(f'\nmarker {mid}')
        print(f'  camera frame : {fmt(cam)}')
        print(f'  base_link    : {fmt(marker_base)}')
        print(f'  TCP target   : {fmt(target)}  ({self.hover * 100:.1f} cm above, same orientation)')
        print(f'  TCP now      : {fmt(pose[:3])}')

        self.status = f'moving to marker {mid}...'
        res = await self.robot.move_to_pose(*target, *pose[3:])
        reached = np.array(self.robot.current_pose()[:3])
        print(f'  move_to_pose : {res.status} {res.message}'.rstrip())
        print(f'  TCP reached  : {fmt(reached)}  (reached - target: '
              f'{np.linalg.norm(reached - target) * 1000:.1f} mm)')
        print('  -> measure how far the TCP is from the marker center')
        self.status = f'marker {mid}: {res.status} - press h to go back'

        log_row([f'{time.time():.3f}', mid, self.hover, *cam, *marker_base, *target, *reached,
                 res.status])

    async def go_home(self):
        self.status = 'returning to start pose...'
        res = await self.robot.move_to_pose(*self.home)
        print(f'\nreturn to start: {res.status} {res.message}'.rstrip())
        self.status = 'click a marker' if res.ok else f'return failed: {res.status}'

    def start(self, coro):
        if self.busy:
            print('still moving - ignored')
            return
        self.task = asyncio.create_task(coro)

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
        await self.robot.stop()
        self.status = 'stopped'

    def draw(self):
        """Latest color frame with every detected marker outlined."""
        pair = self.camera._pair
        if pair is None:
            return None
        rgb = self.camera._bridge.imgmsg_to_cv2(pair[0], desired_encoding='rgb8')
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        markers = detect_all(rgb)
        if markers:
            cv2.aruco.drawDetectedMarkers(
                out, [c.reshape(1, 4, 2) for _, c in markers],
                np.array([[m] for m, _ in markers]))
            for _, c in markers:
                cu, cv_ = c.mean(axis=0)
                cv2.drawMarker(out, (int(cu), int(cv_)), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(out, self.status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(out, 'click: go   h: start pose   s: stop   q: quit',
                    (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    async def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(f'start pose: {fmt(self.home[:3])}  theta {fmt(self.home[3:])}')
        try:
            while True:
                frame = self.draw()
                if frame is not None:
                    cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    if self.busy:
                        await self.stop()
                    return
                if key == ord('s'):
                    await self.stop()
                elif key == ord('h'):
                    self.start(self.go_home())

                if self.click is not None:
                    u, v = self.click
                    self.click = None
                    self.start(self.go_to_marker(u, v))

                if self.task is not None and self.task.done() and not self.task.cancelled():
                    if self.task.exception() is not None:
                        print(f'error: {self.task.exception()!r}')
                        self.status = 'error - see terminal'
                    self.task = None

                await asyncio.sleep(0.01)
        finally:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hover', type=float, default=HOVER,
                   help=f'm above the marker center the TCP is sent to (default {HOVER})')
    args = p.parse_args()

    cfg = load_config()
    print(f'projection frame: {cfg["camera"]["projection_frame"]} -> {cfg["camera"]["base_frame"]}')

    rclpy.init()
    node = Node('aruco_camera_calibration')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = None
    try:
        robot = Robot(node, cfg)
        camera = Camera(node, cfg)
        missing = robot.wait_for_servers(5.0)
        if 'move_to_pose' in missing:
            print('move_to_pose action server not available', file=sys.stderr)
            return
        if not robot.wait_for_state(2.0):
            print('no RobotState received - is the controller running?', file=sys.stderr)
            return
        tool = robot.tool_length()
        print(f'tool length: {tool:.3f} m (TCP is the bare gripper + this)')
        asyncio.run(App(robot, camera, args.hover).run())
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
