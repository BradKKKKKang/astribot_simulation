#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple camera image viewer for testing camera topics.
Compatible with MuJoCo, ManiSkill, and other simulators.
Combines head (top) and left/right wrist (bottom) images in one resizable window.

Usage:
    # Show all cameras (default)
    python3 tests/unit/test_camera_viewer.py

    # Show specific cameras
    python3 tests/unit/test_camera_viewer.py --cameras head_rgbd left_wrist_rgbd
"""

import os
import sys
import cv2
import numpy as np
import time
import threading
import signal
from cv_bridge import CvBridge

ros_version = os.getenv('ROS_VERSION')

if ros_version == '1':
    import rospy
    from sensor_msgs.msg import Image
elif ros_version == '2':
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image
else:
    raise RuntimeError("ROS_VERSION not set. Please source env.sh first.")


def compose_camera_grid(images, camera_names):
    """Preserve camera order even when frames arrive in a different order."""
    head, left, right = 'head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd'
    rows = [[head]] if head in camera_names else []
    if left in camera_names or right in camera_names:
        rows.append([left if left in camera_names else None, right if right in camera_names else None])
    rows.extend([name] for name in camera_names if name not in (head, left, right))
    head_image = images.get(head)
    width = head_image.shape[1] if head_image is not None else 1280
    # Two equal wrist tiles; the usual 1280x720 + 2*640x360 keeps native pixels.
    width += width % 2
    rendered = []
    for row in rows:
        tile_width = width // len(row)
        height = max([round(tile_width * images[name].shape[0] / images[name].shape[1])
                      if name in images and images[name] is not None else round(tile_width * 9 / 16)
                      for name in row])
        strip = np.zeros((height, width, 3), dtype=np.uint8)
        for index, name in enumerate(row):
            if name is None:
                continue
            tile = strip[:, index * tile_width:(index + 1) * tile_width]
            image = images.get(name)
            if image is not None:
                scale = min(tile_width / image.shape[1], height / image.shape[0])
                size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
                fitted = image if size == (image.shape[1], image.shape[0]) else cv2.resize(image, size)
                x, y = (tile_width - size[0]) // 2, (height - size[1]) // 2
                tile[y:y + size[1], x:x + size[0]] = fitted
            label = name + (' (waiting for image)' if image is None else '')
            cv2.rectangle(tile, (0, 0), (min(tile_width, 400), 28), (0, 0, 0), -1)
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(strip)
    return np.vstack(rendered) if rendered else np.zeros((360, 640, 3), dtype=np.uint8)


class CameraViewer:
    def __init__(self, camera_names=None):
        self.bridge = CvBridge()
        self.camera_images = {}

        # Default: all cameras
        if camera_names is None:
            self.camera_names = ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd']
        else:
            self.camera_names = camera_names

        self.last_update = {}
        self.update_lock = threading.Lock()
        self.stop_event = threading.Event()

        if ros_version == '1':
            rospy.init_node('camera_viewer', anonymous=True)
            self.setup_ros1_subscribers()
        elif ros_version == '2':
            # The main thread owns shutdown; SIGTERM must not invalidate the
            # context while the executor thread is constructing a wait set.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self.node = Node('camera_viewer')
            self.setup_ros2_subscribers()
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

    def _spin(self):
        while not self.stop_event.is_set() and rclpy.ok():
            self.executor.spin_once(timeout_sec=.1)

    def setup_ros1_subscribers(self):
        for camera_name in self.camera_names:
            topic = f'astribot_whole_body/camera/{camera_name}/image_raw'
            rospy.Subscriber(topic, Image, self.image_callback, callback_args=camera_name, queue_size=1, buff_size=2**24)

    def setup_ros2_subscribers(self):
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        for camera_name in self.camera_names:
            topic = f'astribot_whole_body/camera/{camera_name}/image_raw'
            self.node.create_subscription(Image, topic,
                                         lambda msg, name=camera_name: self.image_callback(msg, name), qos)

    def image_callback(self, msg, camera_name):
        current_time = time.time()
        if camera_name in self.last_update and (current_time - self.last_update[camera_name]) < 0.033:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.update_lock:
                self.camera_images[camera_name] = cv_image
                self.last_update[camera_name] = current_time
        except Exception as e:
            print(f"Error converting image from {camera_name}: {e}")

    def run(self):
        print("Camera Viewer Started")
        print("Press 'q' to quit")
        print(f"Listening to topics: {self.camera_names}")
        window_sizes = {}
        window = 'Astribot Cameras'

        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            first_frame = True
            while not self.stop_event.is_set():
                with self.update_lock:
                    images = dict(self.camera_images)
                for camera_name, image in images.items():
                    if image is not None:
                        height, width = image.shape[:2]
                        if window_sizes.get(camera_name) != (width, height):
                            window_sizes[camera_name] = (width, height)
                            print(f"{camera_name}: {width}x{height} (source resolution)")
                combined = compose_camera_grid(images, self.camera_names)
                cv2.imshow(window, combined)
                if first_frame:
                    height, width = combined.shape[:2]
                    scale = min(1., 1280 / width, 900 / height)
                    cv2.resizeWindow(window, round(width * scale), round(height * scale))
                    first_frame = False

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close()

    def close(self):
        self.stop_event.set()
        cv2.destroyAllWindows()
        if ros_version == '2':
            self.spin_thread.join(timeout=3.)
            if self.spin_thread.is_alive():
                raise RuntimeError('Camera viewer ROS spin thread did not stop')
            if not self.executor.shutdown(timeout_sec=3.):
                raise RuntimeError('Camera viewer ROS executor shutdown timed out')
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        print('Camera Viewer Stopped', flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Camera image viewer')
    parser.add_argument('--cameras', nargs='+', help='Specific cameras to show')
    args = parser.parse_args()

    camera_names = args.cameras if args.cameras else None

    viewer = CameraViewer(camera_names)
    if ros_version == '2':
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: viewer.stop_event.set())
    viewer.run()
