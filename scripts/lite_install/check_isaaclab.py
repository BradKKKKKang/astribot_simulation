"""Check the installed Isaac wheels and local ROS type support without starting a node."""
import sys
from importlib.metadata import version

import numpy as np
import open3d
import torch
import tf_transformations
from cv_bridge import CvBridge
from isaaclab.app import AppLauncher
from packaging.version import Version
from astribot_msgs.msg import RobotJointController, RobotJointState
from astribot_msgs.srv import RawRequest
from rclpy.serialization import serialize_message, deserialize_message

expected_versions = {
    "isaaclab": "3.0.0",
    "isaacsim": "6.2.0.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "warp-lang": "1.17.0",
    "newton": "1.6.1",
}
for name, expected in expected_versions.items():
    installed = Version(version(name))
    assert Version(installed.public) == Version(expected), f"{name}: expected {expected}, found {installed}"
    print(f"{name}: {installed}")
assert torch.version.cuda == "13.0", torch.version.cuda
assert Version(np.__version__).major == 2, np.__version__

bridge = CvBridge()
rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
image = bridge.cv2_to_imgmsg(rgb, "rgb8")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(image, "bgr8"), rgb[:, :, ::-1])
command = RobotJointController()
command.mode = 1
command.command = [0.1, 0.0]
restored = deserialize_message(serialize_message(command), RobotJointController)
assert restored.mode == 1 and list(restored.command) == [0.1, 0.0]
print(f"ROS 2, camera conversion and Astribot messages OK: {sys.executable}")
