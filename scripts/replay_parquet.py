#!/usr/bin/env python3
"""Temporary ROS2 SDK replay tool; see docs/PARQUET_REPLAY.md for setup."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import numpy as np


COMMAND_COLUMN = "joints_dict.joints_position_command"
CONTROL_HZ = 100.0
CAMERAS = ("head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd")
PARTS = (
    ("astribot_torso", 3, 7),
    ("astribot_arm_left", 7, 14),
    ("astribot_gripper_left", 14, 15),
    ("astribot_arm_right", 15, 22),
    ("astribot_gripper_right", 22, 23),
    ("astribot_head", 23, 25),
)
JOINT_NAMES = [
    prefix + str(i)
    for prefix, size in (
        ("chassis_action_", 3), ("torso_action_", 4), ("left_arm_action_", 7),
        ("left_gripper_action_", 1), ("right_arm_action_", 7), ("right_gripper_action_", 1),
    )
    for i in range(size)
] + ["head_action_yaw", "head_action_pitch"]


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("parquet", type=Path, help="Astribot episode Parquet file")
    result.add_argument("--start-frame", type=int, default=0, help="first row, inclusive")
    result.add_argument("--end-frame", type=int, help="last row, exclusive (default: EOF)")
    result.add_argument("--speed", type=positive_float, default=1.0)
    result.add_argument("--fps", type=positive_float, help="override timestamp timing")
    result.add_argument("--column", default=COMMAND_COLUMN, help="25-DOF joint position column")
    result.add_argument("--metadata", type=Path, help="explicit LeRobot meta/info.json")
    result.add_argument("--initial-duration", type=positive_float, default=3.0)
    result.add_argument("--timeout", type=positive_float, default=20.0,
                        help="seconds to wait for simulation feedback and all three cameras")
    result.add_argument("--dry-run", action="store_true", help="validate data without ROS or GUI")
    result.add_argument("--exit-after-replay", action="store_true",
                        help="close viewers after replay; otherwise press q or Ctrl-C")
    return result


def load_segment(args):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        if error.name != "pyarrow":
            raise
        raise RuntimeError(
            f"PyArrow is not installed for {sys.executable}. Install it in this environment: "
            f"{shlex.quote(sys.executable)} -m pip install 'pyarrow==17.0.0'"
        ) from error

    path = args.parquet.expanduser().resolve(strict=True)
    parquet = pq.ParquetFile(path)
    end = parquet.metadata.num_rows if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= parquet.metadata.num_rows:
        raise ValueError("segment must satisfy 0 <= start-frame < end-frame <= row count")
    columns = [args.column]
    if args.fps is None:
        columns.append("timestamp")
    if "episode_index" in parquet.schema_arrow.names:
        columns.append("episode_index")
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}; use --fps if timestamp is absent")
    table = parquet.read(columns=columns).slice(args.start_frame, end - args.start_frame)
    if "episode_index" in columns and len(set(table["episode_index"].to_pylist())) != 1:
        raise ValueError("segment crosses episode boundaries; select a single episode")
    commands = np.asarray(table[args.column].to_pylist(), dtype=np.float64)
    if commands.shape != (len(table), 25) or not np.isfinite(commands).all():
        raise ValueError("expected finite joint positions with shape (frames, 25)")

    metadata = args.metadata
    if metadata is None:
        metadata = next((p / "meta/info.json" for p in path.parents
                         if (p / "meta/info.json").is_file()), None)
    if metadata is not None:
        info = json.loads(metadata.expanduser().read_text())
        feature = info.get("features", {}).get(args.column, {})
        names = feature.get("names", [])
        if feature.get("shape") != [25] or len(names) != 25 or set(names) != set(JOINT_NAMES):
            raise ValueError("metadata must declare the Astribot 25 joint names for this column")
        commands = commands[:, [names.index(name) for name in JOINT_NAMES]]
    else:
        print("No metadata: assuming SDK order chassis/torso/left arm/left gripper/"
              "right arm/right gripper/head (3/4/7/1/7/1/2).")
    if np.any(commands[:, [14, 22]] < 0) or np.any(commands[:, [14, 22]] > 100):
        raise ValueError("gripper positions must be in dataset/SDK units [0, 100]")

    times = (np.arange(len(table), dtype=np.float64) / args.fps if args.fps is not None
             else np.asarray(table["timestamp"].to_pylist(), dtype=np.float64))
    if times.shape != (len(table),) or not np.isfinite(times).all():
        raise ValueError("timestamps must be finite scalar seconds")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    times = (times - times[0]) / args.speed
    if not np.isfinite(times).all():
        raise ValueError("replay duration overflow")
    return commands, times


def split_command(command):
    return [command[start:end].tolist() for _, start, end in PARTS]


def load_sdk():
    # Native SDK imports can obscure this missing dependency with a misleading
    # robotics_library_py "is not a package" error. Check it explicitly first.
    try:
        importlib.import_module("cryptography.fernet")
    except ImportError as error:
        raise RuntimeError(
            f"SDK dependency cryptography is unavailable for {sys.executable}: {error}. "
            f"Install replay dependencies with {shlex.quote(sys.executable)} -m pip install "
            f"-r {shlex.quote(str(Path(__file__).with_name('requirements-replay.txt')))}. "
            "Use the Python environment where the SDK examples work; "
            "the simulator can run in a separate Conda environment."
        ) from error
    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    from astribot_ros_middleware import astribot_ros_middleware as middleware
    return Astribot, middleware


def wait_for_simulation(timeout):
    """Read-only check before constructing the SDK (which acquires control)."""
    import rclpy
    from astribot_msgs.msg import RobotJointState
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    context = Context()
    rclpy.init(context=context)
    node = None
    executor = None
    seen = set()
    required = {name for name, _, _ in PARTS} | {"astribot_chassis"} | set(CAMERAS)

    def state(msg, name):
        # The simulator's synthetic fixed-chassis publisher leaves frame_id empty.
        fixed_chassis = name == "astribot_chassis" and msg.header.frame_id == ""
        if msg.header.frame_id != "simulation" and not fixed_chassis:
            raise RuntimeError(f"{name}: feedback is not from simulation")
        seen.add(name)

    try:
        node = rclpy.create_node(f"parquet_replay_probe_{os.getpid()}", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        for name in required - set(CAMERAS):
            node.create_subscription(
                RobotJointState, f"/{name}/joint_space_states",
                lambda msg, name=name: state(msg, name), qos_profile_sensor_data)
        for name in CAMERAS:
            node.create_subscription(
                Image, f"/astribot_whole_body/camera/{name}/image_raw",
                lambda msg, name=name: seen.add(name) if msg.width and msg.height else None,
                qos_profile_sensor_data)
        deadline = time.monotonic() + timeout
        while required - seen:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"simulation/camera timeout; missing: {sorted(required - seen)}")
            executor.spin_once(timeout_sec=0.1)
    finally:
        try:
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                context.try_shutdown()


def wait_until(deadline, viewer, middleware):
    while True:
        if viewer.poll() is not None:
            if viewer.returncode:
                raise RuntimeError(f"camera viewer exited with status {viewer.returncode}")
            raise KeyboardInterrupt
        if not middleware.ok():
            raise RuntimeError("ROS shut down during replay")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))


def replay(robot, commands, times, initial_duration, viewer, middleware):
    names = [name for name, _, _ in PARTS]
    current = robot.get_current_joints_position(names)
    if [len(row) for row in current] != [end - start for _, start, end in PARTS]:
        raise RuntimeError("SDK returned an unexpected joint feedback layout")
    initial = np.concatenate(current)
    if not np.isfinite(initial).all():
        raise RuntimeError("SDK returned non-finite joint feedback")

    def send(command):
        if not robot.get_control_rights_status():
            raise RuntimeError("SDK lost control rights")
        robot.set_joints_position(names, split_command(command),
                                  control_way="direct", use_wbc=False)

    start_pose = commands[0].copy()
    start_pose[3:] = initial
    def stream(duration, sample):
        epoch = time.monotonic()
        tick = 0
        while True:
            scheduled = min(tick / CONTROL_HZ, duration)
            wait_until(epoch + scheduled, viewer, middleware)
            elapsed = max(scheduled, time.monotonic() - epoch)
            send(sample(min(elapsed, duration)))
            if elapsed >= duration:
                break
            # Skip missed control ticks instead of sending a burst of stale poses.
            tick = max(tick + 1, math.floor(elapsed * CONTROL_HZ) + 1)

    print(f"Moving to segment start over {initial_duration:g}s...", flush=True)
    def initial_pose(elapsed):
        fraction = elapsed / initial_duration
        blend = fraction * fraction * (3 - 2 * fraction)
        return start_pose + blend * (commands[0] - start_pose)
    stream(initial_duration, initial_pose)

    print("Replaying segment...", flush=True)
    stream(float(times[-1]), lambda elapsed: commands[
        np.searchsorted(times, elapsed, side="right") - 1])
    # Allow the SDK's asynchronous publisher to transmit the final target.
    wait_until(time.monotonic() + 0.1, viewer, middleware)


def run(args, commands, times):
    if os.environ.get("ROS_VERSION") != "2" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("source ROS2/SDK environments, then export ROS_LOCALHOST_ONLY=1 "
                           "in both simulator and replay terminals")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("three camera viewers require a graphical desktop")
    Astribot, middleware = load_sdk()
    print(f"Waiting for simulator and three cameras on ROS_DOMAIN_ID="
          f"{os.environ.get('ROS_DOMAIN_ID', '0')}...", flush=True)
    wait_for_simulation(args.timeout)

    viewer_path = Path(__file__).resolve().parents[1] / "tests/unit/test_camera_viewer.py"
    viewer = subprocess.Popen([sys.executable, str(viewer_path), "--cameras", *CAMERAS])
    robot = None
    try:
        robot = Astribot(freq=CONTROL_HZ)
        replay(robot, commands, times, args.initial_duration, viewer, middleware)
        print("Replay complete. Press q in a camera window or Ctrl-C to exit.", flush=True)
        if not args.exit_after_replay:
            while viewer.poll() is None:
                wait_until(time.monotonic() + 0.1, viewer, middleware)
    finally:
        try:
            if robot is not None:
                robot.stop_robot()
        finally:
            try:
                middleware.shutdown()
            finally:
                if viewer.poll() is None:
                    viewer.terminate()
                    try:
                        viewer.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=3)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        commands, times = load_segment(args)
        print(f"Segment rows [{args.start_frame}, {args.start_frame + len(commands)}): "
              f"{len(commands)} frames, {times[-1]:.3f}s at {args.speed:g}x. "
              "Chassis fixed; replaying torso, arms, grippers and head.")
        if not args.dry_run:
            def interrupt(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupt)
            run(args, commands, times)
        return 0
    except KeyboardInterrupt:
        print("Replay stopped.")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Replay error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
