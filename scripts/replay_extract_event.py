#!/usr/bin/env python3
"""Replay raw Astribot Parquet episodes, capture feedback, and extract Web events.

Isaac Lab runs separately. CPU extraction starts after SDK cleanup.
See docs/DATASET_REPLAY.md for environment setup and extraction-only usage.
"""

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.placement_analysis.robot_export import DEFAULT_DEMO_OUTPUT, DEFAULT_POSE_CONFIG, DEFAULT_USD
from src.placement_analysis.web_publish import task_directory_name


COMMAND_COLUMN = "joints_dict.joints_position_command"
CONTROL_HZ = 100.0
MOVE_HZ = 250.0
FEEDBACK_DRAIN_S = 0.1
FEEDBACK_TIMEOUT_S = 2.0
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


class TqdmProgress:
    """Use tqdm for episode/frame progress without adding a runtime import for dry-run."""

    def __init__(self, stream=None, enabled=None):
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else bool(enabled)
        self.bar = None

    def start_episode(self, number, total, episode_id, frame_total):
        self.finish_episode()
        if not self.enabled:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError as error:
            raise RuntimeError(
                "Replay progress requires tqdm. Install it with "
                f"{sys.executable} -m pip install 'tqdm>=4.66'"
            ) from error
        self.bar = tqdm(total=max(1, int(frame_total)),
                        desc=f"Replay {number}/{total} {episode_id}",
                        unit="frame", dynamic_ncols=True, file=self.stream,
                        position=0, leave=False, mininterval=0.5)

    def update_frame(self, frame):
        if self.bar is None:
            return
        target = max(0, min(int(frame), self.bar.total))
        self.bar.update(target - self.bar.n)

    def finish_episode(self):
        if self.bar is not None:
            self.bar.update(self.bar.total - self.bar.n)
            self.bar.close()
            self.bar = None

    def close(self):
        # An interrupted episode must retain its actual frame count.
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("parquet", nargs="?", type=Path,
                        help="raw episode/dataset; with --extract-only, a completed capture/batch")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true", help="reuse captures without replay or SDK")
    mode.add_argument("--export-robot-only", action="store_true", help="regenerate the persistent S1 Web model")
    mode.add_argument("--check-extraction", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--start-frame", type=int, default=0, help="first row, inclusive")
    result.add_argument("--end-frame", type=int, help="last row, exclusive (default: EOF)")
    result.add_argument("--speed", type=positive_float, default=1.0)
    result.add_argument("--fps", type=positive_float, help="override timestamp timing")
    result.add_argument("--column", default=COMMAND_COLUMN, help="25-DOF joint position column")
    result.add_argument("--metadata", type=Path, help="explicit LeRobot meta/info.json")
    result.add_argument("--initial-duration", type=positive_float, default=3.0)
    result.add_argument("--control-mode", choices=("move", "direct"), default="direct",
                        help="direct (default): timestamped target stream; move: SDK 103 point-to-point calls")
    result.add_argument("--timeout", type=positive_float, default=20.0,
                        help="seconds to wait for simulation feedback and all three cameras")
    result.add_argument("--dry-run", action="store_true", help="validate inputs without ROS, USD loading or output writes")
    result.add_argument("--report", type=Path, help="write the dataset validation report, including with --dry-run")
    result.add_argument("--no-viewer", action="store_true",
                        help="replay without camera viewers or camera readiness checks")
    result.add_argument("--output-root", type=Path, default=ROOT / "results/replay_capture",
                        help="per-episode feedback directory (default: results/replay_capture)")
    result.add_argument("--output", type=Path,
                        help="Web bundle directory (default: results/placement; model-only: public/data/demo)")
    result.add_argument("--task-name", type=task_directory_name,
                        help="Web data subdirectory name (default: output directory name)")
    result.add_argument("--no-web-publish", action="store_true",
                        help="export the bundle without copying it into the Web data directory")
    result.add_argument("--usd", type=Path, default=DEFAULT_USD, help="same fixed-base S1 USD used by Isaac")
    result.add_argument("--detector-config", type=Path, default=ROOT / "config/placement_analysis.json")
    result.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG,
                        help="reference joint targets for --export-robot-only")
    result.add_argument("--extract-python", type=Path,
                        help="Python with NumPy, PyArrow and pxr for CPU extraction (default: current Python)")
    result.add_argument("--resume", action="store_true", help="skip matching completed captures under --output-root")
    result.add_argument("--no-progress", action="store_true",
                        help="disable the interactive terminal replay progress bar")
    return result


def discover_episodes(args):
    path = args.parquet.expanduser().resolve(strict=True)
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no Parquet episodes found in {path}")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("episode filenames must be unique across the dataset")
    if path.is_dir() and (args.start_frame != 0 or args.end_frame is not None):
        raise ValueError("frame selection requires a single episode file; directory replay always uses all rows")
    if args.output_root and path.is_dir() and args.output_root.expanduser().resolve().is_relative_to(path):
        raise ValueError("capture output must be outside the input dataset directory")
    return paths


def replay_settings(args):
    return {key: getattr(args, key) for key in
            ("column", "speed", "fps", "start_frame", "end_frame", "initial_duration", "control_mode")}


def metadata_path(args, path):
    if args.metadata is not None:
        return args.metadata.expanduser().resolve(strict=True)
    return next((p / "meta/info.json" for p in path.parents if (p / "meta/info.json").is_file()), None)


def validate_dataset(args, paths):
    """Read every episode before constructing an SDK client; retain only its report."""
    episodes = []
    for path in paths:
        segment_args = argparse.Namespace(**vars(args))
        segment_args.parquet = path
        commands, times = load_segment(segment_args, quiet=True)
        metadata = metadata_path(args, path)
        episodes.append({"episode_id": path.stem, "input_parquet": str(path),
                         "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "metadata_file": str(metadata) if metadata else None,
                         "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata else None,
                         "frames": len(commands), "duration_s": float(times[-1]),
                         "chassis_command": commands[0, :3].tolist()})
    report = {"episode_count": len(episodes), "frames": sum(e["frames"] for e in episodes),
              "duration_s": sum(e["duration_s"] for e in episodes),
              "replay_settings": replay_settings(args), "episodes": episodes,
              "joint_order": JOINT_NAMES,
              "joint_order_without_metadata": "assumed canonical Astribot order",
              "control_path": "raw Parquet -> Astribot SDK -> ROS2 -> Isaac Lab"}
    return report


def completed_capture(output, source, settings, resume):
    if not output.exists() or not any(output.iterdir()):
        return False
    metadata_path = output / "run_metadata.json"
    if resume and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        matching = (metadata.get("input_sha256") == source["input_sha256"]
                    and metadata.get("source", {}).get("metadata_sha256") == source["metadata_sha256"]
                    and metadata.get("replay_settings") == settings)
        if metadata.get("status") == "failed" and matching:
            return False
        if (metadata.get("status") == "complete" and metadata.get("capture_schema") == "replay_parquet_v1"
                and matching
                and all((output / name).is_file() and
                        metadata.get("capture_sha256", {}).get(name)
                        == hashlib.sha256((output / name).read_bytes()).hexdigest()
                        for name in ("sim_states.parquet", "api_commands.parquet"))):
            return True
    raise FileExistsError(f"refusing to overwrite incomplete or mismatched capture: {output}; "
                          "use a new output root or move this episode directory before retrying")


def load_segment(args, quiet=False):
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

    metadata = metadata_path(args, path)
    if metadata is not None:
        info = json.loads(metadata.expanduser().read_text())
        feature = info.get("features", {}).get(args.column, {})
        names = feature.get("names", [])
        if feature.get("shape") != [25] or len(names) != 25 or set(names) != set(JOINT_NAMES):
            raise ValueError("metadata must declare the Astribot 25 joint names for this column")
        commands = commands[:, [names.index(name) for name in JOINT_NAMES]]
    elif not quiet:
        print("No metadata: assuming SDK order chassis/torso/left arm/left gripper/"
              "right arm/right gripper/head (3/4/7/1/7/1/2).")
    if np.any(commands[:, [14, 22]] < 0) or np.any(commands[:, [14, 22]] > 100):
        raise ValueError("gripper positions must be in dataset/SDK units [0, 100]")
    if np.any(np.ptp(commands[:, :3], axis=0) > 1e-9):
        raise ValueError("moving chassis cannot be reproduced by this fixed-base replay")

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
            "'pyarrow==17.0.0' 'cryptography>=3.4'. "
            "Use the Python environment where the SDK examples work; "
            "the simulator can run in a separate Conda environment."
        ) from error
    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    from astribot_ros_middleware import astribot_ros_middleware as middleware
    return Astribot, middleware


def wait_for_simulation(timeout, cameras=True):
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
    required = {name for name, _, _ in PARTS} | {"astribot_chassis"}
    if cameras:
        required.update(CAMERAS)

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
        for name in (CAMERAS if cameras else ()):
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
        if viewer is not None and viewer.poll() is not None:
            if viewer.returncode:
                raise RuntimeError(f"camera viewer exited with status {viewer.returncode}")
            raise KeyboardInterrupt
        if not middleware.ok():
            raise RuntimeError("ROS shut down during replay")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))


def drain_feedback(capture, viewer, middleware):
    # Always give the asynchronous SDK publisher time to send the final target.
    wait_until(time.monotonic() + FEEDBACK_DRAIN_S, viewer, middleware)
    if capture is None:
        return
    capture.end_window()
    deadline = time.monotonic() + FEEDBACK_TIMEOUT_S
    while not capture.feedback_covers_window():
        now = time.monotonic()
        if now >= deadline:
            # save() records the raw data and reports the synchronization error.
            return
        wait_until(min(deadline, now + 0.01), viewer, middleware)


def replay(robot, commands, times, initial_duration, viewer, middleware,
           control_mode="move", capture=None, progress=None):
    names = [name for name, _, _ in PARTS]
    if control_mode == "move":
        def move(index, duration, phase="dataset"):
            wait_until(time.monotonic(), viewer, middleware)
            if not robot.get_control_rights_status():
                raise RuntimeError("SDK lost control rights")
            started = time.monotonic_ns()
            result = robot.move_joints_position(
                names, split_command(commands[index]), duration=float(duration), use_wbc=False)
            if capture is not None:
                capture.record_call(index, times[index], commands[index], started, time.monotonic_ns(), phase=phase)
            if isinstance(result, str) and "fail" in result.lower():
                raise RuntimeError(f"SDK move failed: {result}")

        print(f"Moving to segment start with SDK 103 interface over {initial_duration:g}s...",
              flush=True)
        move(0, initial_duration, phase="initial")
        if capture is not None:
            capture.begin()
        print("Replaying segment with sequential SDK 103 moves...", flush=True)
        started = time.monotonic()
        for index in range(1, len(commands)):
            # Each blocking call owns its interval. Do not skip frames to catch
            # up with wall time when SDK planning/call overhead delays replay.
            move(index, times[index] - times[index - 1])
            if progress is not None:
                progress(index + 1)
        print(f"SDK moves completed in {time.monotonic() - started:.3f}s "
              f"(dataset duration {times[-1]:.3f}s).", flush=True)
        drain_feedback(capture, viewer, middleware)
        return

    current = robot.get_current_joints_position(names)
    if [len(row) for row in current] != [end - start for _, start, end in PARTS]:
        raise RuntimeError("SDK returned an unexpected joint feedback layout")
    initial = np.concatenate(current)
    if not np.isfinite(initial).all():
        raise RuntimeError("SDK returned non-finite joint feedback")

    last_send = None

    def send(command):
        nonlocal last_send
        # Raw frame deadlines and hold ticks must share one SDK clock. In
        # particular, 30 FPS frame deadlines must not add calls between 100 Hz
        # hold ticks, nor may late frames be sent in a catch-up burst.
        if last_send is not None:
            wait_until(last_send + 1 / CONTROL_HZ, viewer, middleware)
        if not robot.get_control_rights_status():
            raise RuntimeError("SDK lost control rights")
        last_send = time.monotonic()
        started = time.monotonic_ns()
        robot.set_joints_position(names, split_command(command),
                                  control_way="direct", use_wbc=False)
        return started

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
    if capture is not None:
        capture.begin()
    epoch = time.monotonic()
    for index, command in enumerate(commands):
        wait_until(epoch + times[index], viewer, middleware)
        started = send(command)
        if capture is not None:
            capture.record_call(index, times[index], command, started, time.monotonic_ns())
        if progress is not None:
            progress(index + 1)
        # Hold each raw target at the SDK cadence until the next source frame.
        # Even when a call is late, each subsequent source frame is sent once.
        next_time = times[index + 1] if index + 1 < len(times) else times[index]
        tick = math.floor((time.monotonic() - epoch) * CONTROL_HZ) + 1
        while tick / CONTROL_HZ < next_time - 1e-12:
            wait_until(epoch + tick / CONTROL_HZ, viewer, middleware)
            if time.monotonic() - epoch >= next_time:
                break
            started = send(command)
            if capture is not None:
                capture.record_call(index, times[index], command, started, time.monotonic_ns())
            tick = max(tick + 1, math.floor((time.monotonic() - epoch) * CONTROL_HZ) + 1)
    drain_feedback(capture, viewer, middleware)


def run(args, commands=None, times=None, report=None):
    if os.environ.get("ROS_VERSION") != "2" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("source ROS2/SDK environments, then export ROS_LOCALHOST_ONLY=1 "
                           "in both simulator and replay terminals")
    if not args.no_viewer and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("three camera viewers require a graphical desktop")
    Astribot, middleware = load_sdk()
    if not os.environ.get("ROS_DOMAIN_ID"):
        raise RuntimeError("set the same isolated ROS_DOMAIN_ID in simulator and replay terminals")
    print(f"Waiting for simulator on ROS_DOMAIN_ID="
          f"{os.environ.get('ROS_DOMAIN_ID', '0')}...", flush=True)
    wait_for_simulation(args.timeout, cameras=not args.no_viewer)

    viewer = None
    robot = None
    capture = None
    batch = None
    progress = TqdmProgress(enabled=None if not args.no_progress else False)
    try:
        if not args.no_viewer:
            viewer_path = Path(__file__).resolve().parents[1] / "tests/unit/test_camera_viewer.py"
            viewer = subprocess.Popen([sys.executable, str(viewer_path), "--cameras", *CAMERAS])
        robot = Astribot(freq=MOVE_HZ if args.control_mode == "move" else CONTROL_HZ)
        if not robot.get_control_rights_status():
            raise RuntimeError("SDK did not acquire simulation control rights")
        sources = report["episodes"] if report else [None]
        settings = replay_settings(args)
        if args.output_root:
            batch = {"status": "partial", "episodes": sources, "completed": [], "replay_settings": settings}
            args.output_root.mkdir(parents=True, exist_ok=True)
            write_json(args.output_root / "batch_manifest.json", batch)
        for number, source in enumerate(sources, 1):
            output = args.output_root / source["episode_id"] if args.output_root else None
            if output and completed_capture(output, source, settings, args.resume):
                print(f'[{number}/{len(sources)}] Already complete: {source["episode_id"]}', flush=True)
                progress.finish_episode()
            else:
                if source is not None:
                    segment_args = argparse.Namespace(**vars(args))
                    segment_args.parquet = Path(source["input_parquet"])
                    if hashlib.sha256(segment_args.parquet.read_bytes()).hexdigest() != source["input_sha256"]:
                        raise ValueError(f"source changed after validation: {segment_args.parquet}")
                    if source["metadata_file"]:
                        metadata_digest = hashlib.sha256(Path(source["metadata_file"]).read_bytes()).hexdigest()
                        if metadata_digest != source["metadata_sha256"]:
                            raise ValueError(f'metadata changed after validation: {source["metadata_file"]}')
                    commands, times = load_segment(segment_args, quiet=True)
                    print(f'[{number}/{len(sources)}] Replaying {source["episode_id"]}: '
                          f'{len(commands)} frames', flush=True)
                if output:
                    if output.exists() and any(output.iterdir()):
                        # completed_capture() permits retry only for a matching
                        # failed capture. Keep its raw data outside batch IDs.
                        archived = output.parent / '_failed' / f'{output.name}-{time.time_ns()}'
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        output.rename(archived)
                        print(f'Archived failed capture: {archived}', flush=True)
                    # Runtime ROS imports stay out of dataset validation/dry-run.
                    root = str(Path(__file__).resolve().parents[1])
                    if root not in sys.path:
                        sys.path.insert(0, root)
                    from src.placement_analysis.replay_capture import ReplayCapture
                    capture = ReplayCapture(robot)
                progress_started = False

                def update_progress(frame):
                    nonlocal progress_started
                    if not progress_started:
                        progress.start_episode(number, len(sources), source["episode_id"], source["frames"])
                        progress_started = True
                    progress.update_frame(frame)

                try:
                    replay(robot, commands, times, args.initial_duration, viewer, middleware,
                           control_mode=args.control_mode, capture=capture,
                           progress=update_progress)
                    if capture is not None:
                        capture.save(output, source, settings)
                    progress.finish_episode()
                finally:
                    if capture is not None:
                        capture.close()
                        capture = None
            if batch is not None:
                batch["completed"].append(source["episode_id"])
                batch["status"] = "complete" if len(batch["completed"]) == len(sources) else "partial"
                write_json(args.output_root / "batch_manifest.json", batch)
        print("Replay complete.", flush=True)
    finally:
        progress.close()
        try:
            if robot is not None:
                robot.stop_robot()
        finally:
            try:
                middleware.shutdown()
            finally:
                if viewer is not None and viewer.poll() is None:
                    viewer.terminate()
                    try:
                        viewer.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=3)


def write_json(path, value):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_cpu_phase(args, *, check=False):
    """Keep the USD interpreter independent of the SDK interpreter."""
    python = str(args.extract_python.expanduser()) if args.extract_python else sys.executable
    command = [python, str(Path(__file__).resolve()), "--usd", str(args.usd),
               "--detector-config", str(args.detector_config)]
    if check:
        command.append("--check-extraction")
    elif args.export_robot_only:
        command.extend(["--export-robot-only", "--pose-config", str(args.pose_config)])
    else:
        source = args.parquet if args.extract_only else args.output_root
        command.extend(["--extract-only", str(source)])
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    if args.task_name is not None:
        command.extend(["--task-name", args.task_name])
    if args.no_web_publish:
        command.append("--no-web-publish")
    if args.dry_run:
        command.append("--dry-run")
    result = subprocess.run(command, timeout=60 if check else None)
    if result.returncode:
        raise RuntimeError(f"CPU {'preflight' if check else 'extraction/export'} failed "
                           f"with exit code {result.returncode}; captures remain available for --extract-only")


def validate_pipeline_outputs(args, paths):
    """Reject conflicts before any SDK connection or output write."""
    args.output_root = args.output_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    source = args.parquet.expanduser().resolve(strict=True)
    if source.is_dir() and args.output.is_relative_to(source):
        raise ValueError("Web bundle output must be outside the input dataset directory")
    if (args.output.is_relative_to(args.output_root)
            or args.output_root.is_relative_to(args.output)):
        raise ValueError("Web bundle and feedback capture directories must not overlap")
    if args.output in paths or args.output_root in paths:
        raise ValueError("outputs must not overwrite an input Parquet")
    if args.report:
        report = args.report.expanduser().resolve()
        if report in paths or report.is_relative_to(args.output) or report.is_relative_to(args.output_root):
            raise ValueError("validation report must be outside the input files and output directories")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise ValueError("feedback output must be a directory")
    if not args.dry_run and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing bundle: {args.output}")


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not (args.export_robot_only or args.check_extraction) and args.parquet is None:
        cli.error("an input episode, dataset or completed capture is required")
    if (args.export_robot_only or args.check_extraction) and args.parquet is not None:
        cli.error("model export/preflight does not accept an episode input")
    if (args.export_robot_only or args.extract_only) and (args.resume or args.report):
        cli.error("--resume and --report apply only to replay + extraction")
    if args.output is None:
        args.output = DEFAULT_DEMO_OUTPUT if args.export_robot_only else ROOT / "results/placement"
    if args.task_name is not None and not args.task_name.strip():
        cli.error("--task-name must not be empty")
    try:
        if args.check_extraction:
            from src.placement_analysis.extraction import prepare_extraction
            prepare_extraction(args.usd, args.detector_config)
            print("CPU extraction preflight passed (no ROS, SDK, simulation or output writes).", flush=True)
            return 0
        if args.extract_only or args.export_robot_only:
            if args.extract_python:
                run_cpu_phase(args)
            elif args.export_robot_only:
                from src.placement_analysis.robot_export import export_reference_robot
                if args.dry_run:
                    args.usd.expanduser().resolve(strict=True)
                    json.loads(args.pose_config.expanduser().read_text())
                    print("Model inputs valid. No USD loading or output writes.")
                else:
                    export_reference_robot(args.output, usd=args.usd, pose_config=args.pose_config)
            else:
                from src.placement_analysis.extraction import extract_events
                extract_events([args.parquet], args.output, usd=args.usd,
                               detector_config=args.detector_config, dry_run=args.dry_run)
                if not args.dry_run and not args.no_web_publish:
                    from src.placement_analysis.web_publish import publish_bundle
                    publish_bundle(args.output, task_name=args.task_name)
            return 0
        paths = discover_episodes(args)
        validate_pipeline_outputs(args, paths)
        report = validate_dataset(args, paths)
        # Validate detector settings even during dry-run, without importing USD.
        from src.placement_analysis.extraction import detector_settings
        detector_settings(args.detector_config)
        args.usd.expanduser().resolve(strict=True)
        completed = []
        if args.output_root:
            args.output_root = args.output_root.expanduser().resolve()
            for source in report["episodes"]:
                if completed_capture(args.output_root / source["episode_id"], source,
                                     report["replay_settings"], args.resume):
                    completed.append(source["episode_id"])
        print(f'{report["episode_count"]} episodes, {report["frames"]} frames, '
              f'{report["duration_s"]:.3f}s at {args.speed:g}x. '
              'Chassis fixed; replaying torso, arms, grippers and head.', flush=True)
        if any(source["metadata_file"] is None for source in report["episodes"]):
            print("No joint metadata for some episodes: assuming canonical Astribot "
                  "chassis/torso/left arm/left gripper/right arm/right gripper/head order (3/4/7/1/7/1/2).")
        if not args.dry_run:
            run_cpu_phase(args, check=True)
        if args.report:
            write_json(args.report, report)
        if not args.dry_run and args.output_root and len(completed) == len(paths):
            write_json(args.output_root / "batch_manifest.json", {
                "status": "complete", "episodes": report["episodes"], "completed": completed,
                "replay_settings": report["replay_settings"],
            })
            print("All captures already complete. No SDK connection needed.")
            run_cpu_phase(args)
            return 0
        if not args.dry_run:
            def interrupt(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupt)
            run(args, report=report)
            run_cpu_phase(args)
        else:
            print("Dataset valid. No ROS, SDK, camera or simulation started.")
        return 0
    except KeyboardInterrupt:
        print("Replay/extraction stopped; completed captures are preserved.")
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.TimeoutExpired) as error:
        print(f"Replay/extraction error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
