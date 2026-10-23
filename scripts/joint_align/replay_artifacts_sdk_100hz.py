#!/usr/bin/env python3
"""Replay resampled artifacts through the Astribot SDK and record ROS feedback."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PERIOD_NS = 10_000_000
CONTROL_HZ = 100.0
PART_CONFIGS = (
    ("astribot_torso", 4),
    ("astribot_arm_left", 7),
    ("astribot_gripper_left", 1),
    ("astribot_arm_right", 7),
    ("astribot_gripper_right", 1),
    ("astribot_head", 2),
)
PART_NAMES = tuple(name for name, _ in PART_CONFIGS)
JOINTS = tuple(
    f"{part}_joint_{index}"
    for part, count in PART_CONFIGS
    for index in range(1, count + 1)
)


def _canonical_joint(name: str) -> str:
    return name.replace("_joint_L1", "_joint_1")


def _simulation_joint(name: str) -> str:
    return name.replace("_joint_1", "_joint_L1") if "gripper" in name else name


def _header_stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_episode(path: Path):
    path = path.expanduser().resolve(strict=True)
    table = pq.read_table(path)
    commands = table.filter(pc.equal(table["kind"], "command")).to_pylist()
    state_rows = table.filter(pc.equal(table["kind"], "state")).to_pylist()
    grouped_commands = defaultdict(list)
    grouped_states = defaultdict(list)

    for row in commands:
        grouped_commands[row["joint"]].append(row)
    if set(grouped_commands) != set(JOINTS):
        raise ValueError(
            f"{path}: command joints differ; missing={sorted(set(JOINTS) - set(grouped_commands))}, "
            f"extra={sorted(set(grouped_commands) - set(JOINTS))}"
        )

    grid = None
    matrix = []
    for joint in JOINTS:
        rows = sorted(grouped_commands[joint], key=lambda row: row["time"])
        times = np.asarray([row["time"] for row in rows], dtype=np.float64)
        values = np.asarray([row["position"] for row in rows], dtype=np.float64)
        if values.size < 2 or not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid command stream for {joint}")
        if not np.allclose(np.diff(times), 0.01, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: {joint} command stream is not 100 Hz")
        if any(row.get("mode") != 1 for row in rows):
            raise ValueError(f"{path}: position-mode commands required for {joint}")
        if grid is None:
            grid = times
        elif times.shape != grid.shape or not np.allclose(times, grid, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: command joints do not share a time grid")
        matrix.append(values)

    for row in state_rows:
        joint = row["joint"]
        grouped_states[joint].append(
            [row["time"], row["position"], row["velocity"], row["torque"]]
        )
    states = {}
    for joint in JOINTS:
        if joint not in grouped_states:
            raise ValueError(f"{path}: missing real state for {joint}")
        values = np.asarray(sorted(grouped_states[joint]), dtype=np.float64)
        if not np.isfinite(values[:, :3]).all():
            raise ValueError(f"{path}: non-finite real position/velocity for {joint}")
        if not values[0, 0] <= grid[0] <= values[-1, 0]:
            raise ValueError(f"{path}: real state does not bracket command start for {joint}")
        states[joint] = values

    return path, grid, np.asarray(matrix).T, states


def _part_command(command):
    result = []
    offset = 0
    for _, count in PART_CONFIGS:
        result.append(command[offset:offset + count].astype(float).tolist())
        offset += count
    return result


def _state_rows_for_message(msg, part: str, receive_mono_ns: int, receive_system_ns: int):
    expected = [f"{part}_joint_{i}" for i in range(1, dict(PART_CONFIGS)[part] + 1)]
    names = list(msg.name) if msg.name else [_simulation_joint(name) for name in expected]
    columns = (list(msg.position), list(msg.velocity), list(msg.acceleration), list(msg.torque))
    if any(len(values) != len(names) for values in columns):
        raise ValueError(f"{part}: state names and data lengths differ")
    result = []
    for index, raw_name in enumerate(names):
        joint = _canonical_joint(raw_name)
        if joint not in JOINTS:
            raise ValueError(f"{part}: unexpected simulation joint {raw_name}")
        result.append({
            "part": part,
            "joint": joint,
            "position": float(msg.position[index]),
            "velocity": float(msg.velocity[index]),
            "acceleration": float(msg.acceleration[index]),
            "torque": float(msg.torque[index]),
            "header_stamp_ns": _header_stamp_ns(msg.header),
            "received_monotonic_ns": receive_mono_ns,
            "received_system_ns": receive_system_ns,
            "frame_id": str(msg.header.frame_id),
        })
    return result


class TopicRecorder:
    """Capture actual SDK command and simulator state messages on the SDK node."""

    def __init__(self, robot):
        from astribot_msgs.msg import RobotJointController, RobotJointState
        from rclpy.qos import qos_profile_sensor_data

        self.node = robot.astribot_interface.node
        self.lock = threading.Condition()
        self.state_rows = []
        self.command_rows = []
        self.latest_state = {}
        self.latest_state_by_part = {}
        self.callback_errors = []
        self.subscriptions = []

        for part, _ in PART_CONFIGS:
            self.subscriptions.append(self.node.create_subscription(
                RobotJointState,
                f"/{part}/joint_space_states",
                lambda msg, part=part: self._on_state(msg, part),
                10,
            ))
            self.subscriptions.append(self.node.create_subscription(
                RobotJointController,
                f"/{part}/joint_space_command",
                lambda msg, part=part: self._on_command(msg, part),
                qos_profile_sensor_data,
            ))

    def _on_state(self, msg, part):
        receive_mono_ns = time.monotonic_ns()
        receive_system_ns = time.time_ns()
        try:
            if msg.header.frame_id != "simulation":
                raise RuntimeError(f"{part}: rejected non-simulation state")
            rows = _state_rows_for_message(msg, part, receive_mono_ns, receive_system_ns)
        except Exception as error:
            with self.lock:
                self.callback_errors.append(str(error))
                self.lock.notify_all()
            return
        with self.lock:
            self.state_rows.extend(rows)
            for row in rows:
                self.latest_state[row["joint"]] = row
            self.latest_state_by_part[part] = receive_mono_ns
            self.lock.notify_all()

    def _on_command(self, msg, part):
        row = {
            "part": part,
            "mode": int(msg.mode),
            "names": list(msg.name),
            "command": [float(value) for value in msg.command],
            "header_stamp_ns": _header_stamp_ns(msg.header),
            "received_monotonic_ns": time.monotonic_ns(),
            "received_system_ns": time.time_ns(),
        }
        with self.lock:
            self.command_rows.append(row)
            self.lock.notify_all()

    def wait_for_state(self, timeout: float):
        deadline = time.monotonic() + timeout
        with self.lock:
            while set(self.latest_state_by_part) != set(PART_NAMES):
                if self.callback_errors:
                    raise RuntimeError(self.callback_errors[-1])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(PART_NAMES) - set(self.latest_state_by_part))
                    raise TimeoutError(f"missing simulator state topics: {missing}")
                self.lock.wait(min(remaining, 0.1))

    def snapshot(self):
        with self.lock:
            if self.callback_errors:
                raise RuntimeError(self.callback_errors[-1])
            return (
                list(self.state_rows),
                list(self.command_rows),
                {joint: dict(row) for joint, row in self.latest_state.items()},
                dict(self.latest_state_by_part),
            )

    def close(self):
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions.clear()


def wait_for_simulation(timeout: float) -> None:
    import rclpy
    from astribot_msgs.msg import RobotJointState
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    context = Context()
    rclpy.init(context=context)
    node = None
    executor = None
    seen = set()
    errors = []

    def state_callback(msg, part):
        if msg.header.frame_id != "simulation":
            errors.append(f"{part}: state publisher is not marked simulation")
        else:
            seen.add(part)

    try:
        node = rclpy.create_node(f"sdk_replay_probe_{os.getpid()}", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        for part, _ in PART_CONFIGS:
            node.create_subscription(
                RobotJointState,
                f"/{part}/joint_space_states",
                lambda msg, part=part: state_callback(msg, part),
                10,
            )
        deadline = time.monotonic() + timeout
        while seen != set(PART_NAMES):
            if errors:
                raise RuntimeError(errors[-1])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"simulator state topics unavailable: {sorted(set(PART_NAMES)-seen)}")
            executor.spin_once(timeout_sec=min(remaining, 0.1))
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


class VideoController:
    """Start/stop the simulator's third-person recorder through ROS2."""

    def __init__(self, node, service_name: str):
        from std_srvs.srv import SetBool

        self.client = node.create_client(SetBool, service_name)

    def _call(self, enabled: bool, timeout: float):
        from std_srvs.srv import SetBool

        if not self.client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"video service unavailable: {self.client.srv_name}")
        request = SetBool.Request()
        request.data = bool(enabled)
        future = self.client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for video service response")
            time.sleep(0.01)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(response.message if response is not None else "video service failed")
        try:
            return json.loads(response.message)
        except json.JSONDecodeError:
            return {"message": response.message}

    def start(self, timeout: float):
        return self._call(True, timeout)

    def stop(self, timeout: float):
        return self._call(False, timeout)

    def close(self):
        self.client.destroy()


def _initial_positions(grid_start: float, states):
    return np.asarray([
        np.interp(grid_start, states[joint][:, 0], states[joint][:, 1])
        for joint in JOINTS
    ], dtype=np.float64)


def _within_position_tolerance(latest, target):
    for index, joint in enumerate(JOINTS):
        tolerance = 1.0 if "gripper" in joint else 0.1
        if abs(latest[joint]["position"] - target[index]) > tolerance:
            return False
    return True


def preposition(robot, recorder, target, timeout: float):
    """Reach the measured real initial pose through SDK commands before timing."""
    latest = None
    consecutive = 0
    last_signature = None
    started = time.monotonic_ns()
    attempts = 0
    deadline = time.monotonic() + timeout
    next_send = time.monotonic()
    while time.monotonic() < deadline:
        if time.monotonic() >= next_send:
            robot.set_joints_position(
                list(PART_NAMES), _part_command(target),
                control_way="direct", use_wbc=False,
            )
            attempts += 1
            next_send += 1.0 / CONTROL_HZ
        _, _, latest, by_part = recorder.snapshot()
        if latest and set(latest) == set(JOINTS):
            signature = tuple(by_part.get(part, 0) for part in PART_NAMES)
            if signature != last_signature:
                fresh = all(time.monotonic_ns() - value < 100_000_000 for value in signature)
                if fresh and _within_position_tolerance(latest, target):
                    consecutive += 1
                else:
                    consecutive = 0
                last_signature = signature
                if consecutive >= 5:
                    return {
                        "duration_s": (time.monotonic_ns() - started) / 1e9,
                        "sdk_set_calls": attempts,
                        "position_tolerance": "0.1 rad angular, 1.0 percentage point gripper",
                        "position_error_by_joint": {
                            joint: latest[joint]["position"] - float(target[index])
                            for index, joint in enumerate(JOINTS)
                        },
                    }
        time.sleep(0.001)
    latest = latest or {}
    remaining = {
        joint: latest[joint]["position"] - float(target[index])
        for index, joint in enumerate(JOINTS)
        if joint in latest
    }
    raise TimeoutError(f"SDK preposition did not settle within {timeout}s; remaining_error={remaining}")


def _wait_until(deadline_ns: int):
    while True:
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 500_000:
            time.sleep((remaining - 250_000) / 1e9)
        else:
            time.sleep(0)


def _metric_for_rows(states, rows, grid_start, grid_end, epoch_system_ns, epoch_mono_ns, basis):
    per_joint = {}
    for joint in JOINTS:
        selected = [row for row in rows if row["joint"] == joint]
        if basis == "simulator_header_time":
            relative = np.asarray([
                (row["header_stamp_ns"] - epoch_system_ns) / 1e9 for row in selected
            ])
        else:
            relative = np.asarray([
                (row["received_monotonic_ns"] - epoch_mono_ns) / 1e9 for row in selected
            ])
        sim_pos = np.asarray([row["position"] for row in selected], dtype=np.float64)
        sim_vel = np.asarray([row["velocity"] for row in selected], dtype=np.float64)
        real = states[joint]
        source_time = grid_start + relative
        valid = (
            np.isfinite(relative) & np.isfinite(sim_pos) & np.isfinite(sim_vel)
            & (source_time >= max(grid_start, real[0, 0]))
            & (source_time <= min(grid_end, real[-1, 0]))
        )
        if not valid.any():
            per_joint[joint] = None
            continue
        source_time = source_time[valid]
        position_error = sim_pos[valid] - np.interp(source_time, real[:, 0], real[:, 1])
        velocity_error = sim_vel[valid] - np.interp(source_time, real[:, 0], real[:, 2])
        per_joint[joint] = {
            "position_unit": "percentage_points" if "gripper" in joint else "rad",
            "velocity_unit": "percentage_points/s" if "gripper" in joint else "rad/s",
            "position_rmse": float(np.sqrt(np.mean(position_error ** 2))),
            "position_max_abs": float(np.max(np.abs(position_error))),
            "velocity_rmse": float(np.sqrt(np.mean(velocity_error ** 2))),
            "velocity_max_abs": float(np.max(np.abs(velocity_error))),
            "samples": int(len(position_error)),
        }

    groups = {}
    for group, is_gripper in (("angular_joints", False), ("grippers", True)):
        values = [
            result for joint, result in per_joint.items()
            if result is not None and ("gripper" in joint) == is_gripper
        ]
        groups[group] = {
            "joint_count": len(values),
            "position_rmse": float(np.sqrt(np.mean([item["position_rmse"] ** 2 for item in values]))),
            "velocity_rmse": float(np.sqrt(np.mean([item["velocity_rmse"] ** 2 for item in values]))),
        } if values else None
    return {"time_basis": basis, "groups": groups, "joints": per_joint}


def _command_delivery_metrics(api_rows, sdk_rows):
    call_times = np.asarray([row["call_start_monotonic_ns"] for row in api_rows], dtype=np.int64)
    expected = np.asarray([row["position"] for row in api_rows], dtype=np.float64)
    per_joint_errors = defaultdict(list)
    matched_frames = defaultdict(set)
    unmatched_messages = Counter()
    offsets = {}
    offset = 0
    for part, count in PART_CONFIGS:
        offsets[part] = (offset, offset + count)
        offset += count

    candidates = defaultdict(list)
    for row in sdk_rows:
        start, end = offsets[row["part"]]
        local_count = end - start
        if row["mode"] != 1 or len(row["command"]) < local_count:
            unmatched_messages[row["part"]] += 1
            continue
        index = int(np.argmin(np.abs(call_times - row["received_monotonic_ns"])))
        distance_ns = abs(int(call_times[index]) - int(row["received_monotonic_ns"]))
        if distance_ns > 8_000_000:
            unmatched_messages[row["part"]] += 1
            continue
        error = np.asarray(row["command"][:local_count]) - expected[index, start:end]
        candidates[(row["part"], index)].append(
            (float(np.sqrt(np.mean(error ** 2))), error)
        )

    for (part, index), values in candidates.items():
        _, error = min(values, key=lambda item: item[0])
        start, end = offsets[part]
        matched_frames[part].add(index)
        for joint, value in zip(JOINTS[start:end], error):
            per_joint_errors[joint].append(float(value))

    joints = {}
    for joint in JOINTS:
        values = np.asarray(per_joint_errors[joint], dtype=np.float64)
        joints[joint] = {
            "unit": "percentage_points" if "gripper" in joint else "rad",
            "rmse": float(np.sqrt(np.mean(values ** 2))) if values.size else None,
            "max_abs": float(np.max(np.abs(values))) if values.size else None,
            "matched_frames": int(len(values)),
        }
    groups = {}
    for group, is_gripper in (("angular_joints", False), ("grippers", True)):
        values = [joints[joint]["rmse"] for joint in JOINTS
                  if ("gripper" in joint) == is_gripper and joints[joint]["rmse"] is not None]
        groups[group] = float(np.sqrt(np.mean(np.square(values))) if values else np.nan)
        if not values:
            groups[group] = None
    return {
        "comparison": "observed SDK command topic payload against nearest scheduled SDK API target",
        "matched_frames_by_part": {part: len(matched_frames[part]) for part in PART_NAMES},
        "unmatched_messages_by_part": {part: unmatched_messages[part] for part in PART_NAMES},
        "groups": groups,
        "joints": joints,
    }


def _state_message_stats(rows):
    result = {}
    for part in PART_NAMES:
        selected = [row for row in rows if row["part"] == part and row["header_stamp_ns"] > 0]
        stamps = sorted({row["header_stamp_ns"] for row in selected})
        result[part] = {
            "joint_samples": sum(row["part"] == part for row in rows),
            "messages": len(stamps),
            "observed_hz": (
                (len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1e9)
                if len(stamps) > 1 and stamps[-1] > stamps[0] else None
            ),
        }
    return result


def _write_parquet(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty capture: {path.name}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def replay_episode(robot, recorder, input_path, grid, command, states, output_dir,
                   preposition_timeout, video_controller=None, video_timeout=10.0):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = _initial_positions(float(grid[0]), states)
    preposition_meta = preposition(robot, recorder, target, preposition_timeout)
    video_meta = None
    if video_controller is not None:
        # Let the position controller settle before the first recorded frame.
        time.sleep(0.5)
        video_meta = video_controller.start(video_timeout)
    state_before, commands_before, _, _ = recorder.snapshot()
    epoch_mono_ns = time.monotonic_ns()
    epoch_system_ns = time.time_ns()
    api_rows = []

    try:
        for index, frame in enumerate(command):
            deadline_ns = epoch_mono_ns + index * PERIOD_NS
            _wait_until(deadline_ns)
            call_start_ns = time.monotonic_ns()
            robot.set_joints_position(
                list(PART_NAMES), _part_command(frame),
                control_way="direct", use_wbc=False,
            )
            call_end_ns = time.monotonic_ns()
            api_rows.append({
                "frame_index": index,
                "artifact_time_s": float(grid[index]),
                "relative_time_s": float(grid[index] - grid[0]),
                "scheduled_monotonic_ns": int(deadline_ns),
                "call_start_monotonic_ns": int(call_start_ns),
                "call_end_monotonic_ns": int(call_end_ns),
                "call_duration_ms": (call_end_ns - call_start_ns) / 1e6,
                "start_lateness_ms": (call_start_ns - deadline_ns) / 1e6,
                "completion_deadline_miss": call_end_ns > deadline_ns + PERIOD_NS,
                "position": frame.astype(float).tolist(),
            })
            if index % 500 == 0 or index == len(command) - 1:
                print(f"{input_path.parent.name}: SDK frame {index + 1}/{len(command)}", flush=True)
    finally:
        if video_controller is not None:
            video_meta = video_controller.stop(video_timeout)

    time.sleep(0.2)
    state_after, commands_after, _, _ = recorder.snapshot()
    sim_rows = state_after[len(state_before):]
    sdk_rows = commands_after[len(commands_before):]
    if not sim_rows:
        raise RuntimeError(f"{input_path}: no simulator state messages received during replay")
    if not sdk_rows:
        raise RuntimeError(f"{input_path}: no SDK command messages observed on ROS topics")
    state_counts = Counter(row["part"] for row in sim_rows)
    command_counts = Counter(row["part"] for row in sdk_rows)
    if set(state_counts) != set(PART_NAMES) or set(command_counts) != set(PART_NAMES):
        raise RuntimeError(
            f"{input_path}: incomplete ROS capture; "
            f"state parts={sorted(state_counts)}, command parts={sorted(command_counts)}"
        )

    _write_parquet(output_dir / "api_commands.parquet", api_rows)
    _write_parquet(output_dir / "sdk_command_messages.parquet", sdk_rows)
    _write_parquet(output_dir / "sim_states.parquet", sim_rows)
    errors = {
        "simulator_header_time": _metric_for_rows(
            states, sim_rows, float(grid[0]), float(grid[-1]), epoch_system_ns, epoch_mono_ns,
            "simulator_header_time",
        ),
        "sdk_receiver_time": _metric_for_rows(
            states, sim_rows, float(grid[0]), float(grid[-1]), epoch_system_ns, epoch_mono_ns,
            "sdk_receiver_time",
        ),
        "torque_comparable": False,
    }
    _json(output_dir / "tracking_error.json", errors)
    command_delivery = _command_delivery_metrics(api_rows, sdk_rows)
    _json(output_dir / "sdk_command_error.json", command_delivery)

    call_lateness = np.asarray([row["start_lateness_ms"] for row in api_rows])
    call_durations = np.asarray([row["call_duration_ms"] for row in api_rows])
    state_stats = _state_message_stats(sim_rows)
    state_delays = np.asarray([
        (row["received_system_ns"] - row["header_stamp_ns"]) / 1e6
        for row in sim_rows if row["header_stamp_ns"] > 0
    ])
    meta = {
        "status": "complete",
        "input_parquet": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "control_path": "Parquet 100 Hz position frames -> Astribot SDK set_joints_position(direct) -> ROS2 joint_space_command -> Isaac Lab callback -> ROS2 joint_space_states",
        "frequency_hz": CONTROL_HZ,
        "sdk_frequency_hz": CONTROL_HZ,
        "command_frames": len(command),
        "api_calls": len(api_rows),
        "sdk_command_messages_by_part": dict(command_counts),
        "sdk_command_delivery_error": command_delivery["groups"],
        "sim_state_stats_by_part": state_stats,
        "sim_state_sample_rows": len(sim_rows),
        "command_start_artifact_time_s": float(grid[0]),
        "command_end_artifact_time_s": float(grid[-1]),
        "command_span_s": float(grid[-1] - grid[0]),
        "epoch_system_ns": int(epoch_system_ns),
        "epoch_monotonic_ns": int(epoch_mono_ns),
        "preposition": preposition_meta,
        "video": video_meta,
        "api_start_lateness_mean_ms": float(call_lateness.mean()),
        "api_start_lateness_p95_ms": float(np.percentile(call_lateness, 95)),
        "api_call_duration_mean_ms": float(call_durations.mean()),
        "api_call_duration_p95_ms": float(np.percentile(call_durations, 95)),
        "api_completion_deadline_misses": int(sum(row["completion_deadline_miss"] for row in api_rows)),
        "state_message_transport_delay_ms": {
            "median": float(np.median(state_delays)),
            "p95": float(np.percentile(state_delays, 95)),
            "max": float(np.max(state_delays)),
        } if state_delays.size else None,
        "torque_comparable": False,
        "position_semantics": "SDK position-only direct command; no velocity or torque feedforward",
    }
    _json(output_dir / "run_metadata.json", meta)
    _write_summary(output_dir / "summary.md", meta, errors)
    return meta, errors


def _write_summary(path: Path, meta, errors):
    lines = [
        f"# SDK 100 Hz replay: {Path(meta['input_parquet']).parent.name}",
        "",
        "Commands were submitted through the Astribot SDK and observed on ROS2 command topics. "
        "Simulation states were captured from ROS2 state topics with both header and SDK-process receive timestamps.",
        "",
        f"- Frames / SDK API calls: {meta['command_frames']} / {meta['api_calls']}",
        f"- SDK command messages by part: {meta['sdk_command_messages_by_part']}",
        f"- SDK command delivery RMSE: {meta['sdk_command_delivery_error']}",
        f"- Simulation state messages / observed Hz by part: {meta['sim_state_stats_by_part']}",
        f"- Preposition duration: {meta['preposition']['duration_s']:.3f} s",
        f"- Maximum initial angular residual: {max(abs(value) for joint, value in meta['preposition']['position_error_by_joint'].items() if 'gripper' not in joint):.6f} rad",
        f"- API start lateness p95: {meta['api_start_lateness_p95_ms']:.3f} ms",
        f"- API calls completing after the next 10 ms deadline: {meta['api_completion_deadline_misses']}",
        f"- Simulator header-to-receiver delay p95: {meta['state_message_transport_delay_ms']['p95']:.3f} ms",
        "",
        "| Time basis | Group | Position RMSE | Velocity RMSE |",
        "| --- | --- | ---: | ---: |",
    ]
    units = {"angular_joints": ("rad", "rad/s"), "grippers": ("percentage points", "percentage points/s")}
    for basis_name, result in errors.items():
        if not isinstance(result, dict) or "groups" not in result:
            continue
        for group, metrics in result["groups"].items():
            if metrics is None:
                continue
            q_unit, v_unit = units[group]
            lines.append(
                f"| {basis_name} | {group} | {metrics['position_rmse']:.6f} {q_unit} "
                f"| {metrics['velocity_rmse']:.6f} {v_unit} |"
            )
    lines.extend([
        "",
        "Per-joint metrics are in tracking_error.json. Torque is recorded for diagnosis only; "
        "its comparability with real feedback is not established.",
        "",
    ])
    path.write_text("\n".join(lines))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("parquet", nargs="+", type=Path, help="100 Hz resampled episode Parquet files")
    result.add_argument("--output-root", type=Path, default=Path("results/alignment_sdk_100hz"))
    result.add_argument("--connect-timeout", type=float, default=30.0)
    result.add_argument("--preposition-timeout", type=float, default=10.0)
    result.add_argument("--video-service", help="enable third-person capture via this simulator ROS service")
    result.add_argument("--video-timeout", type=float, default=20.0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if os.environ.get("ROS_VERSION") != "2" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("source ROS2 and SDK environments, then set ROS_LOCALHOST_ONLY=1")
    if not os.environ.get("ROS_DOMAIN_ID"):
        raise RuntimeError("set an isolated ROS_DOMAIN_ID before starting simulator and SDK replay")

    episodes = [load_episode(path) for path in args.parquet]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Waiting for simulation state topics in ROS domain {os.environ['ROS_DOMAIN_ID']} "
        f"(localhost-only)...",
        flush=True,
    )
    wait_for_simulation(args.connect_timeout)

    from astribot_ros_middleware import astribot_ros_middleware as middleware
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    robot = None
    recorder = None
    video_controller = None
    results = []
    try:
        robot = Astribot(freq=CONTROL_HZ)
        if not robot.get_control_rights_status():
            raise RuntimeError("SDK did not acquire control rights from the simulation")
        recorder = TopicRecorder(robot)
        recorder.wait_for_state(args.connect_timeout)
        if args.video_service:
            video_controller = VideoController(robot.astribot_interface.node, args.video_service)
        for path, grid, command, states in episodes:
            output_dir = args.output_root / path.parent.name
            meta, errors = replay_episode(
                robot, recorder, path, grid, command, states, output_dir,
                args.preposition_timeout,
                video_controller=video_controller,
                video_timeout=args.video_timeout,
            )
            results.append({"episode": path.parent.name, "metadata": meta, "errors": errors})
    finally:
        if recorder is not None:
            recorder.close()
        if video_controller is not None:
            video_controller.close()
        if robot is not None:
            try:
                robot.stop_robot()
            finally:
                middleware.shutdown()

    _json(args.output_root / "summary.json", results)
    print(f"SDK replay results saved under {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, TimeoutError, ImportError) as error:
        print(f"SDK replay failed: {error}", file=sys.stderr)
        raise SystemExit(1)
