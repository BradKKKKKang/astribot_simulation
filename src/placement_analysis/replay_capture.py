"""Per-episode ROS feedback capture for replay_extract_event; no alignment runner."""
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .robot_profile import default_robot_profile

DEFAULT_PROFILE = default_robot_profile()


def state_rows(msg, part, profile=None):
    """Validate the complete part payload before recording any of its joints."""
    profile = profile or DEFAULT_PROFILE
    expected = [joint for joint, owner in profile.feedback_parts.items() if owner == part]
    if not expected:
        raise ValueError(f'{part}: unknown feedback part')
    names = [profile.feedback_aliases.get(name, name) for name in msg.name]
    if msg.header.frame_id != 'simulation' or len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f'{part}: expected complete named simulation feedback')
    if len(msg.position) != len(names) or not all(math.isfinite(float(v)) for v in msg.position):
        raise ValueError(f'{part}: invalid feedback position payload')
    stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    if stamp <= 0:
        raise ValueError(f'{part}: invalid feedback header stamp')
    return [dict(part=part, joint=name, position=float(value), header_stamp_ns=stamp,
                 frame_id='simulation') for name, value in zip(names, msg.position)]


class ReplayCapture:
    """Subscribe on the SDK's existing executor; keep at most one episode."""

    def __init__(self, robot, profile=None):
        from astribot_msgs.msg import RobotJointState
        from rclpy.qos import qos_profile_sensor_data

        self.profile = profile or DEFAULT_PROFILE
        self.node = robot.astribot_interface.node
        self.lock = threading.Lock()
        self.latest, self.rows, self.calls, self.errors = {}, [], [], []
        self.active = False
        self.window = None
        self.dataset_end_monotonic_ns = None
        self.subscriptions = []
        try:
            for part, _ in self.profile.part_configs:
                self.subscriptions.append(self.node.create_subscription(
                    RobotJointState, f'/{part}/joint_space_states',
                    lambda msg, part=part: self.on_state(msg, part), qos_profile_sensor_data))
        except BaseException:
            self.close()
            raise

    def on_state(self, msg, part):
        try:
            rows = state_rows(msg, part, getattr(self, "profile", DEFAULT_PROFILE))
            with self.lock:
                previous = self.latest.get(part)
                if previous and rows[0]['header_stamp_ns'] <= previous[0]['header_stamp_ns']:
                    raise ValueError(f'{part}: feedback clock repeated or moved backwards')
                self.latest[part] = rows
                if self.active:
                    self.rows.extend(rows)
        except Exception as error:
            with self.lock:
                self.errors.append(str(error))

    def begin(self):
        with self.lock:
            self.epoch_mono = time.monotonic_ns()
            self.epoch_wall = time.time_ns()
            self.window = [self.epoch_mono, None]
            self.rows = [row for rows in self.latest.values() for row in rows]
            self.active = True

    def record_call(self, source_frame, source_time, command, started, ended, *, phase='dataset'):
        self.calls.append(dict(frame_index=len(self.calls), source_frame_index=int(source_frame),
                               source_time_s=float(source_time), phase=phase,
                               call_start_monotonic_ns=int(started), call_end_monotonic_ns=int(ended),
                               position=command[list(getattr(self, 'profile', DEFAULT_PROFILE).command_indices)].astype(float).tolist()))
        if phase != 'initial':
            self.dataset_end_monotonic_ns = int(ended)

    def end_window(self):
        # Feedback collection remains active through the asynchronous drain,
        # but the synchronization window ends at the last dataset command.
        # This keeps terminal state samples available for interpolation without
        # treating a post-command hold as part of the recorded trajectory.
        end = self.dataset_end_monotonic_ns
        if end is None:
            raise ValueError('dataset command window has no command endpoint')
        self.window[1] = end

    def feedback_covers_window(self):
        """Wait for a header sample after the final command on every part."""
        with self.lock:
            if self.errors:
                return True  # save() persists and reports the callback error.
            end_stamp = self.epoch_wall + self.window[1] - self.epoch_mono
            return all(self.latest.get(part)
                       and self.latest[part][0]['header_stamp_ns'] >= end_stamp
                       for part, _ in getattr(self, "profile", DEFAULT_PROFILE).part_configs)

    def save(self, output, source, settings):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from .ros_feedback import synchronize_feedback

        with self.lock:
            self.active = False
            errors = list(self.errors)
            rows, self.rows = self.rows, []
        metadata = {
            'status': 'complete', 'capture_schema': 'replay_parquet_v1',
            'input_parquet': source['input_parquet'], 'input_sha256': source['input_sha256'],
            'source': source, 'replay_settings': settings,
            'control_path': 'raw Parquet -> Astribot SDK -> ROS2 -> Isaac Lab -> ROS2 state capture',
            'command_frames': len(self.calls), 'source_frames': source['frames'],
            'epoch_monotonic_ns': self.epoch_mono, 'epoch_system_ns': self.epoch_wall,
            'dataset_window_monotonic_ns': self.window,
            'position_source': 'Isaac ROS measured joint positions; not dataset targets',
            'time_basis': 'ROS header wall time; not PhysX simulation time',
            'joint_names': list(getattr(self, 'profile', DEFAULT_PROFILE).joint_names),
            'declared_robot_control': getattr(self, 'profile', DEFAULT_PROFILE).control_provenance(),
        }
        if source.get('task_config'):
            metadata['task_config'] = source['task_config']
        # Preserve failed recordings too: strict synchronization can then be
        # diagnosed offline, without rerunning robot motion or inventing data.
        failure = None
        try:
            if errors:
                raise RuntimeError(errors[-1])
            profile = getattr(self, 'profile', DEFAULT_PROFILE)
            _, _, metadata['feedback_quality'] = synchronize_feedback(
                rows, self.calls, metadata, profile.joint_names, settings.get('alignment_max_gap_s', .1),
                anchor_joint=profile.anchor_joint, part_by_joint=profile.feedback_parts)
        except (ValueError, RuntimeError) as error:
            failure = error
            metadata.update(status='failed', error=str(error))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), output / 'sim_states.parquet', compression='zstd')
        pq.write_table(pa.Table.from_pylist(self.calls), output / 'api_commands.parquet', compression='zstd')
        metadata['capture_sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                      for name in ('sim_states.parquet', 'api_commands.parquet')}
        (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        if failure is not None:
            raise RuntimeError(f'{failure}; raw capture saved in {output}') from failure
        return metadata

    def close(self):
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions.clear()
