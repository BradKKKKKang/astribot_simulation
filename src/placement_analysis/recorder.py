"""Isaac Lab adapter and portable event-only output (no trajectory output)."""
import json
from pathlib import Path

import numpy as np

from .events import (DetectorConfig, compose_pose, detect_presses, detect_stops, detect_transitions,
                     initial_phase_mask, motion_metrics, relative_pose)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")) + "\n")


class PlacementRecorder:
    """Collect feedback poses, then persist gripper and configured motion candidates.

    Body poses are link frames, not centers of mass; do not substitute COM poses.
    Gripper state is measured feedback in SDK percentage units.
    """

    def __init__(self, env, settings, episode_id, *, body_names=None, joint_names=None):
        self.env, self.settings, self.episode_id = env, settings, str(episode_id)
        fields = DetectorConfig.__dataclass_fields__
        self.config = DetectorConfig(**{k: v for k, v in settings.items() if k in fields})
        self.body_names = list(env.robot.body_names if env is not None else body_names)
        self.joint_names = list(env.robot.data.joint_names if env is not None else joint_names)
        self.base_index = self.body_names.index(settings['base_body'])
        self.tcp_indices, self.tcp_local, self.gripper_indices = {}, {}, {}
        for side in ('left', 'right'):
            tcp = settings['tcp'][side]
            self.tcp_indices[side] = self.body_names.index(tcp['body'])
            local = np.asarray([*tcp['position'], *tcp['quaternion_wxyz']], float)
            if local.shape != (7,) or not np.isfinite(local).all() or not np.isclose(np.linalg.norm(local[3:]), 1):
                raise ValueError(f'{side}: TCP must contain a finite position and unit quaternion')
            self.tcp_local[side] = local
            self.gripper_indices[side] = self.joint_names.index(f'astribot_gripper_{side}_joint_L1')
        self.times, self.poses, self.joints, self.openings = [], [], [], []

    def sample(self, timestamp):
        data = self.env.robot.data
        # Isaac Lab 2.1 exposes body_link_pose_w; body_pos_w/body_quat_w are
        # link-frame compatibility aliases in this repository's Lab version.
        try:
            tensor = data.body_link_pose_w
        except AttributeError:
            import torch
            tensor = torch.cat((data.body_pos_w, data.body_quat_w), dim=-1)
        poses = tensor[0].detach().cpu().numpy().astype(float, copy=True)
        joints = data.joint_pos[0].detach().cpu().numpy().astype(float, copy=True)
        opening = [self.env._gripper_joint_to_percent(
            f'astribot_gripper_{side}_joint_L1', joints[self.gripper_indices[side]])
            for side in ('left', 'right')]
        self.append_sample(timestamp, poses, joints, opening)

    def append_sample(self, timestamp, poses, joints, opening):
        """Append synchronized feedback/FK data without an Isaac environment."""
        poses, joints, opening = (np.asarray(value, dtype=float) for value in (poses, joints, opening))
        if not np.isfinite(timestamp) or (self.times and timestamp <= self.times[-1]):
            raise ValueError('placement sample time must increase strictly and be finite')
        if (poses.shape != (len(self.body_names), 7) or joints.shape != (len(self.joint_names),)
                or opening.shape != (2,) or not all(np.isfinite(v).all() for v in (poses, joints, opening))
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1)):
            raise ValueError('invalid placement sample shapes, values or rotations')
        self.times.append(float(timestamp))
        self.poses.append(poses.copy())
        self.joints.append(joints.copy())
        # Physical tracking can overshoot the commanded endpoints slightly.
        self.openings.append(np.clip(opening, 0, 100))

    def snapshot(self, index):
        return {'body_poses_world': self.poses[index].tolist(),
                'joint_positions_rad': self.joints[index].tolist()}

    def finish(self):
        if not self.times:
            raise ValueError('no placement samples collected')
        events = []
        for side_index, side in enumerate(('left', 'right')):
            tcp_world = np.asarray([compose_pose(p[self.tcp_indices[side]], self.tcp_local[side])
                                    for p in self.poses])
            tcp_base = np.asarray([relative_pose(p[self.base_index], tcp)
                                   for p, tcp in zip(self.poses, tcp_world)])
            speed, acceleration = motion_metrics(self.times, tcp_world[:, :3], self.config)
            initial, departure, distance = initial_phase_mask(self.times, tcp_base[:, :3], self.config)
            openings = np.asarray(self.openings)[:, side_index]
            motion = (detect_presses(self.times, tcp_world[:, :3], openings, self.config)
                      if self.config.motion_detector == 'press' else
                      detect_stops(self.times, tcp_world[:, :3], speed, acceleration, self.config))
            candidates = detect_transitions(self.times, openings, self.config) + motion
            for transition in candidates:
                i = transition['frame_index']
                base = self.poses[i][self.base_index]
                velocity = float(speed[i]) if np.isfinite(speed[i]) else None
                accel = float(acceleration[i]) if np.isfinite(acceleration[i]) else None
                events.append({
                    'episode_id': self.episode_id, 'side': side, **transition,
                    'timestamp_s': self.times[i],
                    'transition_start_s': self.times[transition['transition_start_index']],
                    'transition_end_s': self.times[transition['transition_end_index']],
                    'confirmed_time_s': self.times[transition['confirmed_index']],
                    'tcp_pose_world': tcp_world[i].tolist(),
                    'tcp_pose_base': tcp_base[i].tolist(),
                    'base_pose_world': base.tolist(),
                    'gripper_percent': float(openings[i]),
                    'speed_m_s': velocity, 'acceleration_m_s2': accel,
                    'low_motion': (velocity is not None and accel is not None and
                                   velocity <= self.config.speed_threshold_m_s and
                                   accel <= self.config.acceleration_threshold_m_s2),
                    'detection_sources': ['motion' if transition['event_type'] in ('dwell', 'press') else 'gripper'],
                    'related_event_ids': [],
                    'episode_elapsed_s': self.times[i] - self.times[0],
                    'initial_phase': bool(initial[i]),
                    'initial_distance_m': float(distance[i]),
                    'departure_time_s': self.times[departure] if departure is not None else None,
                    'exclusion_reasons': (['initial_pose'] if initial[i] else [])
                                         + (['initial_time'] if self.times[i] - self.times[0] < self.config.initial_ignore_s else []),
                    'snapshot': self.snapshot(i),
                })
        events.sort(key=lambda event: (event['timestamp_s'], event['side']))
        for i, event in enumerate(events):
            event['event_id'] = f'{self.episode_id}:{i:04d}'
        # Associate only the same hand and startup phase. A slow frame alone
        # does not prove a stop: the independent sustained window must overlap
        # the gripper transition (with an explicit time tolerance).
        stops = [event for event in events if event['event_type'] in ('dwell', 'press')]
        for event in events:
            if event['event_type'] in ('dwell', 'press'):
                continue
            for stop in stops:
                if stop['side'] != event['side'] or stop['initial_phase'] != event['initial_phase']:
                    continue
                if (event['transition_start_s'] <= stop['transition_end_s'] + self.config.coincidence_tolerance_s
                        and event['confirmed_time_s'] >= stop['transition_start_s'] - self.config.coincidence_tolerance_s):
                    if 'motion' not in event['detection_sources']:
                        event['detection_sources'].append('motion')
                    event['related_event_ids'].append(stop['event_id'])
                    stop['related_event_ids'].append(event['event_id'])
        return events


def save_bundle(output, events, metadata, reference_snapshot):
    """JSON is the browser interchange; Parquet preserves typed analysis fields."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pose = pa.list_(pa.float64(), 7)
    schema = pa.schema([
        ('event_id', pa.string()), ('episode_id', pa.string()), ('side', pa.string()),
        ('event_type', pa.string()), ('timestamp_s', pa.float64()),
        ('transition_start_s', pa.float64()), ('transition_end_s', pa.float64()),
        ('confirmed_time_s', pa.float64()), ('frame_index', pa.int64()),
        ('transition_start_index', pa.int64()), ('transition_end_index', pa.int64()),
        ('confirmed_index', pa.int64()), ('tcp_pose_world', pose), ('tcp_pose_base', pose),
        ('base_pose_world', pose), ('gripper_percent', pa.float64()),
        ('speed_m_s', pa.float64()), ('acceleration_m_s2', pa.float64()),
        ('low_motion', pa.bool_()),
        ('press_depth_m', pa.float64()), ('press_retract_m', pa.float64()),
        ('press_lateral_m', pa.float64()),
        ('detection_sources', pa.list_(pa.string())), ('related_event_ids', pa.list_(pa.string())),
        ('episode_elapsed_s', pa.float64()), ('initial_phase', pa.bool_()),
        ('initial_distance_m', pa.float64()), ('departure_time_s', pa.float64()),
        ('exclusion_reasons', pa.list_(pa.string())),
        ('body_poses_world', pa.list_(pose)),
        ('joint_positions_rad', pa.list_(pa.float64())),
    ])
    rows = [{**{k: v for k, v in event.items() if k != 'snapshot'}, **event['snapshot']}
            for event in events]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / 'events.parquet')
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'placement.json', {
        'schema_version': 3 if metadata.get('detector', {}).get('motion_detector') == 'press' else 2,
        'metadata': metadata, 'robot_model': 'robot.glb',
        'reference_snapshot': reference_snapshot, 'events': events,
    })
