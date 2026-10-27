"""Isaac Lab adapter and portable event-only output (no trajectory output)."""
import json
from pathlib import Path

import numpy as np

from .events import (DetectorConfig, compose_pose, detect_presses, detect_transitions,
                     initial_phase_mask, motion_metrics, relative_pose)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
    temporary.replace(path)


class PlacementRecorder:
    """Collect feedback poses, then persist only the configured task's candidates.

    Body poses are link frames, not centers of mass; do not substitute COM poses.
    Gripper state is measured feedback in SDK percentage units.
    """

    def __init__(self, env, settings, episode_id, *, body_names=None, joint_names=None,
                 snapshot_provider=None):
        self.env, self.settings, self.episode_id = env, settings, str(episode_id)
        self.snapshot_provider = snapshot_provider
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
            if env is not None:
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
        if self.snapshot_provider is not None:
            return self.snapshot_provider(index)
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
            initial, departure, distance = initial_phase_mask(self.times, tcp_base[:, :3], self.config)
            openings = np.asarray(self.openings)[:, side_index]
            candidates = (detect_presses(self.times, tcp_world[:, :3], openings, self.config)
                          if self.config.type == 'press' else
                          detect_transitions(self.times, openings, self.config))
            speed, acceleration = motion_metrics(
                self.times, tcp_world[:, :3], self.config,
                indices=[candidate["frame_index"] for candidate in candidates])
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
        exclude_initial = (self.settings.get('event_rules') or {}).get('exclude_initial', True)
        for i, event in enumerate(events):
            event['analysis_exclusion_reasons'] = list(event['exclusion_reasons']) if exclude_initial else []
            event['analysis_selected'] = not event['analysis_exclusion_reasons']
            event['event_id'] = f'{self.episode_id}:{i:04d}'
        return events


def save_bundle(output, events, metadata, reference_snapshot):
    """JSON is the browser interchange; Parquet preserves typed analysis fields."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for event in events:
        event.setdefault('object', None)
        event.setdefault('identification', {'status': 'not_requested'})
    pose = pa.list_(pa.float64(), 7)
    source = pa.struct([
        *[(key, pa.string()) for key in ('mapping_policy', 'parquet', 'parquet_sha256',
           'metadata_file', 'metadata_sha256', 'camera', 'video', 'video_sha256',
           'original_parquet_sha256', 'source_relocation_file', 'source_relocation_sha256')],
        *[(key, pa.int64()) for key in ('mapping_version', 'parquet_row', 'dataset_frame_index',
           'dataset_episode_index', 'command_index', 'source_frame_index', 'frame_offset',
           'video_frame_count', 'video_frame_index')],
        *[(key, pa.float64()) for key in ('event_time_s', 'command_time_s', 'source_video_time_s',
           'video_fps', 'video_time_s')],
    ])
    identification = pa.struct([
        *[(key, pa.string()) for key in ('status', 'candidate_key', 'screenshot', 'screenshot_sha256',
                                        'error', 'fingerprint', 'raw_response', 'selection_policy')],
        ('probabilities', pa.map_(pa.string(), pa.float64())), ('source', source),
    ])
    schema = pa.schema([
        ('event_id', pa.string()), ('episode_id', pa.string()), ('side', pa.string()),
        ('object', pa.string()), ('identification', identification),
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
    rows = [{**{k: v for k, v in event.items() if k != 'snapshot'}, **(event.get('snapshot') or {})}
            for event in events]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / 'events.parquet.tmp')
    (output / 'events.parquet.tmp').replace(output / 'events.parquet')
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'placement.json', {
        'schema_version': 4,
        'metadata': metadata, 'robot_model': 'robot.glb',
        'reference_snapshot': reference_snapshot, 'events': events,
    })
