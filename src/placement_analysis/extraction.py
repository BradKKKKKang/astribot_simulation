"""Extract placement candidates from completed SDK -> ROS -> Isaac captures.

Invoked by scripts/replay_extract_event.py. Extraction is CPU-only, requires pxr,
and never starts simulation, imports the SDK, or sends ROS commands.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .events import DetectorConfig
from .recorder import PlacementRecorder, save_bundle
from .reference_pose import UsdKinematics
from .ros_feedback import JOINTS, feedback_joint_positions, synchronize_feedback
from .robot_export import DEFAULT_USD, ROOT, open_robot_stage
from .usd_export import export_robot_glb

DEFAULT_DETECTOR_CONFIG = ROOT / 'config/placement_analysis.json'


def load_capture(path, detector):
    path = path.expanduser().resolve(strict=True)
    metadata = json.loads((path / 'run_metadata.json').read_text())
    if 'Astribot SDK' not in metadata.get('control_path', '') or 'Isaac Lab' not in metadata['control_path']:
        raise ValueError(f'{path}: SDK -> ROS -> Isaac capture required')
    rows = pq.read_table(path / 'sim_states.parquet').to_pylist()
    api = pq.read_table(path / 'api_commands.parquet').to_pylist()
    times, positions, quality = synchronize_feedback(rows, api, metadata, JOINTS, detector.max_gap_s)
    return path, metadata, times, positions, quality


def discover_captures(inputs):
    """Expand a replay batch only when every expected episode completed."""
    paths = []
    for value in inputs:
        path = value.expanduser().resolve(strict=True)
        manifest = path / 'batch_manifest.json'
        if (path / 'run_metadata.json').is_file():
            paths.append(path)
        elif manifest.is_file():
            batch = json.loads(manifest.read_text())
            ids = [episode['episode_id'] for episode in batch['episodes']]
            if (not ids or len(set(ids)) != len(ids) or batch.get('status') != 'complete'
                    or batch.get('completed') != ids
                    or any(Path(name).name != name or name in ('.', '..') for name in ids)):
                raise ValueError(f'{path}: batch is incomplete or has invalid episode IDs; finish replay first')
            paths.extend(path / name for name in ids)
        else:
            raise ValueError(f'{path}: expected an episode capture or a completed replay batch')
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('episode directory names must be unique in one bundle')
    return paths


def detector_settings(detector_config):
    settings = json.loads(Path(detector_config).expanduser().read_text())
    detector = DetectorConfig(**{k: v for k, v in settings.items()
                                 if k in DetectorConfig.__dataclass_fields__})
    return settings, detector


def prepare_extraction(usd=DEFAULT_USD, detector_config=DEFAULT_DETECTOR_CONFIG):
    """Check USD/model/TCP contracts on the CPU before replay acquires control."""
    settings, _ = detector_settings(detector_config)
    stage, root, names = open_robot_stage(usd)
    model = UsdKinematics(stage, root.GetPath(), names)
    if set(feedback_joint_positions(JOINTS, [0.0] * len(JOINTS))[0]) != set(model.joint_names):
        raise ValueError('USD joints do not match the SDK feedback mapping')
    PlacementRecorder(None, settings, 'preflight', body_names=names, joint_names=model.joint_names)
    return stage, root, names, model


def extract_events(inputs, output, *, usd=DEFAULT_USD,
                   detector_config=DEFAULT_DETECTOR_CONFIG, dry_run=False):
    settings, detector = detector_settings(detector_config)
    captures = discover_captures(inputs)
    # Validate sequentially; a full dataset must not retain all feedback arrays.
    for capture_path in captures:
        path, _, times, _, quality = load_capture(capture_path, detector)
        print(f'Validated {path.name}: {len(times)} feedback samples; {quality["gap_count"]} gaps', flush=True)
    if dry_run:
        print('Captures valid. No simulation, ROS, SDK, USD loading or output writes.')
        return None
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite existing bundle: {output}')
    model_path = Path(usd).expanduser().resolve(strict=True)
    stage, root, names, model = prepare_extraction(model_path, detector_config)
    output.mkdir(parents=True, exist_ok=True)
    geometry = export_robot_glb(stage, root.GetPath(), names, output / 'robot.glb')
    all_events, episodes, reference = [], [], None
    for capture_path in captures:
        path, capture_meta, times, positions, quality = load_capture(capture_path, detector)
        recorder = PlacementRecorder(None, settings, path.name,
                                     body_names=names, joint_names=model.joint_names)
        for timestamp, values in zip(times, positions):
            joints, grippers = feedback_joint_positions(JOINTS, values)
            snapshot = model.snapshot(joints)
            recorder.append_sample(timestamp, snapshot['body_poses_world'], snapshot['joint_positions_rad'], grippers)
        events = recorder.finish()
        all_events.extend(events)
        if reference is None:
            reference = recorder.snapshot(0)
        episodes.append({'episode_id': path.name, 'source': capture_meta['input_parquet'],
                         'sha256': capture_meta['input_sha256'], 'capture_directory': str(path),
                         'capture_sha256': {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                            for name in ('sim_states.parquet', 'api_commands.parquet', 'run_metadata.json')},
                         'frames': len(times), 'event_count': len(events), 'feedback_quality': quality,
                         'default_excluded_count': sum(bool(e['exclusion_reasons']) for e in events),
                         'candidate_counts': {kind: sum(e['event_type'] == kind for e in events)
                                              for kind in ('grasp', 'release', 'dwell', 'press')},
                         'sample_start_s': float(times[0]), 'sample_end_s': float(times[-1])})
        metadata = {
            'units': 'meters', 'up_axis': 'Z', 'quaternion_order': 'wxyz',
            'position_source': 'USD link-frame FK from synchronized Isaac ROS joint feedback',
            'gripper_source': 'Isaac ROS measured SDK percentage, clipped to [0,100]',
            'replay_mode': 'Astribot SDK -> ROS2 command -> Isaac Lab -> ROS2 state capture; CPU extraction',
            'time_basis': 'ROS header wall time; not PhysX simulation time',
            'base_assumption': 'fixed base at authored USD world pose; no mobile base reconstruction',
            'finger_pose_source': 'other fingers inferred from measured L1 percentage and signed linkage',
            'placement_meaning': 'TCP transition and motion candidates; not object centers or confirmed contact',
            'candidate_policy': 'preserve both detectors and startup candidates; filter only in viewer',
            'motion_basis': ('world Z down/bottom/up with time smoothing, bounded XY excursion and gripper gating'
                             if detector.motion_detector == 'press' else
                             'world TCP translation; both speed and acceleration must be low'),
            'initial_phase_basis': 'per-hand sustained departure from initial TCP in base coordinates',
            'detector': asdict(detector), 'tcp': settings['tcp'],
            'tcp_calibration': settings.get('tcp_calibration', ''),
            'base_body': settings['base_body'], 'body_names': names, 'joint_names': model.joint_names,
            'robot_usd': str(model_path), 'robot_usd_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'robot_geometry': geometry, 'episodes': episodes,
            'status': 'complete' if len(episodes) == len(captures) else 'partial',
        }
        save_bundle(output, all_events, metadata, reference)
        print(f'{path.name}: saved {len(events)} candidates', flush=True)
    print(f'PLACEMENT_SAVED: {output / "placement.json"}', flush=True)
    return output / 'placement.json'
