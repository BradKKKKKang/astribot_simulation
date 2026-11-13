"""CPU USD FK and shared geometry export; never starts SimulationApp."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .common import atomic_json, digest, read_json


def error(a, b):
    return {'position_mm': float(np.linalg.norm(np.asarray(a[:3]) - b[:3]) * 1000),
            'angle_deg': float((Rotation.from_quat(a[3:]).inv() * Rotation.from_quat(b[3:])).magnitude() * 180 / np.pi)}


def statistics(values):
    return {name: float(f(values)) for name, f in [('mean', np.mean), ('p95', lambda a: np.percentile(a, 95)), ('max', np.max)]} if values else None


def model(config):
    from src.placement_analysis.robot_export import open_robot_stage, geometry_cache_key
    from src.placement_analysis.reference_pose import UsdKinematics
    from src.simulation_config.robot_profile import default_robot_profile
    profile = default_robot_profile()
    stage, root, names = open_robot_stage(profile.model_path, root_path=profile.root_path)
    kinematics = UsdKinematics(stage, root.GetPath(), names)
    return profile, stage, root, names, kinematics, geometry_cache_key(profile.model_path, names, root_path=root.GetPath(), stage=stage)


def snapshot(profile, kinematics, q):
    targets = {part: q[first:end] for part, first, end in profile.sdk_parts if 'gripper' not in part}
    joints = profile.sdk_joint_positions({'joint_targets_rad': targets, 'gripper_percent': {'left': q[14], 'right': q[22]}})
    return kinematics.snapshot(joints)['body_poses_world']


def prepare(config):
    from src.placement_analysis.usd_export import export_robot_glb
    from .sdk_worker import initial_joints
    profile, stage, root, names, kinematics, geometry_key = model(config)
    output = Path(config['cache']) / 'models' / geometry_key
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'robot.glb'
    if not path.exists():
        temporary = output / '.robot.glb'
        export_robot_glb(stage, root.GetPath(), names, temporary)
        temporary.replace(path)
    metadata = {'body_names': names, 'base_body': profile.base_body, 'geometry_key': geometry_key,
                'kinematics_fingerprint': kinematics.fingerprint(), 'quaternion_order': 'wxyz',
                'initial_snapshot': {'body_poses_world': snapshot(profile, kinematics, initial_joints(read_json(config['initial_pose'])).tolist())},
                'glb_url': f'/artifacts/models/{geometry_key}/robot.glb'}
    atomic_json(Path(config['cache']) / 'model.json', metadata)
    return metadata


def run(job, directory):
    profile, stage, root, names, kinematics, geometry_key = model(job['config'])
    if kinematics.fingerprint() != job['model']['kinematics_fingerprint']:
        raise ValueError('USD changed during computation')
    sdk = read_json(directory / 'sdk.json')
    frames = []
    consistency = []
    t0 = time.monotonic()
    endpoints = {'torso': 'astribot_torso_link_4', 'left': 'astribot_arm_left_link_7', 'right': 'astribot_arm_right_link_7'}
    for index, row in enumerate(sdk['frames']):
        poses = snapshot(profile, kinematics, row['joints'])
        achieved, errors = {}, {}
        targets = job['trajectory']['frames'][index][job['mode']]['targets']
        for side, name in endpoints.items():
            pose = poses[names.index(name)]
            rotation = Rotation.from_quat([*pose[4:], pose[3]])
            position = np.asarray(pose[:3]) + rotation.apply(job['config']['tool_offsets_m'][side]) - job['config']['sdk_to_usd_translation_m']
            achieved[side] = [*position.tolist(), *rotation.as_quat().tolist()]
            consistency.append(error(achieved[side], row['sdk_fk'][side]))
            errors[side] = error(achieved[side], targets[side]) if row['ik_success'] and targets[side] else None
        thresholds = job['config']['thresholds']
        anomaly = (not row['ik_success'] or bool(row['limit_violations']) or row['joint_jump_deg'] > thresholds['joint_jump_deg']
                   or any(v and (v['position_mm'] > thresholds['position_mm'] or v['angle_deg'] > thresholds['angle_deg']) for v in errors.values()))
        frames.append({**row, 'body_poses_world': poses, 'achieved': achieved, 'errors': errors, 'anomaly': bool(anomaly)})
    summary = {'frames': len(frames), 'success_frames': sum(f['ik_success'] for f in frames),
               'failed_frames': [i for i, f in enumerate(frames) if not f['ik_success']],
               'anomaly_frames': [i for i, f in enumerate(frames) if f['anomaly']],
               'sdk_usd_consistency': {metric: statistics([e[metric] for e in consistency]) for metric in ('position_mm', 'angle_deg')},
               'per_arm': {side: {metric: statistics([f['errors'][side][metric] for f in frames if f['errors'][side]])
                                 for metric in ('position_mm', 'angle_deg')} for side in ('left', 'right')},
               'max_joint_jump_deg': max(f['joint_jump_deg'] for f in frames),
               'max_torso_drift_deg': max(f['torso_drift_deg'] for f in frames),
               'sdk_solve_s': sum(f['solve_s'] for f in frames), 'seed_wait_s': sum(f['seed_wait_s'] for f in frames),
               'usd_fk_s': time.monotonic() - t0, 'command_messages_observed': sdk['command_messages_observed']}
    size = job['config']['chunk_size']
    for start in range(0, len(frames), size):
        atomic_json(directory / f'chunk-{start // size:05d}.json', {'start': start, 'frames': frames[start:start + size]})
    atomic_json(directory / 'result.json', {'key': job['key'], 'episode_id': job['episode_id'], 'mode': job['mode'],
                'summary': summary, 'model': job['model'], 'chunk_size': size,
                'chunks': (len(frames) + size - 1) // size, 'provenance': job['provenance'],
                'basis': 'source base = SDK chassis (unverified across robots); USD z = chassis z - 0.097 m',
                'tool_offsets_m': job['config']['tool_offsets_m']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.prepare:
        prepare(read_json(args.prepare))
    else:
        run(read_json(args.job), Path(args.job).parent)


if __name__ == '__main__':
    main()
