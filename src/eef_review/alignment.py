"""Fixed per-episode SE(3) placement of saved EEF targets into SDK chassis."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('alignment requires a finite xyz/xyzw pose')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def place(trajectory, policy=None, initial_tcp=None):
    """Use State frame zero for both channels; retain the untouched source data."""
    mode = (policy or {}).get('mode', 'chassis')
    result = copy.deepcopy(trajectory)
    result['alignment'] = {'mode': mode, 'target_frame': 'sdk_chassis', 'per_arm': {}}
    if mode == 'chassis':
        return result
    if mode != 'episode_first_state':
        raise ValueError('unknown EEF alignment mode')
    if not initial_tcp:
        raise ValueError('SDK initial TCP poses are required for episode placement')
    first = result['frames'][0]['state']['targets']
    for side in ('left', 'right'):
        active = any(frame[channel]['targets'][side] is not None for frame in result['frames']
                     for channel in ('state', 'action'))
        if not active:
            continue
        if first[side] is None:
            raise ValueError('missing first State EEF pose: ' + side)
        anchor = initial_tcp['astribot_arm_' + side]
        transform = pose_matrix(anchor) @ np.linalg.inv(pose_matrix(first[side]))
        result['alignment']['per_arm'][side] = {
            'data_to_chassis': transform.tolist(), 'source_state0': first[side], 'target_initial': anchor}
    for frame in result['frames']:
        for channel in ('state', 'action'):
            item = frame[channel]
            item['source_targets'] = copy.deepcopy(item['targets'])
            for side, placement in result['alignment']['per_arm'].items():
                pose = item['source_targets'][side]
                if pose is not None:
                    transformed = np.asarray(placement['data_to_chassis']) @ pose_matrix(pose)
                    item['targets'][side] = [*transformed[:3, 3].tolist(),
                                             *Rotation.from_matrix(transformed[:3, :3]).as_quat().tolist()]
    result['alignment'].update(anchor='first State frame; shared by State and Action',
                               preserves_original_bimanual_geometry=False)
    return result
