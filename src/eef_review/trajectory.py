"""Decode already-corrected 34D EEF without applying source transforms twice."""
import numpy as np
from scipy.spatial.transform import Rotation

POSE_SLICES = {'torso': (0, 9), 'left': (9, 18), 'right': (19, 28)}


def decode_pose(values, mask):
    if not np.asarray(mask, dtype=bool).all():
        return None, 'missing_pose'
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return None, 'nonfinite_pose'
    a, b = values[3:6], values[6:9]
    orthogonal_error = float(np.linalg.norm(np.array([a, b]) @ np.array([a, b]).T - np.eye(2)))
    if min(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(np.cross(a, b))) < 1e-6:
        return None, 'degenerate_rotation'
    if orthogonal_error > 0.05:
        return None, 'nonorthogonal_rotation'
    a = a / np.linalg.norm(a)
    b = b - a * np.dot(a, b)
    b /= np.linalg.norm(b)
    quaternion = Rotation.from_matrix(np.vstack([a, b, np.cross(a, b)])).as_quat()
    return [*values[:3].tolist(), *quaternion.tolist()], None


def adapt(table):
    """Keep raw values and timestamp IDs; malformed channels become explicit errors."""
    columns = table.to_pydict() if hasattr(table, 'to_pydict') else table
    times = np.asarray(columns['timestamp'], dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamp must be nonempty, finite and strictly increasing')
    sample = times - times[0]
    if 'sample_timestamp' in columns and not np.allclose(columns['sample_timestamp'], sample, atol=2e-5, rtol=0):
        raise ValueError('sample_timestamp differs from timestamp - timestamp[0]')
    masks = np.asarray(columns['eef_mask'])
    if masks.shape != (len(times), 34) or not np.isin(masks, [0, 1]).all():
        raise ValueError('eef_mask must be Nx34 boolean values')
    arrays = {mode: np.asarray(columns[name], dtype=float)
              for mode, name in [('state', 'observation.state'), ('action', 'action')]}
    if any(a.shape != masks.shape for a in arrays.values()):
        raise ValueError('state/action must be Nx34')
    frames = []
    for i in range(len(times)):
        frame = {'frame_index': int(columns.get('frame_index', list(range(len(times))))[i]),
                 'timestamp': float(times[i]), 'sample_timestamp': float(sample[i]),
                 'mask': masks[i].astype(bool).tolist()}
        for mode, array in arrays.items():
            value = array[i]
            item = {'raw': [float(x) if np.isfinite(x) else None for x in value],
                    'targets': {}, 'issues': {}, 'grippers': {}}
            for side, (start, stop) in POSE_SLICES.items():
                pose, issue = decode_pose(value[start:stop], masks[i, start:stop])
                item['targets'][side] = pose
                if issue:
                    item['issues'][side] = issue
            for side, index in [('left', 18), ('right', 28)]:
                if masks[i, index] and np.isfinite(value[index]) and 0 <= value[index] <= 100:
                    item['grippers'][side] = float(value[index])
                else:
                    item['grippers'][side] = None
                    item['issues'][side + '_gripper'] = 'missing_or_invalid_gripper'
            frame[mode] = item
        frames.append(frame)
    return {'frames': frames, 'frame_count': len(frames), 'duration': float(sample[-1]),
            'time_mapping': 'sample_timestamp = timestamp - timestamp[0]',
            'quaternion_order': 'xyzw', 'gripper_convention': '0=open;100=closed'}
