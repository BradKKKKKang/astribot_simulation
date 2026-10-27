"""CPU conversion of SDK replay captures; never imports ROS or controls a robot."""
import numpy as np

from .reference_pose import sdk_joint_positions

PART_CONFIGS = (
    ('astribot_torso', 4), ('astribot_arm_left', 7), ('astribot_gripper_left', 1),
    ('astribot_arm_right', 7), ('astribot_gripper_right', 1), ('astribot_head', 2),
)
JOINTS = tuple(f'{part}_joint_{i}' for part, count in PART_CONFIGS for i in range(1, count + 1))


def synchronize_feedback(rows, api_rows, metadata, joint_names, max_gap_s, *,
                         anchor_joint=None, part_by_joint=None):
    """Interpolate feedback on torso header times within the actual API window.

    Each joint needs bracketing feedback less than max_gap_s apart. Missing
    intervals are omitted so the detectors see a time gap rather than a
    fabricated stationary segment. Header time is ROS wall time, not PhysX time.
    """
    if metadata.get('status') != 'complete' or not api_rows:
        raise ValueError('a completed SDK replay capture with API commands is required')
    if len(api_rows) != metadata['command_frames']:
        raise ValueError('API command capture is incomplete')
    epoch_mono = int(metadata['epoch_monotonic_ns'])
    epoch_wall = int(metadata['epoch_system_ns'])
    starts = np.asarray([int(row['call_start_monotonic_ns']) - epoch_mono for row in api_rows])
    ends = np.asarray([int(row['call_end_monotonic_ns']) - epoch_mono for row in api_rows])
    if (np.any(np.diff(starts) <= 0) or np.any(ends < starts)
            or [r['frame_index'] for r in api_rows] != list(range(len(api_rows)))):
        raise ValueError('invalid API command timing or ordering')
    if 'dataset_window_monotonic_ns' in metadata:
        # replay_extract_event records the actual dataset window independently of
        # the SDK's blocking initial move and its variable per-frame durations.
        window = metadata['dataset_window_monotonic_ns']
        if (len(window) != 2 or not all(isinstance(v, int) for v in window)
                or not epoch_mono <= window[0] < window[1]
                or any(epoch_mono + stamp < window[0]
                       for stamp, row in zip(starts, api_rows) if row.get('phase', 'dataset') != 'initial')
                or window[1] < epoch_mono + ends[-1]):
            raise ValueError('invalid dataset capture window')
        requested_start, requested_end = [(value - epoch_mono) / 1e9 for value in window]
    else:
        requested_start, requested_end = starts[0] / 1e9, ends[-1] / 1e9 + .01
    anchor_joint = anchor_joint or 'astribot_torso_joint_1'
    if anchor_joint not in joint_names:
        raise ValueError('feedback anchor must be a configured joint')
    part_by_joint = (part_by_joint if part_by_joint is not None else
                     {name: name.rsplit('_joint_', 1)[0] for name in joint_names})
    if set(part_by_joint) != set(joint_names):
        raise ValueError('feedback part mapping must cover all joints')
    streams = {name: [] for name in joint_names}
    for row in rows:
        name = row['joint']
        if (row['frame_id'] != 'simulation' or name not in streams
                or row['part'] != part_by_joint[name]):
            raise ValueError('unexpected joint/part or non-simulation feedback')
        stamp = int(row['header_stamp_ns'])
        value = float(row['position'])
        timestamp = (stamp - epoch_wall) / 1e9
        if stamp <= 0 or not np.isfinite(value):
            raise ValueError('invalid feedback timestamp or joint position')
        if streams[name] and timestamp <= streams[name][-1][0]:
            raise ValueError(f'{name}: feedback header clock repeated or moved backwards')
        streams[name].append((timestamp, value))
    if any(len(values) < 2 for values in streams.values()):
        raise ValueError('missing or insufficient feedback for one or more joints')
    streams = {name: np.asarray(values) for name, values in streams.items()}
    times = streams[anchor_joint][:, 0]
    times = times[(times >= requested_start) & (times <= requested_end)]
    if len(times) < 2:
        raise ValueError('insufficient feedback within the command window')
    valid = np.ones(len(times), dtype=bool)
    positions = []
    for name in joint_names:
        stream = streams[name]
        t, q = stream[:, 0], stream[:, 1]
        # The anchor stream defines the common cropped interval. A joint may
        # begin after the requested command start, but it must still overlap
        # the anchor interval; otherwise synchronization would fabricate data.
        if t[0] > times[-1] + max_gap_s or t[-1] < times[0] - max_gap_s:
            raise ValueError(f'{name}: feedback does not cover the command window '
                             f'[{requested_start:.6f}, {requested_end:.6f}]s; '
                             f'headers [{t[0]:.6f}, {t[-1]:.6f}]s, '
                             f'max_gap_s={max_gap_s:g}')
        right = np.searchsorted(t, times, side='left')
        clipped = np.clip(right, 0, len(t) - 1)
        exact = t[clipped] == times
        left = np.clip(right - 1, 0, len(t) - 1)
        bracket = (right > 0) & (right < len(t)) & (t[clipped] - t[left] <= max_gap_s)
        valid &= exact | bracket
        positions.append(np.interp(times, t, q))
    times, positions = times[valid], np.asarray(positions).T[valid]
    if len(times) < 2:
        raise ValueError('insufficient synchronized feedback')
    # Sparse headers can invalidate anchors near either command boundary.
    # Crop those boundaries and preserve the original timestamp origin.
    start, end = float(times[0]), float(times[-1])
    # Interior gaps remain explicit and reset the candidate detectors.
    quality = {'time_basis': 'ROS state header wall time relative to replay epoch',
               'command_window_s': [float(requested_start), float(requested_end)],
               'synchronized_window_s': [start, end],
               'boundary_trim_s': [start - requested_start, requested_end - end],
               'anchor_samples': int(len(valid)), 'synchronized_samples': len(times),
               'discarded_anchor_samples': int((~valid).sum()),
               'max_sample_gap_s': float(np.max(np.diff(times))),
               'gap_count': int(np.sum(np.diff(times) > max_gap_s))}
    return times, positions, quality


def feedback_joint_positions(joint_names, values):
    """SDK angles are radians; measured gripper percentages expand to fingers.

    Only L1 is published per gripper; other finger joints use the model's
    signed linkage convention. These are inferred visual poses, not separate
    finger measurements. TCP remains attached to the measured wrist chain.
    """
    joints = dict(zip(joint_names, values))
    grippers = {side: float(np.clip(joints.pop(f'astribot_gripper_{side}_joint_1'), 0, 100))
                for side in ('left', 'right')}
    expanded = sdk_joint_positions({'joint_targets_rad': {}, 'gripper_percent': grippers})
    return {**joints, **expanded}, [grippers['left'], grippers['right']]
