"""Task-owned initial robot poses; importing a frame never starts replay or ROS."""
import hashlib
import json
import math
from pathlib import Path


STATE_COLUMN = 'joints_dict.joints_position_state'


def initial_joint_positions(pose, profile):
    """Validate SDK-unit targets and expand them into model joint radians."""
    if not isinstance(pose, dict) or set(pose) - {'joint_targets_rad', 'gripper_percent', 'source'}:
        raise ValueError('task.initial_pose requires joint_targets_rad and gripper_percent')
    targets, grippers = pose.get('joint_targets_rad'), pose.get('gripper_percent')
    if not isinstance(targets, dict) or not isinstance(grippers, dict) or not (targets or grippers):
        raise ValueError('task.initial_pose must contain joint or gripper targets')
    robot = profile.as_dict()
    gripper_parts = {item['part'] for item in robot['grippers'].values()}
    parts = {part['name']: len(part['model_joints']) for part in robot['parts']
             if part['name'] not in gripper_parts}
    for part, values in targets.items():
        if (part not in parts or not isinstance(values, list) or len(values) != parts[part]
                or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
            raise ValueError(f'task.initial_pose invalid joint target: {part}')
    for side, value in grippers.items():
        if side not in robot['grippers'] or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('task.initial_pose gripper_percent must be in [0, 100]')
    if 'source' in pose:
        source = pose['source']
        if (not isinstance(source, dict) or set(source) != {'parquet', 'frame_index', 'column', 'sha256'}
                or not isinstance(source['parquet'], str) or not source['parquet']
                or type(source['frame_index']) is not int or source['frame_index'] < 0
                or source['column'] != STATE_COLUMN
                or not isinstance(source['sha256'], str) or len(source['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in source['sha256'])):
            raise ValueError('task.initial_pose.source requires parquet, frame_index, state column and sha256')
    return profile.sdk_joint_positions(pose)


def load_initial_pose(parquet_path, frame_index, profile):
    """Snapshot one measured frame using LeRobot metadata's named joint order.

    The fixed chassis stays at robot_placement. The returned pose is self-contained
    so a task can retain it without depending on the original dataset at launch.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError('initial pose frame_index must be a nonnegative integer')
    import pyarrow.parquet as pq

    path = Path(parquet_path).expanduser().resolve(strict=True)
    metadata_path = path.parents[2] / 'meta' / 'info.json'
    metadata = json.loads(metadata_path.read_text())
    expected_names = [name.replace('_action_', '_state_') for name in profile.dataset_joint_names]
    feature = metadata.get('features', {}).get(STATE_COLUMN, {})
    names = feature.get('names', [])
    if (feature.get('shape') != [len(expected_names)] or not isinstance(names, list)
            or len(names) != len(expected_names) or set(names) != set(expected_names)):
        raise ValueError('initial pose metadata must declare all configured state joint names exactly once')
    parquet = pq.ParquetFile(path)
    if not {'frame_index', STATE_COLUMN} <= set(parquet.schema_arrow.names):
        raise ValueError('initial pose parquet requires frame_index and measured joint state')
    table = parquet.read(columns=['frame_index', STATE_COLUMN])
    rows = [row for row in table.to_pylist() if row['frame_index'] == frame_index]
    if len(rows) != 1:
        raise ValueError(f'initial pose requires exactly one frame_index={frame_index}; found {len(rows)}')
    values = rows[0][STATE_COLUMN]
    if (not isinstance(values, list) or len(values) != len(names)
            or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
        raise ValueError('initial pose measured joint state must contain finite values for every joint')
    state = dict(zip(names, values))
    ordered = [state[name] for name in expected_names]
    robot = profile.as_dict()
    gripper_parts = {item['part'] for item in robot['grippers'].values()}
    parts = {part['name']: part for part in robot['parts']}
    pose = {
        'joint_targets_rad': {part['name']: [ordered[i] for i in part['command_indices']]
                              for part in robot['parts'] if part['name'] not in gripper_parts},
        'gripper_percent': {side: ordered[parts[item['part']]['command_indices'][0]]
                            for side, item in robot['grippers'].items()},
        'source': {'parquet': str(path), 'frame_index': frame_index, 'column': STATE_COLUMN,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    initial_joint_positions(pose, profile)
    return pose
