"""Measured frame import, task composition and initial/reset pose propagation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.simulation_config.initial_pose import initial_joint_positions, load_initial_pose, STATE_COLUMN
from src.simulation_config.robot_profile import default_robot_profile
from src.simulation_config.experiment_config import resolve_experiment
from src.simulation_config.scene_profile import build_experiment_simulation_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    profile = default_robot_profile()
    names = [name.replace('_action_', '_state_') for name in profile.dataset_joint_names]
    values = [index / 100 for index in range(len(names))]
    values[14], values[22] = 31., 72.
    path = tmp_path / 'data/chunk-000/episode_000000.parquet'
    path.parent.mkdir(parents=True)
    (tmp_path / 'meta').mkdir()
    # Metadata order is deliberately reversed; frame_index is not the row offset.
    metadata = {'features': {STATE_COLUMN: {'shape': [len(names)], 'names': names[::-1]}}}
    (tmp_path / 'meta/info.json').write_text(json.dumps(metadata))
    pq.write_table(pa.Table.from_pylist([
        {'frame_index': 9, STATE_COLUMN: [0.] * len(names)},
        {'frame_index': 0, STATE_COLUMN: values[::-1]},
    ]), path)
    return path, values, profile


def test_measured_frame_import_maps_names_units_and_ignores_fixed_chassis(dataset):
    path, values, profile = dataset
    pose = load_initial_pose(path, 0, profile)
    assert pose['joint_targets_rad']['astribot_torso'] == values[3:7]
    assert pose['joint_targets_rad']['astribot_arm_left'] == values[7:14]
    assert pose['joint_targets_rad']['astribot_arm_right'] == values[15:22]
    assert pose['joint_targets_rad']['astribot_head'] == values[23:25]
    assert pose['gripper_percent'] == {'left': 31., 'right': 72.}
    joints = initial_joint_positions(pose, profile)
    assert joints['astribot_gripper_left_joint_L1'] == pytest.approx(-31 * .0093)
    assert joints['astribot_gripper_right_joint_R1'] == pytest.approx(72 * .0093)
    assert not any('chassis' in name for name in joints)
    assert load_initial_pose(path, 9, profile)['gripper_percent'] == {'left': 0., 'right': 0.}
    with pytest.raises(ValueError, match='exactly one'):
        load_initial_pose(path, 1, profile)


@pytest.mark.parametrize('change,match', [
    (lambda pose: pose['joint_targets_rad'].update(astribot_head=[0]), 'joint target'),
    (lambda pose: pose['joint_targets_rad'].update(astribot_head=[0, float('nan')]), 'joint target'),
    (lambda pose: pose['joint_targets_rad'].update(astribot_gripper_left=[50]), 'joint target'),
    (lambda pose: pose['gripper_percent'].update(left=101), 'gripper_percent'),
    (lambda pose: pose.update(extra=True), 'initial_pose'),
])
def test_invalid_task_pose_rejected(change, match):
    pose = {'joint_targets_rad': {'astribot_head': [0., .8]}, 'gripper_percent': {}}
    change(pose)
    with pytest.raises(ValueError, match=match):
        initial_joint_positions(pose, default_robot_profile())


@pytest.mark.parametrize('fault', ['duplicate_frame', 'nan_state', 'wrong_names'])
def test_ambiguous_or_invalid_dataset_rejected(dataset, fault):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path, values, profile = dataset
    if fault == 'wrong_names':
        meta = path.parents[2] / 'meta/info.json'
        data = json.loads(meta.read_text())
        data['features'][STATE_COLUMN]['names'][0] = 'wrong_joint'
        meta.write_text(json.dumps(data))
    else:
        rows = pq.read_table(path).to_pylist()
        if fault == 'duplicate_frame':
            rows.append(rows[1])
        else:
            rows[1][STATE_COLUMN][0] = float('nan')
        pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        load_initial_pose(path, 0, profile)


def test_task_pose_overrides_robot_defaults_without_mutating_profile(dataset, tmp_path):
    path, _, profile = dataset
    resolved = resolve_experiment(ROOT / 'config/experiments/evaluation/press_cube.json')
    resolved['sections']['robot']['initial_joint_positions'] = {'astribot_head_joint_2': -.4}
    resolved['sections']['task']['initial_pose'] = load_initial_pose(path, 0, profile)
    experiment = tmp_path / 'experiment.json'
    experiment.write_text(json.dumps({'schema_version': 4, 'kind': 'evaluation', **resolved['sections']}))
    resolved = resolve_experiment(experiment)
    before = copy.deepcopy(resolved)
    data = build_experiment_simulation_config(resolved)
    joints = data['robot_profile']['initial_joint_positions']
    assert joints['astribot_head_joint_2'] == .24
    assert data['experiment_provenance']['robot_control']['initial_joint_positions'] == joints
    assert resolved == before
    # Stored task snapshots work even when their dataset is no longer mounted.
    path.unlink()
    assert build_experiment_simulation_config(resolve_experiment(experiment))['robot_profile'] == data['robot_profile']
    invalid = {'schema_version': 4, 'kind': 'evaluation', **resolved['sections']}
    invalid.pop('robot_placement')
    experiment.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='requires robot_placement'):
        resolve_experiment(experiment)


def test_integrated_cli_imports_and_displays_selected_pose(dataset, capsys):
    from tests.unit.test_simulation_integrate import launcher
    path, values, _ = dataset
    data = launcher.main(['--config', str(ROOT / 'config/experiments/evaluation/press_cube.json'),
                          '--initial-pose-parquet', str(path), '--initial-pose-frame', '0', '--check-config'])
    assert data['robot_profile']['initial_joint_positions']['astribot_head_joint_2'] == values[24]
    output = capsys.readouterr().out
    assert 'frame_index=0' in output and STATE_COLUMN in output
    assert 'astribot_head [rad]: 0.230000, 0.240000' in output
    with pytest.raises(SystemExit):
        launcher.main(['--initial-pose-frame', '0', '--check-config'])


def test_reset_restores_imported_joint_state_and_hold_commands(dataset, monkeypatch):
    torch = pytest.importorskip('torch')
    from tests.unit.test_isaaclab_gripper import load_env_class
    path, values, profile = dataset
    initial = initial_joint_positions(load_initial_pose(path, 0, profile), profile)
    env = object.__new__(load_env_class(monkeypatch))
    env.device, env.dt = 'cpu', .01
    env.robot_profile_data = profile.as_dict()
    env.resolved_robot_placement = {'articulation_root_pose_world': [0, 0, .1, 1, 0, 0, 0],
                                    'initial_joint_positions': initial}
    grippers = {item['part']: item for item in env.robot_profile_data['grippers'].values()}
    env.robot_joint_map = {part['name']: ([next(iter(grippers[part['name']]['model_joints']))]
                                         if part['name'] in grippers else part['model_joints'])
                           for part in env.robot_profile_data['parts']}
    env.robot_dict = {part: SimpleNamespace(joint_position_command=[999.]) for part in env.robot_joint_map}
    q = torch.tensor([list(initial.values())])
    writes = []
    env.robot = SimpleNamespace(data=SimpleNamespace(default_root_state=torch.zeros((1, 13)),
        default_joint_pos=q, default_joint_vel=torch.zeros_like(q)),
        write_root_pose_to_sim=lambda *a: None, write_root_velocity_to_sim=lambda *a: None,
        write_joint_state_to_sim=lambda joints, velocities: writes.append(joints.clone()),
        set_joint_position_target=lambda joints: writes.append(joints.clone()))
    env.scene_object_keys, env.task_reset = {'dynamic': {}}, None
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kw: None)
    env.sim = SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None),
                              forward=lambda: None)
    for _ in range(2):
        env._restore_initial_state()
        for part in env.robot_profile_data['parts']:
            assert env.robot_dict[part['name']].joint_position_command == pytest.approx(
                [values[i] for i in part['command_indices']])
            env.robot_dict[part['name']].joint_position_command = [999.]
    for written in writes:
        torch.testing.assert_close(written, q)
