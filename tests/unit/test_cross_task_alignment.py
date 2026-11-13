"""Cross-task selection must not hide a task regression in a pooled average."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def cross(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'scripts/joint_align'))
    import cross_task
    return cross_task


def fake_trial(cross, values, gain):
    joints = cross.replay.JOINTS
    config = {cross.control_key(j): {'stiffness': gain, 'damping': 40.} for j in joints}
    config['joint_command_filters'] = {cross.filter_key(j): {'time_constant_s': 0.} for j in joints}
    episodes = {name: {'meta': {'input_sha256': name},
                       'metrics': {j: dict(position_rmse=q, velocity_rmse=v) for j in joints},
                       'free_metrics': {j: dict(position_rmse=q, velocity_rmse=v) for j in joints if 'gripper' in j}}
                for name, (q, v) in values.items()}
    return {}, config, episodes


def test_task_weighting_preserves_episode_hierarchy(cross):
    manifest = {'inputs': {'a': {'task': 'one'}, 'b': {'task': 'one'}, 'c': {'task': 'two'}}}
    trial = fake_trial(cross, {'a': (1, 1), 'b': (3, 3), 'c': (10, 10)}, 800)
    scores = cross.task_scores(trial, manifest)
    joint = cross.replay.JOINTS[0]
    assert scores['one'][joint]['position_rmse'] == pytest.approx(5 ** .5)
    assert scores['two'][joint]['position_rmse'] == 10


def test_selection_rejects_single_task_velocity_regression(cross, tmp_path, monkeypatch):
    manifest = {'inputs': {'a': {'task': 'one'}, 'b': {'task': 'two'}}}
    trials = {
        'v3': fake_trial(cross, {'a': (1, 1), 'b': (1, 1)}, 800),
        'bad': fake_trial(cross, {'a': (.1, .1), 'b': (.5, 1.5)}, 1600),
        'good': fake_trial(cross, {'a': (.9, .9), 'b': (.9, .9)}, 1200)}
    (tmp_path / 'episodes.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(cross, 'load_training', lambda *_: (manifest, trials))
    cross.select(tmp_path, list(trials), 'selected')
    result = json.loads((tmp_path / 'selected_selection.json').read_text())
    assert all(v['selected'] == 'good' and 'bad' not in v['eligible'] for v in result.values())


def test_fit_rejects_unlisted_or_validation_episode(cross, tmp_path, monkeypatch):
    (tmp_path / 'episodes.json').write_text(json.dumps({'training': ['/inputs/train.hdf5']}))
    trial = fake_trial(cross, {'validation': (1, 1)}, 800)
    monkeypatch.setattr(cross.refine, 'read_trial', lambda *_: trial)
    monkeypatch.setattr(cross.refine, 'comparable', lambda *_: None)
    with pytest.raises(ValueError, match='exactly the declared training'):
        cross.load_training(tmp_path, ['v3'])


def test_contact_partition_requires_sustained_plateau_and_reopening(cross):
    import numpy as np
    t = np.arange(1000) * .004
    command = np.zeros(1000)
    position = np.zeros(1000)
    command[200:600] = 100
    position[200:600] = 80
    state = np.column_stack([t, position, np.zeros(1000)])
    free = cross.free_motion_mask(t, command, state)
    assert free[:150].all()
    assert not free[200:625].any()
    assert free[630:].all()
    # A single outlier must not classify the entire grasp as contact.
    command[:] = 0
    position[:] = 0
    command[300] = 100
    state[:, 1] = position
    assert cross.free_motion_mask(t, command, state).all()
    # A wide object can block fingers near the open end, well below 70%.
    command[200:600] = 30
    state[200:600, 1] = 14
    free = cross.free_motion_mask(t, command, state)
    assert not free[200:625].any()


def test_local_fit_cannot_trade_away_one_task(cross):
    import numpy as np
    residual = np.array([1., 0., 1., 0.])
    jac = np.array([[-1., 0., 0.], [0., 0., 0.], [.2, 0., 0.], [0., 0., 0.]])
    blocks = [(residual[:2], jac[:2], np.array([1.0201, 1.])),
              (residual[2:], jac[2:], np.array([1.0201, 1.]))]
    coefficients = cross.constrained_coefficients(residual, jac, blocks, [-1., -1., 0.], [1., 1., 4.])
    assert .049 < coefficients[0] <= .050001


def test_validation_refuses_configuration_change_after_freeze(cross, tmp_path, monkeypatch):
    trial = fake_trial(cross, {'train': (1., 1.)}, 800)
    (tmp_path / 'episodes.json').write_text(json.dumps({'training': ['train.hdf5'], 'validation': ['held_out.hdf5']}))
    config_path = tmp_path / 'v3.yaml'
    config_path.write_text(cross.yaml.safe_dump(trial[1]))
    monkeypatch.setattr(cross.refine, 'read_trial', lambda *_: trial)
    cross.freeze(tmp_path, ['v3'])
    with pytest.raises(FileExistsError):
        cross.freeze(tmp_path, ['v3'])
    config = cross.yaml.safe_load(config_path.read_text())
    config[cross.control_key(cross.TUNED_JOINTS[0])]['stiffness'] = 1600
    config_path.write_text(cross.yaml.safe_dump(config))
    monkeypatch.setattr(cross.subprocess, 'run', lambda *_a, **_k: pytest.fail('must reject before launching SDK'))
    with pytest.raises(ValueError, match='configuration changed after freeze'):
        cross.run(tmp_path, ['v3'], 'validation', 201)


def test_coupled_wrist_is_selected_as_a_complete_block(cross, tmp_path, monkeypatch):
    manifest = {'inputs': {'a': {'task': 'one'}}}
    base = fake_trial(cross, {'a': (1., 1.)}, 800)
    better = fake_trial(cross, {'a': (.5, .5)}, 1200)
    j5 = cross.COUPLED_GROUPS[0][0]
    better[2]['a']['metrics'][j5] = dict(position_rmse=1.005, velocity_rmse=1.005)
    monkeypatch.setattr(cross, 'load_training', lambda *_: (manifest, {'v3': base, 'group': better}))
    (tmp_path / 'episodes.json').write_text(json.dumps(manifest))
    cross.select(tmp_path, ['v3', 'group'], 'selected')
    report = json.loads((tmp_path / 'selected_selection.json').read_text())
    assert all(report[j]['selected'] == 'group' for j in cross.COUPLED_GROUPS[0])
