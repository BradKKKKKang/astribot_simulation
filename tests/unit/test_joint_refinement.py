"""Continuation trials cannot mix physics protocols or validation identities."""
import copy
from pathlib import Path

import pytest


@pytest.fixture
def refinement(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / 'scripts/joint_align'))
    import refine_response
    return refine_response


def trial(refinement, errors, gain=800):
    config = {refinement.key(j): {'stiffness': gain, 'damping': 40}
              for j in refinement.JOINTS}
    config['joint_command_filters'] = {j: {'time_constant_s': 0}
                                       for j in refinement.JOINTS}
    episodes = {f'episode_{i}': {'meta': {'input_sha256': f'hash_{i}'},
                               'metrics': {j: {'position_rmse': q, 'velocity_rmse': v}
                                           for j in refinement.replay.JOINTS}}
                for i, (q, v) in enumerate(errors)}
    return {'settle_steps': 500, 'source_sha256': {'sim': 'abc'}}, config, episodes


@pytest.mark.parametrize('change', ['settling', 'source', 'episode'])
def test_protocol_or_source_mismatch_is_rejected(refinement, change):
    base = trial(refinement, [(1, 1)])
    candidate = copy.deepcopy(base)
    if change == 'settling':
        candidate[0]['settle_steps'] = 0
    elif change == 'source':
        candidate[0]['source_sha256']['sim'] = 'modified'
    else:
        candidate[2]['episode_0']['meta']['input_sha256'] = 'validation'
    with pytest.raises(ValueError, match='differs'):
        refinement.comparable({'baseline': base, 'candidate': candidate})


def test_equal_episode_rms_and_velocity_gate(refinement):
    base = trial(refinement, [(1, 1), (3, 3)])
    values = refinement.scores(base)
    assert values[refinement.JOINTS[0]]['position_rmse'] == pytest.approx(5 ** .5)
    bad = trial(refinement, [(.1, 4), (.1, 4)], gain=1600)
    good = trial(refinement, [(1, 1), (2, 2)], gain=1200)
    config, decisions = refinement.select({'baseline_settled': base, 'bad': bad, 'good': good})
    assert all(v['selected'] == 'good' for v in decisions.values())
    assert config[refinement.key(refinement.JOINTS[0])]['stiffness'] == 1200
    assert base[1][refinement.key(refinement.JOINTS[0])]['stiffness'] == 800
