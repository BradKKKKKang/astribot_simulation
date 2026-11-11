"""Material validation, connected FEM geometry and origin-preserving resets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulation_config.deformable_geometry import (
    fit_deformable_pose, tetrahedral_volume, validate_deformable,
)
from src.simulation_config.glb_geometry import rotation_matrix
from src.simulation_config.isaac_scene_objects import read_scene_object_states, reset_scene_objects


def physics():
    path = Path(__file__).resolve().parents[2] / 'config/assets/wanjuxiong.json'
    return json.loads(path.read_text())['physics']


@pytest.mark.parametrize('key,value', [
    ('youngs_modulus', 0), ('youngs_modulus', float('nan')), ('poissons_ratio', .5),
    ('elasticity_damping', -1), ('self_collision', 1), ('collision_simplification_remeshing_resolution', True),
    ('solver_position_iteration_count', 0), ('rest_offset', .01), ('core_nodes', ['missing']),
])
def test_invalid_soft_material_or_core_fails_before_runtime(key, value):
    config = physics()
    nodes = [{'node': n} for n in config['deformable']['core_nodes']]
    validate_deformable(config, nodes)
    config['deformable'][key] = value
    with pytest.raises(ValueError, match='deformable'):
        validate_deformable(config, nodes)


def test_cooking_requires_connected_nondegenerate_volume():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert tetrahedral_volume(points, [0, 1, 2, 3]) == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='disconnected'):
        tetrahedral_volume(np.r_[points, points + 3], np.arange(8))
    with pytest.raises(ValueError, match='degenerate'):
        tetrahedral_volume(points, [0, 1, 1, 3])


def test_material_pose_distinguishes_rigid_motion_from_squashing():
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    quaternion = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    delta = rotation_matrix(quaternion)
    current = rest @ delta.T + [4, 5, 6]
    pose, residual = fit_deformable_pose(rest, current, [.2, .3, .1, 1, 0, 0, 0])
    assert pose[:3] == pytest.approx([3.7, 5.2, 6.1])
    np.testing.assert_allclose(rotation_matrix(pose[3:]), delta, atol=1e-7)
    assert residual < 1e-12
    current[:, 2] *= .7
    assert fit_deformable_pose(rest, current, [0, 0, 0, 1, 0, 0, 0])[1] > .05


def test_soft_reset_restores_shape_velocity_free_targets_and_sampled_origin():
    torch = pytest.importorskip('torch')
    initial = [1, 2, 3, np.sqrt(.5), np.sqrt(.5), 0, 0]
    target = [4, 5, 6, np.sqrt(.5), 0, 0, np.sqrt(.5)]
    local = np.array([[0, 0, 0], [.1, 0, 0], [0, .2, 0], [0, 0, .3]])
    rest = local @ rotation_matrix(initial[3:]).T + initial[:3]
    default = torch.tensor(np.c_[rest, np.zeros_like(rest)][None], dtype=torch.float32)
    data = SimpleNamespace(default_nodal_state_w=default,
                           nodal_kinematic_target=torch.zeros(1, 4, 4))
    writes = {}
    body = SimpleNamespace(data=data, write_nodal_state_to_sim=lambda x: writes.update(state=x.clone()),
        write_nodal_kinematic_target_to_sim=lambda x: writes.update(targets=x.clone()), reset=lambda: None)
    entry = {'key': 'soft', 'initial_pose': initial}
    runtime = {'static': {}, 'dynamic': {}, 'deformable': {'bear': entry, 'unselected': copy.copy(entry)}}
    runtime['deformable']['unselected']['key'] = 'absent'
    reset_scene_objects({'soft': body}, runtime, sampled={'bear': {'pose_world': target}})
    expected = local @ rotation_matrix(target[3:]).T + target[:3]
    np.testing.assert_allclose(writes['state'][0, :, :3], expected, atol=1e-6)
    assert torch.count_nonzero(writes['state'][..., 3:]) == 0
    assert torch.all(writes['targets'][..., 3] == 1)
    np.testing.assert_allclose(writes['targets'][..., :3], writes['state'][..., :3])
    assert torch.equal(data.default_nodal_state_w, default)
    runtime['deformable'].pop('unselected')
    data.nodal_pos_w = writes['state'][..., :3]
    state = read_scene_object_states({'soft': body}, runtime)['bear']
    assert state['pose_world'][:3] == pytest.approx(target[:3], abs=1e-6)
    assert state['source'] == 'isaaclab_deformable_object'
    assert state['deformation_rms_m'] < 1e-6
    reset_scene_objects({'soft': body}, runtime)
    assert torch.equal(writes['state'], default)
