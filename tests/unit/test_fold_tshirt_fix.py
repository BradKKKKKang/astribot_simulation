"""Check the fixed garment placement using source geometry, without simulation."""
from pathlib import Path

import numpy as np
import pytest

from src.simulation_config.experiment_config import resolve_experiment
from src.simulation_config.glb_geometry import rotation_matrix
from src.simulation_config.poses import compose_pose, rotate
from src.simulation_config.scene_assets import transformed_primitives
from src.simulation_config.scene_profile import build_experiment_simulation_config


ROOT = Path(__file__).resolve().parents[2]
PRESET = ROOT / 'config/experiments/evaluation/fold_tshirt_fix.yaml'


def test_fixed_garment_is_centered_flat_supported_and_collar_points_robot_right():
    resolved = resolve_experiment(PRESET)
    config = build_experiment_simulation_config(resolved)
    objects = {item['name']: item for item in config['scene_objects']}
    table, shirt = objects['table'], objects['tshirt']
    surface = table['asset']['surfaces']['tabletop']
    tabletop_pose = compose_pose(table['pose_world'], [
        *(np.array(surface['position']) * table['scale']), *surface['quaternion_wxyz']])
    # The source collar points along +Y after the asset's source transform.
    np.testing.assert_allclose(rotate(shirt['pose_world'][3:], [0, 1, 0]), [0, -1, 0], atol=1e-10)
    np.testing.assert_allclose(rotate(shirt['pose_world'][3:], [0, 0, 1]), [0, 0, 1], atol=1e-10)
    center = (np.array(shirt['bounds_world']['min']) + shirt['bounds_world']['max']) / 2
    np.testing.assert_allclose(center[:2], tabletop_pose[:2], atol=1e-4)
    clearance = shirt['bounds_world']['min'][2] - tabletop_pose[2]
    assert clearance == pytest.approx(.004)
    assert clearance > shirt['asset']['physics']['cloth']['rest_offset']
    # Check the complete unfolded geometry against the actual support rectangle.
    vertices = np.concatenate([part['vertices'] for part in
                               transformed_primitives(shirt['asset'], shirt['scale'])])
    world = vertices @ rotation_matrix(shirt['pose_world'][3:]).T + shirt['pose_world'][:3]
    local = (world - tabletop_pose[:3]) @ rotation_matrix(tabletop_pose[3:])
    bounds = {key: np.array(surface['usable_bounds'][key]) * table['scale'][:2]
              for key in ('min', 'max')}
    assert np.all(local[:, :2].min(0) >= bounds['min'])
    assert np.all(local[:, :2].max(0) <= bounds['max'])
    assert shirt['scale'] == [1., 1., 1.]
    assert resolved['sections']['task']['reset'] is None
    assert 'task_reset' not in config


def test_fixed_preset_keeps_original_robot_and_cloth_settings():
    fixed = resolve_experiment(PRESET)
    original = resolve_experiment(PRESET.with_name('fold_tshirt.json'))
    for key in ('robot', 'robot_placement', 'runtime'):
        assert fixed['sections'][key] == original['sections'][key]
    assert fixed['sections']['scene']['objects']['tshirt']['asset'] == original['sections']['scene']['objects']['tshirt']['asset']
    assert fixed['sections']['task']['initial_pose'] == original['sections']['task']['initial_pose']
    assert 'cloth_grasp' in original['sections']['task']['reset']
    assert fixed['sections']['output']['root'] != original['sections']['output']['root']


def test_integrated_launcher_accepts_fixed_yaml_without_runtime_imports(monkeypatch):
    import builtins
    import astribot_simulation_integrate

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(('rclpy', 'rospy', 'isaaclab', 'isaacsim', 'astribot_sdk')):
            raise AssertionError('runtime import during config validation: ' + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    config = astribot_simulation_integrate.main([
        '--config', str(PRESET), '--check-config', '--headless', '--no-cameras'])
    assert config['physics_substeps'] == 4
    assert 'task_reset' not in config
