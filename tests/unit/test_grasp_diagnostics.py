from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.grasp_diagnostics import read_contact_view
from src.evaluation.cloth_grasp_diagnostics import compile_finger_hulls, sample_cloth_grasp


def test_contacts_use_counts_copy_reused_buffers_and_convert_with_dt():
    # Garbage/unoccupied slots must not leak into output. Friction querying
    # overwrites the PhysX count/index buffers used by the preceding query.
    counts = np.array([[2, 0]])
    starts = np.array([[1, 0]])
    received_dt = []
    def normal(dt):
        received_dt.append(dt)
        return (np.array([[np.nan], [2.], [3.], [np.nan]]),
                np.zeros((4, 3)), np.ones((4, 3)), np.zeros((4, 1)), counts, starts)
    def friction(dt):
        received_dt.append(dt)
        counts[:] = [[1, 0]]
        starts[:] = [[0, 0]]
        return np.array([[.1, .2, .3]] * 4), np.zeros((4, 3)), counts, starts
    view = SimpleNamespace(sensor_count=1, filter_count=2, max_contact_data_count=4,
                           get_contact_data=normal, get_friction_data=friction)
    result = read_contact_view(view, .01)
    assert received_dt == [.01, .01]
    assert result['pairs'][0]['normal_force_sum_n'] == 5.
    assert len(result['pairs'][0]['contacts']) == 2
    assert len(result['pairs'][0]['friction_anchors']) == 1
    assert result['pairs'][1]['normal_force_sum_n'] == 0.
    assert result['pairs'][1]['contacts'] == []
    assert not result['possibly_truncated']


def test_contacts_reject_invalid_buffer_ranges():
    view = SimpleNamespace(sensor_count=1, filter_count=1, max_contact_data_count=1,
        get_contact_data=lambda dt: (np.zeros((1, 1)), np.zeros((1, 3)), np.zeros((1, 3)),
                                     np.zeros((1, 1)), np.array([[2]]), np.array([[0]])),
        get_friction_data=lambda dt: (np.zeros((1, 3)), np.zeros((1, 3)),
                                      np.array([[0]]), np.array([[0]])))
    with pytest.raises(RuntimeError, match='capacity'):
        read_contact_view(view, .01)


def box_finger():
    vertices = [[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)]
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return {'colliders': [{'collider_to_body': np.eye(4).tolist(), 'hulls': [{
        'vertices': vertices, 'indices': sum(faces, []),
        'polygons': [{'index_base': i * 4, 'num_vertices': 4} for i in range(6)]}]}]}


def proximity_fixture():
    # Inner faces are at +/-2 mm. All distances are in meters.
    poses = {'L11': [.012, 0, 0, 1, 0, 0, 0], 'R11': [-.012, 0, 0, 1, 0, 0, 0]}
    hulls = {key: compile_finger_hulls(box_finger()) for key in poses}
    return poses, hulls


def test_cloth_proximity_distinguishes_empty_single_and_bilateral_candidates():
    poses, hulls = proximity_fixture()
    points = np.array([[0, 0, 0], [.024, 0, 0], [.1, 0, 0], [.012, 0, 0]])
    result = sample_cloth_grasp(points, poses, hulls, .003, .006)
    assert result['closing_axis_gap_m'] == pytest.approx(.004)
    assert result['rest_candidates'] == {'L11': 3, 'R11': 1, 'both': 1}
    assert result['between_fingers_candidates'] == 1
    assert result['inside_hull_particles'] == {'L11': 1, 'R11': 0}
    empty = sample_cloth_grasp(points[2:3], poses, hulls, .003, .006)
    assert empty['rest_candidates'] == {'L11': 0, 'R11': 0, 'both': 0}
    assert empty['between_fingers_candidates'] == 0


def test_cloth_proximity_is_invariant_under_world_rotation_and_translation():
    from src.simulation_config.glb_geometry import rotation_matrix
    poses, hulls = proximity_fixture()
    points = np.array([[0, 0, 0], [.024, 0, 0], [.1, 0, 0]])
    expected = sample_cloth_grasp(points, poses, hulls, .003, .006)
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    rotation = rotation_matrix(q)
    offset = np.array([2., -1., 3.])
    moved = {key: [*(np.array(pose[:3]) @ rotation.T + offset), *q]
             for key, pose in poses.items()}
    actual = sample_cloth_grasp(points @ rotation.T + offset, moved, hulls, .003, .006)
    assert actual.pop('closing_axis_gap_m') == pytest.approx(expected.pop('closing_axis_gap_m'))
    assert actual == expected


def test_cooked_collider_transform_and_disconnected_hulls_are_preserved():
    body = box_finger()
    body['colliders'][0]['collider_to_body'][3][0] = .1
    body['colliders'] += box_finger()['colliders']
    hulls = compile_finger_hulls(body)
    assert len(hulls) == 2
    np.testing.assert_allclose(hulls[0]['vertices'].mean(0), [.1, 0, 0], atol=1e-12)
    from src.evaluation.cloth_grasp_diagnostics import _plane_clearance
    # Do not fill the empty space between disconnected hulls with a new hull.
    values = _plane_clearance(np.array([[.05, 0, 0], [.1, 0, 0]]), hulls, .006)
    assert np.isinf(values[0]) and values[1] == pytest.approx(-.01)


def test_corner_proximity_is_explicitly_conservative_and_invalid_data_is_rejected():
    poses, hulls = proximity_fixture()
    # This corner is sqrt(3)*2 mm from the box: the 3 mm plane-band test
    # intentionally reports a candidate, not an exact Euclidean contact.
    result = sample_cloth_grasp(np.array([[.024, .012, .012]]), poses, hulls, .003, .006)
    assert result['rest_candidates']['L11'] == 1
    assert result['rest_candidates']['both'] == 0
    with pytest.raises(ValueError, match='finite'):
        sample_cloth_grasp(np.array([[np.nan, 0, 0]]), poses, hulls, .003, .006)
    with pytest.raises(ValueError, match='no cooked'):
        compile_finger_hulls({'colliders': []})


def test_live_cloth_sampler_uses_current_particles_and_reports_missing_view():
    from src.evaluation.grasp_diagnostics import GraspDiagnostics
    poses, hulls = proximity_fixture()
    sampler = GraspDiagnostics.__new__(GraspDiagnostics)
    sampler.cloth_geometry_error = None
    points = np.array([[0., 0, 0]])
    sampler.cloth_entries = {'tshirt': {'view': SimpleNamespace(get_positions=lambda: points)}}
    sampler.cloth_settings = {'tshirt': {'rest_offset': .003, 'contact_offset': .006}}
    sampler.cloth_hulls = {f'astribot_gripper_{side}_Link_{key}': hulls[key]
                           for side in ('left', 'right') for key in poses}
    world_poses = {f'astribot_gripper_{side}_Link_{key}': poses[key]
                   for side in ('left', 'right') for key in poses}
    assert sampler._sample_cloth(world_poses)['tshirt']['left']['rest_candidates']['both'] == 1
    points[:] = [1, 1, 1]
    assert sampler._sample_cloth(world_poses)['tshirt']['left']['rest_candidates']['both'] == 0
    # Both reset and completed-step trace records must carry the new evidence.
    body_names = list(world_poses)
    data = SimpleNamespace(body_state_w=np.array([[world_poses[name] for name in body_names]]),
                           **{key: np.empty((1, 0)) for key in (
                               'joint_pos', 'joint_vel', 'joint_pos_target',
                               'joint_vel_target', 'applied_torque')})
    sampler.env = SimpleNamespace(robot=SimpleNamespace(
        data=data, joint_names=[], body_names=body_names), scene_object_keys={'dynamic': {}})
    sampler.fingers, sampler.failures, sampler.views = body_names, {}, {}
    for reset in (True, False):
        assert sampler.sample(reset=reset)['cloth_proximity']['tshirt']['left']['rest_candidates']['both'] == 0
    del sampler.cloth_entries['tshirt']['view']
    assert sampler._sample_cloth(world_poses)['tshirt']['status'] == 'unavailable'
