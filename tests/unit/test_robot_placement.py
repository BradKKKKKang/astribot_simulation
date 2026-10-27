"""CPU grounding, world anchors and immutable coordinate provenance."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulation_config.robot_placement import (identity, validate_robot_placement,
    validate_placement_attestation, validate_capture_placement)

DECLARATION = {'frame': 'world', 'xy': [1., 2.], 'yaw': .5, 'grounding': 'geometry_min'}


def attestation(stamp=10_000_000_000):
    profile = SimpleNamespace(base_body='base', control_provenance=lambda: {'model': 'fixture'})
    pose = [1., 2., .1, math.cos(.25), 0., 0., math.sin(.25)]
    resolved = {'base_pose_world': pose, 'declaration': DECLARATION}
    resolved['placement_sha256'] = identity(resolved)
    sample = {'source': 'isaaclab.body_pose_w', 'observed_system_ns': stamp,
              'declaration': DECLARATION, 'base_body': 'base', 'actual_base_pose_world': list(pose),
              'robot_control_sha256': identity(profile.control_provenance()), 'resolved': resolved}
    return profile, sample


@pytest.mark.parametrize('change', [lambda p:p.update(yaw=float('nan')),
    lambda p:p.update(xy=[0]), lambda p:p.update(frame='base'), lambda p:p.update(extra=0)])
def test_invalid_declaration(change):
    value = copy.deepcopy(DECLARATION)
    change(value)
    with pytest.raises(ValueError):
        validate_robot_placement(value)


def test_attestation_checks_actual_pose_identity_freshness_and_whole_window():
    profile, sample = attestation()
    validate_placement_attestation(sample, DECLARATION, profile, now_ns=10_000_000_001)
    bad = copy.deepcopy(sample)
    bad['actual_base_pose_world'][0] += .01
    with pytest.raises(ValueError, match='actual'):
        validate_placement_attestation(bad, DECLARATION, profile, now_ns=10_000_000_001)
    with pytest.raises(ValueError, match='stale'):
        validate_placement_attestation(sample, DECLARATION, profile, now_ns=13_000_000_000)
    metadata = {'epoch_monotonic_ns': 0, 'epoch_system_ns': 10_000_000_000,
                'dataset_window_monotonic_ns': [100_000_000, 900_000_000],
                'robot_placement_attestations': [sample, attestation(11_000_000_000)[1]]}
    validate_capture_placement(metadata, DECLARATION, profile)
    metadata['robot_placement_attestations'][-1]['actual_base_pose_world'][3:] = [
        -value for value in sample['actual_base_pose_world'][3:]]
    validate_capture_placement(metadata, DECLARATION, profile)
    metadata['robot_placement_attestations'].pop()
    with pytest.raises(ValueError, match='cover'):
        validate_capture_placement(metadata, DECLARATION, profile)
    metadata['robot_placement_attestations'] = []
    with pytest.raises(ValueError, match='legacy coordinates'):
        validate_capture_placement(metadata, DECLARATION, profile)


def test_complete_geometry_initial_joints_world_anchor_and_local_calibration():
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from src.simulation_config.robot_placement import (resolve_robot_placement, pose_matrix,
        transform_world_fixed_anchors, _geometry_min_z)
    from src.placement_analysis.reference_pose import UsdKinematics
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/robot')
    root.AddTranslateOp().Set((3., 0., 4.))
    body = UsdGeom.Xform.Define(stage, '/robot/base')
    body.AddTranslateOp().Set((0., 0., .3))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    hand = UsdGeom.Xform.Define(stage, '/robot/hand')
    UsdPhysics.RigidBodyAPI.Apply(hand.GetPrim())
    anchor = UsdPhysics.FixedJoint.Define(stage, '/robot/anchor')
    anchor.GetBody1Rel().SetTargets(['/robot/base'])
    anchor.GetLocalPos0Attr().Set((3., 0., 4.3))
    anchor.GetLocalRot0Attr().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(1,0,0), 30).GetQuat()))
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/robot/hinge')
    joint.GetBody0Rel().SetTargets(['/robot/base'])
    joint.GetBody1Rel().SetTargets(['/robot/hand'])
    joint.GetAxisAttr().Set('Y')
    cube = UsdGeom.Cube.Define(stage, '/robot/base/solid')
    cube.GetSizeAttr().Set(.2)
    mesh = UsdGeom.Mesh.Define(stage, '/robot/hand/solid')
    mesh.GetPointsAttr().Set([(0,0,0), (2,0,0), (0,.1,0)])
    profile = SimpleNamespace(root_path='/robot', base_body='base',
        as_dict=lambda: {'initial_joint_positions': {'hinge': math.pi/2}},
        control_provenance=lambda: {'model': 'fixture'})
    placement = resolve_robot_placement(profile, DECLARATION, stage=stage)
    assert placement['geometry_count'] == 2
    np.testing.assert_allclose(placement['base_pose_world'][:3], [1,2,2], atol=1e-6)
    assert placement['root_pose_world'] != placement['base_pose_world']
    original = UsdKinematics(stage, '/robot', ['base', 'hand'])
    moved = UsdKinematics(stage, '/robot', ['base', 'hand'], world_transform=placement['world_transform'])
    joints = {'hinge': math.pi/2}
    a = original.snapshot(joints)['body_poses_world']
    b = moved.snapshot(joints)['body_poses_world']
    np.testing.assert_allclose(pose_matrix(a[1]) * pose_matrix(a[0]).GetInverse(),
                               pose_matrix(b[1]) * pose_matrix(b[0]).GetInverse(), atol=1e-6)
    # Local TCP and camera calibration are unchanged by a whole-robot transform.
    local = pose_matrix([.2,.1,.05,1,0,0,0])
    np.testing.assert_allclose(local * pose_matrix(a[1]) * pose_matrix(a[0]).GetInverse(),
                               local * pose_matrix(b[1]) * pose_matrix(b[0]).GetInverse(), atol=1e-6)
    assert original.fingerprint() != moved.fingerprint()
    minimum = min(_geometry_min_z(cube.GetPrim(), pose_matrix(b[0])),
                  _geometry_min_z(mesh.GetPrim(), pose_matrix(b[1])))
    assert minimum == pytest.approx(0., abs=1e-6)
    # PhysX's world-fixed anchor gets the same transform, not only the spawn prim.
    transform_world_fixed_anchors(stage, '/robot', placement['world_transform'])
    anchored = UsdKinematics(stage, '/robot', ['base', 'hand']).snapshot(joints)
    np.testing.assert_allclose(anchored['body_poses_world'], b, atol=1e-6)


def test_actual_s1_complete_geometry_nonzero_initial_pose():
    pytest.importorskip('pxr')
    from src.simulation_config.robot_profile import default_robot_profile, RobotProfile
    from src.simulation_config.robot_placement import resolve_robot_placement
    value = default_robot_profile().as_dict()
    value['initial_joint_positions'] = {'astribot_torso_joint_1': .29, 'astribot_torso_joint_2': -.58,
                                         'astribot_torso_joint_3': .29}
    placed = resolve_robot_placement(RobotProfile(value), DECLARATION)
    assert placed['geometry_count'] == 59
    assert placed['base_pose_world'][:3] == pytest.approx([1, 2, .10000000149])
    assert placed['initial_joint_positions']['astribot_torso_joint_2'] == -.58


def test_reset_restores_resolved_root_joints_commands_and_refreshes_actual_fk(monkeypatch):
    torch = pytest.importorskip('torch')
    from tests.unit.test_isaaclab_gripper import load_env_class
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.device, env.dt = 'cpu', .01
    env.resolved_robot_placement = {'articulation_root_pose_world': [1, 2, .1, 1, 0, 0, 0],
                                    'initial_joint_positions': {'joint': .4}}
    env.robot_profile_data = {'grippers': {}}
    env.robot_dict = {'arm': SimpleNamespace(joint_position_command=[-1])}
    env.robot_joint_map = {'arm': ['joint']}
    calls = []
    def record(name):
        return lambda *args, **kwargs: calls.append((name, [x.clone() if torch.is_tensor(x) else x for x in args]))
    env.robot = SimpleNamespace(data=SimpleNamespace(default_root_state=torch.zeros((1, 13)),
        default_joint_pos=torch.tensor([[.4]]), default_joint_vel=torch.zeros((1, 1))))
    for name in ('write_root_pose_to_sim', 'write_root_velocity_to_sim',
                 'write_joint_state_to_sim', 'set_joint_position_target'):
        setattr(env.robot, name, record(name))
    env.scene_object_keys = {'dynamic': {}}
    env.scene = SimpleNamespace(write_data_to_sim=record('write'), update=record('update'))
    env.sim = SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=record('fk')),
                              forward=record('forward'))
    env._restore_initial_state()
    first = calls[:]
    calls.clear()
    env.robot_dict['arm'].joint_position_command = [17.]
    env._restore_initial_state()
    assert [name for name, _ in calls] == [name for name, _ in first]
    for (_, left), (_, right) in zip(calls, first):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b)
    assert env.robot_dict['arm'].joint_position_command == [.4]
    assert [name for name, _ in calls][-3:] == ['fk', 'forward', 'update']


def test_online_object_base_coordinates_use_actual_nonzero_robot_pose(monkeypatch):
    torch = pytest.importorskip('torch')
    from tests.unit.test_isaaclab_gripper import load_env_class
    from src.simulation_config import isaac_scene_objects
    from src.evaluation.task_evaluation import evaluate_success
    cls = load_env_class(monkeypatch)
    env = object.__new__(cls)
    env.robot_profile_data = {'base_body': 'base'}
    env.scene, env.scene_object_keys = object(), {}
    q = math.sqrt(.5)
    env.robot = SimpleNamespace(find_bodies=lambda _: ([0], ['base']),
        data=SimpleNamespace(body_state_w=torch.tensor([[[1, 2, .1, q, 0, 0, q]]])))
    monkeypatch.setattr(isaac_scene_objects, 'read_scene_object_states', lambda *args: {
        'pen': {'pose_world': [1, 3, .6, q, 0, 0, q], 'source': 'isaaclab_rigid_object'}})
    state = env.evaluation_state()
    assert state['objects']['pen']['pose_base'][:3] == pytest.approx([1, 0, .5], abs=1e-6)
    scene = {'objects': {'pen': {}}, 'regions': {
        'target': {'frame': 'base', 'min': [.9, -.1, .4], 'max': [1.1, .1, .6]}}}
    success = {'predicates': [{'id': 'inside', 'kind': 'object_in_region', 'object': 'pen', 'region': 'target'}]}
    result = evaluate_success([], scene, {}, success, {'objects': state['objects']})
    assert result['status'] == 'passed'
