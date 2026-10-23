"""CPU USD-to-GLB transform and collision-exclusion contract."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

pytest.importorskip('pxr', reason='optional CPU USD bindings required')
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from src.placement_analysis.usd_export import export_robot_glb
from src.placement_analysis.reference_pose import sdk_joint_positions, usd_reference_snapshot


def test_export_preserves_world_scale_body_transform_and_visual_offset(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    robot = UsdGeom.Xform.Define(stage, '/World/robot')
    body = UsdGeom.Xform.Define(stage, '/World/robot/body')
    body.AddTranslateOp().Set((1, 2, 3))
    body.AddRotateZOp().Set(90)
    visual = UsdGeom.Xform.Define(stage, '/World/robot/body/visuals')
    visual.AddTranslateOp().Set((.1, 0, 0))
    points = [(0, 0, 0), (.2, 0, 0), (0, .3, 0)]
    for path in ['/World/robot/body/visuals/triangle', '/World/robot/body/collisions/triangle']:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    path = tmp_path / 'robot.glb'
    info = export_robot_glb(stage, robot.GetPath(), ['body'], path)
    assert info['triangle_count'] == 1
    data = path.read_bytes()
    magic, version, size, json_size, json_kind = struct.unpack_from('<IIIII', data)
    assert (magic, version, size, json_kind) == (0x46546C67, 2, len(data), 0x4E4F534A)
    gltf = json.loads(data[20:20+json_size])
    binary_start = 20 + json_size + 8
    accessor = gltf['accessors'][gltf['meshes'][0]['primitives'][0]['attributes']['POSITION']]
    view = gltf['bufferViews'][accessor['bufferView']]
    exported = np.frombuffer(data, dtype='<f4', count=9, offset=binary_start+view['byteOffset']).reshape(3, 3)
    np.testing.assert_allclose(exported, np.asarray(points) + [.1, 0, 0], atol=1e-7)
    matrix = np.asarray(gltf['nodes'][0]['matrix']).reshape(4, 4)
    actual_world = (np.c_[exported, np.ones(3)] @ matrix)[:, :3]
    usd_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/robot/body/visuals/triangle'))
    expected_world = [usd_matrix.Transform(Gf.Vec3d(*point)) for point in points]
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-7)
    assert gltf['nodes'][0]['name'] == 'body'


def test_reference_pose_solves_joint_frames_and_preserves_fixed_base():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/robot')
    base = UsdGeom.Xform.Define(stage, '/robot/base')
    base.AddTranslateOp().Set((1, 2, 0))
    UsdGeom.Xform.Define(stage, '/robot/hand')
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/robot/bend')
    joint.GetBody0Rel().SetTargets(['/robot/base'])
    joint.GetBody1Rel().SetTargets(['/robot/hand'])
    joint.GetAxisAttr().Set('Z')
    joint.GetLocalPos0Attr().Set((1, 0, 0))
    joint.GetLocalPos1Attr().Set((-.5, 0, 0))
    snapshot, names = usd_reference_snapshot(stage, '/robot', ['base', 'hand'], {'bend': np.pi/2})
    np.testing.assert_allclose(snapshot['body_poses_world'][0], [1, 2, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(snapshot['body_poses_world'][1][:3], [2, 2.5, 0], atol=1e-7)
    assert names == ['bend']
    assert snapshot['joint_positions_rad'] == [np.pi/2]


def test_sdk103_targets_produce_symmetric_arm_pose_in_actual_usd():
    root = Path(__file__).parents[2]
    settings = json.loads((root / 'config/placement_reference_pose.json').read_text())
    targets = sdk_joint_positions(settings)
    assert [targets[f'astribot_torso_joint_{i}'] for i in range(1, 5)] == [.29, -.58, .29, 0]
    assert targets['astribot_gripper_left_joint_L1'] == -.465
    assert targets['astribot_gripper_right_joint_R1'] == .465
    stage = Usd.Stage.Open(str(root / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1_textured.usda'))
    bodies = ['astribot_torso_base', 'astribot_arm_left_link_7', 'astribot_arm_right_link_7']
    snapshot, _ = usd_reference_snapshot(stage, '/World/astribot_whole_body', bodies, targets)
    base, left, right = snapshot['body_poses_world']
    np.testing.assert_allclose(base, [0, 0, 0, 1, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(left[:3], np.asarray(right[:3]) * [1, -1, 1], atol=2e-5)
    assert left[0] > .2 and left[1] > .15 and left[2] > .5
    # Check all authored joint constraints at the target, without physics.
    names = [p.GetName() for p in stage.GetPrimAtPath('/World/astribot_whole_body').GetChildren()
             if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    full, _ = usd_reference_snapshot(stage, '/World/astribot_whole_body', names, targets)
    assert len(full['body_poses_world']) == 33
    np.testing.assert_allclose(np.linalg.norm(np.asarray(full['body_poses_world'])[:, 3:], axis=1), 1, atol=1e-7)


def test_internal_reference_export_preserves_s1_geometry_and_sdk_pose_contract(tmp_path, monkeypatch):
    from src.placement_analysis import robot_export

    def geometry(stage, root, names, output):
        assert str(root) == '/World/astribot_whole_body'
        assert len(names) == 33
        output.write_bytes(b'geometry fixture')
        return {'body_count': len(names), 'triangle_count': 1, 'bytes': output.stat().st_size}

    monkeypatch.setattr(robot_export, 'export_robot_glb', geometry)
    robot_export.export_reference_robot(tmp_path)
    model = json.loads((tmp_path / 'model.json').read_text())
    assert model['reference_pose']['name'] == 'SDK ROS2 example 103 target'
    assert model['base_body'] == 'astribot_torso_base'
    assert len(model['body_names']) == len(model['reference_snapshot']['body_poses_world']) == 33
    assert len(model['joint_names']) == len(model['reference_snapshot']['joint_positions_rad']) == 32
