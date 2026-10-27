"""Explicit robot placement, computed on CPU from the complete USD geometry.

Matrices use USD row vectors. A placement transforms authored world coordinates;
the robot root and the SDK base remain separate frames. Legacy callers omit it.
"""
import copy
import hashlib
import json
import math

import numpy as np


def validate_robot_placement(value):
    if not isinstance(value, dict) or set(value) != {'frame', 'xy', 'yaw', 'grounding'}:
        raise ValueError('robot_placement requires frame, xy, yaw and grounding')
    if value['frame'] != 'world' or value['grounding'] != 'geometry_min':
        raise ValueError('robot_placement supports world frame and geometry_min grounding')
    xy, yaw = value['xy'], value['yaw']
    if (not isinstance(xy, list) or len(xy) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in xy)
            or type(yaw) not in (int, float) or not math.isfinite(yaw)):
        raise ValueError('robot_placement xy/yaw must be finite metric/radian values')
    return copy.deepcopy(value)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _same_pose(first, second, tolerance=1e-4):
    first, second = np.asarray(first), np.asarray(second)
    return (np.allclose(first[:3], second[:3], atol=tolerance, rtol=0)
            and min(np.linalg.norm(first[3:] - second[3:]),
                    np.linalg.norm(first[3:] + second[3:])) <= tolerance)


def pose_matrix(pose):
    from pxr import Gf
    result = Gf.Matrix4d(1)
    result.SetRotate(Gf.Quatd(pose[3], Gf.Vec3d(*pose[4:])).GetNormalized())
    result.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
    return result


def matrix_pose(matrix):
    from pxr import Gf
    rotation = Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
    return [*matrix.ExtractTranslation(), rotation.GetReal(), *rotation.GetImaginary()]


def _geometry_min_z(prim, matrix):
    """Exact linear support for meshes and USD analytic solid primitives."""
    from pxr import UsdGeom
    transform = np.asarray(matrix)
    direction, offset = transform[:3, 2], transform[3, 2]
    if prim.IsA(UsdGeom.Mesh):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f'invalid grounding mesh: {prim.GetPath()}')
        return float(np.min(points @ direction) + offset)
    if prim.IsA(UsdGeom.Cube):
        return float(offset - UsdGeom.Cube(prim).GetSizeAttr().Get() * .5 * np.abs(direction).sum())
    if prim.IsA(UsdGeom.Sphere):
        return float(offset - UsdGeom.Sphere(prim).GetRadiusAttr().Get() * np.linalg.norm(direction))
    for schema in (UsdGeom.Cylinder, UsdGeom.Capsule, UsdGeom.Cone):
        if prim.IsA(schema):
            shape = schema(prim)
            axis = {'X': 0, 'Y': 1, 'Z': 2}[shape.GetAxisAttr().Get()]
            radius, half = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get() * .5
            radial = radius * np.linalg.norm(np.delete(direction, axis))
            axial = direction[axis] * half
            if schema == UsdGeom.Cone:
                return float(offset + min(axial, -axial - radial))
            if schema == UsdGeom.Capsule:
                radial = radius * np.linalg.norm(direction)
            return float(offset - abs(axial) - radial)
    raise ValueError(f'unsupported solid geometry for grounding: {prim.GetTypeName()} at {prim.GetPath()}')


def resolve_robot_placement(profile, declaration, *, initial_joint_positions=None, stage=None):
    """Resolve base XY/yaw and full visual+collision min Z at initial joints.

The default initial joint configuration is the simulator's zero-radian pose.
This intentionally does not use reference/replay target poses for grounding.
"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from src.placement_analysis.reference_pose import UsdKinematics
    from src.placement_analysis.robot_export import open_robot_stage, validate_usd_composition
    declaration = validate_robot_placement(declaration)
    if stage is None:
        stage, root, names = open_robot_stage(profile.model_path, root_path=profile.root_path)
    else:
        root = stage.GetPrimAtPath(profile.root_path)
        names = [p.GetName() for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    validate_usd_composition(stage)
    model = UsdKinematics(stage, root.GetPath(), names)
    joints = {name: 0. for name in model.joint_names}
    joints.update(profile.as_dict().get('initial_joint_positions', {}))
    joints.update(initial_joint_positions or {})
    poses = model.snapshot(joints)['body_poses_world']
    solved = {model.bodies[name]: pose_matrix(pose) for name, pose in zip(names, poses)}
    base = solved[model.bodies[profile.base_body]]
    target = Gf.Matrix4d(1)
    target.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(declaration['yaw'])))
    target.SetTranslateOnly(Gf.Vec3d(*declaration['xy'], 0))
    delta = base.GetInverse() * target
    cache, minima = UsdGeom.XformCache(), []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        # Curves/points are annotations, not solid robot geometry.
        if prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.BasisCurves):
            continue
        ancestor = prim
        while ancestor and str(ancestor.GetPath()) not in solved:
            ancestor = ancestor.GetParent()
        authored = cache.GetLocalToWorldTransform(prim)
        transform = (authored * cache.GetLocalToWorldTransform(ancestor).GetInverse()
                     * solved[str(ancestor.GetPath())] if ancestor else authored)
        minima.append(_geometry_min_z(prim, transform * delta))
    if not minima or not np.isfinite(minima).all():
        raise ValueError('robot grounding requires finite complete solid geometry')
    translation = delta.ExtractTranslation()
    delta.SetTranslateOnly(Gf.Vec3d(translation[0], translation[1], translation[2] - min(minima)))
    anchors = [entry for entry in model.incoming if model.incoming[entry][0] is None]
    if len(anchors) != 1 or anchors[0] not in solved:
        raise ValueError('grounded robot requires exactly one world-fixed articulation root')
    result = {'version': 1, 'declaration': declaration, 'base_body': profile.base_body,
              'initial_joint_positions': joints,
              'world_transform': [list(row) for row in delta],
              'spawn_pose_world': matrix_pose(delta),
              'root_pose_world': matrix_pose(cache.GetLocalToWorldTransform(root) * delta),
              'articulation_root_body': anchors[0].rsplit('/', 1)[-1],
              'articulation_root_pose_world': matrix_pose(solved[anchors[0]] * delta),
              'base_pose_world': matrix_pose(base * delta), 'geometry_min_z': 0.,
              'geometry_count': len(minima),
              'basis': 'complete USD solid visual and collision geometry at initial joints',
              'robot_control_sha256': identity(profile.control_provenance())}
    # Composed geometry and anchors can come from sublayers, not just the root file.
    from pathlib import Path
    result['model_composition_sha256'] = identity(sorted(
        hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
        if layer.realPath else hashlib.sha256(layer.ExportToString().encode()).hexdigest()
        for layer in stage.GetUsedLayers() if not layer.anonymous))
    result['placement_sha256'] = identity(result)
    return result


def transform_world_fixed_anchors(stage, root_path, world_transform):
    """Move world-side joint frames after USD instantiation, before physics starts."""
    from pxr import Gf, Usd, UsdPhysics
    delta = Gf.Matrix4d(world_transform)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for side in (0, 1):
            if getattr(joint, f'GetBody{side}Rel')().GetTargets():
                continue
            position = getattr(joint, f'GetLocalPos{side}Attr')()
            rotation = getattr(joint, f'GetLocalRot{side}Attr')()
            q = rotation.Get()
            pose = [*position.Get(), q.GetReal(), *q.GetImaginary()]
            transformed = matrix_pose(pose_matrix(pose) * delta)
            position.Set(Gf.Vec3f(*transformed[:3]))
            rotation.Set(Gf.Quatf(transformed[3], Gf.Vec3f(*transformed[4:])))


def validate_placement_attestation(value, declaration, profile, *, now_ns=None):
    """Check actual simulator feedback; a declaration alone is never evidence."""
    import time
    expected = validate_robot_placement(declaration)
    if (not isinstance(value, dict) or value.get('source') != 'isaaclab.body_pose_w'
            or value.get('declaration') != expected
            or value.get('robot_control_sha256') != identity(profile.control_provenance())
            or value.get('base_body') != profile.base_body):
        raise ValueError('running simulator robot placement/profile does not match experiment')
    now_ns = time.time_ns() if now_ns is None else now_ns
    if type(value.get('observed_system_ns')) is not int or not 0 <= now_ns - value['observed_system_ns'] <= 2_000_000_000:
        raise ValueError('robot placement attestation is stale')
    actual, resolved = np.asarray(value.get('actual_base_pose_world', [])), value.get('resolved', {})
    if identity({k: v for k, v in resolved.items() if k != 'placement_sha256'}) != resolved.get('placement_sha256'):
        raise ValueError('resolved simulator placement identity is invalid')
    planned = np.asarray(resolved.get('base_pose_world', []))
    if (actual.shape != (7,) or planned.shape != (7,) or not np.isfinite(actual).all()
            or not np.isfinite(planned).all() or not _same_pose(actual, planned)
            or resolved.get('declaration') != expected or not resolved.get('placement_sha256')):
        raise ValueError('actual simulator base pose differs from resolved grounded placement')
    yaw_quaternion = np.asarray([math.cos(expected['yaw']/2), 0., 0., math.sin(expected['yaw']/2)])
    if (not np.allclose(actual[:2], expected['xy'], atol=1e-4, rtol=0)
            or min(np.linalg.norm(actual[3:] - yaw_quaternion), np.linalg.norm(actual[3:] + yaw_quaternion)) > 1e-4):
        raise ValueError('actual base XY/yaw/upright orientation differs from experiment')
    return copy.deepcopy(value)


def validate_capture_placement(metadata, declaration, profile):
    """Require fresh, stable actual feedback spanning a completed command window."""
    samples = metadata.get('robot_placement_attestations', [])
    if not samples:
        raise ValueError('capture has no actual robot placement evidence; legacy coordinates cannot be reinterpreted')
    start, end = metadata['dataset_window_monotonic_ns']
    epoch_mono, epoch_wall = metadata['epoch_monotonic_ns'], metadata['epoch_system_ns']
    start, end = (epoch_wall + value - epoch_mono for value in (start, end))
    stamps = [sample.get('observed_system_ns', 0) for sample in samples]
    if (stamps[0] > start or stamps[-1] < end or stamps[0] < 0
            or any(not 0 < b-a <= 2_000_000_000 for a, b in zip(stamps, stamps[1:]))):
        raise ValueError('actual placement feedback does not continuously cover the capture window')
    first = samples[0]
    for sample in samples:
        validate_placement_attestation(sample, declaration, profile, now_ns=sample['observed_system_ns'])
        if sample['resolved'] != first['resolved'] or not _same_pose(
                sample['actual_base_pose_world'], first['actual_base_pose_world']):
            raise ValueError('actual robot placement changed during capture')
    return copy.deepcopy(first)
