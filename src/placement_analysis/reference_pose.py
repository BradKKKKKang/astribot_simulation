"""CPU-only reference poses from authored USD physics joint frames."""
import math

import numpy as np


def sdk_joint_positions(settings):
    """Expand SDK targets into articulation joint radians, including fingers."""
    joints = {f'{part}_joint_{i}': float(value)
              for part, values in settings['joint_targets_rad'].items()
              for i, value in enumerate(values, 1)}
    directions = {'L1': -1, 'L2': -1, 'R11': -1, 'R1': 1, 'R2': 1, 'L11': 1}
    for side, value in settings['gripper_percent'].items():
        if side not in ('left', 'right') or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('invalid reference gripper target')
        for suffix, direction in directions.items():
            joints[f'astribot_gripper_{side}_joint_{suffix}'] = value * direction * .93 / 100
    if not all(math.isfinite(value) for value in joints.values()):
        raise ValueError('reference joint targets must be finite')
    return joints


class UsdKinematics:
    """Compile USD joint frames once; evaluate feedback poses on the CPU.

    USD row vectors: child_world = inverse(child_frame) * motion
    * parent_frame * parent_world. The authored fixed base is retained.
    """

    def __init__(self, stage, robot_path, body_names, *, world_transform=None):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self.Gf = Gf
        self.world_transform = Gf.Matrix4d(world_transform) if world_transform is not None else None
        root = stage.GetPrimAtPath(robot_path)
        if not root:
            raise ValueError(f'missing USD robot: {robot_path}')
        self.bodies = {p.GetName(): str(p.GetPath()) for p in root.GetChildren()}
        self.body_names = list(body_names)
        self.incoming, self.joint_names = {}, []
        cache = UsdGeom.XformCache()
        self.authored = {str(p.GetPath()): cache.GetLocalToWorldTransform(p)
                         for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Xformable)}

        def frame(position, quaternion):
            q = Gf.Quatd(quaternion.GetReal(), Gf.Vec3d(*quaternion.GetImaginary()))
            matrix = Gf.Matrix4d(1)
            matrix.SetRotate(q.GetNormalized())
            matrix.SetTranslateOnly(Gf.Vec3d(*position))
            return matrix

        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody1Rel().GetTargets()
            if not targets:
                continue
            child = str(targets[0])
            if child in self.incoming:
                raise ValueError('reference pose requires a tree without closed joint loops')
            parents = joint.GetBody0Rel().GetTargets()
            first = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            second = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            axis = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis_name = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                axis = Gf.Vec3d(*{'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[axis_name])
                self.joint_names.append(prim.GetName())
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f'unsupported reference joint: {prim.GetPath()}')
            self.incoming[child] = (str(parents[0]) if parents else None,
                                    second.GetInverse(), first, prim.GetName(), axis)

    def fingerprint(self):
        """Hash only compiled kinematics, independent of mesh/texture changes."""
        import hashlib
        import json
        def matrix(value):
            return [[float(v) for v in row] for row in value]
        roots = (set(self.bodies.values()) |
                 {entry[0] for entry in self.incoming.values() if entry[0] is not None}) - set(self.incoming)
        value = {
            'version': 1, 'bodies': self.bodies, 'joint_names': self.joint_names,
            'authored_roots': {path: matrix(transform) for path, transform in self.authored.items()
                               if path in roots},
            'incoming': {path: [parent, matrix(second), matrix(first), name,
                                list(axis) if axis is not None else None]
                         for path, (parent, second, first, name, axis) in self.incoming.items()},
        }
        if self.world_transform is not None:
            value['world_transform'] = matrix(self.world_transform)
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def snapshot(self, joint_positions, *, body_names=None):
        Gf = self.Gf
        unknown = set(joint_positions) - set(self.joint_names)
        if unknown or not all(math.isfinite(v) for v in joint_positions.values()):
            raise ValueError(f'invalid reference joints: {sorted(unknown)}')
        solved, visiting = {}, set()

        def solve(path):
            if path in solved:
                return solved[path]
            if path in visiting:
                raise ValueError('cycle in USD reference joint tree')
            visiting.add(path)
            if path not in self.incoming:
                result = self.authored[path]
            else:
                parent_path, second_inv, first, name, axis = self.incoming[path]
                parent = solve(parent_path) if parent_path else Gf.Matrix4d(1)
                motion = Gf.Matrix4d(1)
                if axis is not None:
                    motion.SetRotate(Gf.Rotation(axis, math.degrees(joint_positions.get(name, 0))))
                result = second_inv * motion * first * parent
            visiting.remove(path)
            solved[path] = result
            return result

        poses = []
        for name in self.body_names if body_names is None else body_names:
            matrix = solve(self.bodies[name])
            if self.world_transform is not None:
                matrix = matrix * self.world_transform
            rotation = Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
            poses.append([*matrix.ExtractTranslation(), rotation.GetReal(), *rotation.GetImaginary()])
        if not np.isfinite(poses).all():
            raise ValueError('non-finite reference pose')
        return {'body_poses_world': poses,
                'joint_positions_rad': [joint_positions.get(name, 0) for name in self.joint_names]}


def usd_reference_snapshot(stage, robot_path, body_names, joint_positions):
    """Solve an authored reference pose without SimulationApp or physics."""
    model = UsdKinematics(stage, robot_path, body_names)
    return model.snapshot(joint_positions), model.joint_names
