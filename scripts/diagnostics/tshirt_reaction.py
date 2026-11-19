"""Process-local collision-shell intervention for the T-shirt grasp experiment.

No cloth pins, forces, or attachment constraints. The same native collision
shape and mass distribution are used in both shell modes. An ordinary fixed
joint (excluded from articulation) lets the PBD impulse reach the robot.
"""
import json
import numpy as np

FINGERS = ['astribot_gripper_right_Link_' + side for side in ('L11', 'R11')]


class GapFeedback:
    """Diagnostic closure compensation against a completed reference gap trace.

Uses measured geometry at the preceding completed physics boundary. The same
controller runs in both matched-gap arms; no estimated contact force is used.
"""
    def __init__(self, reference_trace, current_trace):
        self.targets = {}
        for line in reference_trace.open():
            row = json.loads(line)
            if row['event'] == 'state':
                self.targets[row['step_id']] = row['diagnostics']['cloth_proximity']['tshirt']['right']['closing_axis_gap_m']
        if set(self.targets) != set(range(1231)):
            raise ValueError('Gap reference must be a complete 1230-step trial')
        self.stream = current_trace.open()
        self.bias = 0.
        self.records = []

    def command(self, step, nominal):
        previous = step - 1
        for line in self.stream:
            row = json.loads(line)
            if row['event'] == 'state' and row['step_id'] == previous:
                measured = row['diagnostics']['cloth_proximity']['tshirt']['right']['closing_axis_gap_m']
                break
        else:
            raise RuntimeError(f'Completed gap feedback missing at step {previous}')
        error_mm = (measured-self.targets[previous])*1000
        self.bias += float(np.clip(.2*error_mm, -.25, .25))
        value = float(np.clip(nominal+self.bias, 0., 100.))
        self.bias = value-nominal  # anti-windup at the physical command limits
        self.records.append({'step': step, 'feedback_step': previous,
                             'reference_gap_mm': self.targets[previous]*1000,
                             'measured_gap_mm': measured*1000, 'error_mm': error_mm,
                             'nominal_percent': nominal, 'command_percent': value})
        return value

    def close(self):
        self.stream.close()


def author_shells(stage, mode, directory):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    excluded = mode == 'rigid_shell'
    if mode not in ('art_shell', 'rigid_shell'):
        raise ValueError('Unknown reaction mode: ' + mode)
    cache = UsdGeom.XformCache()
    paths = {p.GetName(): p.GetPath() for p in stage.Traverse() if p.GetName() in FINGERS}
    if set(paths) != set(FINGERS):
        raise RuntimeError('Expected both original right finger bodies')
    # Capture adjacent-body collision exclusions before adding new fixed joints.
    adjacent = {name: [] for name in FINGERS}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            pair = j.GetBody0Rel().GetTargets() + j.GetBody1Rel().GetTargets()
            for name, path in paths.items():
                if path in pair:
                    adjacent[name].extend(p for p in pair if p != path)
    parent = '/World/ReactionShells'
    UsdGeom.Xform.Define(stage, parent)
    records = []
    for name, path in paths.items():
        original = stage.GetPrimAtPath(path)
        mass_api = UsdPhysics.MassAPI(original)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass <= 0 or min(inertia) <= 0:
            raise RuntimeError('Shell experiment needs explicit positive finger inertia')
        world = cache.GetLocalToWorldTransform(original)
        shell_path = Sdf.Path(parent + '/' + name + '_shell')
        shell = stage.DefinePrim(shell_path, 'Xform')
        shell.GetReferences().AddInternalReference(path)
        xf = UsdGeom.Xformable(shell)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(world)
        # Split co-located mass tensors, preserving the combined rigid inertia.
        mass_api.CreateMassAttr(mass * .5)
        mass_api.CreateDiagonalInertiaAttr(inertia * .5)
        shell_mass = UsdPhysics.MassAPI(shell)
        shell_mass.CreateMassAttr(mass * .5)
        shell_mass.CreateDiagonalInertiaAttr(inertia * .5)
        colliders = [p for p in Usd.PrimRange(original, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1 or colliders[0].IsInstanceProxy():
            raise RuntimeError('Unexpected finger collider hierarchy')
        collider = colliders[0]
        relative = collider.GetPath().MakeRelativePath(path)
        UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
        cloned = stage.GetPrimAtPath(shell_path.AppendPath(relative))
        UsdPhysics.CollisionAPI(cloned).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(original).CreateVisibilityAttr('invisible')
        UsdGeom.Imageable(shell).CreateVisibilityAttr('inherited')
        UsdPhysics.FilteredPairsAPI.Apply(shell).CreateFilteredPairsRel().SetTargets(
            list(dict.fromkeys([path, *adjacent[name]])))
        joint = UsdPhysics.FixedJoint.Define(stage, parent + '/' + name + '_fixed')
        joint.CreateBody0Rel().SetTargets([path])
        joint.CreateBody1Rel().SetTargets([shell_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateExcludeFromArticulationAttr(excluded)
        records.append({'finger': name, 'original': str(path), 'shell': str(shell_path),
                        'joint': str(joint.GetPath()), 'exclude_from_articulation': excluded,
                        'total_mass_kg': mass, 'total_diagonal_inertia': list(inertia),
                        'filtered_pairs': [str(p) for p in [path, *adjacent[name]]],
                        'source_collider': str(collider.GetPath()),
                        'shell_collider': str(cloned.GetPath())})
    (directory/'reaction_setup.json').write_text(json.dumps(records, indent=2)+'\n')
    return records


class ShellState:
    def __init__(self, env, records):
        self.env, self.records = env, records
        self.view = env.sim.physics_sim_view.create_rigid_body_view([r['shell'] for r in records])
        if self.view.count != 2:
            raise RuntimeError('Shell rigid-body view failed')
        self.indices = [list(self.view.prim_paths).index(r['shell']) for r in records]
        self.names = [r['finger'] for r in records]
        self.robot_indices = [env.robot.body_names.index(n) for n in self.names]
        self.excluded = records[0]['exclude_from_articulation']

    def reset(self):
        if not self.excluded:
            return
        import torch
        states = self.env.robot.data.body_state_w[0, self.robot_indices].clone()
        poses = self.view.get_transforms().clone()
        velocities = self.view.get_velocities().clone()
        poses[self.indices] = states[:, [0, 1, 2, 4, 5, 6, 3]]
        velocities[self.indices] = states[:, 7:13]
        ids = torch.arange(2, dtype=torch.int32, device=poses.device)
        self.view.set_transforms(poses, ids)
        self.view.set_velocities(velocities, ids)

    def poses(self):
        poses = self.view.get_transforms().cpu().numpy()[self.indices]
        return poses[:, [0, 1, 2, 6, 3, 4, 5]]

    def particle_body_poses(self, poses):
        poses = poses.copy()
        poses[self.robot_indices] = self.poses()
        return poses

    def sample(self):
        proxy = self.poses()
        original = self.env.robot.data.body_state_w[0, self.robot_indices, :7].cpu().numpy()
        dot = np.abs(np.sum(original[:, 3:] * proxy[:, 3:], axis=1))
        return {'mode': 'rigid_shell' if self.excluded else 'art_shell',
                'finger_names': self.names, 'shell_pose_world': proxy.tolist(),
                'original_pose_world': original.tolist(),
                'translation_error_m': np.linalg.norm(proxy[:, :3]-original[:, :3], axis=1).tolist(),
                'rotation_error_deg': np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.))).tolist(),
                'wrench_body_names': list(self.env.robot.body_names)}


def install(directory, mode):
    """Called before simulator creation, only in an owned diagnostic process."""
    from src.astribot_envs import astribot_isaaclab_env as module
    from src.evaluation.grasp_diagnostics import GraspDiagnostics
    env_type = module.AstribotIsaacLabEnv
    original_setup = env_type.setup_simulation_and_scene
    original_restore = env_type._restore_initial_state
    original_geometry = GraspDiagnostics._geometry
    original_sample = GraspDiagnostics.sample

    def setup(self):
        original_scene = module.InteractiveScene

        def scene(cfg):
            import omni.usd
            value = original_scene(cfg)
            self._reaction_records = author_shells(omni.usd.get_context().get_stage(), mode, directory)
            return value

        module.InteractiveScene = scene
        try:
            return original_setup(self)
        finally:
            module.InteractiveScene = original_scene

    def restore(self):
        original_restore(self)
        if not hasattr(self, '_reaction_bridge'):
            self._reaction_bridge = ShellState(self, self._reaction_records)
        self._reaction_bridge.reset()
        self.sim.forward()

    def geometry(self):
        for record in self.env._reaction_records:
            self.body_paths[record['finger']] = record['shell']
        return original_geometry(self)

    def sample(self, reset=False):
        result = original_sample(self, reset=reset)
        bridge = self.env._reaction_bridge
        for name, pose in zip(bridge.names, bridge.poses()):
            result['finger_pose_world'][name] = pose.tolist()
        result['cloth_proximity'] = self._sample_cloth(result['finger_pose_world'])
        result['reaction_shell'] = bridge.sample()
        return result

    env_type.setup_simulation_and_scene = setup
    env_type._restore_initial_state = restore
    GraspDiagnostics._geometry = geometry
    GraspDiagnostics.sample = sample
