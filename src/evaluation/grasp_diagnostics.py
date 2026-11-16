"""Read-only Isaac/PhysX grasp evidence; imported only inside a running simulator."""
import json

import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    # PhysX reuses its count/index buffers between normal and friction queries.
    return np.array(value, copy=True)


def read_contact_view(view, dt):
    """Decode only occupied slots, converting impulses to N with the physics dt."""
    forces, points, normals, distances, counts, starts = map(_array, view.get_contact_data(dt))
    friction, friction_points, friction_counts, friction_starts = map(_array, view.get_friction_data(dt))
    pairs = []
    for sensor in range(view.sensor_count):
        for target in range(view.filter_count):
            count, start = int(counts[sensor, target]), int(starts[sensor, target])
            fcount, fstart = int(friction_counts[sensor, target]), int(friction_starts[sensor, target])
            if (count and (start < 0 or start + count > len(forces))) or (
                    fcount and (fstart < 0 or fstart + fcount > len(friction))):
                raise RuntimeError('PhysX contact buffer range exceeds capacity')
            contacts = [{'normal_force_n': float(forces[i].item()),
                         'point_world_m': points[i].tolist(), 'normal_world': normals[i].tolist(),
                         'separation_m': float(distances[i].item())}
                        for i in range(start, start + count)]
            pairs.append({'sensor_index': sensor, 'filter_index': target,
                          'normal_force_sum_n': sum(abs(c['normal_force_n']) for c in contacts),
                          'contacts': contacts,
                          'friction_anchors': [{'force_world_n': friction[i].tolist(),
                                                'point_world_m': friction_points[i].tolist()}
                                               for i in range(fstart, fstart + fcount)]})
    return {'status': 'ok', 'pairs': pairs, 'capacity': view.max_contact_data_count,
            'possibly_truncated': bool(counts.sum() >= view.max_contact_data_count
                                       or friction_counts.sum() >= view.max_contact_data_count)}


class GraspDiagnostics:
    def __init__(self, env, directory):
        import omni.usd
        self.env = env
        self.stage = omni.usd.get_context().get_stage()
        self.views = {}
        self.failures = {}
        self.fingers = [name for name in env.robot.body_names
                        if 'gripper' in name and name.endswith(('_Link_L11', '_Link_R11'))]
        # The articulation prim is nested under the spawned robot wrapper.
        self.body_paths = {prim.GetName(): str(prim.GetPath()) for prim in self.stage.Traverse()
                           if prim.GetName() in self.fingers}
        # GPU contact filters support rigid bodies. A static mesh's Xform path
        # is unsupported and would otherwise silently report zero contacts.
        self.filters = ['/World/SceneObjects/' + spec['name'] for spec in env.scene_object_specs
                        if spec['asset']['physics']['body_type'] == 'dynamic']
        for name in self.fingers:
            try:
                if not self.filters:
                    raise RuntimeError('scene has no dynamic objects to monitor')
                path = self.body_paths[name]
                view = env.sim.physics_sim_view.create_rigid_contact_view(
                    path, filter_patterns=self.filters, max_contact_data_count=512)
                if view.sensor_count != 1 or view.filter_count != len(self.filters):
                    raise RuntimeError(f'contact view matched {view.sensor_count} sensors, {view.filter_count} filters')
                self.views[name] = view
            except Exception as error:
                self.failures[name] = str(error)
        self.metadata = {
            'status': 'ok' if self.views and not self.failures else 'unavailable_or_partial',
            'errors': self.failures, 'finger_paths': self.body_paths, 'filter_paths': self.filters,
            'filter_scope': 'dynamic scene bodies only; static table contacts are not measured by these pair filters',
            'contact_source': 'PhysX RigidContactView; forces averaged over dt, world-frame points/normals',
            'torque_note': 'implicit_pd_estimate is Isaac Lab pre-step PD estimate, NOT measured grip force',
            'state_units': 'SI; joint positions rad for revolute, m for prismatic; quaternions wxyz',
            'controls': {name: getattr(env, name) for name in env._CONTROL_PATTERNS},
            'geometry': str(directory / 'collision_geometry.json'),
        }
        geometry = self._geometry()
        with (directory / 'collision_geometry.json').open('x', encoding='utf-8') as stream:
            json.dump(geometry, stream, allow_nan=False)
        self.metadata['geometry_status'] = geometry['status']
        self.cloth_entries = env.scene_object_keys.get('cloth', {})
        self.cloth_settings = {spec['name']: spec['asset']['physics']['cloth']
                               for spec in env.scene_object_specs
                               if spec['asset']['physics']['body_type'] == 'cloth'}
        self.cloth_hulls, self.cloth_geometry_error = {}, None
        if self.cloth_entries:
            from .cloth_grasp_diagnostics import compile_finger_hulls
            self.metadata['cloth_proximity_note'] = (
                'Geometric candidates from current particles and cooked finger hulls; '
                'cloth-only offset bands, excluding rigid collider offsets. Offset '
                'half-spaces overestimate proximity at corners. Counts are NOT '
                'contact forces or proof of grasp. Gap is a closing-axis projection.')
            try:
                self.cloth_hulls = {name: compile_finger_hulls(geometry['bodies'][name])
                                    for name in self.fingers}
            except Exception as error:
                self.cloth_geometry_error = str(error)

    def _sample_cloth(self, finger_poses):
        from .cloth_grasp_diagnostics import sample_cloth_grasp
        result = {}
        for name, entry in self.cloth_entries.items():
            try:
                if self.cloth_geometry_error:
                    raise RuntimeError(self.cloth_geometry_error)
                points = _array(entry['view'].get_positions()).reshape(-1, 3)
                settings = self.cloth_settings[name]
                result[name] = {}
                for side in ('left', 'right'):
                    names = {key: f'astribot_gripper_{side}_Link_{key}' for key in ('L11', 'R11')}
                    result[name][side] = sample_cloth_grasp(
                        points, {key: finger_poses[value] for key, value in names.items()},
                        {key: self.cloth_hulls[value] for key, value in names.items()},
                        settings['rest_offset'], settings['contact_offset'])
            except Exception as error:
                result[name] = {'status': 'unavailable', 'error': str(error)}
        return result

    def _geometry(self):
        """Export cooked PhysX hulls, not an independent approximation of visuals."""
        from pxr import Usd, UsdGeom, UsdPhysics
        from omni.physx import get_physx_cooking_interface
        cooking = get_physx_cooking_interface()
        bodies = {**self.body_paths, **{spec['name']: '/World/SceneObjects/' + spec['name']
                                     for spec in self.env.scene_object_specs
                                     if spec['asset']['physics']['body_type'] == 'dynamic'
                                     and spec['asset']['physics']['collision'] == 'convex_hull'}}
        result = {'status': 'ok', 'source': 'PhysX cooked convex mesh data',
                  'coordinates': 'vertices in collider local frame; row-vector USD collider_to_body matrix',
                  'bodies': {}}
        cache = UsdGeom.XformCache()
        for name, path in bodies.items():
            body = self.stage.GetPrimAtPath(path)
            entry = {'path': path, 'colliders': []}
            result['bodies'][name] = entry
            try:
                inverse = cache.GetLocalToWorldTransform(body).GetInverse()
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if not (prim.HasAPI(UsdPhysics.CollisionAPI) or
                            (prim.IsA(UsdGeom.Mesh) and '/collisions/' in str(prim.GetPath()))):
                        continue
                    collider = {'path': str(prim.GetPath()),
                                'approximation': str(prim.GetAttribute('physics:approximation').Get()),
                                'collider_to_body': np.asarray(cache.GetLocalToWorldTransform(prim) * inverse).tolist(),
                                'hulls': []}
                    entry['colliders'].append(collider)
                    count = cooking.get_nb_convex_mesh_data(str(prim.GetPath()))
                    for index in range(count):
                        hull = cooking.get_convex_mesh_data(str(prim.GetPath()), index)
                        collider['hulls'].append({
                            'vertices': [[float(v) for v in point] for point in hull['vertices']],
                            'indices': [int(v) for v in hull['indices']],
                            'polygons': [{'index_base': int(p['index_base']),
                                          'num_vertices': int(p['num_vertices'])} for p in hull['polygons']],
                        })
                if not any(c['hulls'] for c in entry['colliders']):
                    entry['error'] = 'No cooked convex hull returned; geometry unavailable (not an empty collider)'
                    result['status'] = 'partial'
            except Exception as error:
                entry['error'] = str(error)
                result['status'] = 'partial'
        return result

    def sample(self, reset=False):
        robot = self.env.robot
        data = robot.data
        result = {'status': 'ok', 'joint_names': list(robot.joint_names),
                  'joints': {key: _array(getattr(data, attr)[0]).tolist() for key, attr in (
                      ('position', 'joint_pos'), ('velocity', 'joint_vel'),
                      ('position_target', 'joint_pos_target'), ('velocity_target', 'joint_vel_target'))},
                  'finger_pose_world': {name: _array(data.body_state_w[0, robot.body_names.index(name), :7]).tolist()
                                        for name in self.fingers},
                  'objects': {name: _array(self.env.scene[key].data.root_state_w[0]).tolist()
                              for name, key in self.env.scene_object_keys['dynamic'].items()},
                  'contacts': {name: {'status': 'unavailable', 'error': error}
                               for name, error in self.failures.items()}}
        if self.cloth_entries:
            result['cloth_proximity'] = self._sample_cloth(result['finger_pose_world'])
        if reset:
            result['contacts'] = {'status': 'not_sampled', 'reason': 'reset has no completed physics step'}
            return result
        result['joints']['implicit_pd_estimate'] = _array(data.applied_torque[0]).tolist()
        for name, view in self.views.items():
            try:
                result['contacts'][name] = read_contact_view(view, self.env.dt)
            except Exception as error:
                result['contacts'][name] = {'status': 'unavailable', 'error': str(error)}
        if self.failures or not self.views or any(v['status'] != 'ok' for v in result['contacts'].values()):
            result['status'] = 'partial'
        return result
