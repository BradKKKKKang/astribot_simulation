"""PhysX FEM core and a single material-subset skin; import only after AppLauncher."""
import numpy as np

from .deformable_geometry import fit_deformable_pose, tetrahedral_volume
from .glb_geometry import rotation_matrix


def _mesh(stage, path, primitives):
    from pxr import UsdGeom, Vt
    vertices, faces, offset = [], [], 0
    for primitive in primitives:
        vertices.append(primitive['vertices'])
        faces.append(primitive['faces'] + offset)
        offset += len(primitive['vertices'])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(vertices).astype(np.float32)))
    triangles = np.concatenate(faces).astype(np.int32)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def author_deformable_object(stage, path, primitives, document, physics, material_for):
    from omni.physx import get_physx_cooking_interface
    from omni.physx.scripts import deformableUtils
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    cfg = physics['deformable']
    # Cook only the padded forms. Fur fibers, nose and cord are render detail,
    # embedded into this connected volume, not independent simulated bodies.
    core_path = path + '/CookingCore'
    core = _mesh(stage, core_path, [p for p in primitives if p['node'] in cfg['core_nodes']])
    try:
        if not deformableUtils.add_physx_deformable_body(
                stage, core_path, collision_simplification=True, collision_simplification_remeshing=True,
                simulation_hexahedral_resolution=cfg['simulation_hexahedral_resolution'],
                collision_simplification_remeshing_resolution=cfg['collision_simplification_remeshing_resolution'],
                collision_simplification_target_triangle_count=cfg['collision_simplification_target_triangle_count']):
            raise RuntimeError('Failed to author deformable cooking core: ' + path)
        get_physx_cooking_interface().cook_deformable_body_mesh(core_path)
        api = PhysxSchema.PhysxDeformableBodyAPI(core.GetPrim())
        sim_points = api.GetSimulationRestPointsAttr().Get()
        sim_indices = PhysxSchema.PhysxDeformableAPI(api).GetSimulationIndicesAttr().Get()
        collision_points = api.GetCollisionRestPointsAttr().Get()
        collision_indices = api.GetCollisionIndicesAttr().Get()
        if not sim_points or not sim_indices or not collision_points or not collision_indices:
            raise RuntimeError('PhysX failed to cook deformable core: ' + path)
        volume = tetrahedral_volume(sim_points, sim_indices)
        tetrahedral_volume(collision_points, collision_indices)
    finally:
        stage.RemovePrim(core_path)

    mesh = _mesh(stage, path + '/SoftBody', primitives)
    normals, colors, uvs, face_offset = [], [], [], 0
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    for index, primitive in enumerate(primitives):
        attrs, count = primitive['attributes'], len(primitive['vertices'])
        normals.append(attrs.get('NORMAL', np.zeros((count, 3))))
        uv = attrs.get('TEXCOORD_0', np.zeros((count, 2))).copy()
        uv[:, 1] = 1 - uv[:, 1]
        uvs.append(uv)
        color = attrs.get('COLOR_0', np.ones((count, 3)))[:, :3].copy()
        if 'COLOR_0' in attrs and primitive['material'] is not None:
            factor = document['materials'][primitive['material']].get('pbrMetallicRoughness', {}).get(
                'baseColorFactor', [1, 1, 1, 1])[:3]
            color *= factor
        colors.append(color)
        face_count = len(primitive['faces'])
        subset = binding.CreateMaterialBindSubset(f'part_{index}',
                    Vt.IntArray.FromNumpy(np.arange(face_offset, face_offset + face_count, dtype=np.int32)))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            material_for(primitive['material'], 'COLOR_0' in attrs))
        face_offset += face_count
    binding.SetMaterialBindSubsetsFamilyType('partition')
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(normals).astype(np.float32)))
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(np.concatenate(colors).astype(np.float32)))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
        Vt.Vec2fArray.FromNumpy(np.concatenate(uvs).astype(np.float32)))
    if not deformableUtils.add_physx_deformable_body(
            stage, str(mesh.GetPath()), simulation_rest_points=sim_points, simulation_indices=sim_indices,
            collision_rest_points=collision_points, collision_indices=collision_indices,
            solver_position_iteration_count=cfg['solver_position_iteration_count'],
            vertex_velocity_damping=cfg['vertex_velocity_damping'], self_collision=cfg['self_collision']):
        raise RuntimeError('Failed to author deformable skin: ' + path)
    collision = PhysxSchema.PhysxCollisionAPI(mesh.GetPrim())
    collision.CreateContactOffsetAttr(cfg['contact_offset'])
    collision.CreateRestOffsetAttr(cfg['rest_offset'])
    material_cfg = sim_utils.DeformableBodyMaterialCfg(
        density=physics['mass'] / volume, dynamic_friction=physics.get('friction', .6),
        youngs_modulus=cfg['youngs_modulus'], poissons_ratio=cfg['poissons_ratio'],
        elasticity_damping=cfg['elasticity_damping'])
    material_path = path + '/PhysicsMaterial'
    material_cfg.func(material_path, material_cfg)
    binding.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)), materialPurpose='physics')
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(physics['mass'])
    mesh.GetPrim().SetCustomDataByKey('core_volume_m3', volume)


def reset_deformable_objects(scene, entries, sampled):
    for name, entry in entries.items():
        if sampled is not None and name not in sampled:
            continue
        body = scene[entry['key']]
        state = body.data.default_nodal_state_w.clone()
        if sampled is not None:
            initial, target = entry['initial_pose'], sampled[name]['pose_world']
            # Rotate about the asset origin, not the nodal centroid. This keeps
            # geometry_min grounding consistent for offset/asymmetric objects.
            delta = rotation_matrix(target[3:]) @ rotation_matrix(initial[3:]).T
            state[..., :3] = ((state[..., :3] - state.new_tensor(initial[:3]))
                              @ state.new_tensor(delta.T) + state.new_tensor(target[:3]))
        state[..., 3:] = 0
        body.write_nodal_state_to_sim(state)
        targets = body.data.nodal_kinematic_target.clone()
        targets[..., :3], targets[..., 3] = state[..., :3], 1.
        body.write_nodal_kinematic_target_to_sim(targets)
        body.reset()


def read_deformable_states(scene, entries):
    objects = {}
    for name, entry in entries.items():
        body = scene[entry['key']]
        current = body.data.nodal_pos_w[0].detach().cpu().numpy()
        rest = body.data.default_nodal_state_w[0, :, :3].detach().cpu().numpy()
        pose, residual = fit_deformable_pose(rest, current, entry['initial_pose'])
        objects[name] = {'pose_world': pose, 'source': 'isaaclab_deformable_object',
                         'pose_method': 'best_fit_material_frame', 'deformation_rms_m': residual,
                         'simulation_bounds_world': {'min': current.min(0).tolist(), 'max': current.max(0).tolist()}}
    return objects
