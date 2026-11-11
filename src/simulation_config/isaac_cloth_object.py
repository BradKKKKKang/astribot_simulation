"""Live PhysX particle cloth. Import only after AppLauncher.

The source is a sewn, single midsurface with UVs. That same surface is rendered,
so contacts and visual deformation share vertices (no independent skin offsets).
"""
import numpy as np
from .glb_geometry import rotation_matrix


def author_cloth(stage, path, primitives, physics, pose, material_for):
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    from omni.physx.scripts import particleUtils, physicsUtils
    p = primitives[0]
    c = physics['cloth']
    system = path + '/ParticleSystem'
    particleUtils.add_physx_particle_system(stage, system, simulation_owner=Sdf.Path('/physicsScene'),
        contact_offset=c['contact_offset'], rest_offset=c['rest_offset'],
        particle_contact_offset=c['contact_offset'], solid_rest_offset=c['rest_offset'], fluid_rest_offset=0.,
        solver_position_iterations=c['solver_iterations'], enable_ccd=True, max_velocity=8.,
        max_depenetration_velocity=1., global_self_collision_enabled=True)
    material = path + '/ClothMaterial'
    particleUtils.add_pbd_particle_material(stage, material, friction=physics['friction'],
        damping=c['velocity_damping'], drag=0., lift=0.)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), Sdf.Path(material))
    UsdGeom.Xformable(stage.GetPrimAtPath(path)).ClearXformOpOrder()
    mesh = UsdGeom.Mesh.Define(stage, path+'/Cloth')
    points = p['vertices'] @ rotation_matrix(pose[3:]).T + pose[:3]
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(p['faces']),3,np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(p['faces'].astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    if 'TEXCOORD_0' in p['attributes']:
        uv=p['attributes']['TEXCOORD_0'].copy(); uv[:,1]=1-uv[:,1]
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material_for(p['material'],False))
    # A render-only copy shares the exact physical topology. Updating it cannot
    # re-author solver state or reset particle velocities during a snapshot.
    Sdf.CopySpec(stage.GetRootLayer(),mesh.GetPath(),stage.GetRootLayer(),Sdf.Path(path+'/Visual'))
    mesh.CreateVisibilityAttr('invisible')
    particleUtils.add_physx_particle_cloth(stage,str(mesh.GetPath()),None,Sdf.Path(system),
        spring_stretch_stiffness=c['stretch_stiffness'],spring_shear_stiffness=c['shear_stiffness'],
        spring_bend_stiffness=c['bend_stiffness'],spring_damping=c['spring_damping'],
        self_collision=True,self_collision_filter=True)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(physics['mass'])
    return str(mesh.GetPath())


def _initialize(entry):
    if 'view' in entry: return
    import omni.physics.tensors as tensors
    import omni.usd
    sim=tensors.create_simulation_view('torch')
    view=sim.create_particle_cloth_view(entry['path'])
    if view.count != 1: raise RuntimeError('Cloth is not initialized: '+entry['path'])
    prim=omni.usd.get_context().get_stage().GetPrimAtPath(entry['path'])
    points=entry['rest']
    if view.max_particles_per_cloth != len(points):
        mapping=np.asarray(prim.GetAttribute('physxParticle:weldedVerticesRemapToOrig').Get(),dtype=int)
        if mapping.shape != (view.max_particles_per_cloth,): raise RuntimeError('Invalid cloth welding map')
        points=points[mapping]
    entry.update(sim_view=sim,view=view,particle_rest=points)


def reset_cloth(entries, sampled):
    import torch
    for name,entry in entries.items():
        if sampled is not None and name not in sampled: continue
        _initialize(entry)
        pose=entry['initial_pose'] if sampled is None else sampled[name]['pose_world']
        view=entry['view']; current=view.get_positions()
        points=entry['particle_rest'] @ rotation_matrix(pose[3:]).T + pose[:3]
        values=torch.as_tensor(points,dtype=current.dtype,device=current.device).reshape_as(current)
        indices=torch.zeros(1,dtype=torch.int32,device=current.device)
        view.set_positions(values,indices)
        view.set_velocities(torch.zeros_like(values),indices)
        entry['reset_pose']=list(pose)
        entry['reset_region']=None if sampled is None else sampled[name].get('reset_region')


def read_cloth(entries):
    from .deformable_geometry import fit_deformable_pose
    result={}
    for name,entry in entries.items():
        _initialize(entry)
        points=entry['view'].get_positions().cpu().numpy().reshape(-1,3)
        if not np.isfinite(points).all(): raise RuntimeError('Nonfinite cloth state: '+name)
        initial=entry['initial_pose']
        rest=entry['particle_rest'] @ rotation_matrix(initial[3:]).T + initial[:3]
        pose,residual=fit_deformable_pose(rest,points,initial)
        result[name]={'pose_world':pose,'source':'physx_particle_cloth','deformation_rms_m':residual,
          'simulation_bounds_world':{'min':points.min(0).tolist(),'max':points.max(0).tolist()},
          'simulation_particles':len(points),'reset_region':entry.get('reset_region'),
          'release_pose_world':entry.get('reset_pose')}
    return result


def sync_cloth_visuals(entries):
    """Explicit tensor-to-USD refresh; managed snapshots never advance physics."""
    import omni.usd
    from pxr import UsdGeom,Vt
    for entry in entries.values():
        _initialize(entry)
        prim=omni.usd.get_context().get_stage().GetPrimAtPath(entry['path'])
        points=entry['view'].get_positions().cpu().numpy().reshape(-1,3)
        if len(points)!=len(entry['rest']):
            mapping=np.asarray(prim.GetAttribute('physxParticle:weldedVerticesRemapToWeld').Get(),dtype=int)
            if mapping.shape!=(len(entry['rest']),): raise RuntimeError('Invalid cloth render welding map')
            points=points[mapping]
        visual=UsdGeom.Mesh.Get(omni.usd.get_context().get_stage(),prim.GetParent().GetPath().AppendChild('Visual'))
        visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        triangles=np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        face_normals=np.cross(points[triangles[:,1]]-points[triangles[:,0]],
                              points[triangles[:,2]]-points[triangles[:,0]])
        normals=np.zeros_like(points)
        for column in range(3): np.add.at(normals,triangles[:,column],face_normals)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        visual.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        visual.SetNormalsInterpolation('vertex')
