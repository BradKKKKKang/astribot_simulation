"""Author GLB scene bodies after AppLauncher; no runtime imports at module load."""
from pathlib import Path
import tempfile

import numpy as np

from .scene_assets import asset_geometry, transformed_primitives


def author_scene_object(stage, specification, texture_directory):
    """Create a static collider or an independent dynamic rigid body on a USD stage."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    name, asset = specification['name'], specification['asset']
    path = '/World/SceneObjects/' + name
    if stage.GetPrimAtPath(path):
        raise ValueError('scene object prim already exists: ' + path)
    root = UsdGeom.Xform.Define(stage, path)
    pose = specification['pose_world']
    root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
    root.AddOrientOp().Set(Gf.Quatf(pose[3], Gf.Vec3f(*pose[4:])))
    root.GetPrim().SetCustomDataByKey('source_sha256', asset['source']['sha256'])
    root.GetPrim().SetCustomDataByKey('initial_placement_frame', specification['initial_frame'])
    geometry = asset_geometry(asset)
    document = geometry['document']
    material_cache = {}

    def material_for(index, vertex_colors):
        key = (index, vertex_colors)
        if key in material_cache:
            return material_cache[key]
        definition = document.get('materials', [])[index] if index is not None else {}
        pbr = definition.get('pbrMetallicRoughness', {})
        material_path = f'{path}/Materials/material_{len(material_cache)}'
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        color = pbr.get('baseColorFactor', [1., 1., 1., 1.])
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(color[3]))
        overrides = asset.get('material', {})
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(overrides.get('roughness', pbr.get('roughnessFactor', 1.))))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(overrides.get('metallic', pbr.get('metallicFactor', 1.))))
        if vertex_colors:
            reader = UsdShade.Shader.Define(stage, material_path + '/Color')
            reader.CreateIdAttr('UsdPrimvarReader_float3')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
            shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(), 'result')
        if 'baseColorTexture' in pbr:
            texture = document['textures'][pbr['baseColorTexture']['index']]
            image_index = texture['source']
            image = document['images'][image_index]
            if 'bufferView' not in image or image.get('mimeType') not in ('image/png', 'image/jpeg'):
                raise ValueError('scene textures require embedded PNG or JPEG images')
            view = document['bufferViews'][image['bufferView']]
            offset = view.get('byteOffset', 0)
            extension = '.png' if image['mimeType'] == 'image/png' else '.jpg'
            image_path = Path(texture_directory) / f'{geometry["sha256"]}_{image_index}{extension}'
            if not image_path.exists():
                image_path.write_bytes(geometry['binary'][offset:offset + view['byteLength']])
            reader = UsdShade.Shader.Define(stage, material_path + '/UV')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            texture_shader = UsdShade.Shader.Define(stage, material_path + '/Texture')
            texture_shader.CreateIdAttr('UsdUVTexture')
            texture_shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(image_path))
            texture_shader.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            texture_shader.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            texture_shader.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color))
            shader.GetInput('diffuseColor').ConnectToSource(texture_shader.ConnectableAPI(), 'rgb')
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        material_cache[key] = material
        return material

    all_vertices, all_faces, count = [], [], 0
    for index, primitive in enumerate(transformed_primitives(asset, specification['scale'])):
        vertices, faces = primitive['vertices'], primitive['faces']
        mesh = UsdGeom.Mesh.Define(stage, f'{path}/Visuals/mesh_{index}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        attributes = primitive['attributes']
        if 'NORMAL' in attributes:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(attributes['NORMAL'].astype(np.float32)))
            mesh.SetNormalsInterpolation('vertex')
        if 'TEXCOORD_0' in attributes:
            uv = attributes['TEXCOORD_0'].copy()
            uv[:, 1] = 1 - uv[:, 1]
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
                Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
        if 'COLOR_0' in attributes:
            colors = attributes['COLOR_0'][:, :3]
            definition = document['materials'][primitive['material']] if primitive['material'] is not None else {}
            factor = definition.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1])[:3]
            mesh.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy((colors * factor).astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material_for(primitive['material'], 'COLOR_0' in attributes))
        all_vertices.append(vertices)
        all_faces.append(faces + count)
        count += len(vertices)

    # One complete collision mesh preserves the open vessel. Splitting by visual
    # material would produce open strips, and a single convex hull closes its mouth.
    collision = UsdGeom.Mesh.Define(stage, path + '/Collision')
    vertices, faces = np.concatenate(all_vertices), np.concatenate(all_faces)
    collision.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    collision.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    collision.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    collision.CreateSubdivisionSchemeAttr('none')
    collision.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    physics = asset['physics']
    approximation = {'triangle_mesh': 'none', 'convex_hull': 'convexHull', 'sdf': 'sdf'}[physics['collision']]
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(approximation)
    if physics['collision'] == 'sdf':
        from pxr import PhysxSchema
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collision.GetPrim()).CreateSdfResolutionAttr(256)
    material = UsdShade.Material.Define(stage, path + '/PhysicsMaterial')
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(physics.get('friction', .6)))
    material_api.CreateDynamicFrictionAttr(float(physics.get('friction', .6)))
    material_api.CreateRestitutionAttr(float(physics.get('restitution', 0.)))
    UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(material, materialPurpose='physics')
    if physics['body_type'] == 'dynamic':
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(physics['mass']))
    return path


def attach_scene_objects(scene_cfg, specifications):
    """Populate InteractiveScene with independent bodies before its construction."""
    if not specifications:
        return {'dynamic': {}, 'static': {}, 'resources': None}
    import omni.usd
    from isaaclab.assets import RigidObjectCfg
    resources = tempfile.TemporaryDirectory(prefix='astribot-scene-textures-')
    runtime = {'dynamic': {}, 'static': {}, 'resources': resources}
    try:
        stage = omni.usd.get_context().get_stage()
        for specification in specifications:
            path = author_scene_object(stage, specification, resources.name)
            name, pose = specification['name'], specification['pose_world']
            if specification['asset']['physics']['body_type'] == 'static':
                runtime['static'][name] = path
                continue
            key = 'scene_object_' + name
            setattr(scene_cfg, key, RigidObjectCfg(prim_path=path, spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:]))))
            runtime['dynamic'][name] = key
        return runtime
    except Exception:
        resources.cleanup()
        raise


def reset_scene_objects(scene, runtime, *, sampled=None):
    for name, key in runtime['dynamic'].items():
        if sampled is not None and name not in sampled:
            continue
        body = scene[key]
        state = body.data.default_root_state.clone()
        if sampled is not None:
            state[:, :7] = state.new_tensor(sampled[name]['pose_world'])
            state[:, 7:] = 0.
        body.write_root_pose_to_sim(state[:, :7])
        body.write_root_velocity_to_sim(state[:, 7:])
        body.reset()


def read_scene_object_states(scene, runtime):
    objects = {}
    if runtime['static']:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        for name, path in runtime['static'].items():
            prim = stage.GetPrimAtPath(path)
            if not prim:
                continue
            transform = cache.GetLocalToWorldTransform(prim)
            quaternion = transform.ExtractRotationQuat()
            objects[name] = {'pose_world': [*transform.ExtractTranslation(), quaternion.GetReal(),
                                           *quaternion.GetImaginary()], 'source': 'isaac_usd_stage'}
    for name, key in runtime['dynamic'].items():
        objects[name] = {'pose_world': scene[key].data.root_state_w[0, :7].detach().cpu().tolist(),
                         'source': 'isaaclab_rigid_object'}
    return objects


def close_scene_objects(runtime):
    if runtime.get('resources') is not None:
        runtime['resources'].cleanup()
