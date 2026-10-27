"""Geometry support placement, asset provenance and independent USD scene bodies."""
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulation_config.experiment_config import _section, read_json
from src.simulation_config.glb_geometry import read_glb, rotation_matrix
from src.simulation_config.scene_assets import (
    expand_scene_assets, resolve_scene_objects, transformed_primitives, validate_asset,
)

ROOT = Path(__file__).resolve().parents[2]


def write_glb(path, vertices, node=None):
    vertices = np.asarray(vertices, dtype='<f4')
    binary = vertices.tobytes()
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}],
           'bufferViews': [{'buffer': 0, 'byteLength': len(binary)}],
           'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}],
           'nodes': [{'mesh': 0, 'name': 'panel', **(node or {})}], 'scenes': [{'nodes': [0]}]}
    encoded = json.dumps(doc).encode()
    encoded += b' ' * (-len(encoded) % 4)
    data = (struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary))
            + struct.pack('<II', len(encoded), 0x4e4f534a) + encoded
            + struct.pack('<II', len(binary), 0x004e4942) + binary)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def profile(path, vertices=None, *, static=False, node=None):
    vertices = vertices if vertices is not None else [[-.03, -.01, -.1], [.03, -.01, -.1], [0, .02, .1]]
    digest = write_glb(path, vertices, node)
    return {'name': path.stem, 'source': {'path': str(path), 'sha256': digest},
            'source_transform': {'quaternion_wxyz': [1, 0, 0, 0], 'scale': [1, 1, 1]},
            'physics': {'body_type': 'static' if static else 'dynamic',
                        'collision': 'triangle_mesh' if static else 'convex_hull', 'mass': .02}}


@pytest.fixture
def scene(tmp_path):
    table = profile(tmp_path/'table.glb', [[-1,-.5,.4], [1,-.5,.4], [1,.5,.4],
                     [-1,-.5,.4], [1,.5,.4], [-1,.5,.4], [-1,-.5,-.4], [1,-.5,-.4], [1,.5,-.4]], static=True)
    table['surfaces'] = {'tabletop': {'position': [0,0,.4], 'quaternion_wxyz': [1,0,0,0],
        'normal':[0,0,1], 'long_direction':[1,0,0], 'short_direction':[0,1,0],
        'geometry_node':'panel', 'usable_bounds':{'min':[-.9,-.4], 'max':[.9,.4]}}}
    pen = profile(tmp_path/'pen.glb')
    return {'objects': {
        'pen': {'asset':pen,'scale':[2,1,1.5], 'placement':{'frame':'table.tabletop','xy':[.2,0],
                 'quaternion_wxyz':[math.sqrt(.5),0,math.sqrt(.5),0], 'grounding':'geometry_min','clearance':.001}},
        'table': {'asset':table,'placement':{'frame':'world','xy':[1,2],
                 'quaternion_wxyz':[math.sqrt(.5),0,0,math.sqrt(.5)],'grounding':'geometry_min'}}}}


def test_grounding_uses_actual_rotated_scaled_geometry_and_parent_frame(scene):
    expanded = expand_scene_assets(scene, ROOT)
    objects = {o['name']:o for o in resolve_scene_objects(expanded)[0]}
    assert objects['table']['bounds_world']['min'][2] == pytest.approx(0)
    assert objects['pen']['bounds_world']['min'][2] == pytest.approx(.801)
    assert objects['pen']['pose_world'][:2] == pytest.approx([1,2.2])
    assert expanded['frames']['table.tabletop']['parent'] == 'table'
    # Independently project all transformed vertices, rather than assuming a half-height.
    pen = objects['pen']
    vertices = np.concatenate([p['vertices'] for p in transformed_primitives(pen['asset'], pen['scale'])])
    world = vertices @ rotation_matrix(pen['pose_world'][3:]).T + pen['pose_world'][:3]
    assert world[:,2].min() == pytest.approx(.801)


def test_read_glb_applies_node_transforms_and_rejects_changed_source(tmp_path):
    asset = profile(tmp_path/'node.glb', node={'translation':[3,4,5], 'scale':[2,3,4]})
    vertices = read_glb(asset['source']['path'])['primitives'][0]['vertices']
    assert vertices.min(0) == pytest.approx([2.94,3.97,4.6])
    asset['source']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='digest changed'):
        validate_asset(asset)


def test_rotated_height_uses_vertices_instead_of_aabb_corners(tmp_path):
    asset = profile(tmp_path/'triangle.glb', [[1,0,1],[0,1,0],[0,0,0]])
    q = [math.cos(math.pi/8),0,math.sin(math.pi/8),0]
    scene = {'objects':{'shape':{'asset':asset,'placement':{'frame':'world','xy':[0,0],
                          'quaternion_wxyz':q,'grounding':'geometry_min'}}}}
    obj = expand_scene_assets(scene,tmp_path)['frames']['shape']
    # Every true vertex has z >= 0 after rotation; an AABB corner would give -sqrt(.5).
    assert obj['position'][2] == pytest.approx(0,abs=1e-7)
    assert resolve_scene_objects(scene)[0][0]['bounds_world']['min'][2] == pytest.approx(0)


def test_support_bounds_check_full_footprint_not_only_origin(scene):
    scene['objects']['pen']['placement']['xy'] = [.85,0]
    with pytest.raises(ValueError, match='beyond usable'):
        expand_scene_assets(scene, ROOT)
    scene['objects']['pen']['placement']['frame'] = 'pen.tabletop'
    with pytest.raises(ValueError, match='cyclic'):
        expand_scene_assets(scene, ROOT)


def test_usable_surface_corners_need_real_support(tmp_path, scene):
    asset = profile(tmp_path/'triangular_panel.glb', [[-1,-1,.4],[1,-1,.4],[0,1,.4]], static=True)
    asset['surfaces'] = copy.deepcopy(scene['objects']['table']['asset']['surfaces'])
    asset['surfaces']['tabletop']['usable_bounds'] = {'min':[-.5,-.5], 'max':[.5,.5]}
    with pytest.raises(ValueError, match='supporting top-face'):
        validate_asset(asset)


def test_nested_asset_and_scene_refs_use_each_declaration_directory(tmp_path, scene):
    base, child = tmp_path/'base', tmp_path/'child'
    base.mkdir(); child.mkdir()
    first = profile(base/'first.glb')
    first['source']['path'] = 'first.glb'
    (base/'asset.json').write_text(json.dumps(first))
    (child/'nested.json').write_text(json.dumps({'ref':'../base/asset.json'}))
    loaded = expand_scene_assets({'objects':{'pen':{'asset':{'ref':'nested.json'},
                       'placement':{'frame':'world','xy':[0,0],'grounding':'geometry_min'}}}},child)
    assert loaded['objects']['pen']['asset']['source']['path'] == str(base/'first.glb')
    second = profile(child/'second.glb', [[0,0,-.3],[.1,0,.1],[0,.1,.1]])
    second['source']['path'] = 'second.glb'
    (child/'asset.json').write_text(json.dumps(second))
    (base/'scene.json').write_text(json.dumps({'objects':{'pen':{'asset':{'ref':'asset.json'},
                         'placement':{'frame':'world','xy':[0,0],'grounding':'geometry_min'}}}}))
    (child/'scene.json').write_text(json.dumps({'ref':'../base/scene.json','overrides':{
                     'objects':{'pen':{'asset':{'ref':'asset.json'}}}}}))
    result = _section(read_json(child/'scene.json'), child, scene_assets=True)
    assert result['objects']['pen']['asset']['source']['path'] == str(child/'second.glb')
    assert result['frames']['pen']['position'][2] == pytest.approx(.3)
    override = {'ref':'nested.json','overrides':{'source':second['source']}}
    loaded['objects']['pen']['asset'] = override
    result = expand_scene_assets(loaded,child)
    assert result['objects']['pen']['asset']['source']['path'] == str(child/'second.glb')


def test_pure_asset_resolution_never_imports_runtime(monkeypatch, scene):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('isaaclab','isaacsim','omni','pxr','rclpy','rospy'):
            raise AssertionError('runtime import: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    expanded = expand_scene_assets(scene,ROOT)
    expanded['objects']['table']['placement']['xy'] = [3,4]
    updated = expand_scene_assets(expanded,ROOT)
    assert updated['frames']['pen']['position'][:2] == pytest.approx([3,4.2])


def real_scene():
    if not (Path.home()/'assets/white_table.glb').is_file():
        pytest.skip('user-supplied GLB assets are not installed')
    path = ROOT/'config/scenes/table_penholder.json'
    return _section(read_json(path),path.parent,scene_assets=True)


def test_real_white_panel_geometry_and_scaled_objects():
    scene = real_scene()
    objects = {o['name']:o for o in resolve_scene_objects(scene)[0]}
    table, pen, holder = (objects[key] for key in ('table','pen','penholder'))
    assert table['bounds_world']['min'][2] == pytest.approx(0,abs=1e-7)
    top = table['bounds_world']['max'][2]
    assert top == pytest.approx(.736,abs=1e-7)
    assert pen['bounds_world']['min'][2] == pytest.approx(top+.001)
    assert holder['bounds_world']['min'][2] == pytest.approx(top+.001)
    assert holder['asset']['physics']['collision'] == 'sdf'


def test_usd_authoring_independent_bodies_and_open_vessel_collision(tmp_path):
    Usd = pytest.importorskip('pxr.Usd')
    PhysxSchema = pytest.importorskip('pxr.PhysxSchema')
    from pxr import UsdGeom, UsdPhysics
    from src.simulation_config.isaac_scene_objects import author_scene_object
    stage = Usd.Stage.CreateInMemory()
    objects = resolve_scene_objects(real_scene())[0]
    for item in objects:
        author_scene_object(stage,item,tmp_path)
    table = stage.GetPrimAtPath('/World/SceneObjects/table')
    pen = stage.GetPrimAtPath('/World/SceneObjects/pen')
    holder = stage.GetPrimAtPath('/World/SceneObjects/penholder')
    assert not table.HasAPI(UsdPhysics.RigidBodyAPI)
    assert pen.HasAPI(UsdPhysics.RigidBodyAPI) and holder.HasAPI(UsdPhysics.RigidBodyAPI)
    assert pen.GetParent() == table.GetParent() == holder.GetParent()
    collision = stage.GetPrimAtPath('/World/SceneObjects/penholder/Collision')
    assert UsdPhysics.MeshCollisionAPI(collision).GetApproximationAttr().Get() == 'sdf'
    assert PhysxSchema.PhysxSDFMeshCollisionAPI(collision).GetSdfResolutionAttr().Get() == 256
    # Collision topology is the original vessel, without a generated cap/hull.
    source = next(o for o in objects if o['name']=='penholder')
    count = sum(len(p['faces']) for p in transformed_primitives(source['asset']))
    assert len(UsdGeom.Mesh(collision).GetFaceVertexCountsAttr().Get()) == count
    assert list(tmp_path.glob('*.png'))


def test_reset_restores_pose_and_velocity():
    from src.simulation_config.isaac_scene_objects import reset_scene_objects
    state = np.arange(13,dtype=float).reshape(1,13)
    calls = []
    body = SimpleNamespace(data=SimpleNamespace(default_root_state=SimpleNamespace(clone=lambda:state.copy())),
        write_root_pose_to_sim=lambda value:calls.append(('pose',value.copy())),
        write_root_velocity_to_sim=lambda value:calls.append(('velocity',value.copy())),
        reset=lambda:calls.append(('reset',None)))
    reset_scene_objects({'pen':body},{'dynamic':{'pen':'pen'}})
    assert np.array_equal(calls[0][1],state[:,:7])
    assert np.array_equal(calls[1][1],state[:,7:])
    assert calls[2][0]=='reset'


def test_static_evidence_reads_current_stage_transform(monkeypatch):
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, Gf
    from src.simulation_config.isaac_scene_objects import read_scene_object_states
    stage = Usd.Stage.CreateInMemory()
    table = UsdGeom.Xform.Define(stage,'/World/SceneObjects/table')
    translate = table.AddTranslateOp()
    translate.Set(Gf.Vec3d(1,2,3))
    omni = ModuleType('omni'); usd = ModuleType('omni.usd'); omni.usd = usd
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda:stage)
    monkeypatch.setitem(sys.modules,'omni',omni)
    monkeypatch.setitem(sys.modules,'omni.usd',usd)
    runtime = {'static':{'table':'/World/SceneObjects/table'},'dynamic':{}}
    assert read_scene_object_states({},runtime)['table']['pose_world'][:3] == [1,2,3]
    translate.Set(Gf.Vec3d(4,5,6))
    assert read_scene_object_states({},runtime)['table']['pose_world'][:3] == [4,5,6]
