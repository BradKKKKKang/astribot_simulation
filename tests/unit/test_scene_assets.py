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


@pytest.mark.parametrize('material', [{'metallic': -0.1}, {'roughness': 1.1},
                                     {'roughness': True}, {'roughness': float('nan')}, {'gloss': .5}])
def test_material_override_rejects_invalid_fields_and_values(tmp_path, material):
    asset = profile(tmp_path/'material.glb')
    asset['material'] = material
    with pytest.raises(ValueError, match='asset.material'):
        validate_asset(asset)


def test_material_override_is_written_to_usd_without_changing_geometry(tmp_path):
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdShade
    from src.simulation_config.isaac_scene_objects import author_scene_object
    asset = profile(tmp_path/'material.glb', static=True)
    asset['material'] = {'metallic': 0., 'roughness': .55}
    validate_asset(asset)
    scene = {'objects': {'vessel': {'asset': asset, 'placement': {
        'frame': 'world', 'xy': [0, 0], 'grounding': 'geometry_min'}}}}
    obj = resolve_scene_objects(scene)[0][0]
    stage = Usd.Stage.CreateInMemory()
    author_scene_object(stage, obj, tmp_path)
    shader = UsdShade.Shader(stage.GetPrimAtPath('/World/SceneObjects/vessel/Materials/material_0/Surface'))
    assert shader.GetInput('metallic').Get() == 0.
    assert shader.GetInput('roughness').Get() == pytest.approx(.55)
    assert tuple(shader.GetInput('diffuseColor').Get()) == (1., 1., 1.)


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
    if not all((Path.home()/'assets'/name).is_file() for name in ('red_markpen.glb', 'yellow_penholder.glb')):
        pytest.skip('user-supplied GLB assets are not installed')
    path = ROOT/'config/scenes/penholder.json'
    return _section(read_json(path),path.parent,scene_assets=True)


def test_real_white_panel_geometry_and_scaled_objects():
    scene = real_scene()
    objects = {o['name']:o for o in resolve_scene_objects(scene)[0]}
    table, pen, holder = (objects[key] for key in ('table','pen','penholder'))
    assert table['bounds_world']['min'][2] == pytest.approx(0,abs=1e-7)
    top = table['bounds_world']['max'][2]
    assert top == pytest.approx(.75,abs=1e-7)
    assert np.subtract(table['bounds_world']['max'], table['bounds_world']['min']) == pytest.approx([.7, 1.4, .75])
    assert np.linalg.norm(scene['frames']['table']['position'][:2]) == pytest.approx(.6)
    assert pen['bounds_world']['min'][2] == pytest.approx(top+.001)
    assert holder['bounds_world']['min'][2] == pytest.approx(top+.001)
    assert holder['asset']['physics']['collision'] == 'sdf'


def test_penholder_reset_is_seeded_supported_and_separated():
    from src.simulation_config.experiment_config import resolve_experiment
    from src.simulation_config.task_reset import sample_task_objects
    real_scene()  # Skip when the external pen/holder assets are not installed.
    resolved = resolve_experiment(ROOT/'config/experiments/evaluation/penholder.json')
    scene, task = resolved['sections']['scene'], resolved['sections']['task']
    assert task['reset']['placement']['type'] == 'empirical'
    assert all(len(scene['objects'][name]['placement']['reset_xy_samples']) == 310
               for name in task['reset']['objects'])
    before = copy.deepcopy(scene)
    first = sample_task_objects(scene, task['reset'], np.random.default_rng(42))
    again = sample_task_objects(scene, task['reset'], np.random.default_rng(42))
    assert first == again
    assert scene == before
    positions = []
    for seed in range(30):
        sampled = sample_task_objects(scene, task['reset'], np.random.default_rng(seed))
        assert set(sampled) == {'pen', 'penholder'}
        pen, holder = sampled['pen'], sampled['penholder']
        paired_rows = np.ones(310, dtype=bool)
        for name, item in sampled.items():
            low, high = np.array(item['bounds_world']['min']), np.array(item['bounds_world']['max'])
            assert low[2] == pytest.approx(.751)
            assert np.all(low[:2] >= [.27, -.68])
            assert np.all(high[:2] <= [.93, .68])
            # Independently undo the table's world XY [0.6, 0] and +90 degree yaw.
            x, y = item['pose_world'][:2]
            xy = np.array([y, .6 - x])
            limits = scene['objects'][name]['placement']['reset_xy_bounds']
            assert np.all(xy >= limits['min']) and np.all(xy <= limits['max'])
            observed = scene['objects'][name]['placement']['reset_xy_samples']
            paired_rows &= np.all(np.isclose(observed, xy, atol=1e-9, rtol=0), axis=1)
        assert paired_rows.any()  # Both objects must come from the same episode.
        a, b = pen['bounds_world'], holder['bounds_world']
        assert (np.any(np.array(a['max'])[:2] + .02 <= np.array(b['min'])[:2])
                or np.any(np.array(b['max'])[:2] + .02 <= np.array(a['min'])[:2]))
        positions.append(pen['pose_world'][:2])
    assert len({tuple(x) for x in positions}) > 20


@pytest.mark.parametrize('bounds', [
    {}, {'min': [0, 0], 'max': [1]}, {'min': [0, 0], 'max': [1, float('nan')]},
    {'min': [0, 0], 'max': [1, True]}, {'min': [0, 0], 'max': [1, 0]},
    {'min': [1, 0], 'max': [0, 1]}, {'min': [0, 0], 'max': [1, 1], 'z': .75},
])
def test_reset_xy_bounds_reject_invalid_limits(scene, bounds):
    scene['objects']['pen']['placement']['reset_xy_bounds'] = bounds
    with pytest.raises(ValueError, match='reset_xy_bounds'):
        expand_scene_assets(scene, ROOT)


def test_bounded_reset_requires_limits_and_never_falls_back_to_whole_table(scene):
    from src.simulation_config.task_reset import sample_task_objects, validate_task_reset
    reset = {'objects': ['pen'], 'placement': {'type': 'uniform_in_bounds', 'surface': 'table.tabletop'}}
    with pytest.raises(ValueError, match='requires placement.reset_xy_bounds'):
        validate_task_reset(scene, reset)
    scene['objects']['pen']['placement']['reset_xy_bounds'] = {'min': [2, 2], 'max': [3, 3]}
    with pytest.raises(ValueError, match='no supported placement'):
        sample_task_objects(scene, reset, np.random.default_rng(1))
    reset['placement']['type'] = 'uniform_on_surface'
    with pytest.raises(ValueError, match='requires uniform_in_bounds'):
        validate_task_reset(scene, reset)


def test_bounded_reset_clips_support_without_shrinking_origin_bounds_twice(scene):
    from src.simulation_config.task_reset import sample_task_objects
    reset = {'objects': ['pen'], 'placement': {'type': 'uniform_in_bounds', 'surface': 'table.tabletop'}}
    scene['objects']['pen']['placement']['reset_xy_bounds'] = {'min': [.70, .30], 'max': [.85, .39]}
    calls = []
    def uniform(low, high):
        calls.append((low.copy(), high.copy()))
        return high.copy()
    # The scaled/rotated pen footprint is [-.15, .15] x [-.01, .02].
    # Support permits origins up to [.75, .38], while the requested lower bound stays [.70, .30].
    result = sample_task_objects(scene, reset, SimpleNamespace(uniform=uniform))['pen']
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx([.70, .30])
    assert calls[0][1] == pytest.approx([.75, .38])
    assert result['pose_world'][:2] == pytest.approx([1 - .38, 2 + .75])
    assert result['bounds_world']['min'][2] == pytest.approx(.801)
    assert result['pose_world'][3:] == pytest.approx([.5, -.5, .5, .5])
    # Changing the table height changes only the grounded Z, not the calibrated XY.
    scene['objects']['table']['scale'] = [1, 1, 2]
    taller = sample_task_objects(scene, reset, SimpleNamespace(uniform=uniform))['pen']
    assert taller['pose_world'][:2] == pytest.approx(result['pose_world'][:2])
    assert taller['bounds_world']['min'][2] == pytest.approx(1.601)


def test_bounded_reset_keeps_pair_separation_and_attempt_limit(scene):
    from src.simulation_config.task_reset import sample_task_objects
    scene['objects']['holder'] = copy.deepcopy(scene['objects']['pen'])
    for obj in ('pen', 'holder'):
        scene['objects'][obj]['placement']['reset_xy_bounds'] = {'min': [0, 0], 'max': [.001, .001]}
    reset = {'objects': ['pen', 'holder'], 'placement': {'type': 'uniform_in_bounds',
             'surface': 'table.tabletop', 'min_separation': .02, 'max_attempts': 2}}
    with pytest.raises(ValueError, match='max_attempts'):
        sample_task_objects(scene, reset, np.random.default_rng(1))


@pytest.fixture
def empirical_scene(scene):
    scene['objects']['holder'] = copy.deepcopy(scene['objects']['pen'])
    for name, frequent, rare in [('pen', [-.45, -.2], [.45, -.2]),
                                  ('holder', [.45, .2], [-.45, .2])]:
        scene['objects'][name]['placement'].update(
            reset_xy_bounds={'min': [-.6, -.3], 'max': [.6, .3]},
            reset_xy_samples=[frequent[:] for _ in range(9)] + [rare[:]])
    reset = {'objects': ['pen', 'holder'], 'placement': {'type': 'empirical',
             'surface': 'table.tabletop', 'min_separation': .02, 'max_attempts': 10}}
    return scene, reset


def test_empirical_reset_retains_density_pairing_and_support(empirical_scene):
    from src.simulation_config.task_reset import sample_task_objects
    scene, reset = empirical_scene
    before = copy.deepcopy(scene)
    rng = np.random.default_rng(123)
    frequent_count = 0
    for _ in range(1000):
        sampled = sample_task_objects(scene, reset, rng)
        p, h = sampled['pen'], sampled['holder']
        # Independently invert world XY [1, 2], yaw +90 degrees.
        pen_xy = np.array([p['pose_world'][1] - 2, 1 - p['pose_world'][0]])
        holder_xy = np.array([h['pose_world'][1] - 2, 1 - h['pose_world'][0]])
        assert abs(pen_xy[0]) == pytest.approx(.45)
        assert pen_xy[1] == pytest.approx(-.2)
        assert holder_xy == pytest.approx(-pen_xy)
        assert p['bounds_world']['min'][2] == pytest.approx(.801)
        frequent_count += pen_xy[0] < 0
    # 9 observations in one cluster and 1 in the other must retain their mass.
    assert .85 < frequent_count / 1000 < .95
    assert scene == before


@pytest.mark.parametrize('samples', [None, [], [[0]], [[0, 0, 0]],
                                    [[True, 0]], [[0, float('inf')]], [[.7, 0]], [[0, 0]]])
def test_empirical_reset_rejects_invalid_or_unpaired_samples(empirical_scene, samples):
    from src.simulation_config.task_reset import validate_task_reset
    scene, reset = empirical_scene
    scene['objects']['pen']['placement']['reset_xy_samples'] = samples
    with pytest.raises(ValueError, match='reset_xy_samples|equal sample counts'):
        validate_task_reset(scene, reset)


def test_empirical_reset_redraws_whole_pair_on_overlap(empirical_scene):
    from src.simulation_config.task_reset import sample_task_objects
    scene, reset = empirical_scene
    scene['objects']['pen']['placement']['reset_xy_samples'] = [[0, 0], [-.45, -.2]]
    scene['objects']['holder']['placement']['reset_xy_samples'] = [[0, 0], [.45, .2]]
    rows = iter([0, 1])
    sampled = sample_task_objects(scene, reset, SimpleNamespace(integers=lambda n: next(rows)))
    assert sampled['pen']['pose_world'][:2] == pytest.approx([1.2, 1.55])
    assert sampled['holder']['pose_world'][:2] == pytest.approx([.8, 2.45])
    with pytest.raises(ValueError, match='max_attempts'):
        sample_task_objects(scene, reset, SimpleNamespace(integers=lambda n: 0))


def test_empirical_reset_rejects_unsupported_row_without_clipping(scene):
    from src.simulation_config.task_reset import sample_task_objects
    scene['objects']['pen']['placement'].update(
        reset_xy_bounds={'min': [.7, 0], 'max': [.85, .2]},
        reset_xy_samples=[[.8, 0], [.7, .1]])
    reset = {'objects': ['pen'], 'placement': {'type': 'empirical', 'surface': 'table.tabletop'}}
    rows = iter([0, 1])
    sampled = sample_task_objects(scene, reset, SimpleNamespace(integers=lambda n: next(rows)))
    assert sampled['pen']['pose_world'][:2] == pytest.approx([.9, 2.7])


def test_empirical_samples_cannot_silently_use_uniform_sampling(empirical_scene):
    from src.simulation_config.task_reset import validate_task_reset
    scene, reset = empirical_scene
    reset['placement']['type'] = 'uniform_in_bounds'
    with pytest.raises(ValueError, match='requires empirical'):
        validate_task_reset(scene, reset)


def test_reset_rejects_static_selection_and_impossible_sampling(scene):
    from src.simulation_config.task_reset import sample_task_objects, validate_task_reset
    reset = {'objects': ['table'], 'placement': {'type': 'uniform_on_surface', 'surface': 'table.tabletop'}}
    with pytest.raises(ValueError, match='dynamic'):
        validate_task_reset(scene, reset)
    reset['objects'] = ['pen']
    scene['objects']['obstacle'] = copy.deepcopy(scene['objects']['pen'])
    reset['placement'].update(min_separation=100., max_attempts=2)
    with pytest.raises(ValueError, match='max_attempts'):
        sample_task_objects(scene, reset, np.random.default_rng(1))


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


def test_sampled_reset_writes_selected_pose_and_zero_velocity_only():
    torch = pytest.importorskip('torch')
    from src.simulation_config.isaac_scene_objects import reset_scene_objects
    state = torch.ones((1, 13))
    calls = []
    body = SimpleNamespace(data=SimpleNamespace(default_root_state=state),
        write_root_pose_to_sim=lambda value:calls.append(('pose',value.clone())),
        write_root_velocity_to_sim=lambda value:calls.append(('velocity',value.clone())),
        reset=lambda:calls.append(('reset',None)))
    pose = [.4, .2, .8, 1, 0, 0, 0]
    reset_scene_objects({'pen':body}, {'dynamic':{'pen':'pen','unselected':'absent'}},
                        sampled={'pen':{'pose_world':pose}})
    assert calls[0][1][0].tolist() == pytest.approx(pose)
    assert torch.count_nonzero(calls[1][1]) == 0
    assert calls[2][0] == 'reset'
    assert torch.equal(state, torch.ones((1, 13)))


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
