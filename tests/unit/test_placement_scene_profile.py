"""Scene composition, calibrated coordinates and launcher/config boundaries on CPU."""
import ast
import builtins
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.simulation_config.isaac_scene_config import normalize_scene_simulation, spawn_scene_environment
from src.simulation_config.experiment_config import resolve_experiment
from src.simulation_config.scene_profile import build_experiment_simulation_config
from src.placement_analysis.task_config import DEFAULT_DETECTOR_CONFIG
from src.evaluation.task_evaluation import evaluate_success

ROOT = Path(__file__).resolve().parents[2]


def experiment(tmp_path, change=None):
    resolved = resolve_experiment(ROOT / 'config/experiments/evaluation/press_cube.json')
    if change:
        change(resolved['sections'])
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps({'schema_version': 4, 'kind': 'evaluation', **resolved['sections']}))
    return resolve_experiment(path), path


def test_scene_changes_launch_config_without_placement_dependency(tmp_path):
    before, _ = experiment(tmp_path)
    changed, _ = experiment(tmp_path, lambda sections: sections['scene'].update(simulation={
        'ground': {'enabled': False}, 'light': {'intensity': 1500., 'color': [.8,.7,.6]},
        'cameras': {'head_rgbd': {'width': 320, 'height': 240}, 'left_wrist_rgbd': {'enabled': False}}}))
    config = build_experiment_simulation_config(changed)
    assert config['scene_simulation']['ground']['enabled'] is False
    assert config['scene_simulation']['light']['intensity'] == 1500
    assert config['camera_names'] == ['head_rgbd', 'right_wrist_rgbd']
    assert config['camera_resolutions']['head_rgbd'] == {'width':320, 'height':240}
    assert changed['fingerprints']['scene_simulation'] != before['fingerprints']['scene_simulation']
    assert 'candidates' not in changed['sections']


def test_nested_calibrated_frames_are_consumed_by_region_evaluation(tmp_path):
    def configure(sections):
        sections['scene']['frames'] = {
            'table': {'parent':'base','position':[1,0,0], 'quaternion_wxyz':[1,0,0,0]},
            'target': {'parent':'table','position':[0,2,0], 'quaternion_wxyz':[1,0,0,0]}}
        sections['scene']['regions'] = {'destination': {'frame':'target','min':[-.1,-.1,-.1],'max':[.1,.1,.1]}}
        sections['task']['events'] = {'press':{'event_type':'press'}}
        sections['task']['stages'] = {}
        sections['task']['success'] = {'predicates':[{'id':'region','kind':'in_region','event':'press','region':'destination'}]}
    resolved, _ = experiment(tmp_path, configure)
    sections = resolved['sections']
    event = {'event_id':'event', 'event_type':'press', 'tcp_pose_base':[1,2,0,1,0,0,0]}
    result = evaluate_success([event],sections['scene'],sections['task'],sections['task']['success'],{'events_complete':True})
    assert result['status'] == 'passed'
    assert result['predicates'][0]['evidence']['positions'] == [[0.,0.,0.]]
    event['tcp_pose_base'][0] = 2
    assert evaluate_success([event],sections['scene'],sections['task'],sections['task']['success'],{'events_complete':True})['status'] == 'failed'
    original = resolve_experiment(ROOT / 'config/experiments/evaluation/press_cube.json')
    assert resolved['fingerprints']['scene_simulation'] == original['fingerprints']['scene_simulation']
    assert 'extraction' not in resolved['fingerprints']


@pytest.mark.parametrize('scene', [
    {'frames':{'a':{'parent':'b'},'b':{'parent':'a'}}},
    {'frames':{'a':{'parent':'missing'}}},
    {'frames':{'a':{'parent':'world','quaternion_wxyz':[0,0,0,0]}}},
    {'simulation':{'light':{'brightness':100}}},
    {'simulation':{'cameras':{'unknown':{}}}},
    {'simulation':{'cameras':{'head_rgbd':{'body':'../escape'}}}},
    {'simulation':{'cameras':{'head_rgbd':{'width':0}}}},
])
def test_invalid_scene_rejected_before_launch(tmp_path, scene):
    with pytest.raises(ValueError):
        experiment(tmp_path, lambda sections: sections['scene'].update(scene))


def test_background_and_supported_robot_root(tmp_path):
    usd = tmp_path/'background.usda'; usd.write_text('#usda 1.0\n')
    resolved, _ = experiment(tmp_path, lambda sections: sections['scene'].update(simulation={
        'background':{'usd_path':str(usd),'position':[1,2,3]}}))
    config = build_experiment_simulation_config(resolved)
    assert config['scene_simulation']['background']['usd_path'] == str(usd)
    assert config['experiment_provenance']['scene_simulation']['background']['usd_sha256']
    resolved['sections']['robot']['root_path'] = '/World/OtherRobot'
    with pytest.raises(ValueError, match='robot root'):
        build_experiment_simulation_config(resolved)


def fake_sim_utils(calls):
    def constructor(kind):
        def build(**kwargs):
            config = SimpleNamespace(**kwargs)
            config.func = lambda path, cfg, **transform: calls.append((kind,path,kwargs,transform))
            return config
        return build
    return SimpleNamespace(**{name:constructor(name) for name in ('GroundPlaneCfg','DomeLightCfg','UsdFileCfg')})


def test_runtime_spawn_adapter_keeps_defaults_and_consumes_overrides():
    calls=[]
    spawn_scene_environment(fake_sim_utils(calls),{})
    assert calls == [('GroundPlaneCfg','/World/ground',{},{}),
                     ('DomeLightCfg','/World/light',{'intensity':3000.,'color':(.75,.75,.75)}, {})]
    calls=[]
    spawn_scene_environment(fake_sim_utils(calls),{'ground':{'enabled':False},'light':{'enabled':False},
        'background':{'usd_path':'/scene.usda','position':[1,2,3]}})
    assert calls == [('UsdFileCfg','/World/scene_background',{'usd_path':'/scene.usda'},
                      {'translation':(1,2,3),'orientation':(1.,0.,0.,0.)})]


def test_backend_scene_method_consumes_config_without_importing_isaac(monkeypatch):
    """Execute the actual method AST with configuration-class fakes only."""
    import sys
    from src.simulation_config.isaac_scene_config import camera_prim_path
    tree=ast.parse((ROOT/'src/astribot_envs/astribot_isaaclab_env.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='AstribotIsaacLabEnv')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='setup_simulation_and_scene')
    calls=[]
    def cfg(**kwargs): return SimpleNamespace(**kwargs)
    def interactive(config):
        calls.append(('scene',config))
        return {'robot':object()}
    sensors=ModuleType('isaaclab.sensors');sensors.CameraCfg=cfg
    torch=ModuleType('torch');torch.cuda=SimpleNamespace(is_available=lambda:False)
    monkeypatch.setitem(sys.modules,'isaaclab.sensors',sensors)
    monkeypatch.setitem(sys.modules,'torch',torch)
    namespace={'SimulationCfg':cfg,'SimulationContext':lambda config:SimpleNamespace(reset=lambda:None),
               'sim_utils':fake_sim_utils(calls),'InteractiveSceneCfg':cfg,'InteractiveScene':interactive,
               'astribot_simu_log':lambda *args:None,'spawn_scene_environment':spawn_scene_environment,
               'camera_prim_path':camera_prim_path}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<scene-method>','exec'),namespace)
    env=SimpleNamespace(physics_device='cpu',dt=.01,decimation=1,use_fabric=True,
        robot_placement_declaration=None, resolved_robot_placement=None, scene_object_specs=[],
        _restore_initial_state=lambda:calls.append(('restore',)),
        scene_simulation=normalize_scene_simulation({'ground':{'enabled':False},'light':{'intensity':123}}),
        _create_robot_config=lambda:object(),camera_names=['head_rgbd'],model_path='/robot.usda',
        camera_bodies={'head_rgbd':'custom_head'},camera_resolutions={'head_rgbd':(320,240)},
        robot_root_path='/World/astribot_whole_body',third_person_camera_cfg=None)
    namespace['setup_simulation_and_scene'](env)
    scene=next(item[1] for item in calls if item[0]=='scene')
    assert scene.head_rgbd.prim_path.endswith('/custom_head/head_rgbd')
    assert scene.head_rgbd.width == 320 and scene.head_rgbd.height == 240
    assert scene.head_rgbd.spawn is None
    assert not any(item[0]=='GroundPlaneCfg' for item in calls)
    assert ('restore',) in calls


def test_entrypoint_help_and_check_config_import_no_runtime(tmp_path, monkeypatch):
    resolved,path=experiment(tmp_path)
    spec=importlib.util.spec_from_file_location('_simulation_entry',ROOT/'astribot_simulation.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('rclpy','rospy','isaaclab','astribot_sdk','src.placement_analysis')) or name=='src.astribot_envs.astribot_envs_factory':
            raise AssertionError('runtime import during config validation: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(SystemExit) as stop:
        entry.main(['--help'])
    assert stop.value.code == 0
    config=entry.main(['--experiment',str(path),'--check-config','--headless','--no-cameras'])
    assert config['mode']=='rgb_array' and config['camera_names']==[]
    assert config['experiment_provenance']['robot_control']


def test_scene_frame_rotation_and_missing_pose_evidence():
    from src.simulation_config.scene_profile import frame_coordinates
    import math
    scene={'frames':{'turned':{'parent':'world','position':[1,0,0],
                              'quaternion_wxyz':[math.sqrt(.5),0,0,math.sqrt(.5)]}}}
    point=frame_coordinates({'tcp_pose_world':[1,1,0,1,0,0,0]},scene,'turned')
    assert point == pytest.approx([1,0,0])
    assert frame_coordinates({},scene,'turned') is None
