"""Evaluation launcher composition and lifecycle without ROS or Isaac startup."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('integrated_launcher', ROOT/'astribot_simulation_integrate.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('name', ['cap_pen', 'press_cube', 'penholder', 'table_penholder'])
def test_evaluation_configs_compose_before_runtime_import(name, monkeypatch):
    if 'penholder' in name:
        from tests.unit.test_scene_assets import real_scene
        real_scene()
    original = builtins.__import__
    def guarded(module, *args, **kwargs):
        if module.split('.')[0] in ('gymnasium', 'isaaclab', 'isaacsim', 'rclpy', 'rospy', 'omni', 'pxr'):
            raise AssertionError('unexpected runtime import: ' + module)
        return original(module, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    data = launcher.main(['--config', str(ROOT/f'config/experiments/evaluation/{name}.json'),
                          '--check-config', '--headless', '--no-cameras'])
    assert data['simulator_type'] == 'IsaacLab'
    assert data['mode'] == 'rgb_array' and data['camera_names'] == []
    if 'penholder' in name:
        assert {obj['name'] for obj in data['scene_objects']} == {'table', 'pen', 'penholder'}
        assert data['task_reset']['objects'] == ['pen', 'penholder']
        assert all(Path(obj['asset']['source']['path']).is_file() for obj in data['scene_objects'])


def test_default_evaluation_is_relative_to_script_not_working_directory(tmp_path, monkeypatch):
    from tests.unit.test_scene_assets import real_scene
    real_scene()
    monkeypatch.chdir(tmp_path)
    data = launcher.main(['--check-config'])
    assert len(data['scene_objects']) == 3


def test_placement_experiment_is_rejected():
    with pytest.raises(ValueError, match='requires an evaluation experiment'):
        launcher.main(['--config', str(ROOT/'config/placement_distribution/penholder.json'), '--check-config'])


def test_removed_step_budget_argument_is_rejected():
    with pytest.raises(SystemExit):
        launcher.main(['--max-steps', '2', '--check-config'])


@pytest.mark.parametrize('flags,show_viewer', [([], True), (['--headless'], False),
                                            (['--no-cameras'], False), (['--no-viewer'], False)])
def test_factory_receives_assets_and_writes_evaluation_before_kit_close(tmp_path, monkeypatch, flags, show_viewer):
    from tests.unit.test_scene_assets import real_scene
    real_scene()
    calls, factories = [], []
    def popen(argv, **kwargs):
        calls.append(('viewer', argv))
        return SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append(('terminate',)),
                               wait=lambda **kw: calls.append(('wait',)))
    monkeypatch.setattr('subprocess.Popen', popen)
    monkeypatch.setenv('DISPLAY', ':test')
    class Factory:
        def __init__(self):
            self.astribot_simu_env = None
            factories.append(self)
        def create_simulation_env(self, data):
            calls.append(('create', data))
            env = SimpleNamespace(node=None)
            def close():
                assert (tmp_path/'evaluation.json').is_file()
                assert self.stop_event.is_set()
                if show_viewer:
                    assert calls[-1] == ('wait',)
                calls.append(('close',))
            self.astribot_simu_env = SimpleNamespace(unwrapped=env, close=close)
            self.evaluator.observe(env, reset=True)
            assert not self.evaluator.observe(env)
            assert not self.evaluator.observe(env)
    module = ModuleType('src.astribot_envs.astribot_envs_factory')
    module.AstribotEnvsFactory = Factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('ROS_VERSION', 'test')
    monkeypatch.setattr('signal.signal', lambda *args: None)
    launcher.main(['--output-dir', str(tmp_path), *flags])
    assert calls[-1] == ('close',)
    created = next(call[1] for call in calls if call[0] == 'create')
    assert {obj['name'] for obj in created['scene_objects']} == {'table', 'pen', 'penholder'}
    assert any(call[0] == 'viewer' for call in calls) == show_viewer
    if show_viewer:
        assert calls[0][1][-3:] == ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd']
    if '--no-cameras' not in flags:
        assert len(created['camera_names']) == 3
    assert factories[0].stop_event.is_set()
    report = json.loads((tmp_path/'evaluation.json').read_text())
    assert report['steps'] == 2
    assert report['config']['task']['instruction'] == '拿起笔放进笔筒'
    from src.simulation_config.experiment_config import fingerprint
    assert report['fingerprints']['runtime'] == fingerprint(factories[0].evaluator.sections['runtime'])


def test_owned_viewer_is_stopped_when_environment_creation_fails(monkeypatch):
    calls = []
    class Factory:
        astribot_simu_env = None
        def create_simulation_env(self, data):
            raise RuntimeError('test initialization failure')
    module = ModuleType('src.astribot_envs.astribot_envs_factory')
    module.AstribotEnvsFactory = Factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('ROS_VERSION', 'test')
    monkeypatch.setenv('DISPLAY', ':test')
    monkeypatch.setattr('signal.signal', lambda *args: None)
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs: SimpleNamespace(
        poll=lambda: None, terminate=lambda: calls.append('terminate'),
        wait=lambda **kw: calls.append('wait')))
    with pytest.raises(RuntimeError, match='test initialization failure'):
        launcher.main(['--config', str(ROOT/'config/experiments/evaluation/press_cube.json')])
    assert calls == ['terminate', 'wait']


def test_continuous_launcher_removes_legacy_budget_and_keeps_evaluating(tmp_path, monkeypatch):
    from src.simulation_config.experiment_config import resolve_experiment, fingerprint
    resolved = resolve_experiment(ROOT/'config/experiments/evaluation/press_cube.json')
    resolved['sections']['runtime'] = {'evaluation': {'max_steps': 2}}
    path = tmp_path/'legacy_budget.json'
    path.write_text(json.dumps({'schema_version': 4, 'kind': 'evaluation', **resolved['sections']}))
    class Factory:
        astribot_simu_env = None
        def create_simulation_env(self, data):
            assert self.evaluator.max_steps is None
            assert 'max_steps' not in self.evaluator.sections['runtime']['evaluation']
            assert self.evaluator.fingerprints['runtime'] == fingerprint(self.evaluator.sections['runtime'])
            for _ in range(1001):
                assert not self.evaluator.observe(SimpleNamespace())
            assert not self.stop_event.is_set()
    module = ModuleType('src.astribot_envs.astribot_envs_factory')
    module.AstribotEnvsFactory = Factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('ROS_VERSION', 'test')
    monkeypatch.setattr('signal.signal', lambda *args: None)
    launcher.main(['--config', str(path), '--no-viewer', '--output-dir', str(tmp_path)])
    assert json.loads((tmp_path/'evaluation.json').read_text())['steps'] == 1001
