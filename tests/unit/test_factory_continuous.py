"""Exercise the real factory and Gym wrappers without starting Isaac or ROS."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import pytest


@pytest.mark.parametrize('stop_kind', ['window', 'signal', 'terminated'])
def test_factory_runs_past_registered_limit_and_still_stops(monkeypatch, stop_kind):
    monkeypatch.setenv('ROS_VERSION', 'test')
    monkeypatch.setenv('ASTRIBOT_ISAAC_CONTROL_HZ', '0')
    source = Path(__file__).resolve().parents[2] / 'src/astribot_envs/astribot_envs_factory.py'
    spec = importlib.util.spec_from_file_location('continuous_factory', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = module.AstribotEnvsFactory()
    observations = []

    def observe(env, *, reset=False):
        observations.append(env.steps)
        return False

    factory.evaluator = SimpleNamespace(observe=observe)

    class FakeIsaacEnv(gym.Env):
        def __init__(self, param):
            self.steps = 0
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Discrete(1)
            window_close_step = 1001 if stop_kind == 'window' else 1002
            self.simulation_app = SimpleNamespace(is_running=lambda: self.steps < window_close_step)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return 0, {}

        def step(self, action):
            self.steps += 1
            if stop_kind == 'signal' and self.steps == 1001:
                factory.stop_event.set()
            terminated = stop_kind == 'terminated' and self.steps == 12
            return 0, 0.0, terminated, False, {}

    backend = ModuleType('src.astribot_envs.astribot_isaaclab_env')
    backend.AstribotIsaacLabEnv = FakeIsaacEnv
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    logger = ModuleType('src.simu_utils.simu_common_tools')
    logger.astribot_simu_log = lambda message: None
    monkeypatch.setitem(sys.modules, logger.__name__, logger)

    # Keep the actual registration's 300-step budget to reproduce the regression.
    assert gym.spec('astribot_envs/AstribotIsaacLabEnv-v0').max_episode_steps == 300
    try:
        assert factory.create_simulation_env({'simulator_type': 'IsaacLab'}) is None
        expected_steps = 12 if stop_kind == 'terminated' else 1001
        assert observations == list(range(expected_steps + 1))
        assert factory.astribot_simu_env.spec.max_episode_steps is None
        assert factory.stop_event.is_set() == (stop_kind != 'window')
    finally:
        if factory.astribot_simu_env is not None:
            factory.astribot_simu_env.close()
