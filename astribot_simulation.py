#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

import sys

def main(argv=None):
    import argparse
    import signal
    import threading
    import os

    parser = argparse.ArgumentParser(description="Astribot Isaac Lab simulation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--config', default=None, help='existing simulator YAML configuration')
    selection.add_argument('--experiment', help='compose a robot replay environment or evaluation scene from an experiment')
    parser.add_argument('--check-config', action='store_true',
                        help='validate configuration on CPU without importing ROS or Isaac')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-cameras', action='store_true',
                        help='disable configured camera sensors for state-only replay')
    args = parser.parse_args(argv)
    if args.experiment:
        from src.simulation_config.experiment_config import resolve_experiment
        from src.simulation_config.scene_profile import build_experiment_simulation_config
        data = build_experiment_simulation_config(resolve_experiment(args.experiment, require_ready=False))
    else:
        # Keep the existing YAML behavior while keeping --help/check-config free
        # of factory, ROS, gymnasium and simulator imports.
        import yaml
        from pathlib import Path
        config = args.config or 'config/astribot_s1/simulation_isaaclab_param.yaml'
        data = yaml.safe_load(Path(config).expanduser().read_text())
        if not isinstance(data, dict):
            raise ValueError('simulation configuration must be an object')
        root = os.getenv('ASTRIBOT_SIMU_ROOT')
        if root is None:
            raise ValueError('ASTRIBOT_SIMU_ROOT must be set for legacy --config YAML')
        data['model_path'] = root + data['model_path']
    if args.headless:
        data['mode'] = 'rgb_array'
    if args.no_cameras:
        data['camera_names'] = []
    if args.check_config:
        print('Simulation configuration resolved; no ROS, SDK or simulator imported.')
        return data
    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory
    factory = AstribotEnvsFactory()
    factory.stop_event = threading.Event()
    def install_stop_handlers():
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: factory.stop_event.set())
    factory.install_stop_handlers = install_stop_handlers
    install_stop_handlers()
    try:
        simulation_thread = factory.create_simulation_env(data)
        if simulation_thread is not None:
            simulation_thread.join()
    finally:
        factory.stop_event.set()
        errors = []
        def cleanup(action):
            try:
                action()
            except Exception as error:
                errors.append(str(error))
        if hasattr(factory, 'executor'):
            def stop_executor():
                if not factory.executor.shutdown(timeout_sec=3.0):
                    raise RuntimeError('Simulation ROS executor shutdown timed out')
            cleanup(stop_executor)
        if hasattr(factory, 'spin_thread'):
            cleanup(lambda: factory.spin_thread.join(3.0))
            if factory.spin_thread.is_alive():
                errors.append('Simulation ROS spin thread did not stop')
        if factory.astribot_simu_env is not None:
            env = factory.astribot_simu_env.unwrapped
            cleanup(factory.astribot_simu_env.close)
            node = getattr(env, 'node', None)
            if node is not None and os.getenv('ROS_VERSION') == '2':
                cleanup(node.destroy_node)
        if os.getenv('ROS_VERSION') == '2':
            import rclpy
            if rclpy.ok():
                cleanup(rclpy.shutdown)
        if errors:
            raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
