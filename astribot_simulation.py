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
from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

def main():
    import argparse
    import signal
    import threading
    import os

    parser = argparse.ArgumentParser(description="Astribot Isaac Lab simulation")
    parser.add_argument('--config', default='config/astribot_s1/simulation_isaaclab_param.yaml')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-cameras', action='store_true',
                        help='disable configured camera sensors for state-only replay')
    args = parser.parse_args()
    data = AstribotEnvsFactory.load_yaml_file(args.config)
    if args.headless:
        data['mode'] = 'rgb_array'
    if args.no_cameras:
        data['camera_names'] = []
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
