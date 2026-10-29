#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""Launch an evaluation experiment with its robot, scene assets and task."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'config/experiments/evaluation/penholder.json'


def main(argv=None):
    import argparse
    import signal
    import threading
    import os
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Astribot integrated evaluation simulation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--config', default=None,
                           help='evaluation JSON path (default: config/experiments/evaluation/penholder.json)')
    selection.add_argument('--experiment', help='alias for --config')
    parser.add_argument('--check-config', action='store_true',
                        help='validate configuration on CPU without importing ROS or Isaac')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--managed-evaluation', action='store_true',
                        help='pause physics until Atlas acquires an evaluation or Replay lease')
    parser.add_argument('--no-cameras', action='store_true',
                        help='disable configured camera sensors for state-only evaluation')
    parser.add_argument('--no-viewer', action='store_true',
                        help='disable camera preview windows while keeping camera sensors enabled')
    parser.add_argument('--output-dir', type=Path,
                        help='override the directory containing evaluation.json')
    parser.add_argument('--initial-pose-parquet', type=Path,
                        help='override task initial pose with measured joints from a LeRobot parquet')
    parser.add_argument('--initial-pose-frame', type=int, default=None,
                        help='frame_index to import with --initial-pose-parquet (default: 0)')
    args = parser.parse_args(argv)
    if args.initial_pose_frame is not None and args.initial_pose_parquet is None:
        parser.error('--initial-pose-frame requires --initial-pose-parquet')
    # Resolve/validate the full scene before importing ROS or starting Isaac.
    from src.simulation_config.experiment_config import resolve_experiment, fingerprint
    from src.simulation_config.scene_profile import build_experiment_simulation_config
    config = Path(args.config or args.experiment or DEFAULT_CONFIG).expanduser().resolve(strict=True)
    resolved = resolve_experiment(config)
    if resolved['kind'] != 'evaluation':
        raise ValueError('astribot_simulation_integrate.py requires an evaluation experiment')
    sections = resolved['sections']
    if args.initial_pose_parquet is not None:
        from src.simulation_config.initial_pose import load_initial_pose
        from src.simulation_config.robot_profile import RobotProfile
        sections['task']['initial_pose'] = load_initial_pose(
            args.initial_pose_parquet, 0 if args.initial_pose_frame is None else args.initial_pose_frame,
            RobotProfile(sections['robot']))
        resolved['fingerprints']['task'] = fingerprint(sections['task'])
    # This interactive entrypoint stays alive until the user closes/stops it.
    # Ignore historical evaluation budgets, including those in external configs.
    evaluation = sections.get('runtime', {}).get('evaluation', {})
    if not args.managed_evaluation and 'max_steps' in evaluation:
        evaluation.pop('max_steps')
        resolved['fingerprints']['runtime'] = fingerprint(sections['runtime'])
    if args.output_dir is not None:
        sections['output']['root'] = str(args.output_dir.expanduser().resolve())
        resolved['fingerprints']['output'] = fingerprint(sections['output'])
    data = build_experiment_simulation_config(resolved)
    if args.headless:
        data['mode'] = 'rgb_array'
    if args.no_cameras:
        data['camera_names'] = []
        data.pop('third_person_camera', None)
    if args.managed_evaluation:
        if args.initial_pose_parquet is not None:
            raise ValueError('managed evaluation initial pose must come from the task configuration')
        from src.evaluation.managed import validate_managed_config
        validate_managed_config(resolved, data)
    objects = ', '.join(obj['name'] for obj in data['scene_objects']) or '(none configured)'
    cameras = ', '.join(data['camera_names']) or '(disabled)'
    print(f'Evaluation config: {config}\nScene assets: {objects}\nCameras: {cameras}', flush=True)
    initial_pose = sections['task'].get('initial_pose')
    if initial_pose is not None:
        print('Initial robot pose: task.initial_pose (restored on reset)', flush=True)
        if 'source' in initial_pose:
            source = initial_pose['source']
            print(f"  Source: {source['parquet']}, frame_index={source['frame_index']}, {source['column']}", flush=True)
        for part, values in initial_pose['joint_targets_rad'].items():
            print(f"  {part} [rad]: " + ', '.join(f'{value:.6f}' for value in values), flush=True)
        print(f"  Grippers [%]: {initial_pose['gripper_percent']}; base: experiment robot_placement", flush=True)
    print(('Managed simulation: waiting for Atlas control.' if args.managed_evaluation else
           'Continuous simulation: close the GUI or press Ctrl+C to stop.'), flush=True)
    if args.check_config:
        print('Simulation configuration resolved; no ROS, SDK or simulator imported.')
        return data
    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory
    factory = AstribotEnvsFactory()
    factory.stop_event = threading.Event()
    from src.evaluation.online import OnlineEvaluator
    if args.managed_evaluation:
        if os.getenv('ROS_VERSION') != '2':
            raise ValueError('managed evaluation requires ROS_VERSION=2')
        from src.evaluation.managed import ManagedEvaluation
        factory.managed_controller = ManagedEvaluation(resolved)
    else:
        factory.evaluator = OnlineEvaluator(resolved)
    def install_stop_handlers():
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: factory.stop_event.set())
    factory.install_stop_handlers = install_stop_handlers
    install_stop_handlers()
    viewer = None
    try:
        if data['mode'] == 'human' and data['camera_names'] and not args.no_viewer:
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                raise RuntimeError('camera previews require a graphical desktop; use --headless or --no-viewer')
            viewer = subprocess.Popen([sys.executable, '-u', str(ROOT/'tests/unit/test_camera_viewer.py'),
                                       '--cameras', *data['camera_names']], start_new_session=True)
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
        if viewer is not None and viewer.poll() is None:
            def stop_viewer():
                viewer.terminate()
                try:
                    viewer.wait(timeout=8.)
                except subprocess.TimeoutExpired:
                    viewer.kill()
                    viewer.wait(timeout=3.)
            cleanup(stop_viewer)
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
            node = getattr(env, 'node', None)
            if node is not None and os.getenv('ROS_VERSION') == '2':
                cleanup(node.destroy_node)
        if os.getenv('ROS_VERSION') == '2':
            import rclpy
            if rclpy.ok():
                cleanup(rclpy.shutdown)
        evaluator = getattr(factory, 'evaluator', None)
        if evaluator is not None and evaluator.result is not None:
            def save_report():
                report_path = evaluator.write_report()
                print('Evaluation: ' + evaluator.result['status']
                      + (f'; report: {report_path}' if report_path is not None else ''), flush=True)
            cleanup(save_report)
        # Kit can terminate the process in close(); finish reporting and ROS first.
        if errors:
            print('Evaluation cleanup errors: ' + '; '.join(errors), flush=True)
        if factory.astribot_simu_env is not None:
            cleanup(factory.astribot_simu_env.close)
        if errors:
            raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
