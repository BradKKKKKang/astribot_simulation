"""Controlled right-hand cloth pinch/lift experiments through managed ROS steps.

Each trial owns a fresh Isaac process in a loopback-only DDS domain. Production
physics settings are preserved. Position interventions translate the flat shirt
in world XY, keeping the recorded robot approach unchanged. Not a policy replay
or an SDK dataset replay entry point.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREFIX_END, CLOSE_END, ANCHOR, LIFT_END, END = 850, 880, 930, 1130, 1230
FPS = 20
VIEWS = {
    'overview': {'eye': [1.35, -1.3, 1.65], 'target': [.35, 0., .85]},
    'closeup': {'eye': [.59, -.47, 1.02], 'target': [.32, -.20, .795]},
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def conditions(reaction_study=False):
    if reaction_study:
        result = [{'name': name, 'reaction_mode': mode, 'closure_percent': 95.5,
                 'shirt_world_xy_m': [-.01, 0.]}
                for name, mode in [('native_reference', 'native'), ('art_shell', 'art_shell'),
                                   ('rigid_shell', 'rigid_shell'),
                                   ('art_shell_repeat2', 'art_shell'),
                                   ('rigid_shell_repeat2', 'rigid_shell')]]
        result.extend({'name': mode + '_gap_matched', 'reaction_mode': mode,
                       'closure_percent': 95.5, 'shirt_world_xy_m': [-.01, 0.],
                       'gap_reference_case': 'art_shell'} for mode in ('art_shell', 'rigid_shell'))
        result.append({**result[-1], 'name': 'rigid_shell_gap_matched_repeat2'})
        return result
    result = [{'name': 'baseline', 'closure_percent': 95.5, 'shirt_world_xy_m': [0., 0.]}]
    for value in (90., 93., 98., 100.):
        result.append({'name': f'closure_{value:g}', 'closure_percent': value,
                       'shirt_world_xy_m': [0., 0.]})
    for axis in (0, 1):
        for sign in (-1, 1):
            offset = [0., 0.]
            offset[axis] = sign * .01
            result.append({'name': f'position_{"xy"[axis]}_{"minus" if sign < 0 else "plus"}10mm',
                           'closure_percent': 95.5, 'shirt_world_xy_m': offset})
    for index in (2, 3):
        result.append({**result[0], 'name': f'baseline_repeat{index}'})
    return result


def prepare(args):
    import numpy as np
    from src.simulation_config.experiment_config import resolve_experiment
    from src.simulation_config.glb_geometry import rotation_matrix
    from src.simulation_config.scene_profile import build_experiment_simulation_config

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source_trace.resolve(strict=True)
    prefix = []
    command_steps, state_steps = [], set()
    for line in source.open():
        row = json.loads(line)
        if row['event'] == 'command' and row['step_id'] <= PREFIX_END:
            prefix.append(row['commands'])
            command_steps.append(row['step_id'])
        if row['event'] == 'state' and row['step_id'] <= PREFIX_END:
            state_steps.add(row['step_id'])
    if command_steps != list(range(1, PREFIX_END+1)) or not set(command_steps) <= state_steps:
        raise ValueError('source trace must contain 850 complete approach commands')
    write_json(out / 'approach.json', prefix)
    base = resolve_experiment(args.config)['sections']
    configs = []
    for condition in conditions(args.reaction_study):
        directory = out / condition['name']
        directory.mkdir()
        config = {'schema_version': 4, 'kind': 'evaluation', **copy.deepcopy(base)}
        # Placement XY is table-local. Convert the requested WORLD translation.
        table_q = config['scene']['objects']['table']['placement']['quaternion_wxyz']
        local = rotation_matrix(table_q).T @ np.array([*condition['shirt_world_xy_m'], 0.])
        placement = config['scene']['objects']['tshirt']['placement']
        placement['xy'] = (np.array(placement['xy']) + local[:2]).tolist()
        config['runtime']['evaluation'].update(episodes=1, max_steps=END, seed=0)
        config['scene']['simulation']['cameras'] = {
            name: {'width': 128, 'height': 96}
            for name in ('head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd')}
        config['runtime']['simulation']['third_person_camera'] = {
            **VIEWS['overview'], 'width': 640, 'height': 480, 'fps': FPS,
            'output_path': str(directory / 'unused.mp4')}
        config['output']['root'] = str(directory / 'evaluation')
        write_json(directory / 'experiment.json', config)
        resolved = resolve_experiment(directory / 'experiment.json')
        built = build_experiment_simulation_config(resolved)
        shirt = next(s for s in built['scene_objects'] if s['name'] == 'tshirt')
        configs.append({**condition, 'shirt_pose_world': shirt['pose_world']})
    write_json(out / 'manifest.json', {
        'source_trace': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'config': str(args.config.resolve()), 'conditions': configs, 'seed': 0,
        'reaction_study': args.reaction_study,
        'reaction_intervention': (
            'Right-finger colliders are internally referenced onto co-located shells. '
            'Each original finger and shell receives half the original mass and inertia. '
            'The paired modes differ only in physics:excludeFromArticulation on their '
            'fixed joints. Both total mass and inertia, collision geometry, material, '
            'controls, placement and timing are preserved. Native reference calibrates '
            'shell introduction. Actual shell poses are used for geometric metrics. '
            'This is an alternative native solver path, not a global PhysX two-way switch.'
            if args.reaction_study else None),
        'dt_s': .01, 'physics_substeps': 4, 'video_fps': FPS, 'views': VIEWS,
        'phases': {'approach_end': PREFIX_END, 'close_end': CLOSE_END, 'closed_hold_end': ANCHOR,
                   'lift_end': LIFT_END, 'lifted_hold_end': END},
        'controls': 'Identical 850-command approach, 0.3 s close, 0.5 s hold, +Z 20 mm/2 s, 1 s hold. '
                    'Feedback FK/IK holds wrist orientation and accounts for torso tracking. '
                    'All cloth, contact, actuator and timing parameters unchanged.',
        'position_definition': 'Translate the shirt in world XY at reset, with the robot approach unchanged. '
                               'A +10 mm shirt translation shifts the gripper-to-shirt vector by -10 mm along the same world axis. '
                               'This includes the changed approach contact history; not an isolated final-contact perturbation.',
        'interpretation': 'Geometric proximity and material tracking are not grip-force measurements or task success.'})
    print('PREPARED', out, len(configs), flush=True)


class FeedbackLift:
    def __init__(self, snapshot, previous, usd_dependencies):
        import numpy as np
        sys.path.append(str(usd_dependencies))
        from pxr import UsdPhysics
        from src.placement_analysis.robot_export import open_robot_stage
        from src.placement_analysis.reference_pose import UsdKinematics
        from src.simulation_config.glb_geometry import rotation_matrix
        stage, robot, names = open_robot_stage(
            ROOT / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1_textured.usda')
        self.model = UsdKinematics(stage, robot.GetPath(), names)
        self.names = previous['astribot_arm_right']['joint_names']
        limits = {p.GetName(): [UsdPhysics.RevoluteJoint(p).GetLowerLimitAttr().Get(),
                               UsdPhysics.RevoluteJoint(p).GetUpperLimitAttr().Get()]
                  for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        self.bounds = np.deg2rad(np.array([limits[n] for n in self.names])).T
        self.prev = np.array(previous['astribot_arm_right']['position'])
        self.reference = self.pose(self.joints(snapshot))
        self.rotation = rotation_matrix(self.reference[3:])
        self.records = []

    @staticmethod
    def joints(snapshot):
        return {n: p for part, v in snapshot['joints'].items()
                if part != 'astribot_chassis' and 'gripper' not in part
                for n, p in zip(v['joint_names'], v['position'])}

    def pose(self, joints):
        import numpy as np
        return np.array(self.model.snapshot(joints, body_names=['astribot_arm_right_link_7'])[
            'body_poses_world'][0])

    def command(self, snapshot, step, command):
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        from src.simulation_config.glb_geometry import rotation_matrix
        q = self.joints(snapshot)
        arm = np.array([q[n] for n in self.names])
        u = min((step - ANCHOR) / (LIFT_END - ANCHOR), 1.)
        target = self.reference[:3] + [0, 0, .02 * u*u*(3-2*u)]

        def residual(x):
            p = self.pose({**q, **dict(zip(self.names, x))})
            rv = Rotation.from_matrix(self.rotation.T @ rotation_matrix(p[3:])).as_rotvec()
            return np.r_[p[:3] - target, .15*rv, 1e-5*(x-arm)]

        solved = least_squares(residual, arm, bounds=self.bounds, xtol=1e-10,
                               ftol=1e-10, gtol=1e-10, max_nfev=40)
        err = np.linalg.norm(residual(solved.x)[:3])
        delta = solved.x - self.prev
        if not solved.success or err >= 1e-5 or np.max(np.abs(delta)) >= .03:
            raise RuntimeError(f'IK failed at {step}: residual={err}, delta={delta}')
        command['astribot_arm_right']['position'] = solved.x.tolist()
        command['astribot_arm_right']['velocity'] = (delta/.01).tolist()
        self.prev = solved.x
        self.records.append({'step': step, 'target_xyz': target.tolist(), 'ik_residual_m': float(err)})
        return command


def server(args):
    """Instrumentation lives only in this owned diagnostic process."""
    import threading
    import numpy as np
    directory = args.output.resolve() / args.case
    import faulthandler
    stack_log = (directory / 'startup-stacks.log').open('w')
    faulthandler.dump_traceback_later(60, repeat=True, file=stack_log)
    if os.environ.get('ROS_DOMAIN_ID') != str(args.domain) or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('diagnostic server requires isolated loopback DDS')
    condition = next(c for c in json.loads((args.output.resolve() / 'manifest.json').read_text())['conditions']
                     if c['name'] == args.case)
    if condition.get('reaction_mode', 'native') != 'native':
        from scripts.diagnostics.tshirt_reaction import install
        install(directory, condition['reaction_mode'])
    from src.evaluation.grasp_diagnostics import GraspDiagnostics
    original_sample = GraspDiagnostics.sample

    def sample(self, reset=False):
        result = original_sample(self, reset=reset)
        self.study_step = 0 if reset else self.study_step + 1
        step = self.study_step
        if step == 0 or step >= PREFIX_END and step % 5 == 0:
            data = self.env.robot.data
            points = self.env.scene_object_keys['cloth']['tshirt']['view'].get_positions().cpu().numpy().reshape(-1, 3)
            poses = data.body_state_w[0, :, :7].cpu().numpy()
            if hasattr(self.env, '_reaction_bridge'):
                poses = self.env._reaction_bridge.particle_body_poses(poses)
            np.savez_compressed(directory / 'particles' / f'{step:06d}.npz', points=points,
                body_names=np.array(self.env.robot.body_names), body_poses=poses)
        if step >= PREFIX_END:
            result['incoming_joint_wrench'] = self.env.robot.root_physx_view.get_link_incoming_joint_force()[0].cpu().tolist()
        return result

    GraspDiagnostics.sample = sample
    from src.astribot_envs import astribot_isaaclab_env as env_module
    AstribotIsaacLabEnv = env_module.AstribotIsaacLabEnv
    original_setup = AstribotIsaacLabEnv.setup_simulation_and_scene

    def setup(self):
        # Author both cameras before the first render. Moving one render
        # product between views is stale in this installed Hydra pipeline.
        from isaaclab.sensors import CameraCfg
        from scipy.spatial.transform import Rotation
        original_scene = env_module.InteractiveScene

        def scene_with_closeup(cfg):
            view = VIEWS['closeup']
            z = np.array(view['eye']) - view['target']
            z /= np.linalg.norm(z)
            x = np.cross([0., 0., 1.], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            q = Rotation.from_matrix(np.stack([x,y,z], axis=1)).as_quat()
            cfg.study_closeup = CameraCfg(
                prim_path='{ENV_REGEX_NS}/study_closeup',
                offset=CameraCfg.OffsetCfg(pos=tuple(view['eye']), rot=tuple(q[[3,0,1,2]]), convention='opengl'),
                spawn=env_module.sim_utils.PinholeCameraCfg(focal_length=24., clipping_range=(.01,30.)),
                data_types=['rgb'], width=640, height=480)
            return original_scene(cfg)

        env_module.InteractiveScene = scene_with_closeup
        try:
            return original_setup(self)
        finally:
            env_module.InteractiveScene = original_scene

    AstribotIsaacLabEnv.setup_simulation_and_scene = setup
    original_snapshot = AstribotIsaacLabEnv.managed_snapshot

    def snapshot(self, *, capture_images):
        faulthandler.cancel_dump_traceback_later()
        value = original_snapshot(self, capture_images=capture_images)
        if capture_images:
            import cv2
            diagnostics = getattr(self, '_grasp_diagnostics', None)
            capture_step = getattr(diagnostics, 'study_step', -1) + 1
            rendered = []
            # Both views are rendered at the SAME paused physics boundary. The
            # original snapshot already refreshed the live cloth visual mesh.
            for name, sensor in [('overview','third_person_camera'), ('closeup','study_closeup')]:
                camera = self.scene[sensor]
                camera.update(0., force_recompute=True)
                rgb = camera.data.output['rgb'][0].cpu().numpy()
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError('diagnostic camera JPEG failed')
                (directory / 'frames' / f'{capture_step:06d}-{name}.jpg').write_bytes(encoded.tobytes())
                rendered.append(rgb.copy())
            if np.abs(rendered[0].astype(float)-rendered[1]).mean() < 5.:
                raise RuntimeError('diagnostic views are not distinct; refuse misleading closeup video')
        return value

    AstribotIsaacLabEnv.managed_snapshot = snapshot
    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory
    original_loop = AstribotEnvsFactory.managed_env_loop

    def loop(self):
        def watch():
            while not self.stop_event.wait(.2):
                if (directory / 'stop').exists():
                    self.stop_event.set()
                    return
        threading.Thread(target=watch, daemon=True).start()
        return original_loop(self)

    AstribotEnvsFactory.managed_env_loop = loop
    import astribot_simulation_integrate
    astribot_simulation_integrate.main(['--config', str(directory / 'experiment.json'),
        '--managed-evaluation', '--headless', '--no-viewer'])


def run_case(args):
    import cv2
    import numpy as np
    out = args.output.resolve()
    directory = out / args.case
    if (directory / 'run_status.json').exists():
        raise FileExistsError('trial already attempted; use a fresh output or case directory')
    condition = next(c for c in json.loads((out / 'manifest.json').read_text())['conditions'] if c['name'] == args.case)
    (directory / 'particles').mkdir()
    (directory / 'frames').mkdir()
    sys.path.insert(0, str(args.atlas / 'src/eval'))
    from astribot_eval.ros2 import dds_environment, dds_profile
    settings = {'interface_ipv4': '127.0.0.1', 'domain_id': args.domain, 'rmw_implementation': 'rmw_fastrtps_cpp'}
    profile = directory / 'loopback.xml'
    profile.write_text(dds_profile(settings))
    environment = dds_environment(os.environ, settings, profile)
    os.environ.clear()
    os.environ.update(environment)
    os.environ['ROS_LOG_DIR'] = str(directory / 'roslog')
    import rclpy
    rclpy.init()
    node = rclpy.create_node('tshirt_sweep_isolation_probe')
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        others = [n for n in node.get_node_names() if n != node.get_name()]
        if others:
            raise RuntimeError(f'DDS domain occupied: {others}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print('ISOLATION_OK', settings, flush=True)
    from astribot_eval.simulation.rpc import SimulationRPC, validate_description
    prefix = json.loads((out / 'approach.json').read_text())
    base = copy.deepcopy(prefix[-1])
    for v in base.values():
        v['velocity'] = [0.] * len(v['position'])
    start_closure = base['astribot_gripper_right']['position'][0]
    status = {'status': 'starting', 'condition': condition, 'isolation': settings}
    process = rpc = identity = writer = None
    gap_feedback = None
    step = 0
    frames = 0
    deadline = time.monotonic() + args.timeout

    def interrupted(signum, frame):
        raise RuntimeError(f'Owned experiment interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def video(snapshot, phase):
        nonlocal frames
        panels = []
        for name in ('overview', 'closeup'):
            frame = cv2.imread(str(directory / 'frames' / f'{step:06d}-{name}.jpg'))
            if frame is None or frame.shape != (480, 640, 3):
                raise RuntimeError(f'invalid camera frame {name}')
            panels.append(frame)
        frame = np.hstack(panels)
        header = np.zeros((64, 1280, 3), dtype=np.uint8)
        closure_label = f'closure target={condition["closure_percent"]:g}%'
        if gap_feedback is not None and gap_feedback.records:
            closure_label = f'gap feedback cmd={gap_feedback.records[-1]["command_percent"]:.2f}%'
        label = f'{args.case} | {phase} | t={step*.01:.2f}s | {closure_label}'
        cv2.putText(header, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(header, f'Shirt offset world XY (mm): {np.array(condition["shirt_world_xy_m"])*1000} | Actual simulation, 1x',
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, (210,210,210), 1, cv2.LINE_AA)
        writer.write(np.vstack((header, frame)))
        if step in (0, PREFIX_END, ANCHOR, LIFT_END, END):
            cv2.imwrite(str(directory / f'frame-{step:06d}.jpg'), np.vstack((header, frame)))
        frames += 1
        with (directory / 'video.frames.jsonl').open('a') as stream:
            stream.write(json.dumps({'frame': frames-1, 'step': step, 'simulation_time_s': step*.01}) + '\n')

    try:
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'server',
                '--output', str(out), '--case', args.case, '--domain', str(args.domain)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['server_pid'] = process.pid
        write_json(directory / 'run_status.json', status)
        rpc = SimulationRPC(timeout_s=240, cancelled=lambda: process.poll() is not None or time.monotonic() > deadline)
        desc = validate_description(rpc.call('describe'))
        from src.simulation_config.experiment_config import resolve_experiment, fingerprint
        expected = fingerprint(resolve_experiment(directory / 'experiment.json')['sections'])
        if desc['config_fingerprint'] != expected:
            raise RuntimeError('live simulator configuration differs from this trial')
        identity = {'run_id': f'tshirt-sweep-{args.case}-{os.getpid()}', 'episode_id': 1,
                    'config_fingerprint': desc['config_fingerprint']}
        status.update(status='running', identity=identity)
        writer = cv2.VideoWriter(str(directory / 'interaction.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 544))
        if not writer.isOpened():
            raise RuntimeError('video encoder unavailable')
        snapshot = rpc.call('reset', **identity, step_id=0, payload={'seed': 0})['snapshot']
        status['trace'] = snapshot['diagnostic_artifacts']['steps']
        if 'gap_reference_case' in condition:
            from scripts.diagnostics.tshirt_reaction import GapFeedback
            reference = json.loads((out/condition['gap_reference_case']/'run_status.json').read_text())
            if reference['status'] != 'completed' or reference['server_exit_code'] != 0:
                raise RuntimeError('Gap-reference run did not complete')
            gap_feedback = GapFeedback(Path(reference['trace']), Path(status['trace']))
        write_json(directory / 'run_status.json', status)
        video(snapshot, 'reset')
        rpc.timeout_s = 90
        for step in range(1, END + 1):
            if step <= PREFIX_END:
                command = prefix[step-1]
                phase = 'identical approach'
            else:
                command = copy.deepcopy(base)
                u = min((step-PREFIX_END)/(CLOSE_END-PREFIX_END), 1.)
                command['astribot_gripper_right']['position'] = [start_closure + (condition['closure_percent']-start_closure)*u*u*(3-2*u)]
                if gap_feedback is not None:
                    command['astribot_gripper_right']['position'][0] = gap_feedback.command(
                        step, command['astribot_gripper_right']['position'][0])
                phase = 'close' if step <= CLOSE_END else 'hold closed'
                if step == ANCHOR + 1:
                    lift = FeedbackLift(snapshot, previous, args.usd_dependencies)
                if step > ANCHOR:
                    command = lift.command(snapshot, step, command)
                    phase = 'lift 20 mm' if step <= LIFT_END else 'hold lifted'
            snapshot = rpc.call('step', **identity, step_id=step,
                payload={'dt': .01, 'commands': command, 'capture_images': step % 5 == 0})['snapshot']
            previous = command
            if step % 5 == 0:
                video(snapshot, phase)
            if step % 100 == 0:
                print('PROGRESS', args.case, step, phase, flush=True)
        write_json(directory / 'ik.json', lift.records)
        if gap_feedback is not None:
            write_json(directory / 'gap_feedback.json', gap_feedback.records)
        if snapshot['terminal'] != {'done': True, 'reason': 'timeout'}:
            raise RuntimeError('trial did not reach the expected fixed-step boundary')
        status.update(status='completed', completed_steps=step, video_frames=frames)
    except BaseException:
        status.update(status='error', completed_steps=step, error=traceback.format_exc())
        raise
    finally:
        if gap_feedback is not None:
            gap_feedback.close()
        if writer is not None:
            writer.release()
        if rpc is not None and identity is not None and process.poll() is None:
            try:
                rpc.call('stop', **identity, step_id=step, payload={'reason': 'aborted' if status['status']=='completed' else 'error'})
            except Exception as error:
                status['stop_error'] = str(error)
        if rpc is not None:
            rpc.close()
        (directory / 'stop').write_text('stop owned trial\n')
        if process is not None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            status['server_exit_code'] = process.returncode
        write_json(directory / 'run_status.json', status)
    print('TRIAL_COMPLETE', args.case, frames, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run-case', 'server'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--domain', type=int, default=203)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--atlas', type=Path, default=ROOT.parent/'astribot_atlas')
    parser.add_argument('--usd-dependencies', type=Path, default=Path('/tmp/astribot-identification-deps'))
    parser.add_argument('--config', type=Path, default=ROOT/'config/experiments/evaluation/fold_tshirt_fix.yaml')
    parser.add_argument('--source-trace', type=Path)
    parser.add_argument('--reaction-study', action='store_true',
                        help='fixed best geometry; native reference and paired collision-shell reaction experiment')
    args = parser.parse_args()
    if args.mode == 'prepare' and args.source_trace is None:
        parser.error('prepare requires --source-trace')
    if args.mode != 'prepare' and args.case is None:
        parser.error('run-case/server requires --case')
    {'prepare': prepare, 'run-case': run_case, 'server': server}[args.mode](args)


if __name__ == '__main__':
    main()
