#!/usr/bin/env python3
"""Replay complete resampled command frames through Isaac Lab env.step at 100 Hz.

This is a same-process physics baseline, not an SDK/ROS transport test. ROS2
interfaces exist on an explicitly isolated local domain, without an executor.
State is read directly after each physics step and retains its physics time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

PERIOD_S = 0.01
PART_CONFIGS = [("astribot_torso", 4), ("astribot_arm_left", 7),
                ("astribot_gripper_left", 1), ("astribot_arm_right", 7),
                ("astribot_gripper_right", 1), ("astribot_head", 2)]
JOINTS = tuple(f"{part}_joint_{j}" for part, dof in PART_CONFIGS for j in range(1, dof + 1))


def _json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_episode(parquet: Path):
    """Require the existing common 10 ms grid; never fill a missing joint."""
    table = pq.read_table(parquet)
    command = table.filter(pc.equal(table['kind'], 'command'))
    real = table.filter(pc.equal(table['kind'], 'state'))
    streams = defaultdict(list)
    states = defaultdict(list)
    for row in command.to_pylist():
        streams[row['joint']].append(row)
    if set(streams) != set(JOINTS):
        raise ValueError(f"command joints mismatch: missing={set(JOINTS)-set(streams)}, extra={set(streams)-set(JOINTS)}")
    grid = None
    positions = []
    for joint in JOINTS:
        rows = sorted(streams[joint], key=lambda r: r['time'])
        times = np.asarray([r['time'] for r in rows], dtype=np.float64)
        pos = np.asarray([r['position'] for r in rows], dtype=np.float64)
        if times.size < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(pos)):
            raise ValueError(f"invalid command stream: {joint}")
        if not np.allclose(np.diff(times), PERIOD_S, rtol=0, atol=1e-9):
            raise ValueError(f"command stream is not 100 Hz: {joint}")
        if any(r['mode'] != 1 for r in rows):
            raise ValueError(f"position-mode commands required: {joint}")
        if grid is None:
            grid = times
        elif grid.shape != times.shape or not np.allclose(grid, times, rtol=0, atol=1e-9):
            raise ValueError(f"command streams do not share a grid: {joint}")
        positions.append(pos)
    for row in real.to_pylist():
        states[row['joint']].append([row['time'], row['position'], row['velocity'], row['torque']])
    for joint in JOINTS:
        if joint not in states:
            raise ValueError(f"missing real state for initialization: {joint}")
        states[joint] = np.asarray(sorted(states[joint], key=lambda r: r[0]), dtype=np.float64)
        if not states[joint][0, 0] <= grid[0] <= states[joint][-1, 0]:
            raise ValueError(f"real state does not bracket initial time: {joint}")
        if not np.all(np.isfinite(states[joint][:, :3])):
            raise ValueError(f"non-finite real position/velocity: {joint}")
    return grid, np.asarray(positions).T, states, real


def save_real_states(path, states):
    """Flat rows + offsets preserve unequal per-joint lengths without pickle."""
    names = sorted(states)
    lengths = np.asarray([len(states[n]) for n in names], dtype=np.int64)
    np.savez_compressed(path, joint_names=np.asarray(names),
                        offsets=np.r_[0, np.cumsum(lengths)],
                        columns=np.asarray(['artifact_time_s', 'position', 'velocity', 'torque']),
                        values=np.concatenate([states[n] for n in names]))


def tracking_errors(states, sim_time, position, velocity, torque, end_time):
    result = {}
    for j, joint in enumerate(JOINTS):
        real = states[joint]
        mask = (sim_time >= real[0, 0]) & (sim_time <= min(end_time, real[-1, 0]))
        t = sim_time[mask]
        if not len(t):
            raise ValueError(f"no comparison overlap: {joint}")
        units = ('percentage_points', 'percentage_points/s') if 'gripper' in joint else ('rad', 'rad/s')
        metrics = {}
        for col, name, observed, unit in [(1, 'position', position, units[0]),
                                           (2, 'velocity', velocity, units[1]),
                                           (3, 'torque_diagnostic', torque, 'unverified artifact vs simulated actuator units')]:
            delta = observed[mask, j] - np.interp(t, real[:, 0], real[:, col])
            delta = delta[np.isfinite(delta)]
            metrics[name] = {'rmse': float(np.sqrt(np.mean(delta**2))),
                             'max_abs': float(np.max(np.abs(delta))), 'unit': unit,
                             'samples': int(len(delta))} if len(delta) else None
        result[joint] = {'time_start_s': float(t[0]), 'time_end_s': float(t[-1]),
                         'torque_comparable': False, **metrics}
    groups = {}
    for name, gripper in [('angular_joints', False), ('grippers', True)]:
        selected = [v for k, v in result.items() if ('gripper' in k) == gripper]
        groups[name] = {'joint_count': len(selected), 'position_unit': selected[0]['position']['unit'],
                        'position_rmse_across_joints': float(np.sqrt(np.mean([v['position']['rmse']**2 for v in selected]))),
                        'velocity_rmse_across_joints': float(np.sqrt(np.mean([v['velocity']['rmse']**2 for v in selected]))),
                        'velocity_unit': selected[0]['velocity']['unit']}
    return {'comparison': 'linear real-state interpolation at known post-step physics times, clipped to command/real overlap',
            'torque_comparable': False, 'groups': groups, 'joints': result}


def env_joint(joint):
    return joint[:-1] + 'L1' if 'gripper' in joint else joint


def set_commands(env, frame):
    for part, dof in PART_CONFIGS:
        interface = env.robot_dict[part]
        interface.controller_mode = 1
        interface.joint_position_command = [float(frame[JOINTS.index(f'{part}_joint_{j}')]) for j in range(1, dof + 1)]
        # Match the previous SDK set_joints_position position-only baseline.
        interface.joint_velocity_command = [0.0] * dof
        interface.joint_torque_command = [0.0] * dof


def initialize(env, grid, states):
    import torch
    q = env.robot.data.default_joint_pos.clone()
    v = env.robot.data.default_joint_vel.clone()
    initial = {}
    for joint in JOINTS:
        real = states[joint]
        p = float(np.interp(grid[0], real[:, 0], real[:, 1]))
        vel = float(np.interp(grid[0], real[:, 0], real[:, 2]))
        initial[joint] = {'position': p, 'velocity': vel}
        if 'gripper' in joint:
            prefix = joint.rsplit('_', 1)[0] + '_'
            for suffix, direction in env._GRIPPER_JOINT_DIRECTIONS.items():
                idx = env.joint_name_to_index[prefix + suffix]
                q[0, idx] = p * direction * env._GRIPPER_RAD_PER_PERCENT
                v[0, idx] = vel * direction * env._GRIPPER_RAD_PER_PERCENT
        else:
            idx = env.joint_name_to_index[joint]
            q[0, idx], v[0, idx] = p, vel
    env.scene.reset()
    env.robot.write_root_state_to_sim(env.robot.data.default_root_state.clone())
    env.robot.write_joint_state_to_sim(q, v)
    env.robot.set_joint_position_target(q)
    env.robot.set_joint_velocity_target(torch.zeros_like(v))
    env.scene.write_data_to_sim()
    env.scene.update(0.0)
    return initial


def sample(env):
    # Each tensor is copied after scene.update; no cached SDK state is involved.
    arrays = [getattr(env.robot.data, field)[0].detach().cpu().numpy().copy()
              for field in ('joint_pos', 'joint_vel', 'applied_torque')]
    output = np.empty((3, len(JOINTS)), dtype=np.float64)
    for j, joint in enumerate(JOINTS):
        idx = env.joint_name_to_index[env_joint(joint)]
        for col in range(3):
            value = arrays[col][idx]
            if col < 2 and 'gripper' in joint:
                value = env._gripper_joint_to_percent(env_joint(joint), value)
            output[col, j] = value
    if not np.all(np.isfinite(output)):
        raise RuntimeError('non-finite simulated state')
    return output


def wait_until(deadline_ns):
    # Absolute deadlines avoid accumulating sleep or execution time per step.
    # A short final spin reduces scheduler jitter without changing physics dt.
    remaining = deadline_ns - time.monotonic_ns()
    if remaining > 200_000:
        time.sleep((remaining - 150_000) / 1e9)
    while time.monotonic_ns() < deadline_ns:
        pass


def replay(env, parquet, output, config):
    grid, command, states, real_table = load_episode(parquet)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing results: {output}')
    output.mkdir(parents=True, exist_ok=True)
    dt = float(env.sim.get_physics_dt())
    if not np.isclose(dt * env.frame_skip, PERIOD_S, rtol=0, atol=1e-12):
        raise ValueError(f'expected 10 ms environment step, actual dt={dt}, frame_skip={env.frame_skip}')
    for joint in JOINTS:
        if env_joint(joint) not in env.joint_name_to_index:
            raise ValueError(f'joint absent from simulation: {joint}')
    # Warm lazy GPU/ROS paths, then replace q/v with the recorded initial state.
    set_commands(env, command[0])
    initialize(env, grid, states)
    for _ in range(30):
        env.step(np.zeros(7))
    initial = initialize(env, grid, states)
    actual_initial = sample(env)
    expected_initial = np.asarray([[initial[j]['position'], initial[j]['velocity']] for j in JOINTS]).T
    if not np.allclose(actual_initial[:2], expected_initial, atol=2e-5, rtol=0):
        raise RuntimeError('recorded-state initialization did not reach the articulation')
    n = len(grid)
    samples = np.empty((n + 1, 3, len(JOINTS)), dtype=np.float64)
    samples[0] = actual_initial
    samples[0, 2] = np.nan  # Initial actuator torque has no corresponding step.
    physics_time = np.empty(n + 1)
    physics_time[0] = float(env.sim.current_time)
    physics_index = np.empty(n + 1, dtype=np.int64)
    physics_index[0] = int(env.sim.current_time_step_index)
    state_ns = np.empty(n + 1, dtype=np.int64)
    state_ns[0] = time.monotonic_ns()
    start_ns = time.monotonic_ns() + 20_000_000
    deadline = start_ns + np.arange(n, dtype=np.int64) * 10_000_000
    sent_ns, end_ns = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    for i in range(n):
        wait_until(int(deadline[i]))
        set_commands(env, command[i])
        sent_ns[i] = time.monotonic_ns()
        env.step(np.zeros(7))
        physics_time[i + 1] = float(env.sim.current_time)
        physics_index[i + 1] = int(env.sim.current_time_step_index)
        samples[i + 1] = sample(env)
        state_ns[i + 1] = end_ns[i] = time.monotonic_ns()
        if i % 500 == 0 or i == n - 1:
            print(f'{parquet.parent.name}: {i+1}/{n}, physics={physics_time[i+1]-physics_time[0]:.3f}s, wall={(end_ns[i]-start_ns)/1e9:.3f}s', flush=True)
    relative = physics_time - physics_time[0]
    # PhysX supplies the callback step as float32, so its accumulated clock can
    # differ from decimal .01 by ~2.24e-10 per step. Check each actual delta and
    # the integer step counter; preserve the original callback times verbatim.
    np.savez_compressed(output / 'physics_validation_trace.npz', physics_time_s=physics_time,
                        physics_step_index=physics_index, expected_physics_dt_s=dt)
    if not np.allclose(np.diff(physics_time), dt, rtol=0, atol=1e-9):
        raise RuntimeError(f'physics timestamps do not advance one 10 ms step per command: {np.diff(physics_time).min()}..{np.diff(physics_time).max()}')
    if not np.all(np.diff(physics_index) == env.frame_skip):
        raise RuntimeError('unexpected physics-step count')
    sim_time = float(grid[0]) + relative
    errors = tracking_errors(states, sim_time, samples[:, 0], samples[:, 1], samples[:, 2], float(grid[-1]))
    intervals = np.diff(sent_ns) / 1e6
    late = (sent_ns - deadline) / 1e6
    work = (end_ns - sent_ns) / 1e6
    missed = end_ns > deadline + 10_000_000
    meta = {
        'status': 'trajectory_complete',
        'wallclock_deadline_status': 'all_met' if not missed.any() else 'overruns_recorded',
        'replay_path': 'same-process Isaac Lab env.step physics baseline; no SDK/ROS transport replay; no ROS executor',
        'input_parquet': str(parquet), 'input_sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(),
        'command': shlex.join(sys.argv), 'python': sys.executable,
        'frequency_hz': 100.0, 'physics_dt_s': dt, 'frame_skip': env.frame_skip,
        'physics_steps': int(physics_index[-1] - physics_index[0]), 'command_frames': n, 'sim_state_samples': n + 1,
        'observed_physics_step_min_s': float(np.diff(physics_time).min()),
        'observed_physics_step_max_s': float(np.diff(physics_time).max()),
        'command_time_start_s': float(grid[0]), 'command_time_end_s': float(grid[-1]),
        'command_time_span_s': float(grid[-1]-grid[0]), 'executed_physics_duration_s': float(relative[-1]),
        'last_state_artifact_time_s': float(sim_time[-1]),
        'timing_semantics': 'command i applied over [grid[i], grid[i]+0.01]; state i+1 read after env.step at grid[i]+0.01; initial q/v at grid[0]',
        'initialization': 'linear interpolation of recorded real q/v at first command time; root/model defaults; 30 warmup steps excluded',
        'command_semantics': 'position only; recorded velocity and torque fields are not feedforward, matching prior SDK position-only baseline',
        'pd_baseline': {'stiffness': 800.0, 'damping': 40.0, 'effort_limit': 400.0},
        'torque_comparable': False, 'torque_note': 'diagnostic only; original metadata does not establish comparability; gripper torque is L1 actuator effort',
        'ros': {key: os.environ.get(key) for key in ('ROS_VERSION', 'ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY')},
        'wall_duration_s': float((end_ns[-1]-start_ns)/1e9),
        'wall_mean_send_period_ms': float(intervals.mean()), 'wall_p95_send_period_ms': float(np.percentile(intervals, 95)),
        'wall_max_send_period_ms': float(intervals.max()), 'wall_mean_work_ms': float(work.mean()),
        'wall_p95_work_ms': float(np.percentile(work, 95)), 'wall_max_work_ms': float(work.max()),
        'deadline_completion_overruns': int(missed.sum()), 'deadline_completion_met_fraction': float((~missed).mean()),
        'work_over_10ms_count': int((work > 10).sum()), 'start_lateness_mean_ms': float(late.mean()),
        'start_lateness_p95_ms': float(np.percentile(late, 95)), 'start_lateness_max_ms': float(late.max()),
        'units': {'angular_position': 'rad', 'angular_velocity': 'rad/s', 'gripper_position': 'percentage_points',
                  'gripper_velocity': 'percentage_points/s', 'wall_clock': 'monotonic_ns', 'physics_time': 's'},
        'tracking_groups': errors['groups'],
    }
    np.savez_compressed(output / 'sim_states.npz', joint_names=np.asarray(JOINTS),
                        artifact_time_s=sim_time, physics_time_s=physics_time,
                        physics_step_index=physics_index, state_monotonic_ns=state_ns,
                        position=samples[:, 0], velocity=samples[:, 1], torque=samples[:, 2])
    np.savez_compressed(output / 'command_trace.npz', joint_names=np.asarray(JOINTS), artifact_time_s=grid,
                        position=command, deadline_monotonic_ns=deadline, applied_monotonic_ns=sent_ns,
                        completion_monotonic_ns=end_ns, start_lateness_ms=late, work_ms=work,
                        completion_overrun=missed)
    save_real_states(output / 'real_states.npz', states)
    pq.write_table(real_table, output / 'real_states.parquet', compression='zstd')
    _json(output / 'tracking_error.json', errors)
    _json(output / 'initial_state.json', initial)
    _json(output / 'run_metadata.json', meta)
    (output / 'actual_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = parquet.parent / 'resample_manifest.json'
    if manifest.exists():
        (output / manifest.name).write_bytes(manifest.read_bytes())
    provenance = {}
    for filename in ('metadata.json', 'source_metadata.json'):
        source = parquet.parent / filename
        if source.exists():
            (output / filename).write_bytes(source.read_bytes())
            provenance[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    model = Path(config['model_path'])
    provenance['model_path'] = str(model)
    provenance['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
    provenance['config_sha256'] = hashlib.sha256((output / 'actual_config.yaml').read_bytes()).hexdigest()
    provenance['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['provenance'] = provenance
    _json(output / 'run_metadata.json', meta)
    angular = errors['groups']['angular_joints']
    gripper = errors['groups']['grippers']
    (output / 'summary.txt').write_text(
        f"Isaac Lab 100 Hz physics baseline (same process, env.step)\nFrames: {n}; command span: {meta['command_time_span_s']:.6f} s; physics: {relative[-1]:.6f} s\n"
        f"Wall: {meta['wall_duration_s']:.6f} s; send interval mean/p95/max: {intervals.mean():.3f}/{np.percentile(intervals,95):.3f}/{intervals.max():.3f} ms\n"
        f"Completion deadline overruns: {missed.sum()}/{n}; work >10ms: {(work>10).sum()}/{n}\n"
        f"Angular position RMSE: {angular['position_rmse_across_joints']:.6f} rad\nGripper position RMSE: {gripper['position_rmse_across_joints']:.6f} percentage points\nTorque: diagnostic only, not comparable\n")
    print(json.dumps(meta), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquets', nargs='+', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('results/alignment_100hz'))
    parser.add_argument('--config', type=Path, default=Path('config/astribot_s1/simulation_isaaclab_param.yaml'))
    args = parser.parse_args()
    # Preflight runs before creating ROS publishers or a CUDA simulation.
    for parquet in args.parquets:
        load_episode(parquet)
    if os.getenv('ROS_VERSION') != '2' or os.getenv('ROS_LOCALHOST_ONLY') != '1' or os.getenv('ROS_DOMAIN_ID') != '197':
        raise RuntimeError('isolated simulation requires ROS_VERSION=2, ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=197 (export after env.sh)')
    os.environ['ASTRIBOT_FRAME_SKIP'] = '1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    config = yaml.safe_load(args.config.read_text())
    if config['simulator_type'] != 'IsaacLab':
        raise ValueError('IsaacLab config required')
    config['model_path'] = str(root / config['model_path'].lstrip('/'))
    config['camera_names'] = []
    config['mode'] = 'rgb_array'
    from src.astribot_envs.astribot_isaaclab_env import AstribotIsaacLabEnv
    env = AstribotIsaacLabEnv(config)
    failure = None
    try:
        env.reset()
        reports = []
        for parquet in args.parquets:
            reports.append(replay(env, parquet, args.output_root / parquet.parent.name, config))
        _json(args.output_root / 'run_manifest.json', {'status': 'complete', 'episodes': reports})
        # Persist all evidence before close: Kit may terminate the interpreter.
        print('ALL_EPISODES_SAVED', flush=True)
    except BaseException as error:
        failure = error
        args.output_root.mkdir(parents=True, exist_ok=True)
        _json(args.output_root / 'failure.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        import rclpy
        env.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        env.close()
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
