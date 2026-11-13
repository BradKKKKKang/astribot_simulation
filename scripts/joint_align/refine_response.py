#!/usr/bin/env python3
"""Refine completed SDK/ROS trials, with equal episode weighting and audit checks.

The local proposal does not model all cross-joint coupling. Always replay the
complete selected configuration and inspect every joint before acceptance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
import yaml

import replay_artifacts_sdk_100hz as replay

JOINTS = [j for j in replay.JOINTS if 'gripper' not in j]
METRICS = ('position_rmse', 'velocity_rmse')
WEIGHT = .02


def key(joint):
    return joint.removeprefix('astribot_') + '_control'


def read_trial(root):
    trial = json.loads((root / 'trial.json').read_text())
    if (trial['status'] != 'complete' or trial.get('sdk_exit_code') != 0
            or trial.get('simulator_exit_code') != 0 or trial.get('time_mode') != 'physics_step'):
        raise ValueError(f'{root}: completed clean physics-step trial required')
    snapshot = root / 'simulation_config.yaml'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != trial['config_sha256']:
        raise ValueError(f'{root}: configuration hash mismatch')
    episodes = {}
    for path in sorted((root / 'replay').glob('*/run_metadata.json')):
        meta = json.loads(path.read_text())
        if (meta['status'] != 'complete' or not meta.get('simulator_verified_each_position_target')
                or meta.get('time_mode') != 'physics_step'
                or meta['preposition'].get('settle_physics_steps', 0) != trial.get('settle_steps', 0)):
            raise ValueError(f'{path}: invalid replay protocol')
        episodes[path.parent.name] = {
            'meta': meta, 'folder': path.parent,
            'metrics': json.loads((path.parent / 'tracking_error.json').read_text())['physics_time']['joints']}
    if not episodes:
        raise ValueError(f'{root}: no episodes')
    return trial, yaml.safe_load(snapshot.read_text()), episodes


def comparable(trials):
    identity = None
    for label, (trial, _, episodes) in trials.items():
        current = (trial.get('settle_steps', 0), trial.get('source_sha256'),
                   {name: ep['meta']['input_sha256'] for name, ep in episodes.items()})
        if identity is not None and identity != current:
            raise ValueError(f'{label}: source, implementation or settling protocol differs')
        identity = current


def scores(trial):
    episodes = trial[2]
    return {j: {m: float(np.sqrt(np.mean([e['metrics'][j][m] ** 2 for e in episodes.values()])))
                for m in METRICS} for j in replay.JOINTS}


def read_curves(folder):
    data = pq.read_table(folder / 'sim_step_states.parquet').to_pydict()
    names = np.asarray(data['joint'])
    a = np.column_stack([data[k] for k in ('physics_time_s', 'position', 'velocity')])
    return {j: a[names == j] for j in JOINTS}


def propose(base, variants):
    trials = {'base': base, **{str(i): v for i, v in enumerate(variants)}}
    comparable(trials)
    config = copy.deepcopy(base[1])
    errors, jacobians = {j: [] for j in JOINTS}, {j: [] for j in JOINTS}
    replay.CONTROL_HZ = 250
    for name, ep in base[2].items():
        source = Path(ep['meta']['input_file'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != ep['meta']['input_sha256']:
            raise ValueError(f'changed input: {source}')
        _, grid, _, real = replay.load_episode(source)
        observed = read_curves(ep['folder'])
        alternatives = [read_curves(v[2][name]['folder']) for v in variants]
        for j in JOINTS:
            a = observed[j]
            t = grid[0] + a[:, 0]
            keep = (t >= grid[0]) & (t <= grid[-1])
            target = np.column_stack([np.interp(t, real[j][:, 0], real[j][:, i]) for i in (1, 2)])
            weight = np.array([1., WEIGHT]) / np.sqrt(keep.sum())
            errors[j].append(((a[keep, 1:] - target[keep]) * weight).ravel())
            columns = []
            for alt in alternatives:
                if not np.array_equal(alt[j][:, 0], a[:, 0]):
                    raise ValueError('physics grids differ')
                columns.append(((alt[j][keep, 1:] - a[keep, 1:]) * weight).ravel())
            jacobians[j].append(np.column_stack(columns))
    report = {}
    for j in JOINTS:
        residual, jac = np.concatenate(errors[j]), np.concatenate(jacobians[j])
        regularizer = max(float(np.linalg.norm(jac)), 1e-9) * .001
        solution = lsq_linear(np.vstack([jac, np.eye(3) * regularizer]),
                              np.r_[-residual, np.zeros(3)],
                              bounds=([-.5, -.5, 0], [1., 1., 2.]))
        coeff = solution.x
        for i, parameter in enumerate(('stiffness', 'damping')):
            b = base[1][key(j)][parameter]
            delta = variants[i][1][key(j)][parameter] - b
            config[key(j)][parameter] = float(b + coeff[i] * delta)
        b = base[1]['joint_command_filters'][j]['time_constant_s']
        delta = variants[2][1]['joint_command_filters'][j]['time_constant_s'] - b
        config['joint_command_filters'][j]['time_constant_s'] = float(b + coeff[2] * delta)
        report[j] = {'coefficients': coeff.tolist(),
                     'observed_objective': float(np.mean(residual ** 2)),
                     'predicted_objective': float(np.mean((residual + jac @ coeff) ** 2))}
    return config, report


def select(trials, reference='baseline_settled'):
    comparable(trials)
    values = {label: scores(t) for label, t in trials.items()}
    config, decisions = copy.deepcopy(trials[reference][1]), {}
    for j in JOINTS:
        eligible = [label for label, v in values.items()
                    if all(v[j][m] <= values[reference][j][m] for m in METRICS)]
        best = min(eligible, key=lambda label: values[label][j]['position_rmse'] ** 2
                   + WEIGHT ** 2 * values[label][j]['velocity_rmse'] ** 2)
        config[key(j)] = copy.deepcopy(trials[best][1][key(j)])
        config['joint_command_filters'][j] = copy.deepcopy(trials[best][1]['joint_command_filters'][j])
        decisions[j] = {'selected': best, 'metrics': values[best][j], 'reference': values[reference][j]}
    return config, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'fit', 'select', 'scores'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--set', default='training')
    parser.add_argument('--sim-python', default='/home/zihengkang/miniconda3/envs/astribot_simu_il/bin/python')
    parser.add_argument('--domain', type=int, default=198)
    parser.add_argument('--output', default='proposal')
    args = parser.parse_args()
    root = args.root
    if args.operation == 'run':
        episodes = json.loads((root / 'episodes.json').read_text())[args.set]
        for label in args.labels:
            config = root / f'{label}.yaml'
            print(f'RUN {label} {args.set}', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('run_trial.py')),
                            '--config', str(config), '--output', str(root / label),
                            '--sim-python', args.sim_python, '--sdk-python', '/usr/bin/python3',
                            '--domain', str(args.domain), '--settle-steps', '500', *episodes], check=True)
            print(json.dumps({'label': label, 'scores': scores(read_trial(root / label))}), flush=True)
        return
    trials = {label: read_trial(root / label) for label in args.labels}
    comparable(trials)
    if args.operation == 'scores':
        print(json.dumps({label: scores(t) for label, t in trials.items()}, indent=2))
        return
    if args.operation == 'fit':
        if len(trials) != 4:
            parser.error('fit labels must be base, stiffness, damping, filter in that order')
        ordered = list(trials.values())
        config, report = propose(ordered[0], ordered[1:])
    else:
        config, report = select(trials)
    (root / f'{args.output}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (root / f'{args.output}.json').write_text(json.dumps({
        'training': {label: {n: e['meta']['input_sha256'] for n, e in t[2].items()}
                     for label, t in trials.items()},
        'velocity_weight_s': WEIGHT, 'joints': report,
        'fitter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'requires_combined_replay': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
