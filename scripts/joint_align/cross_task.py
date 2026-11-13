#!/usr/bin/env python3
"""Prepare and evaluate task-balanced SDK/ROS response calibration experiments."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
import yaml

import replay_artifacts_sdk_100hz as replay
import refine_response as refine

ROOT = Path(__file__).resolve().parents[2]
TRAIN_TASKS = ('penholder', 'press_cube', 'press_cube_v2', 'cap_pen',
               'put_block_back', 'plug_in_Ethernet_cable_v2', 'item_classification')
UNSEEN_TASKS = ('press_button', 'fold_T_shirt')
WEIGHT = .02
TUNED_JOINTS = tuple(j for j in replay.JOINTS if 'arm_' in j or 'gripper' in j)
# Complete seed replay exposed right-joint-5 velocity regression after mixing
# individually eligible wrist drives. Select this coupled wrist as one block.
COUPLED_GROUPS = (tuple(f'astribot_arm_right_joint_{i}' for i in (5, 6, 7)),)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def control_key(joint):
    if 'gripper' in joint:
        return joint.removeprefix('astribot_').replace('_joint_1', '_control')
    return refine.key(joint)


def filter_key(joint):
    return replay._simulation_joint(joint)


def free_motion_mask(grid, command, state):
    """Conservative contact-like exclusion, defined entirely by recorded signals.

    100 is closed. A sustained closing target with a stationary, separated
    encoder suggests contact, but does not prove it. Exclude 150 ms before the
    plateau through reopening, plus 100 ms recovery; retain full metrics too.
    """
    position = np.interp(grid, state[:, 0], state[:, 1])
    velocity = np.interp(grid, state[:, 0], state[:, 2])
    plateau = (command > 10) & (command - position > 5) & (abs(velocity) < 5)
    edges = np.diff(np.r_[False, plateau, False].astype(int))
    mask = np.ones(len(grid), dtype=bool)
    dt = 1 / 250  # All calibration grids are 250 Hz; avoid epoch-float rounding.
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (end - start) * dt < .12:
            continue
        recovered = np.flatnonzero((command[end:] <= 5) & (position[end:] <= 5))
        stop = end + recovered[0] + int(np.ceil(.1 / dt)) if recovered.size else len(grid)
        mask[max(0, start - int(np.ceil(.15 / dt))):min(stop, len(grid))] = False
    return mask


def add_free_metrics(trial):
    """Attach diagnostic partitions; full metrics remain untouched."""
    for ep in trial[2].values():
        path = Path(ep['meta']['input_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != ep['meta']['input_sha256']:
            raise ValueError(f'input changed since replay: {path}')
        _, grid, command, states = replay.load_hdf5_episode(path)
        ep['free_metrics'], ep['contact_metrics'] = {}, {}
        for i, j in enumerate(replay.JOINTS):
            if 'gripper' not in j:
                continue
            source_mask = free_motion_mask(grid, command[:, i], states[j])
            data = pq.read_table(ep['folder'] / 'sim_step_states.parquet',
                                 columns=['joint', 'physics_time_s', 'position', 'velocity'],
                                 filters=[('joint', '=', j)]).to_pydict()
            t = grid[0] + np.asarray(data['physics_time_s'])
            valid = (t >= grid[0]) & (t <= grid[-1])
            index = np.clip(np.rint((t - grid[0]) * 250).astype(int), 0, len(grid) - 1)
            for mode, keep in [('free', valid & source_mask[index]), ('contact', valid & ~source_mask[index])]:
                errors = [np.asarray(data[field])[keep] - np.interp(t[keep], states[j][:, 0], states[j][:, k])
                          for k, field in [(1, 'position'), (2, 'velocity')]]
                ep[f'{mode}_metrics'][j] = {
                    'samples': int(keep.sum()),
                    **{m: float(np.sqrt(np.mean(e ** 2))) if len(e) else None
                       for m, e in zip(refine.METRICS, errors)}}
            if not ep['free_metrics'][j]['samples']:
                raise ValueError(f'no free-motion samples for {path}/{j}')


def extract(source, target):
    """Copy joint streams verbatim, retaining timestamps and per-dataset hashes."""
    checksums = {}
    with h5py.File(source, 'r') as src, h5py.File(target, 'x') as dst:
        group = dst.create_group('joints_dict')
        def visit(name, item):
            if not isinstance(item, h5py.Dataset):
                return
            data = item[:]
            digest = hashlib.sha256(data.tobytes()).hexdigest()
            checksums[name] = {'shape': list(data.shape), 'dtype': str(data.dtype), 'sha256': digest}
            group.create_dataset(name, data=data, compression='gzip')
        src['joints_dict'].visititems(visit)
        dst.attrs['source_file'] = str(source)
        dst.attrs['source_size_bytes'] = source.stat().st_size
    return checksums


def prepare(root, source_root, baseline):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    inventory = []
    for directory in sorted(source_root.iterdir()):
        if directory.is_dir():
            paths = sorted(directory.glob('*.hdf5'))
            inventory.append({'recording_group': directory.name, 'episodes': len(paths),
                              'bytes': sum(p.stat().st_size for p in paths)})
    write_json(root / 'inventory.json', inventory)
    manifest = {'training': [], 'validation': [], 'unseen_tasks': [], 'inputs': {}}
    for split, tasks, number in [('training', TRAIN_TASKS, 0),
                                  ('validation', TRAIN_TASKS, 1),
                                  ('unseen_tasks', UNSEEN_TASKS, 0)]:
        for task in tasks:
            group = f'real_eval-{task}-pi05_sync-id'
            episode_number = 5 if split == 'validation' and task == 'penholder' else number
            source = source_root / group / f'{group}_episode_abs_{episode_number}.hdf5'
            target = root / 'inputs' / source.name
            hashes = extract(source, target)
            _, grid, command, states = replay.load_hdf5_episode(target)
            manifest[split].append(str(target.resolve()))
            manifest['inputs'][target.stem] = {
                'task': task, 'split': split, 'source': str(source),
                'dataset_sha256': hashes,
                'copy_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'duration_s': float(grid[-1] - grid[0]), 'frames': len(grid),
                'command_range': np.ptp(command, axis=0).tolist(),
                'measured_initial': replay._initial_positions(grid[0], states).tolist()}
            print(split, task, len(grid), flush=True)
    manifest['gpu_validation'] = [p for p in manifest['validation']
                                  if manifest['inputs'][Path(p).stem]['task'] in ('press_cube_v2', 'cap_pen', 'put_block_back')]
    manifest['protocol'] = {
        'input': 'recorded commands; measured position/velocity are evaluation only',
        'frequency_hz': 250, 'settle_steps': 500, 'physics_device': 'cpu',
        'fit_weighting': 'equal task then equal complete episode',
        'velocity_weight_s': WEIGHT,
        'gate': 'report every task and joint; <=1% + numeric floor position/velocity regression',
        'source_config': str(baseline), 'source_config_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest()}
    write_json(root / 'episodes.json', manifest)
    config = yaml.safe_load(baseline.read_text())
    config['physics_device'] = 'cpu'
    (root / 'v3.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for label, parameter, factor in [('stiffness', 'stiffness', 2.), ('damping', 'damping', 2.),
                                      ('filter', None, None)]:
        candidate = copy.deepcopy(config)
        for j in TUNED_JOINTS:
            if parameter:
                candidate[control_key(j)][parameter] *= factor
            else:
                candidate['joint_command_filters'][filter_key(j)]['time_constant_s'] += .02
        (root / f'{label}.yaml').write_text(yaml.safe_dump(candidate, sort_keys=False))


def run(root, labels, split, domain):
    manifest = json.loads((root / 'episodes.json').read_text())
    for label in labels:
        if split != 'training':
            frozen = json.loads((root / 'freeze.json').read_text())
            if hashlib.sha256((root / f'{label}.yaml').read_bytes()).hexdigest() != frozen['configs'][label]:
                raise ValueError('configuration changed after freeze')
            if hashlib.sha256((root / 'episodes.json').read_bytes()).hexdigest() != frozen['manifest_sha256']:
                raise ValueError('episode manifest changed after freeze')
        output = root / f'{label}_{split}'
        command = [sys.executable, str(Path(__file__).with_name('run_trial.py')),
                   '--config', str(root / f'{label}.yaml'), '--output', str(output),
                   '--sim-python', '/home/zihengkang/miniconda3/envs/astribot_simu_il/bin/python',
                   '--sdk-python', '/usr/bin/python3', '--domain', str(domain),
                   '--settle-steps', '500', '--timeout', '3600', *manifest[split]]
        print('RUN', label, split, flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


def task_scores(trial, manifest, free_grippers=False):
    tasks = {}
    for name, episode in trial[2].items():
        task = manifest['inputs'][name]['task']
        metrics = {**episode['metrics'], **(episode['free_metrics'] if free_grippers else {})}
        tasks.setdefault(task, []).append(metrics)
    return {task: {j: {m: float(np.sqrt(np.mean([e[j][m] ** 2 for e in episodes])))
                       for m in refine.METRICS} for j in replay.JOINTS}
            for task, episodes in tasks.items()}


def load_training(root, labels):
    manifest = json.loads((root / 'episodes.json').read_text())
    expected = {Path(p).stem for p in manifest['training']}
    trials = {label: refine.read_trial(root / f'{label}_training') for label in labels}
    refine.comparable(trials)
    if any(set(t[2]) != expected for t in trials.values()):
        raise ValueError('fit/selection requires exactly the declared training episodes')
    # Parameter proposals may vary drives and filters, never the physics backend,
    # model, time step or scene while sharing one finite-difference fit.
    def dynamics(config):
        return {k: v for k, v in config.items()
                if not k.endswith('_control') and k != 'joint_command_filters'}
    configurations = [dynamics(t[1]) for t in trials.values()]
    if any(c != configurations[0] for c in configurations[1:]):
        raise ValueError('physics, model or scene differs between candidates')
    for t in trials.values():
        add_free_metrics(t)
    return manifest, trials


def constrained_coefficients(residual, jacobian, task_blocks, lower, upper):
    """Bound the local quadratic prediction separately for each task/metric."""
    regularizer = max(np.linalg.norm(jacobian), 1e-9) * .003
    scale = max(float(residual @ residual), 1e-12)
    def objective(x):
        error = residual + jacobian @ x
        return (error @ error + regularizer ** 2 * (x @ x)) / scale
    def gradient(x):
        return 2 * (jacobian.T @ (residual + jacobian @ x) + regularizer ** 2 * x) / scale
    def constraints(x):
        return np.concatenate([1 - np.sum((r + a @ x).reshape(-1, 2) ** 2, axis=0) / limit
                               for r, a, limit in task_blocks])
    result = minimize(objective, np.zeros(3), jac=gradient, method='SLSQP',
                      bounds=list(zip(lower, upper)),
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'ftol': 1e-10, 'maxiter': 200})
    if not result.success or np.min(constraints(result.x)) < -1e-6:
        raise RuntimeError(f'constrained response fit failed: {result.message}')
    return result.x


def fit(root, labels, output):
    manifest, trials = load_training(root, labels)
    base, *variants = trials.values()
    if len(variants) != 3:
        raise ValueError('fit requires base, stiffness, damping, filter trials')
    residuals, jacobians = {j: [] for j in replay.JOINTS}, {j: [] for j in replay.JOINTS}
    counts = {}
    for name in base[2]:
        task = manifest['inputs'][name]['task']
        counts[task] = counts.get(task, 0) + 1
    def curves(folder):
        data = pq.read_table(folder / 'sim_step_states.parquet').to_pydict()
        names = np.asarray(data['joint'])
        rows = np.column_stack([data[k] for k in ('physics_time_s', 'position', 'velocity')])
        return {j: rows[names == j] for j in replay.JOINTS}
    for name, ep in base[2].items():
        path = Path(ep['meta']['input_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != ep['meta']['input_sha256']:
            raise ValueError('input changed')
        _, grid, command, real = replay.load_hdf5_episode(path)
        observed = curves(ep['folder'])
        alternatives = [curves(v[2][name]['folder']) for v in variants]
        for j in TUNED_JOINTS:
            a = observed[j]
            t = grid[0] + a[:, 0]
            keep = t <= grid[-1]
            if 'gripper' in j:
                mask = free_motion_mask(grid, command[:, replay.JOINTS.index(j)], real[j])
                index = np.clip(np.rint((t - grid[0]) * 250).astype(int), 0, len(grid) - 1)
                keep &= mask[index]
            target = np.column_stack([np.interp(t, real[j][:, 0], real[j][:, k]) for k in (1, 2)])
            weight = np.array([1., WEIGHT]) / np.sqrt(keep.sum() * counts[manifest['inputs'][name]['task']])
            residuals[j].append(((a[keep, 1:] - target[keep]) * weight).ravel())
            columns = []
            for alt in alternatives:
                if not np.array_equal(alt[j][:, 0], a[:, 0]):
                    raise ValueError('physics grids differ')
                columns.append(((alt[j][keep, 1:] - a[keep, 1:]) * weight).ravel())
            jacobians[j].append(np.column_stack(columns))
    config, report = copy.deepcopy(base[1]), {}
    base_scores = task_scores(base, manifest, free_grippers=True)
    for j in TUNED_JOINTS:
        residual, jac = np.concatenate(residuals[j]), np.concatenate(jacobians[j])
        tau = base[1]['joint_command_filters'][filter_key(j)]['time_constant_s']
        delta_tau = variants[2][1]['joint_command_filters'][filter_key(j)]['time_constant_s'] - tau
        lower_tau = -tau / delta_tau
        grouped = {}
        for name, error, derivative in zip(base[2], residuals[j], jacobians[j]):
            task = manifest['inputs'][name]['task']
            grouped.setdefault(task, []).append((error, derivative))
        floors = (.01, .1) if 'gripper' in j else (.0001, .002)
        blocks = []
        for task, rows in grouped.items():
            limit = np.square([base_scores[task][j][m] * 1.01 + floor
                               for m, floor in zip(refine.METRICS, floors)]) * np.array([1., WEIGHT ** 2])
            blocks.append((np.concatenate([r for r, _ in rows]),
                           np.concatenate([a for _, a in rows]), limit))
        coeff = constrained_coefficients(residual, jac, blocks, [-.5, -.75, lower_tau], [2., 2., 4.])
        for k, parameter in enumerate(('stiffness', 'damping')):
            old = base[1][control_key(j)][parameter]
            config[control_key(j)][parameter] = float(old + coeff[k] * (variants[k][1][control_key(j)][parameter] - old))
        config['joint_command_filters'][filter_key(j)]['time_constant_s'] = max(0., float(tau + coeff[2] * delta_tau))
        report[j] = {'coefficients': coeff.tolist(), 'observed_sse': float(residual @ residual),
                     'predicted_sse': float(np.sum((residual + jac @ coeff) ** 2))}
    (root / f'{output}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / f'{output}_fit.json', report)


def select(root, labels, output):
    manifest, trials = load_training(root, labels)
    baseline = labels[0]
    values = {label: task_scores(trial, manifest, free_grippers=True) for label, trial in trials.items()}
    config, decisions = copy.deepcopy(trials[baseline][1]), {}
    for j in replay.JOINTS:
        floor = {'position_rmse': .01 if 'gripper' in j else .0001,
                 'velocity_rmse': .1 if 'gripper' in j else .002}
        eligible = [label for label in labels if all(
            values[label][task][j][m] <= values[baseline][task][j][m] * 1.01 + floor[m]
            for task in values[baseline] for m in refine.METRICS)]
        def objective(label):
            return np.mean([row[j]['position_rmse'] ** 2 + WEIGHT ** 2 * row[j]['velocity_rmse'] ** 2
                            for row in values[label].values()])
        best = min(eligible, key=objective)
        config[control_key(j)] = copy.deepcopy(trials[best][1][control_key(j)])
        config['joint_command_filters'][filter_key(j)] = copy.deepcopy(trials[best][1]['joint_command_filters'][filter_key(j)])
        decisions[j] = {'selected': best, 'eligible': eligible, 'scores': {k: float(objective(k)) for k in labels}}
    for group in COUPLED_GROUPS:
        eligible = [label for label in labels if all(
            values[label][task][j][m] <= values[baseline][task][j][m] * 1.01 + floor
            for task in values[baseline] for j in group
            for m, floor in zip(refine.METRICS, (.0001, .002)))]
        def group_objective(label):
            return float(np.mean([row[j]['position_rmse'] ** 2 + WEIGHT ** 2 * row[j]['velocity_rmse'] ** 2
                                  for row in values[label].values() for j in group]))
        best = min(eligible, key=group_objective)
        for j in group:
            config[control_key(j)] = copy.deepcopy(trials[best][1][control_key(j)])
            config['joint_command_filters'][filter_key(j)] = copy.deepcopy(trials[best][1]['joint_command_filters'][filter_key(j)])
            decisions[j].update(selected=best, eligible=eligible, group=list(group),
                                group_scores={k: group_objective(k) for k in labels})
    (root / f'{output}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / f'{output}_selection.json', decisions)
    write_json(root / f'{output}_selection_provenance.json', {
        'labels': labels, 'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((root / 'episodes.json').read_bytes()).hexdigest(),
        'training_inputs': {n: e['meta']['input_sha256'] for n, e in trials[baseline][2].items()}})


def freeze(root, labels):
    target = root / 'freeze.json'
    if target.exists():
        raise FileExistsError('refusing to overwrite a pre-validation freeze')
    if (root / 'model_sdk_hashes.json').exists():
        for path, expected in json.loads((root / 'model_sdk_hashes.json').read_text()).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'model or SDK implementation changed: {path}')
    hashes = {}
    for label in labels:
        config_path = root / f'{label}.yaml'
        config = yaml.safe_load(config_path.read_text())
        source = label.removesuffix('_gpu')
        trial = refine.read_trial(root / f'{source}_training')
        reference = copy.deepcopy(trial[1])
        if label.endswith('_gpu'):
            reference['physics_device'] = 'cuda:0'
        if config != reference:
            raise ValueError(f'{label}: configuration differs from completed training replay')
        hashes[label] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    write_json(target, {'created_utc': datetime.now(timezone.utc).isoformat(), 'configs': hashes,
                       'manifest_sha256': hashlib.sha256((root / 'episodes.json').read_bytes()).hexdigest(),
                       'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'selection_uses_validation': False})


def report(root, labels, split):
    """Recompute full-episode metrics from raw physics feedback, without time shifts."""
    import csv
    manifest = json.loads((root / 'episodes.json').read_text())
    expected = {Path(p).stem for p in manifest[split]}
    trials = {label: refine.read_trial(root / f'{label}_{split}') for label in labels}
    refine.comparable(trials)
    if split != 'training':
        frozen = json.loads((root / 'freeze.json').read_text())
        for label, trial in trials.items():
            if trial[0]['config_sha256'] != frozen['configs'][label]:
                raise ValueError('reported trial differs from frozen configuration')
    for t in trials.values():
        add_free_metrics(t)
    records = []
    for label, trial in trials.items():
        if set(trial[2]) != expected:
            raise ValueError('report episode set differs from manifest')
        for name, ep in trial[2].items():
            path = Path(ep['meta']['input_file'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != ep['meta']['input_sha256']:
                raise ValueError('input changed')
            _, grid, _, states = replay.load_hdf5_episode(path)
            rows = pq.read_table(ep['folder'] / 'sim_step_states.parquet').to_pylist()
            recomputed = replay._metric_for_rows(states, rows, grid[0], grid[-1], 0, 0, 'physics_time')['joints']
            for joint, metrics in recomputed.items():
                for metric in refine.METRICS:
                    if not np.isclose(metrics[metric], ep['metrics'][joint][metric], rtol=1e-10, atol=1e-10):
                        raise ValueError(f'saved metric mismatch: {label}/{name}/{joint}/{metric}')
                    records.append({'candidate': label, 'task': manifest['inputs'][name]['task'],
                                    'episode': name, 'joint': joint, 'metric': metric,
                                    'value': metrics[metric]})
    with (root / f'per_joint_{split}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    values = {label: task_scores(t, manifest) for label, t in trials.items()}
    write_json(root / f'scores_{split}.json', values)
    free_values = {label: task_scores(t, manifest, free_grippers=True) for label, t in trials.items()}
    write_json(root / f'free_scores_{split}.json', free_values)
    changes = []
    for label in labels[1:]:
        for task in values[labels[0]]:
            for j in replay.JOINTS:
                for basis, scores in [('full', values), *([('free', free_values)] if 'gripper' in j else [])]:
                    for metric in refine.METRICS:
                        b, c = scores[labels[0]][task][j][metric], scores[label][task][j][metric]
                        changes.append({'candidate': label, 'task': task, 'joint': j, 'basis': basis,
                                        'metric': metric, 'baseline': b, 'value': c, 'difference': c - b,
                                        'change_percent': (c / b - 1) * 100 if b > 1e-12 else None,
                                        'any_increase': c > b})
    if changes:
        with (root / f'comparison_{split}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
            writer.writeheader()
            writer.writerows(changes)
    write_json(root / f'gripper_partitions_{split}.json', {
        label: {name: {mode: ep[f'{mode}_metrics'] for mode in ('free', 'contact')}
                for name, ep in t[2].items()} for label, t in trials.items()})
    groups = ('arm_left', 'arm_right', 'gripper_left', 'gripper_right', 'torso', 'head')
    lines = [f'# Cross-task {split} results', '',
             'Full episodes; equal task weighting. Angular position/velocity are rad and rad/s; '
             'grippers are percentage points and percentage points/s. No fitted time shift. '
             'All saved metrics were recomputed from physics feedback.', '',
             '| Task | Group | Candidate | Position RMSE | Velocity RMSE |',
             '| --- | --- | --- | ---: | ---: |']
    for task in values[labels[0]]:
        for group in groups:
            for label in labels:
                scores = [float(np.sqrt(np.mean([v[m] ** 2 for j, v in values[label][task].items() if group in j])))
                          for m in refine.METRICS]
                lines.append(f'| {task} | {group} | {label} | {scores[0]:.6f} | {scores[1]:.6f} |')
    regressions = {}
    for label in labels[1:]:
        failures = []
        for task in values[labels[0]]:
            for j in replay.JOINTS:
                floors = (.01, .1) if 'gripper' in j else (.0001, .002)
                for m, floor in zip(refine.METRICS, floors):
                    b, c = free_values[labels[0]][task][j][m], free_values[label][task][j][m]
                    if c > b * 1.01 + floor:
                        failures.append({'task': task, 'joint': j, 'metric': m, 'baseline': b, 'candidate': c})
        regressions[label] = failures
    write_json(root / f'regressions_{split}.json', regressions)
    lines += ['', '## Free-motion grippers', '',
              '| Task | Gripper | Candidate | Position RMSE | Velocity RMSE |',
              '| --- | --- | --- | ---: | ---: |']
    for task in free_values[labels[0]]:
        for j in replay.JOINTS:
            if 'gripper' in j:
                for label in labels:
                    v = free_values[label][task][j]
                    lines.append(f'| {task} | {j} | {label} | {v["position_rmse"]:.6f} | {v["velocity_rmse"]:.6f} |')
    lines += ['', '## Regression gate', '',
              'Angular joints use full episodes; grippers use the predeclared free-motion partition. '
              'Contact-like intervals are a heuristic based only on recorded signals, not labeled contact truth.', '',
              'The gate allows 1% plus numerical floors: angular 0.0001 rad / 0.002 rad/s; '
              'gripper 0.01 percentage points / 0.1 percentage points/s. '
              'Passing this relative gate is not an absolute alignment claim.', '']
    for label, failures in regressions.items():
        lines.append(f'- {label}: {len(failures)} task/joint/metric regressions beyond the gate.')
    (root / f'report_{split}.md').write_text('\n'.join(lines) + '\n')
    print({label: len(v) for label, v in regressions.items()})


def plot(root, labels, split):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str((root / 'plot_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = json.loads((root / 'episodes.json').read_text())
    episodes = [Path(p) for p in manifest[split]]
    colors = ['#cf8032', '#167d57', '#7552ac', '#b33142']
    if len(labels) > len(colors):
        raise ValueError('plot at most four candidates at once for legible curves')
    for kind in ('grippers', 'arms'):
        fig, axes = plt.subplots(len(episodes), 2, figsize=(16, 2.5 * len(episodes)), squeeze=False)
        for row, path in enumerate(episodes):
            _, grid, command, real = replay.load_hdf5_episode(path)
            base_folder = root / f'{labels[0]}_{split}' / 'replay' / path.stem
            metric = json.loads((base_folder / 'tracking_error.json').read_text())['physics_time']['joints']
            for col, side in enumerate(('left', 'right')):
                if kind == 'grippers':
                    joint = f'astribot_gripper_{side}_joint_1'
                    scale, unit = 1., 'percentage points'
                else:
                    joint = max((j for j in replay.JOINTS if f'arm_{side}' in j), key=lambda j: metric[j]['position_rmse'])
                    scale, unit = 180 / np.pi, 'degrees'
                index, state = replay.JOINTS.index(joint), real[joint]
                ax = axes[row, col]
                ax.plot(grid - grid[0], command[:, index] * scale, '--', color='#929ba5', lw=.7, label='recorded command')
                ax.plot(state[:, 0] - grid[0], state[:, 1] * scale, color='#284d80', lw=1., label='real feedback')
                for label, color in zip(labels, colors):
                    folder = root / f'{label}_{split}' / 'replay' / path.stem
                    data = pq.read_table(folder / 'sim_step_states.parquet', columns=['joint', 'physics_time_s', 'position'],
                                         filters=[('joint', '=', joint)]).to_pydict()
                    ax.plot(data['physics_time_s'], np.asarray(data['position']) * scale, color=color, lw=.8, label=label)
                if kind == 'grippers':
                    mask = free_motion_mask(grid, command[:, index], state)
                    ax.fill_between(grid - grid[0], 0, 100, where=~mask, color='#d7b16a', alpha=.18, label='contact-like excluded from fit')
                ax.set_title(f'{manifest["inputs"][path.stem]["task"]}: {joint.removeprefix("astribot_")}')
                ax.set_xlim(0, grid[-1] - grid[0])
                ax.set_ylabel(unit)
                ax.grid(alpha=.2)
                if row == 0:
                    ax.legend(fontsize=7)
        axes[-1, 0].set_xlabel('source-relative time (s)')
        axes[-1, 1].set_xlabel('source-relative time (s)')
        fig.suptitle(f'{split}: same recorded commands, no fitted time shift', fontsize=13)
        fig.tight_layout()
        fig.savefig(root / f'{kind}_{split}.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'run', 'scores', 'fit', 'select', 'report', 'freeze', 'plot'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('/mnt/nfs_share/temp-data/real_eval_records'))
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/joint_alignment/penholder_250hz/right_gripper_candidate_250hz_v3.yaml')
    parser.add_argument('--labels', nargs='+', default=['v3'])
    parser.add_argument('--split', choices=['training', 'validation', 'unseen_tasks', 'gpu_validation'], default='training')
    parser.add_argument('--followup-splits', nargs='*', choices=['validation', 'unseen_tasks', 'gpu_validation'], default=[])
    parser.add_argument('--domain', type=int, default=201)
    parser.add_argument('--output', default='proposal')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.root, args.source_root, args.baseline)
    elif args.operation == 'run':
        for split in [args.split, *args.followup_splits]:
            run(args.root, args.labels, split, args.domain)
    elif args.operation in ('fit', 'select'):
        globals()[args.operation](args.root, args.labels, args.output)
    elif args.operation == 'report':
        report(args.root, args.labels, args.split)
    elif args.operation == 'freeze':
        freeze(args.root, args.labels)
    elif args.operation == 'plot':
        plot(args.root, args.labels, args.split)
    else:
        manifest = json.loads((args.root / 'episodes.json').read_text())
        values = {label: task_scores(refine.read_trial(args.root / f'{label}_{args.split}'), manifest)
                  for label in args.labels}
        write_json(args.root / f'scores_{args.split}.json', values)
        for label, tasks in values.items():
            for task, joints in tasks.items():
                print(label, task, {group: [round(float(np.sqrt(np.mean([v[m] ** 2 for j, v in joints.items() if group in j]))), 5)
                                           for m in refine.METRICS]
                                    for group in ('arm_left', 'arm_right', 'gripper_left', 'gripper_right')})


if __name__ == '__main__':
    main()
