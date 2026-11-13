#!/usr/bin/env python3
"""Fit bounded per-joint PD/filter proposals from SDK/ROS physics-step trials.

Finite differences are a local proposal model, not a claim of actuator parameter
identification. Every proposed configuration must be rerun through the simulator.
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

from replay_artifacts_sdk_100hz import load_hdf5_episode

JOINTS = [f'astribot_arm_{side}_joint_{i}' for side in ('left', 'right') for i in range(1, 8)]
JOINTS += [f'astribot_torso_joint_{i}' for i in range(1, 4)]
VELOCITY_WEIGHT_S = .02


def control_key(joint):
    return joint.removeprefix('astribot_') + '_control'


def curves(folder):
    meta = json.loads((folder / 'run_metadata.json').read_text())
    if meta['status'] != 'complete' or meta.get('time_mode') != 'physics_step':
        raise ValueError(f'{folder}: completed physics-step trial required')
    data = pq.read_table(folder / 'sim_step_states.parquet').to_pydict()
    names = np.asarray(data['joint'])
    t = np.asarray(data['physics_time_s'])
    q, v = np.asarray(data['position']), np.asarray(data['velocity'])
    return meta, {j: np.column_stack((t[names == j], q[names == j], v[names == j])) for j in JOINTS}


def fit(baseline, variants, base_config):
    residuals, jacobians = {j: [] for j in JOINTS}, {j: [] for j in JOINTS}
    provenance = []
    for folder in sorted((baseline / 'replay').iterdir()):
        if not folder.is_dir():
            continue
        meta, base = curves(folder)
        _, grid, _, real = load_hdf5_episode(Path(meta['input_file']))
        alternatives = [curves(v / 'replay' / folder.name) for v in variants]
        if any(m['input_sha256'] != meta['input_sha256'] for m, _ in alternatives):
            raise ValueError('finite difference trials have different source data')
        provenance.append({'episode': folder.name, 'input_sha256': meta['input_sha256']})
        for joint in JOINTS:
            values = base[joint]
            source_t = grid[0] + values[:, 0]
            keep = source_t <= grid[-1]
            target = np.column_stack([np.interp(source_t, real[joint][:, 0], real[joint][:, i])
                                      for i in (1, 2)])
            weights = np.array([1., VELOCITY_WEIGHT_S])
            residuals[joint].append(((values[keep, 1:] - target[keep]) * weights).ravel())
            columns = []
            for _, alternative in alternatives:
                if not np.array_equal(alternative[joint][:, 0], values[:, 0]):
                    raise ValueError('physics-step grids differ')
                columns.append(((alternative[joint][keep, 1:] - values[keep, 1:]) * weights).ravel())
            jacobians[joint].append(np.column_stack(columns))
    if not provenance:
        raise ValueError('no training episodes')
    config, report = copy.deepcopy(base_config), {}
    for joint in JOINTS:
        residual = np.concatenate(residuals[joint])
        jacobian = np.concatenate(jacobians[joint])
        # Restrict extrapolation; double-P/double-D and 20 ms LPF are measured probes.
        scale = max(float(np.linalg.norm(jacobian)), 1e-9)
        solution = lsq_linear(np.vstack((jacobian, np.eye(3) * scale * .001)),
                              np.concatenate((-residual, np.zeros(3))),
                              bounds=([-.5, -.5, 0.], [1., 1., 2.]))
        coeff = solution.x
        key = control_key(joint)
        config[key]['stiffness'] = float(base_config[key]['stiffness'] * (1 + coeff[0]))
        config[key]['damping'] = float(base_config[key]['damping'] * (1 + coeff[1]))
        config['joint_command_filters'][joint]['time_constant_s'] = float(.02 * coeff[2])
        report[joint] = {'stiffness': config[key]['stiffness'], 'damping': config[key]['damping'],
                         'time_constant_s': float(.02 * coeff[2]),
                         'baseline_objective': float(np.mean(residual ** 2)),
                         'predicted_objective': float(np.mean((residual + jacobian @ coeff) ** 2)),
                         'coefficients': coeff.tolist()}
    return config, {'training': provenance, 'velocity_weight_s': VELOCITY_WEIGHT_S,
                    'method': 'bounded local finite-difference proposal; requires physical replay validation',
                    'joints': report}


def select_completed_trials(trials, configs):
    """Select on training episodes only; a mixed configuration still needs replay."""
    scores = {}
    identity = None
    for label, root in trials.items():
        metrics, sources = [], {}
        for path in sorted((root / 'replay').glob('*/run_metadata.json')):
            meta = json.loads(path.read_text())
            if meta.get('time_mode') != 'physics_step' or meta['status'] != 'complete':
                raise ValueError('selection requires completed physics-step trials')
            sources[path.parent.name] = meta['input_sha256']
            metrics.append(json.loads((path.parent / 'tracking_error.json').read_text())['physics_time']['joints'])
        if not metrics or (identity is not None and identity != sources):
            raise ValueError('selection trials must use the same complete training episodes')
        identity = sources
        scores[label] = {j: {key: float(np.sqrt(np.mean([m[j][key] ** 2 for m in metrics])))
                            for key in ('position_rmse', 'velocity_rmse')} for j in JOINTS}
    config, selected = copy.deepcopy(configs['baseline']), {}
    for joint in JOINTS:
        base = scores['baseline'][joint]
        eligible = [label for label in trials if all(scores[label][joint][key] <= base[key] for key in base)]
        best = min(eligible, key=lambda label: scores[label][joint]['position_rmse'] ** 2
                   + VELOCITY_WEIGHT_S ** 2 * scores[label][joint]['velocity_rmse'] ** 2)
        key = control_key(joint)
        config[key] = copy.deepcopy(configs[best][key])
        config['joint_command_filters'][joint] = copy.deepcopy(configs[best]['joint_command_filters'][joint])
        selected[joint] = {'selected': best, 'baseline': base, 'selected_trial': scores[best][joint]}
    return config, {'training_sources': identity, 'joints': selected,
                    'rule': 'Minimum weighted error among candidates improving both training position and velocity RMS; validate combination independently.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sim-python', type=Path, required=True)
    parser.add_argument('--sdk-python', type=Path, default=Path('/usr/bin/python3'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(args.config.read_text())
    episodes = [Path(json.loads(p.read_text())['input_file'])
                for p in sorted((args.baseline / 'replay').glob('*/run_metadata.json'))]
    if not episodes:
        raise ValueError('baseline has no completed episodes')
    runner = Path(__file__).with_name('run_trial.py')
    def run(config_path, output):
        subprocess.run([sys.executable, str(runner), '--config', str(config_path), '--output', str(output),
                        '--sim-python', str(args.sim_python), '--sdk-python', str(args.sdk_python),
                        '--timeout', '900', *map(str, episodes)], check=True)
    variants = []
    configs, trials = {'baseline': base_config}, {'baseline': args.baseline}
    for name in ('stiffness', 'damping', 'filter'):
        config = copy.deepcopy(base_config)
        for joint in JOINTS:
            if name == 'filter':
                config['joint_command_filters'][joint]['time_constant_s'] = .02
            else:
                config[control_key(joint)][name] *= 2
        path = args.output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[name] = config
        trial = args.output / name
        if not trial.exists():
            run(path, trial)
        elif json.loads((trial / 'trial.json').read_text())['status'] != 'complete':
            raise ValueError(f'incomplete trial must be reviewed: {trial}')
        elif json.loads((trial / 'trial.json').read_text()).get('config_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'trial configuration snapshot does not match: {trial}')
        variants.append(trial)
        trials[name] = trial
    config, report = fit(args.baseline, variants, base_config)
    proposal = args.output / 'proposal.yaml'
    proposal.write_text(yaml.safe_dump(config, sort_keys=False))
    (args.output / 'fit.json').write_text(json.dumps(report, indent=2) + '\n')
    run(proposal, args.output / 'proposal')
    configs['proposal'], trials['proposal'] = config, args.output / 'proposal'
    selected, selection = select_completed_trials(trials, configs)
    final = args.output / 'selected.yaml'
    final.write_text(yaml.safe_dump(selected, sort_keys=False))
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    run(final, args.output / 'selected')


if __name__ == '__main__':
    main()
