#!/usr/bin/env python3
"""Compare complete physics or SDK replays at their recorded timestamps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from replay_artifacts_data import JOINTS, load_episode

FOCUS = ('astribot_torso_joint_1', 'astribot_torso_joint_2', 'astribot_torso_joint_3',
         'astribot_gripper_left_joint_1', 'astribot_gripper_right_joint_1')


def compare(runs):
    rows, curves, sources = [], {}, {}
    episodes, episode_hashes = {}, {}
    for label, root in runs.items():
        paths = sorted(root.glob('*/run_metadata.json'))
        if not paths:
            raise ValueError(f'No completed episodes under {root}')
        for path in paths:
            folder, episode = path.parent, path.parent.name
            meta = json.loads(path.read_text())
            if meta['status'] not in ('complete', 'trajectory_complete'):
                raise ValueError(f'Incomplete episode: {path}')
            source = Path(meta['input_parquet'])
            if source not in episodes:
                if hashlib.sha256(source.read_bytes()).hexdigest() != meta['input_sha256']:
                    raise ValueError(f'Replay input changed: {source}')
                episodes[source] = load_episode(source)
            if episode_hashes.setdefault(episode, meta['input_sha256']) != meta['input_sha256']:
                raise ValueError(f'Runs use different inputs for {episode}')
            grid, command, real, _ = episodes[source]
            errors = json.loads((folder / 'tracking_error.json').read_text())
            sdk = (folder / 'sim_states.parquet').exists()
            if sdk:
                observed = pq.read_table(folder / 'sim_states.parquet').to_pydict()
                joint_names = np.asarray(observed['joint'])
                timestamp = grid[0] + (np.asarray(observed['header_stamp_ns'], dtype=np.int64)
                                       - meta['epoch_system_ns']) / 1e9
                saved = errors['simulator_header_time']['joints']
            else:
                observed = np.load(folder / 'sim_states.npz')
                joint_names = observed['joint_names'].tolist()
                timestamp = observed['artifact_time_s']
                saved = errors['joints']
            sources[f'{label}/{episode}'] = {'directory': str(folder),
                                           'time_basis': 'simulator_header_time' if sdk else 'physics_time',
                                           'input_sha256': meta['input_sha256']}
            for j, joint in enumerate(JOINTS):
                if sdk:
                    indices = joint_names == joint
                    t = timestamp[indices]
                    q = np.asarray(observed['position'])[indices]
                    v = np.asarray(observed['velocity'])[indices]
                else:
                    index = joint_names.index(joint)
                    t, q, v = timestamp, observed['position'][:, index], observed['velocity'][:, index]
                r = real[joint]
                valid = ((t >= max(grid[0], r[0, 0])) & (t <= min(grid[-1], r[-1, 0]))
                         & np.isfinite(q) & np.isfinite(v))
                t, q, v = t[valid], q[valid], v[valid]
                if not len(t):
                    raise ValueError(f'No overlap: {label}/{episode}/{joint}')
                real_q = np.interp(t, r[:, 0], r[:, 1])
                real_v = np.interp(t, r[:, 0], r[:, 2])
                dq, dv = q - real_q, v - real_v
                q_rmse, v_rmse = float(np.sqrt(np.mean(dq**2))), float(np.sqrt(np.mean(dv**2)))
                stored = saved[joint]
                expected_q = stored['position_rmse'] if sdk else stored['position']['rmse']
                expected_v = stored['velocity_rmse'] if sdk else stored['velocity']['rmse']
                samples = stored['samples'] if sdk else stored['position']['samples']
                if (samples != len(t) or not np.allclose([q_rmse, v_rmse], [expected_q, expected_v],
                                                       rtol=1e-9, atol=1e-10)):
                    raise ValueError(f'Recomputed metrics disagree: {label}/{episode}/{joint}')
                factor = 1.0 if 'gripper' in joint else 180 / np.pi
                rows.append(dict(
                    run=label, episode=episode, joint=joint, samples=len(t),
                    position_unit='percentage_points' if 'gripper' in joint else 'deg',
                    position_rmse=q_rmse * factor,
                    position_bias=float(dq.mean()) * factor,
                    position_p95=float(np.percentile(np.abs(dq), 95)) * factor,
                    position_max=float(np.max(np.abs(dq))) * factor,
                    peak_time_s=float(t[np.argmax(np.abs(dq))] - grid[0]),
                    velocity_unit='percentage_points/s' if 'gripper' in joint else 'rad/s',
                    velocity_rmse=v_rmse,
                    simulated_span=float(np.ptp(q)) * factor,
                    real_span=float(np.ptp(real_q)) * factor,
                ))
                if joint in FOCUS:
                    curves[label, episode, joint] = (t - grid[0], q * factor, real_q * factor,
                                                     grid - grid[0], command[:, j] * factor)
    return rows, curves, sources


def write_report(output, rows, curves, sources, reference):
    output.mkdir(parents=True, exist_ok=True)
    before = {(r['episode'], r['joint']): r for r in rows if r['run'] == reference}
    for row in rows:
        baseline = before[row['episode'], row['joint']]
        row['position_rmse_delta'] = row['position_rmse'] - baseline['position_rmse']
        row['velocity_rmse_delta'] = row['velocity_rmse'] - baseline['velocity_rmse']
    with (output / 'per_joint.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'per_joint.json').write_text(json.dumps(dict(
        comparison='Real feedback interpolated at recorded simulator times; full overlap including startup; no fitted time shift',
        reference=reference, sources=sources, joints=rows), indent=2) + '\n')
    episodes = sorted({row['episode'] for row in rows})
    labels = list(dict.fromkeys(row['run'] for row in rows))
    lines = ['# Torso and gripper replay alignment', '',
             'Real feedback is interpolated at recorded physics or simulator header timestamps. '
             'Startup samples are retained; no time shift is fitted. Angular position uses degrees; '
             'grippers use percentage points. Torque is diagnostic only.', '',
             '| Run | Episode | Joint | Position RMSE | Bias | Max | Velocity RMSE |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['joint'] in FOCUS:
            lines.append(f"| {row['run']} | {row['episode'].removeprefix('replay_train_')} | "
                         f"{row['joint'].removeprefix('astribot_')} | {row['position_rmse']:.3f} | "
                         f"{row['position_bias']:+.3f} | {row['position_max']:.3f} | {row['velocity_rmse']:.5f} |")
    lines.extend(['', 'All 22 joints, sample counts, p95, motion spans and reference deltas are in '
                  '`per_joint.csv` and `per_joint.json`. Every position/velocity RMSE and sample '
                  'count was checked against the saved replay error files.', ''])
    (output / 'report.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for episode in episodes:
        fig, axes = plt.subplots(len(FOCUS), 2, figsize=(14, 14), squeeze=False)
        for i, joint in enumerate(FOCUS):
            for n, label in enumerate(labels):
                t, q, real_q, ct, command = curves[label, episode, joint]
                if n == 0:
                    axes[i, 0].plot(t, real_q, 'k', lw=1.5, label='Real feedback')
                    axes[i, 0].plot(ct, command, ':', color='gray', label='Command')
                axes[i, 0].plot(t, q, lw=1, alpha=0.8, label=label)
                axes[i, 1].plot(t, q - real_q, lw=1, label=label)
            unit = 'percentage points' if 'gripper' in joint else 'deg'
            axes[i, 0].set_title(joint.removeprefix('astribot_'))
            axes[i, 1].set_title('Simulation minus real feedback')
            axes[i, 1].axhline(0, c='k', lw=0.5)
            for ax in axes[i]:
                ax.set_ylabel(unit)
                ax.set_xlabel('Replay time (s)')
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7)
        fig.suptitle(episode)
        fig.tight_layout()
        fig.savefig(output / f'{episode}.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=directory containing episode folders')
    parser.add_argument('--reference', default='baseline')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    runs = dict(item.split('=', 1) for item in args.run)
    if len(runs) != len(args.run) or args.reference not in runs:
        raise ValueError('Run labels must be unique and include the reference')
    rows, curves, sources = compare({label: Path(root) for label, root in runs.items()})
    write_report(args.output_root, rows, curves, sources, args.reference)
    for row in rows:
        if row['joint'] in FOCUS:
            print(f"{row['run']}/{row['episode']}/{row['joint']}: {row['position_rmse']:.4f} {row['position_unit']}")


if __name__ == '__main__':
    main()
