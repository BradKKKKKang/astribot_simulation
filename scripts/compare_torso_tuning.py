#!/usr/bin/env python3
"""Compare isolated torso trials at the recorded physics timestamps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

JOINT = 'astribot_torso_joint_1'


def collect(root):
    rows = []
    for path in sorted(root.glob('*/*/tracking_error.json')):
        folder = path.parent
        errors = json.loads(path.read_text())
        meta = json.loads((folder / 'run_metadata.json').read_text())
        settings = meta['pd_torso_joint_1']
        sim = np.load(folder / 'sim_states.npz')
        real = np.load(folder / 'real_states.npz')
        i = sim['joint_names'].tolist().index(JOINT)
        r = real['joint_names'].tolist().index(JOINT)
        values = real['values'][real['offsets'][r]:real['offsets'][r + 1]]
        t = sim['artifact_time_s']
        mask = (t >= values[0, 0]) & (t <= min(values[-1, 0], meta['command_time_end_s']))
        real_q = np.interp(t[mask], values[:, 0], values[:, 1])
        q = sim['position'][mask, i]
        trace = np.load(folder / 'torso_control_trace.npz')['values']
        others = [v for k, v in errors['joints'].items() if k != JOINT and 'gripper' not in k]
        rows.append(dict(
            trial=folder.parent.name, episode=folder.name,
            **settings,
            position_rmse_deg=np.rad2deg(errors['joints'][JOINT]['position']['rmse']),
            position_max_deg=np.rad2deg(errors['joints'][JOINT]['position']['max_abs']),
            velocity_rmse_rad_s=errors['joints'][JOINT]['velocity']['rmse'],
            mean_error_deg=np.rad2deg(np.mean(q - real_q)),
            simulated_span_deg=np.rad2deg(np.ptp(q)), real_span_deg=np.rad2deg(np.ptp(real_q)),
            other_angular_rmse_rad=float(np.sqrt(np.mean([v['position']['rmse'] ** 2 for v in others]))),
            estimated_saturation_fraction=float(np.mean(np.abs(trace[:, 1]) >= settings['effort_limit'])),
            max_abs_gravity_feedforward_Nm=float(np.abs(trace[:, 0]).max()),
            deadline_overruns=meta['deadline_completion_overruns'],
            physics_steps=meta['physics_steps'], command_frames=meta['command_frames'],
        ))
    return rows


def plot(root, rows, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    episodes = sorted({row['episode'] for row in rows})
    fig, axes = plt.subplots(len(episodes), 2, figsize=(14, 3.6 * len(episodes)), squeeze=False)
    for n, episode in enumerate(episodes):
        for j, label in enumerate(labels):
            folder = root / label / episode
            if not (folder / 'sim_states.npz').exists():
                continue
            s = np.load(folder / 'sim_states.npz')
            r = np.load(folder / 'real_states.npz')
            idx = s['joint_names'].tolist().index(JOINT)
            ridx = r['joint_names'].tolist().index(JOINT)
            v = r['values'][r['offsets'][ridx]:r['offsets'][ridx + 1]]
            meta = json.loads((folder / 'run_metadata.json').read_text())
            t = s['artifact_time_s']
            mask = (t >= v[0, 0]) & (t <= min(v[-1, 0], meta['command_time_end_s']))
            real_q = np.interp(t[mask], v[:, 0], v[:, 1])
            if j == 0:
                axes[n, 0].plot(t[mask] - t[0], np.rad2deg(real_q), 'k', linewidth=2, label='Real feedback')
                c = np.load(folder / 'command_trace.npz')
                axes[n, 0].plot(c['artifact_time_s'] - t[0], np.rad2deg(c['position'][:, idx]),
                                color='gray', linestyle=':', label='Position command')
            axes[n, 0].plot(t[mask] - t[0], np.rad2deg(s['position'][mask, idx]), label=label, alpha=0.85)
            axes[n, 1].plot(t[mask] - t[0], np.rad2deg(s['position'][mask, idx] - real_q), label=label)
        axes[n, 0].set_title(episode + ' — torso_joint_1')
        axes[n, 1].set_title('Simulation minus real feedback')
        axes[n, 0].set_ylabel('Joint position (deg)')
        axes[n, 1].set_ylabel('Position error (deg)')
        axes[n, 1].axhline(0, color='black', linewidth=0.7)
        for ax in axes[n]:
            ax.set_xlabel('Replay time (s)')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--plot', nargs='*', default=[])
    args = p.parse_args()
    rows = collect(args.root)
    if not rows:
        raise ValueError('No completed trials')
    with (args.root / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('trial / episode: position RMSE deg, max deg, velocity RMSE rad/s, mean error deg')
    for r in rows:
        print(f"{r['trial']} / {r['episode']}: {r['position_rmse_deg']:.4f}, "
              f"{r['position_max_deg']:.4f}, {r['velocity_rmse_rad_s']:.5f}, {r['mean_error_deg']:.4f}")
    if args.plot:
        plot(args.root, rows, args.plot)


if __name__ == '__main__':
    main()
