"""Paired material tracking and intervention audits for reaction-shell trials."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.analyze_tshirt_sweep import analyze
from scripts.diagnostics.export_tshirt_sweep_videos import validate, comparison
from src.evaluation.cloth_grasp_diagnostics import compile_finger_hulls
from src.simulation_config.glb_geometry import rotation_matrix

FINGERS = ['astribot_gripper_right_Link_' + s for s in ('L11', 'R11')]


def trace(directory):
    status = json.loads((directory/'run_status.json').read_text())
    rows = [json.loads(line) for line in Path(status['trace']).read_text().splitlines()]
    return {r['step_id']: r for r in rows if r['event'] == 'state'}, {
        r['step_id']: r['commands'] for r in rows if r['event'] == 'command'}


def body(snapshot, name):
    return snapshot['body_poses'][snapshot['body_names'].tolist().index(name)]


def patch_metrics(directory, patch):
    start = np.load(directory/'particles/000930.npz')
    end = np.load(directory/'particles/001230.npz')
    wrist0, wrist1 = [body(a, 'astribot_arm_right_link_7') for a in (start, end)]
    relative = [(a['points'][patch]-p[:3]) @ rotation_matrix(p[3:])
                for a, p in ((start, wrist0), (end, wrist1))]
    dz = (end['points'][patch, 2] - start['points'][patch, 2]) * 1000
    return {'patch_particles': len(patch), 'patch_lift_mm': float(np.median(dz)),
            'patch_lift_p10_p90_mm': np.percentile(dz, [10, 90]).tolist(),
            'patch_slip_mm': float(np.median(np.linalg.norm(relative[1]-relative[0], axis=1)) * 1000)}


def patch_series(directory, patch):
    anchor = np.load(directory/'particles/000930.npz')['points']
    result = []
    for path in sorted((directory/'particles').glob('*.npz')):
        step = int(path.stem)
        if step >= 930:
            points = np.load(path)['points']
            result.append({'step': step, 'time_s': step*.01,
                           'patch_lift_mm': float(np.median(points[patch, 2]-anchor[patch, 2])*1000)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--videos', action='store_true')
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out/'manifest.json').read_text())
    summaries, states, commands, hulls = {}, {}, {}, {}
    for condition in manifest['conditions']:
        name = condition['name']
        directory = out/name
        status_path = directory/'run_status.json'
        if args.partial and (not status_path.exists() or json.loads(status_path.read_text())['status'] != 'completed'):
            continue
        summaries[name] = analyze(directory)
        summaries[name]['note'] = (
            'Trial-specific patch: 15 mm region selected at step 850. Shared material patch: '
            'art_shell reference IDs reused only after verifying identical initial particle arrays. '
            'Reaction is active during approach, so preclosure states need not match. '
            'Lift/slip measure steps 930 to 1230. Candidates use a 6 mm geometric contact band '
            'with each trial own anchor IDs; they are not measured forces or task success. '
            'Shell experiments use measured shell poses and different native solver participation.')
        metrics = json.loads((directory/'metrics.json').read_text())
        summaries[name]['final_candidates'] = metrics[-1]['bilateral_candidates']
        states[name], commands[name] = trace(directory)
        status = json.loads(status_path.read_text())
        geometry = json.loads((Path(status['trace']).parent/'collision_geometry.json').read_text())
        hulls[name] = {f: compile_finger_hulls(geometry['bodies'][f]) for f in FINGERS}
        if args.videos:
            validate(directory)
        shell_samples = [r['diagnostics']['reaction_shell'] for r in states[name].values()
                         if 'reaction_shell' in r['diagnostics']]
        if shell_samples:
            summaries[name]['max_shell_translation_error_mm'] = max(
                max(r['translation_error_m']) for r in shell_samples) * 1000
            angles = []
            for r in shell_samples:
                a = np.asarray(r['original_pose_world'])[:, 3:]
                b = np.asarray(r['shell_pose_world'])[:, 3:]
                dot = np.abs(np.sum(a*b, axis=1)) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
                angles.extend(np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.))))
            summaries[name]['max_shell_rotation_error_deg'] = float(max(angles))
        if 'gap_reference_case' in condition:
            reference_states, _ = trace(out/condition['gap_reference_case'])
            errors = [(states[name][i]['diagnostics']['cloth_proximity']['tshirt']['right']['closing_axis_gap_m'] -
                       reference_states[i]['diagnostics']['cloth_proximity']['tshirt']['right']['closing_axis_gap_m'])*1000
                      for i in range(930, 1231)]
            feedback = json.loads((directory/'gap_feedback.json').read_text())
            summaries[name]['gap_control'] = {
                'lift_hold_rms_error_mm': float(np.sqrt(np.mean(np.square(errors)))),
                'lift_hold_max_error_mm': float(np.max(np.abs(errors))),
                'max_command_percent': max(r['command_percent'] for r in feedback),
                'command_at_930_percent': next(r['command_percent'] for r in feedback if r['step'] == 930),
                'saturated_steps': sum(r['command_percent'] == 100. for r in feedback)}
        print(name, {k: summaries[name][k] for k in ('motion_valid', 'closed_gap_mm',
              'patch_lift_mm', 'patch_slip_mm', 'retained_candidates')}, flush=True)
    controls = []
    repeats = []
    if 'art_shell' in summaries:
        ref = 'art_shell'
        initial = np.load(out/ref/'particles/000000.npz')['points']
        approach = np.load(out/ref/'particles/000850.npz')['points']
        patch = np.array(summaries[ref]['patch_ids'])
        for name, summary in summaries.items():
            other_initial = np.load(out/name/'particles/000000.npz')['points']
            other_approach = np.load(out/name/'particles/000850.npz')['points']
            same_initial = np.array_equal(initial, other_initial)
            audit = {'case': name, 'initial_particles_bitwise_equal': same_initial,
                     'particle_count': len(other_initial),
                     'commands_equal_through_close_hold': all(commands[name][i] == commands[ref][i]
                                                              for i in range(1, 931)),
                     'geometry_max_vertex_error_m': max(
                         np.max(np.abs(a['vertices']-b['vertices']))
                         for f in FINGERS for a, b in zip(hulls[ref][f], hulls[name][f])),
                     'hull_counts_equal': all(len(hulls[name][f]) == len(hulls[ref][f]) for f in FINGERS)}
            if same_initial:
                audit['preclose_particle_max_difference_mm'] = float(np.max(np.linalg.norm(
                    other_approach-approach, axis=1)) * 1000)
                audit['preclose_patch_median_difference_mm'] = float(np.median(np.linalg.norm(
                    other_approach[patch]-approach[patch], axis=1)) * 1000)
                summary['shared_material_patch'] = patch_metrics(out/name, patch)
                (out/name/'shared_patch_metrics.json').write_text(json.dumps(patch_series(out/name, patch), indent=2)+'\n')
            for step in (850, 930):
                a = np.load(out/ref/f'particles/{step:06d}.npz')
                b = np.load(out/name/f'particles/{step:06d}.npz')
                audit[f'wrist_offset_at_{step}_mm'] = float(np.linalg.norm(
                    body(a, 'astribot_arm_right_link_7')[:3]-body(b, 'astribot_arm_right_link_7')[:3])*1000)
            # Joint arrays retain the same named DOFs in both articulation topologies.
            joint_diff = []
            for step in range(931):
                a, b = states[ref][step]['diagnostics'], states[name][step]['diagnostics']
                if a['joint_names'] != b['joint_names']:
                    raise ValueError('Joint orders changed; explicit mapping required')
                joint_diff.append(np.max(np.abs(np.array(a['joints']['position'])-b['joints']['position'])))
            audit['prelift_max_joint_difference_rad'] = float(max(joint_diff))
            controls.append(audit)
    for base, repeat in [('art_shell', 'art_shell_repeat2'), ('rigid_shell', 'rigid_shell_repeat2'),
                         ('art_shell', 'art_shell_gap_matched'),
                         ('rigid_shell_gap_matched', 'rigid_shell_gap_matched_repeat2')]:
        if base not in summaries or repeat not in summaries:
            continue
        a_files = sorted((out/base/'particles').glob('*.npz'))
        equal_particles = equal_bodies = True
        for path in a_files:
            a, b = np.load(path), np.load(out/repeat/'particles'/path.name)
            equal_particles &= np.array_equal(a['points'], b['points'])
            equal_bodies &= np.array_equal(a['body_poses'], b['body_poses'])
        repeats.append({'base': base, 'repeat_or_sham': repeat,
                        'sampled_boundaries': len(a_files), 'particles_bitwise_equal': bool(equal_particles),
                        'body_poses_bitwise_equal': bool(equal_bodies),
                        'all_commands_equal': commands[base] == commands[repeat]})
    (out/'reaction_summary.json').write_text(json.dumps(list(summaries.values()), indent=2)+'\n')
    for name, summary in summaries.items():
        (out/name/'analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'reaction_controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    (out/'repeatability.json').write_text(json.dumps(repeats, indent=2)+'\n')
    fields = ['case', 'motion_valid', 'closed_gap_mm', 'wrist_lift_mm', 'initial_candidates',
              'retained_candidates', 'final_candidates', 'patch_lift_mm', 'patch_slip_mm',
              'shared_patch_lift_mm', 'shared_patch_slip_mm', 'max_shell_translation_error_mm']
    with (out/'reaction_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for summary in summaries.values():
            shared = summary.get('shared_material_patch', {})
            writer.writerow({**summary, 'shared_patch_lift_mm': shared.get('patch_lift_mm'),
                             'shared_patch_slip_mm': shared.get('patch_slip_mm')})
    if args.videos and all(n in summaries for n in ('native_reference', 'art_shell', 'rigid_shell')):
        cases = ['native_reference', 'art_shell', 'rigid_shell']
        if 'rigid_shell_repeat2' in summaries:
            cases.append('rigid_shell_repeat2')
        comparison(out, 'reaction_comparison', cases)
    if args.videos and all(n in summaries for n in ('art_shell_gap_matched', 'rigid_shell_gap_matched')):
        cases = ['art_shell_gap_matched', 'rigid_shell_gap_matched']
        if 'rigid_shell_gap_matched_repeat2' in summaries:
            cases.append('rigid_shell_gap_matched_repeat2')
        comparison(out, 'matched_gap_comparison', cases)
    if summaries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for name in summaries:
            metrics = json.loads((out/name/'metrics.json').read_text())
            for ax, key in zip(axes, ('gap_mm', 'patch_lift_mm', 'retained_candidates')):
                plot_metrics = (json.loads((out/name/'shared_patch_metrics.json').read_text())
                                if key == 'patch_lift_mm' and 'shared_material_patch' in summaries[name] else metrics)
                ax.plot([r['time_s'] for r in plot_metrics], [r[key] for r in plot_metrics], label=name)
                ax.set_xlabel('Simulation time (s)')
                ax.set_ylabel(key)
                ax.grid(alpha=.25)
        axes[1].set_ylabel('Shared material patch lift (mm)')
        axes[0].set_ylabel('Actual finger gap (mm)')
        axes[2].set_ylabel('Retained own-anchor candidates')
        axes[-1].legend(fontsize=7)
        fig.savefig(out/'reaction_comparison.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
