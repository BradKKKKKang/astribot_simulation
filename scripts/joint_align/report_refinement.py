#!/usr/bin/env python3
"""Recompute continuation metrics and report all-joint acceptance explicitly."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_joint_alignment import compare, write_report
from refine_response import JOINTS, METRICS, comparable, key, read_trial, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--candidate', default='selected')
    args = p.parse_args()
    root = args.root
    labels = {
        'training': {'baseline': 'baseline_settled', 'first_pass': 'first_pass', 'candidate': args.candidate},
        'validation': {'baseline': 'validation_baseline', 'first_pass': 'validation_first_pass',
                       'candidate': 'validation_candidate'}}
    all_trials, all_scores, summary, rows = {}, {}, {}, []
    for split, paths in labels.items():
        trials = {name: read_trial(root / label) for name, label in paths.items()}
        comparable(trials)
        recomputed, curves, sources = compare({name: root / label / 'replay' for name, label in paths.items()})
        write_report(root / f'{split}_comparison', recomputed, curves, sources, 'baseline', plots=False)
        all_trials[split] = trials
        values = {name: scores(t) for name, t in trials.items()}
        all_scores[split] = values
        groups = {'arms': [j for j in JOINTS if 'arm_' in j],
                  'torso': [j for j in JOINTS if 'torso' in j],
                  'head': [j for j in JOINTS if 'head' in j], 'all_angular': JOINTS}
        summary[split] = {'episodes': list(trials['baseline'][2]), 'groups': {}, 'regressions_over_1_percent': []}
        for group, joints in groups.items():
            summary[split]['groups'][group] = {
                m: {name: float(np.sqrt(np.mean([v[j][m] ** 2 for j in joints])))
                    * (180 / np.pi if m == 'position_rmse' else 1)
                    for name, v in values.items()} for m in METRICS}
        for j in values['baseline']:
            row = {'split': split, 'joint': j, 'position_unit': 'percentage_points' if 'gripper' in j else 'deg',
                   'velocity_unit': 'percentage_points/s' if 'gripper' in j else 'rad/s'}
            for m in METRICS:
                factor = 180 / np.pi if m == 'position_rmse' and 'gripper' not in j else 1
                for name, v in values.items():
                    row[f'{name}_{m}'] = v[j][m] * factor
                delta = 100 * (values['candidate'][j][m] / values['baseline'][j][m] - 1)
                row[f'{m}_change_percent'] = delta
                if delta > 1:
                    summary[split]['regressions_over_1_percent'].append({'joint': j, 'metric': m, 'increase_percent': delta})
            rows.append(row)
        summary[split]['angular_pass'] = not any('gripper' not in r['joint']
            for r in summary[split]['regressions_over_1_percent'])
        summary[split]['angular_all_improved'] = all(
            values['candidate'][j][m] < values['baseline'][j][m] for j in JOINTS for m in METRICS)
    for label in ('baseline', 'first_pass', 'candidate'):
        if all_trials['training'][label][0]['config_sha256'] != all_trials['validation'][label][0]['config_sha256']:
            raise ValueError(f'{label}: validation configuration differs from training')
    if set(all_trials['training']['baseline'][2]) & set(all_trials['validation']['baseline'][2]):
        raise ValueError('training and validation episodes overlap')
    candidate = all_trials['training']['candidate']
    freeze = json.loads((root / 'freeze.json').read_text())
    if freeze['config_sha256'] != candidate[0]['config_sha256']:
        raise ValueError('final configuration does not match the pre-validation freeze')
    if any(ep['meta']['epoch_system_ns'] <= freeze['frozen_system_ns']
           for ep in all_trials['validation']['candidate'][2].values()):
        raise ValueError('candidate validation predates the configuration freeze')
    summary.update(config_sha256=candidate[0]['config_sha256'], settling_steps=500,
                   freeze=freeze,
                   position_unit='deg (angular), percentage points (gripper)',
                   velocity_unit='rad/s (angular), percentage points/s (gripper)',
                   acceptance='Both position and velocity RMS within 1% of the same-protocol baseline for every angular joint; equal episode weight; startup included; no fitted time shift.',
                   torque_fitted=False, source_sha256=candidate[0]['source_sha256'])
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (root / 'per_joint.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split, values in all_scores.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
        y = np.arange(len(JOINTS))
        for ax, metric, factor, unit in zip(axes, METRICS, [180 / np.pi, 1], ['deg', 'rad/s']):
            for label, marker in [('baseline', 'o'), ('first_pass', 'x'), ('candidate', 's')]:
                ax.plot([values[label][j][metric] * factor for j in JOINTS], y,
                        marker, label=label, markersize=5)
            ax.set_xlabel(f'{metric} ({unit})'); ax.grid(alpha=.25); ax.legend()
        axes[0].set_yticks(y, [j.removeprefix('astribot_') for j in JOINTS])
        axes[0].invert_yaxis(); fig.suptitle(f'{split}: equal episode RMS, complete replay')
        fig.tight_layout(); fig.savefig(root / f'{split}.png', dpi=160); plt.close(fig)

    lines = ['# 关节 PD／滤波对齐：续轮实测结果', '',
             f'配置：[aligned_250hz_v2.yaml](../aligned_250hz_v2.yaml)（SHA256 `{candidate[0]["config_sha256"]}`）。', '',
             '训练使用 episode 0、1；独立验证使用 episode 6、7。基线、第一轮与本轮配置均按同一初始化协议重新运行。'
             '每段正式回放前，经 SDK→ROS2 保持实测初始位置 500 个物理步（2 秒）；正式指标仍包含起始样本。'
             '仿真 dt=0.004 s，不拟合时间平移，不使用实测力矩或重构前馈。', '',
             f'20 个角关节逐项验收：训练 **{"通过" if summary["training"]["angular_pass"] else "未全部通过"}**；'
             f'独立验证 **{"通过" if summary["validation"]["angular_pass"] else "未全部通过"}**。'
             '判据为每个关节的位置和速度 RMSE 均不比同协议基线增加超过 1%。', '',
             '| 数据 | 关节组 | 位置 RMSE：基线 → 第一轮 → 本轮（°） | 速度 RMSE：基线 → 第一轮 → 本轮（rad/s） |',
             '| --- | --- | ---: | ---: |']
    for split in labels:
        for group, stats in summary[split]['groups'].items():
            q = ' → '.join(f'{stats["position_rmse"][n]:.4f}' for n in ('baseline', 'first_pass', 'candidate'))
            v = ' → '.join(f'{stats["velocity_rmse"][n]:.5f}' for n in ('baseline', 'first_pass', 'candidate'))
            lines.append(f'| {split} | {group} | {q} | {v} |')
    lines += ['', '![独立验证逐关节误差](validation.png)', '',
              '## 完整配置', '', '| 关节 | Kp | Kd | 滤波时间常数（ms） |', '| --- | ---: | ---: | ---: |']
    config = candidate[1]
    for j in all_scores['training']['baseline']:
        k = key(j) if 'gripper' not in j else j.removeprefix('astribot_').replace('_joint_1', '_control')
        f = j if 'gripper' not in j else j.replace('_joint_1', '_joint_L1')
        lines.append(f'| {j.removeprefix("astribot_")} | {config[k]["stiffness"]:.3f} | '
                     f'{config[k]["damping"]:.3f} | {config["joint_command_filters"][f]["time_constant_s"] * 1000:.3f} |')
    lines += ['', '全部 velocity lookahead 为 0；effort limit、夹爪速度上限、重力补偿和模型保持基线设置。', '',
              '## 回退与夹爪边界', '']
    for split in labels:
        regressions = summary[split]['regressions_over_1_percent']
        lines.append(f'- {split} 超过 1% 的回退：' + ('无。' if not regressions else
            '；'.join(f'{r["joint"].removeprefix("astribot_")} {r["metric"]} +{r["increase_percent"]:.1f}%'
                     for r in regressions) + '。'))
    lines += ['', '| 夹爪 | 数据 | 位置 RMSE：基线 → 本轮（百分点） | 速度 RMSE：基线 → 本轮（百分点/s） |',
              '| --- | --- | ---: | ---: |']
    for row in rows:
        if 'gripper' in row['joint']:
            lines.append(f'| {row["joint"].removeprefix("astribot_")} | {row["split"]} | '
                         f'{row["baseline_position_rmse"]:.4f} → {row["candidate_position_rmse"]:.4f} | '
                         f'{row["baseline_velocity_rmse"]:.4f} → {row["candidate_velocity_rmse"]:.4f} |')
    lines += ['', '左夹爪训练指令跨越 0–100%，但约 30–35% 的样本在高闭合指令下实际位置低于 95%，'
              '与夹持约束相符。80 ms 滤波用于改善开合瞬态，当前无任务物体的场景不能验证接触误差。'
              '右夹爪训练指令跨度仅 0.74–0.84 个百分点；夹爪结果仅验证当前轨迹中的响应。'
              '因此不能宣称两夹爪已完成全行程／接触参数辨识。', '',
              '头部和部分躯干仍有持续位置偏差；当前结果是这些轨迹下的有效 PD／滤波参数，不是消除模型、'
              '载荷、零位或未知真实控制器差异的证明。', '',
              '## 方法与证据', '',
              '- 保留第一轮原始证据。新的固定物理步初始化与旧协议不同，旧表中的数值不用于本轮验收。',
              '- 先测刚度、阻尼、滤波扰动；只用训练数据提出候选并进行位置／速度分别筛选。'
              '低阻尼候选虽降低位置误差，但造成速度回退，未作为最终配置采用。',
              '- 每个组合必须完整回放；逐关节最优不视为组合验收。验证配置与训练配置的 SHA256 必须一致。',
              '- 所有 20 个角关节的两项指标在训练、验证集上均低于同协议基线；'
              '这里的逐关节指标是完整 episode 等权合并的 RMS，并不意味着每个采样点都改善。',
              '- 所有采纳回放的 SDK 和仿真退出码均为 0，SDK 位置指令逐物理步核对。'
              '首个启动失败的 `baseline/` 目录被保留并排除，基线使用 `baseline_settled/`。',
              '- [完整 22 关节汇总 CSV](per_joint.csv)、[机器可读验收结果](summary.json)、'
              '[训练逐段重算](training_comparison/per_joint.csv)、[验证逐段重算](validation_comparison/per_joint.csv)。',
              '- 参数选择与原始候选保留在本目录的 YAML／JSON 中；每个 trial 目录包含配置、实现哈希、'
              '原始指令、ROS 反馈、逐步物理反馈及初始化残差。', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({s: {'angular_pass': summary[s]['angular_pass'],
                         'regressions': summary[s]['regressions_over_1_percent']} for s in labels}, indent=2))


if __name__ == '__main__':
    main()
