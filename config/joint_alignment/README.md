# Cross-task joint response presets

The v4 candidate is one shared set of effective simulation PD/filter parameters.
Its commands are replayed through the public SDK and ROS. Training uses seven
tasks; held-out episodes, two additional tasks, and GPU results are reported
separately in the [experiment report](../../outputs/joint_alignment/cross_task_20261112/report.md).
The report includes regressions and full-episode errors, including contact-like
gripper intervals. These presets are candidates, not a claim that every metric
or every recording group has been aligned.

- `multitask_v4_candidate_250hz_cpu.yaml`: CPU PhysX calibration configuration.
- `multitask_v4_candidate_250hz_gpu.yaml`: same controls, GPU PhysX configuration.

Both require a 4 ms command/physics interval, one physics substep and zero
velocity lookahead. Wall-clock throughput may be below 250 Hz; use verified
simulation-time stepping for reproducible replay. The full YAML includes the
robot and initial pose used by the calibration launcher. That launcher reaches
each recording's initial pose through SDK commands and holds it for 500 steps.
Task scene composition and other integration intervals need separate validation.

For example, after confirming ROS domain 205 is unused:

```bash
/usr/bin/python3 scripts/joint_align/run_trial.py \
  --config config/joint_alignment/multitask_v4_candidate_250hz_gpu.yaml \
  --output outputs/joint_alignment/recheck_v4 \
  --sim-python /home/zihengkang/miniconda3/envs/astribot_simu_il/bin/python \
  --sdk-python /usr/bin/python3 --domain 205 --settle-steps 500 \
  outputs/joint_alignment/cross_task_20261112/inputs/real_eval-press_cube_v2-pi05_sync-id_episode_abs_1.hdf5
```

Use a fresh output directory. Parameter snapshots and their pre-validation hashes
are retained in `outputs/joint_alignment/cross_task_20261112/freeze.json`.
The [cross-task implementation](../../scripts/joint_align/cross_task.py) defines
data semantics, task weighting, gripper partitioning and configuration selection.
Experiment results and remaining regressions are recorded under 2026-11-12 in the
[development log](../../docs/DEVELOPMENT_LOG.md).
