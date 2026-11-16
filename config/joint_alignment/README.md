# Cross-task joint response preset

The sole preset is `multitask_v4_candidate_100hz_gpu.yaml`: v4 candidate PD/filter
parameters with GPU PhysX (`physics_device: cuda:0`) at 100 Hz simulation time.
It uses `dt: 0.01`, `frame_skip: 1`, `physics_substeps: 1`, `decimation: 1`, and
zero velocity lookahead. Filter time constants remain in seconds and are not
rescaled. Simulation frequency does not imply the same wall-clock throughput.

The same control parameters and timing are installed in
[`simulation_isaaclab_param.yaml`](../astribot_s1/simulation_isaaclab_param.yaml),
which is referenced by the shared [robot profile](../robots/astribot_s1.json).
The robot profile explicitly selects GPU physics and 100 Hz. Normal launches
retain their cameras and task-specific initialization. The standalone preset
retains the calibration robot pose and disables cameras; it uses the regular
simulation loop with `joint_alignment_step_service: false`.
Experiment runtime overrides still apply: `fold_tshirt` retains four physics
substeps within each 10 ms control interval (100 Hz control, 400 Hz physics).

After loading the project environment and confirming ROS domain 205 is unused,
launch the standalone preset with the Isaac Lab Python environment:

```bash
ROS_DOMAIN_ID=205 ROS_LOCALHOST_ONLY=1 python astribot_simulation.py \
  --config config/joint_alignment/multitask_v4_candidate_100hz_gpu.yaml \
  --headless --no-cameras
```

The historical calibration launcher `scripts/joint_align/run_trial.py` and its
verified step service require 250 Hz; this 100 Hz preset does not use that path.
SDK replay against this preset should use `--control-hz 100` without
`--physics-step`.

The original v4 fitting and validation used 250 Hz. Their results, including
remaining regressions, do not establish alignment at 100 Hz. Training used seven
tasks; held-out episodes, two additional tasks, and GPU results are reported
separately in the [experiment report](../../outputs/joint_alignment/cross_task_20261112/report.md).
The original parameter snapshots and pre-validation hashes remain in
`outputs/joint_alignment/cross_task_20261112/freeze.json`; the CPU and 250 Hz
presets have been removed from this directory.
The [cross-task implementation](../../scripts/joint_align/cross_task.py) defines
data semantics, task weighting, gripper partitioning and configuration selection.
Historical results and the 100 Hz configuration change are recorded in the
[development log](../../docs/DEVELOPMENT_LOG.md).
