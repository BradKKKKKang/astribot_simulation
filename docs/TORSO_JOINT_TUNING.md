# Torso joint 1 tuning at 100 Hz

`astribot_torso_joint_1` is the pitch joint between the fixed chassis and the
upper body. Its encoder feedback is present in all three replay artifacts.
This experiment changes only this joint's implicit PD gains and gravity
feedforward. Other joint gains, masses, inertias, contacts and the physics
step remain unchanged.

## Control settings

Isaac YAML configurations accept an optional `torso_joint_1_control` mapping:

```yaml
torso_joint_1_control:
  stiffness: 800.0                  # N m / rad
  damping: 40.0                     # N m s / rad
  effort_limit: 400.0               # PhysX drive limit, N m
  gravity_compensation_scale: 0.0   # 1.0 = full model gravity compensation
```

These values show the historical baseline. Omitting the mapping preserves
that baseline. Gravity feedforward also requires `gravity_compensation: true`.
It uses PhysX `get_gravity_compensation_forces()` at the current pose and
applies only the selected DOF, alongside its implicit position drive.
This does not disable gravity, change mass, or compensate the other joints.
The drive force limit and additional feedforward are separate PhysX inputs;
the recorded implicit-actuator torque is an estimate, not a force sensor.

## Reproduce the comparison

The focused tuning runner uses the shared validated replay input and
initialization helpers. It runs in the same process as physics, with no ROS
executor. The normal SDK replay entrypoint remains
`scripts/replay_artifacts_sdk_100hz.py`; use it to check the selected settings
over SDK/ROS after the physics comparison.

From the repository root, with ROS2 Humble, the built message overlay, and the
installed Isaac environment available:

```bash
bash scripts/run_torso_tuning.sh \
  artifacts/resampled_100hz/replay_train_pour/samples.parquet \
  artifacts/resampled_100hz/replay_train_guess/samples.parquet \
  --trials config/torso_tuning_trials.json \
  --output-root results/torso_tuning_new_run

python scripts/compare_torso_tuning.py results/torso_tuning_new_run \
  --plot baseline gravity_only pd_gravity_full
```

The shell helper defaults to `$HOME/miniconda3` and the `astribot_simu_il`
environment; set `ASTRIBOT_CONDA_ROOT` or `ASTRIBOT_ISAAC_ENV` for a different
local installation. It sets localhost-only ROS domain 197 and uses a temporary
ROS log directory. It requires a new output directory and verifies a complete
manifest after Kit exits. Camera rendering is disabled for parameter fitting.

Every command frame advances one 10 ms physics step. Each trial starts at
the same interpolated real position and velocity; 30 warmup steps are excluded
and the initial state is reapplied before recording. Comparisons interpolate
real feedback at the actual post-step physics timestamp and exclude the final
state outside the command time span. No time shift is fitted to reduce error.
Wall-clock deadline overruns are recorded separately from physics tracking.

Each trial saves all 22 simulated joint streams, unchanged real feedback,
command timing, actual settings and resolved PhysX gains, source hashes,
gravity feedforward and estimated actuator effort. `comparison.csv` and
`comparison.json` also report the 19 other angular joints, position bias,
motion span and estimated effort saturation. Grippers retain SDK percentage
units and are excluded from angular aggregates.

## Experiment results

The initial eight candidates are in `config/torso_tuning_trials.json`; the
four subsequent local refinements are in `config/torso_tuning_refinement.json`.
Raw local evidence is under `results/torso_tuning_round1_verified/` and
`results/torso_tuning_round2/`. Partial startup/debug runs are not comparison
evidence.

Selected settings are **Kp=3200, Kd=320, effort limit=400 N m, gravity scale=1.1**.
They are now explicit in the default Isaac YAML. The 1.1 factor is an empirical
correction for this model and these trajectories, not a measured hardware
gravity-compensation setting. Full compensation (1.0) gives similar results;
1.1 slightly reduces the remaining positive bias without worsening the held-out
trajectory. Increasing Kd to 640 worsened position lag on `guess`, so it was rejected.

Physics comparison (degrees for position, rad/s for velocity):

| Episode | Position RMSE before → after | Max position error before → after | Velocity RMSE before → after |
| --- | ---: | ---: | ---: |
| 20261020_213902 (held out during screening) | 1.050 → 0.174 | 2.863 → 0.472 | 0.0751 → 0.0524 |
| guess | 1.768 → 0.157 | 3.539 → 0.410 | 0.0791 → 0.0497 |
| pour | 3.529 → 0.185 | 5.982 → 0.569 | 0.0849 → 0.0359 |

Position RMSE decreased by 83–95%. The selected joint's motion spans are
16.39°, 23.61°, and 2.64°, compared with real feedback spans of 16.27°, 23.36°,
and 2.84°. Thus the improvement is not obtained by freezing the joint.

The selected parameters were then checked using the existing
`scripts/replay_artifacts_sdk_100hz.py` through SDK and ROS2, against a newly
run baseline in the same localhost-only domain 198:

| Episode | SDK position RMSE before → after | SDK max position error before → after | SDK velocity RMSE before → after |
| --- | ---: | ---: | ---: |
| 20261020_213902 | 1.353 → 0.180 | 3.968 → 0.619 | 0.0852 → 0.0527 |
| guess | 2.189 → 0.292 | 6.806 → 1.588 | 0.1027 → 0.0528 |
| pour | 3.552 → 0.183 | 6.002 → 0.467 | 0.0847 → 0.0371 |

Both SDK runs completed all 1,584/2,314/3,597 API frames. Selected-run torso
state observation rates were 97.55/96.81/96.31 Hz; this is not a hard-real-time
100 Hz guarantee. SDK `guess` peaks around 0.40 s after replay starts; its
initial torso residual was −0.01866 rad. The existing preposition tolerance
is 0.1 rad and does not reset the articulation q/v, unlike the physics runner.
No transient samples were removed from the reported SDK metrics.

Local comparison artifacts:

- [Physics curves](../results/torso_joint_1_tuning/comparison.png),
  [CSV](../results/torso_joint_1_tuning/comparison.csv), and
  [JSON](../results/torso_joint_1_tuning/comparison.json).
- [SDK curves](../results/torso_joint_1_tuning/sdk_comparison.png) and
  [SDK metrics](../results/torso_joint_1_tuning/sdk_comparison.json).
- SDK source recordings, configs and owned-process shutdown evidence:
  `results/torso_tuning_sdk_baseline/` and `results/torso_tuning_sdk_selected/`.

Only joint 1 was tuned. Coupling slightly worsens the other 19 angular joints'
combined position RMSE: 0.02227→0.02263, 0.02023→0.02161, and
0.01836→0.01973 rad for the three physics episodes. The largest regression is
`torso_joint_2` on `pour` (+0.01078 rad RMSE). Those joints require their own
calibration; this change does not claim whole-robot alignment. Joint 1 velocity
RMSE on the held-out trajectory remains slightly above 0.05 rad/s. No measured
hardware torque equivalence or real-robot validation is claimed.
