## 2026-10-22：Headless GPU 基准与 100 Hz 完整轨迹对齐

- **性能边界**：RTX 3090、4 ms 物理步长的 headless 长测中，完整 `env.step`（含 ROS executor、无相机）约 118.6 Hz，三相机约 29.9 Hz；physics-only 约 283.0 Hz，不能据此宣称完整回放稳定达到 250 Hz。
- **重采样与回放**：新增 `resample_replay_commands.py` 与 `replay_artifacts_100hz.py`；三组记录各生成 1,584 /2,314 /3,597 个 22 关节共用 100 Hz 指令帧，保留全部原始反馈与来源元数据。完整 `env.step` 采用 dt=0.01 s、frame_skip=1、headless、关闭相机，保存 N+1 状态及实际物理时间；共同覆盖区间裁边与最终网格取整记录在 manifest。
- **结果与验证**：三轨迹全部完成（进程退出码均 0），10 ms 完成期限超时分别 5 /34 /20 次；首组采用独占 GPU 重跑，原并发 GPU 诊断结果保留但排除。仅建立同进程 position-only 物理基线，未验证 SDK/ROS 传输，未调整 800/40/400 PD；力矩仅作诊断。汇总与哈希/状态/误差复算审计见 `results/alignment_100hz/`，离线测试 4 项通过。

## 2026-10-15: Isaac Lab gripper dataset units

- Reference: `cognition_press_three_buttons`, episode 000000 (326 frames).
  Metadata identifies joint gripper columns 14/22 and Cartesian gripper columns
  18/28. Both command representations use the 0–100 scale; the left command
  spans the full range and the right command stays at zero.
- Root cause: Isaac Lab treated these commands as radians and clamped positive
  values to zero in `[-0.5, 0]`. Convert the compensated command once from
  0–100 to 0–0.93 rad, then apply each USD/URDF follower joint's direction.
  This matches the existing MuJoCo actuator gain/bias ratio and Genesis /
  ManiSkill position-command conversion; those backends are unchanged.
- Convert Isaac position, velocity and acceleration feedback with the same
  signed scale. Preserve small out-of-range physical feedback instead of
  folding it through absolute value. Fix velocity-mode scalar indexing so
  either gripper can advance from its feedback position in dataset units.
- Validation: `astribot_simu_il` Python, PyTorch 2.7.0+cu128, CPU tensors;
  `python -m pytest tests/unit/test_isaaclab_gripper.py -o addopts='' -q`: 11
  passed. Tests execute actual control/feedback and base compensation methods
  with mocked ROS/Isaac startup dependencies. Both sides, six follower signs,
  endpoints, intermediate values, clamping, compensation and signed feedback
  are covered.
- Read the reference parquet with PyArrow in `atlas`, then checked all 326
  command pairs through the same CPU control methods. Maximum command-to-
  target-to-feedback error: 3.95e-6 scale units; left L1 targets span
  -0.93–0 rad, right L1 remains zero. No dataset files were modified.
- Scope: syntax and whitespace checks only beyond these focused checks; no
  Isaac physics/GUI, ROS transport, other-backend runtime or hardware run.
  Existing unrelated workspace changes are preserved.

## 2026-10-15: Replay runtime dependencies and SDK control cadence

- Reproduced missing PyArrow in the working system-Python SDK environment loaded
  with `astribot_sdk_ros2/env.zsh`. Also reproduced missing `cryptography` in the
  simulator's `astribot_simu_il` Python; the native SDK masked it with a
  `robotics_library_py ... is not a package` import error. Installed PyArrow 17
  for the local system Python and added `scripts/requirements-replay.txt` for
  repeatable replay dependency setup.
- Check `cryptography.fernet` and import SDK before waiting for simulation;
  missing-dependency errors identify the interpreter and installation command.
  Keep SDK client construction after simulation/camera readiness checks.
- Send both the initial ramp and dataset targets at 100 Hz, matching the SDK
  configuration instead of pairing 100/30 Hz calls with a 250 Hz SDK. Hold the
  current dataset target between timestamps and skip delayed control ticks.
- Restored the missing replay setup guide, linked from README, with separate
  simulator/SDK Python environments and ROS exports after SDK `env.zsh`.
- Validation: 18 offline pytest cases passed; syntax and whitespace checks passed.
  Real 326-frame/10.833-second episode replay used ROS2 Humble, Isaac Lab in
  `astribot_simu_il`, and `/usr/bin/python` with the SDK's `env.zsh`. Domain 187
  was isolated with localhost-only discovery and a loopback UDP transport.
  Each of six moving parts received 1,386 commands; final command errors versus
  the dataset were zero. Each part emitted 211–212 feedback samples with motion;
  final feedback errors were at most 0.076 rad for torso/arms/head and 0.125 SDK
  units for grippers. The observer received 64/94/106 head/left/right RGB images.
  Replay, viewer and owned simulator exited; replay and simulator returned zero.
  No calling-frequency mismatch warnings occurred in this run.
- Limits: SDK-internal desired-state-sync subscriptions still log QoS reliability
  warnings. They did not prevent the simulator or observer receiving commands;
  this change does not modify the external SDK. Only this Isaac Lab/ROS2 setup
  was exercised, with no real hardware or other backend validation.

## 2026-10-22: Isolated torso joint 1 PD and gravity tuning

- Added optional Isaac `torso_joint_1_control` settings and a separate implicit
  actuator for this DOF; only its gains and PhysX gravity feedforward change.
  Default YAML now selects Kp=3200, Kd=320, effort limit=400 N m, gravity scale=1.1.
  Other joint gains remain 800/40. See `docs/TORSO_JOINT_TUNING.md` for semantics,
  reproduction commands, curves and coupled-joint limitations.
- Ran 8 initial candidates on pour/guess, 4 local refinements, and baseline plus
  two candidates on the held-out 20261020_213902 trajectory: 27 complete physics
  replays. Every command advances one 10 ms step; recorded initial q/v are
  restored before every trial. Position RMSE for the selected joint decreased
  from 1.050/1.768/3.529 degrees to 0.174/0.157/0.185 degrees.
- Used the existing SDK 100 Hz runner for a new baseline and selected-parameter
  replay of all three trajectories in localhost-only ROS domain 198. Position
  RMSE decreased from 1.353/2.189/3.552 degrees to 0.180/0.292/0.183 degrees;
  selected-run state observation was 96.31–97.55 Hz. Both owned simulator
  processes shut down with exit code zero. No hardware was controlled.
- Evidence: `results/torso_joint_1_tuning/`, raw trial directories referenced
  there, and `results/torso_tuning_sdk_{baseline,selected}/`. Incomplete debug
  directories are not counted. Implicit actuator torque is diagnostic only.
- Limits: the other 19 angular joints' combined position RMSE rises slightly
  through coupling; torso joint 2 on pour worsens by 0.01078 rad RMSE. SDK guess
  retains an early 1.588-degree peak with nonzero preposition residual. Joint 1
  velocity RMSE on the held-out physics trajectory remains 0.05244 rad/s.
- Focused control tests: 22 passed (existing gripper plus new torso cases).
  Python syntax checks, shell syntax and whitespace checks passed. This does
  not establish whole-robot calibration or hard-real-time SDK feedback.

## 2026-10-22: Per-joint SDK replay alignment audit, including grippers

- Audited the selected torso-joint-1 configuration against all three saved SDK
  replay recordings: 22 joints per episode, 66 joint/episode comparisons.
  Used simulator header timestamps and interpolated real feedback, retaining
  startup transients without fitting a time offset. Angular position errors
  are reported in degrees; gripper errors use percentage points on the SDK
  0–100 scale. Velocity errors retain rad/s or percentage points/s. Torque
  equivalence remains unverified and was not included in the comparison.
- Confirmed substantial remaining misalignment. In episode order
  20261020_213902 / guess / pour, torso joint 2 position RMSE is
  5.008 / 4.925 / 4.027 degrees, with signed simulation-minus-real mean errors
  of −4.910 / −4.837 / −3.883 degrees. Torso joint 3 position RMSE is
  1.650 / 1.114 / 1.728 degrees. These are sustained offsets, not only isolated
  peaks; tuning joint 1 does not establish whole-body alignment.
- Left gripper position RMSE is 16.084 / 8.284 / 4.322 percentage points, with
  a worst instantaneous error of 75.664 points. Right gripper RMSE is
  0.131 / 1.502 / 6.947 points, with a worst instantaneous error of 51.155 points.
  Curves show substantial disagreement during opening/closing transitions,
  including slower simulated left-gripper response in 20261020_213902.
  These discrepancies already existed before joint 1 tuning; the audit does
  not identify a unique cause or change the gripper controller.
- Also identified remaining position offsets in head joint 2, left-arm joints
  1/4, and right-arm joint 4. Suggested follow-up order is torso joint 2,
  torso joint 3, and gripper transient response; no additional tuning was
  performed in this audit.
- Saved the complete per-joint table, all per-episode position/velocity errors,
  signed bias, p95, peak time, sample counts, and baseline differences in
  [the audit report](../results/torso_joint_1_tuning/joint_alignment/report.md),
  [CSV](../results/torso_joint_1_tuning/joint_alignment/per_joint.csv), and
  [JSON](../results/torso_joint_1_tuning/joint_alignment/per_joint.json).
  [Comparison curves](../results/torso_joint_1_tuning/joint_alignment/largest_mismatches.png)
  cover torso joint 2 and both grippers across all three episodes.
- Validation: recomputed all 66 position RMSE values and matched their sample
  counts against the saved SDK tracking-error files; visually inspected the
  generated curves. The preceding tuning change's final focused regression
  included 30 passing tests (gripper, torso control, SDK replay, and simulation
  timing). This audit only analyzed existing recordings; no new simulation,
  SDK control, real-hardware operation, or runtime-code change was performed.
