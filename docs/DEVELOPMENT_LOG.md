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
