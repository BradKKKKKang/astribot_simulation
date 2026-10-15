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
