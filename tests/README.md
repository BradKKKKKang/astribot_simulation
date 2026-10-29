# Testing Guide

## Test Philosophy

**We test what users care about:**
1. ✅ Simulators installed correctly
2. ✅ Simulation pipeline works (create → reset → step)
3. ✅ ROS topics are published

**We don't test:**
- ❌ YAML parsing (PyYAML's job)
- ❌ Path concatenation (trivial)
- ❌ Factory pattern (Python's job)

## Test Structure

```
tests/
├── unit/
│   └── test_simulator_imports.py    # Verify installations
└── integration/
    └── test_simulation_pipeline.py  # Test complete data flow
```

## Running Tests

Managed evaluation queue, identity, atomic batch validation, terminal latch,
reporting, reset cancellation, Replay exclusivity and fixed-step tests run on CPU:
`python -m pytest tests/unit/test_managed_evaluation.py -o addopts='' -q`.
They use mock environments and extracted backend methods. They do not verify
DDS delivery, native WBC, Isaac physics or rendered image correctness.


Placement publication tests run without ROS or USD:
`python -m pytest tests/unit/test_placement_web_publish.py -o addopts='' -q`.
In `web/placement-viewer`, run `npm test` for data/filter tests and
`npx playwright test` with the dev server running on port 5173 for browser tests.
Browser tests require at least one task subdirectory under `public/data` with
`placement.json` and `robot.glb`;
they use its matching robot GLB with synthetic events served only through test routes.
Chrome uses CPU SwiftShader; the longer interaction test allows three minutes.

The gripper conversion regression suite uses CPU PyTorch tensors and mocked
ROS/Isaac startup dependencies. It checks the actual Isaac control and feedback
methods, including velocity compensation, both grippers, follower directions,
range clamping, and feedback units. It does not start a simulator or ROS node:

```bash
python -m pytest tests/unit/test_isaaclab_gripper.py -o addopts='' -q
```

```bash
# Quick: Only check installations
pytest tests/unit/ -v

# Full: Test complete pipeline (requires ROS2)
source env.sh
pytest tests/integration/ -v

# All tests
pytest -v
```

## What Each Test Does

### Unit Tests (Fast, No ROS Required)
- `test_simulator_imports.py`: Verify MuJoCo, Genesis, ManiSkill, Isaac Lab can be imported

### Integration Tests (Slow, Requires ROS2)
- `test_simulation_pipeline.py`:
  - Create environment
  - Call reset() and step()
  - Verify ROS topics published:
    - `/robot/joint_space_states`
    - `/robot/camera/*/image_raw`
    - `/robot/camera/*/depth`
    - `/robot/camera/*/point_cloud`


## Placement configuration migration checks

Use a CPU Python with NumPy, PyArrow, PyYAML, OpenCV, pytest and a compatible
`pxr` build; CPU USD 25 is used for the complete FK/export checks. PyTorch is
needed for the mocked backend reset and base-coordinate tests.
Run:

```bash
python -m pytest -o addopts='' tests/unit/test_placement*.py \
  tests/unit/test_identification*.py tests/unit/test_parquet_replay.py \
  tests/unit/test_replay_artifacts_sdk_100hz.py tests/unit/test_online_evaluation.py \
  tests/unit/test_robot_placement.py tests/unit/test_scene_assets.py \
  tests/unit/test_grounded_capture_extraction.py
```

The v3 tests cover independent experiment schemas, flat/v1/v2 compatibility,
path-only relocation identity and explicit placement changes, simulator config
resolution without placement imports, and CLI replay
orchestration through actual CPU preflight/extraction subprocesses. That last test
mocks SDK feedback acquisition only, using explicit recorded-feedback fixtures;
it exercises real USD FK and artifact export. These tests do not start Isaac or
prove a live SDK/ROS/Isaac episode. Run the simulator import check separately;
missing optional backends are skips, not successful backend execution.

The geometry and runtime-state coverage is split by dependency:

- `test_simulation_integrate.py` checks all evaluation launcher configurations,
  runtime-free preflight, default paths, continuous operation past legacy step
  budgets, asset handoff and report
  persistence before closing Isaac. Factory tests use mocks; actual rendering
  and physics require a separate isolated Isaac run.
- `test_camera_viewer_lifecycle.py` exercises a real thread with mocked ROS/GUI
  dependencies, checking that context shutdown follows spin-thread termination
  and executor shutdown. `test_camera_viewer.py` itself remains the live viewer.
- `test_camera_grid.py` verifies head/wrist layout, partial frame arrival, pixel
  preservation, aspect ratio and one-window cleanup without a graphical desktop.
- `test_initial_pose.py` checks measured parquet frame selection and named joint
  mapping, task overrides, CLI display and reset of joint state/control targets.
  Parquet fixtures need PyArrow; the backend reset mock also needs PyTorch.
- `test_robot_placement.py` checks complete-geometry grounding at nonzero initial
  joints, world-fixed USD anchors, local TCP/camera invariance, placement evidence
  and capture-window validity. CPU PyTorch mocks exercise the actual backend reset
  and measured world-to-base conversion methods without starting ROS or Isaac.
- `test_scene_assets.py` uses synthetic GLBs for axis/scale/support geometry,
  nested references and usable tabletop bounds. Actual asset checks skip when the
  configured source GLBs are absent. The USD authoring case additionally requires
  Isaac Kit's `pxr.PhysxSchema`; run it separately in that environment to verify
  independent rigid bodies and the holder's SDF collision. A CPU-only skip of
  this case is not evidence that PhysX authoring or simulation passed.
- `test_grounded_capture_extraction.py` runs synthetic, explicitly marked mock
  captures through real CPU USD FK, GLB export and schema-5 publication. It checks
  nonzero translation/yaw, preservation of a measured pose offset, rejection of
  legacy-coordinate reinterpretation and changed FK/cache identity while reusing
  body-local geometry. It does not record actual ROS feedback or replay an SDK
  dataset.

Headless scene physics and isolated ROS backend smoke runs are separate runtime
checks, with their environment and evidence recorded in the development log.
