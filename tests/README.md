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
`pxr` build (the geometry exporter uses `Usd.Stage.GetCompositionErrors`).
Run:

```bash
python -m pytest -o addopts='' tests/unit/test_placement*.py \
  tests/unit/test_identification*.py tests/unit/test_parquet_replay.py \
  tests/unit/test_replay_artifacts_sdk_100hz.py
```

The v2 tests cover independent experiment schemas, v1/v2 replay/extraction/request
identity, simulator config resolution without placement imports, and CLI replay
orchestration through actual CPU preflight/extraction subprocesses. That last test
mocks SDK feedback acquisition only, using explicit recorded-feedback fixtures;
it exercises real USD FK and artifact export. These tests do not start Isaac or
prove a live SDK/ROS/Isaac episode. Run the simulator import check separately;
missing optional backends are skips, not successful backend execution.
