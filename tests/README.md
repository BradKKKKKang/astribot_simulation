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
