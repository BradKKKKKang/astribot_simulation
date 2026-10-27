# Astribot Simulation

**A unified robot simulation platform** built on top of **MuJoCo**, **Genesis**, **ManiSkill** and **Isaac Lab**.
This project provides a common abstraction layer so that you can run your robot across different simulators with a consistent API.
It is designed as a **research tool** for users of Astribot.

<p align="center">
  <img src="docs/system.png" alt="Astribot Simulation System" width="600">
</p>

---

## Demo

<table align="center">
  <tr>
    <td align="center">
      <p><b>ManiSkill Environment</b></p>
      <img src="docs/maniskill.gif" alt="ManiSkill Demo" width="400">
    </td>
    <td align="center">
      <p><b>MuJoCo Simulation</b></p>
      <img src="docs/mujoco.gif" alt="MuJoCo Demo" width="400">
    </td>
  </tr>
</table>

---

## Features

- **Multi-backend environments**: Switch between **MuJoCo**, **Genesis**, **ManiSkill**, and **NVIDIA Isaac Lab** with one line of code.
- **ROS & ROS2 Integration**: Comprehensive support for both ROS1 and ROS2 interfaces to bridge simulation and real robots.
- **Plug-and-play robot models**: Full support for **URDF**, **MJCF** and **USD** formats.
- **Config-driven setup**: Easily manage each simulator and robot variant via structured **YAML** configurations.
- **Research-friendly utilities**: Includes logging, common analysis tools, and a unified environment factory.
- **Placement analysis**: Extract grasp/release TCP events from Isaac Lab replay and inspect metric robot geometry and 3D density in a read-only Three.js viewer. See [placement analysis](docs/PLACEMENT_ANALYSIS.md).

---

## Project Structure

```
astribot_simulation/
├── astribot_simulation.py         # Main entry point
├── env.sh                         # Environment setup (sources ROS, sets ASTRIBOT_SIMU_ROOT)
├── pyproject.toml                 # Project metadata & version
├── requirements.txt               # Python dependencies
│
├── src/                           # Source code (src layout)
│   ├── version.py                 # Version string
│   ├── exceptions.py              # Custom exceptions
│   │
│   ├── astribot_envs/             # Environment wrappers for each simulator
│   │   ├── astribot_base_env.py       # Abstract base (extends gym.Env)
│   │   ├── astribot_mujoco_env.py     # MuJoCo backend
│   │   ├── astribot_genesis_env.py    # Genesis backend
│   │   ├── astribot_maniskill_env.py  # ManiSkill backend
│   │   ├── astribot_isaaclab_env.py   # Isaac Lab backend
│   │   └── astribot_envs_factory.py   # Factory: loads YAML, creates gym env
│   │
│   ├── simu_utils/                # Utility modules
│   │   ├── simu_common_tools.py       # Pose math, point cloud, logging
│   │   ├── robot_ros_interface.py     # ROS1/ROS2 publishers & subscribers
│   │   └── chassis_kinematics.py      # Omni-wheel kinematics
│   │
│   └── sim_assets_tools/          # Mesh & texture processing tools
│       ├── stl_to_dae.py, obj_to_dae.py, glb_to_dae.py
│       ├── add_logo_to_texture.py, add_logo_to_dae.py
│       ├── process_mujoco_xml.py, reduce_faces.py
│       └── lib/                       # Shared helpers for asset tools
│
├── config/                        # YAML configs for different robots & simulators
│   ├── astribot_s0/
│   ├── astribot_s1/
│   └── astribot_t1/
│
├── astribot_descriptions/         # Robot model files
│   ├── mjcf/                          # MuJoCo XML models (s0, s1, t1, worldbody)
│   ├── urdf/                          # URDF/SDF models (s0, s1, t1)
│   └── usd/                           # USD models (s1, for Isaac Lab)
│
├── astribot_msgs/                 # Custom ROS message/service/action definitions
│   ├── msg/                           # 22 message types
│   ├── srv/                           # 6 service types
│   ├── action/                        # 1 action type
│   ├── CMakeLists_ros1.txt / CMakeLists_ros2.txt
│   └── package_ros1.xml / package_ros2.xml
│
├── scripts/                       # Setup & build scripts
│   ├── install.sh                     # Full environment setup
│   ├── build.sh                       # Build astribot_msgs
│   ├── depends/                       # Miniconda installer
│   ├── lite_install/                  # MuJoCo-only installers
│   └── bugfix/                        # GLIBCXX / LIBFFI patches
│
├── tests/                         # Test suite
│   ├── unit/
│   └── integration/
│
└── docs/                          # Images and diagrams
```

---

## Installation

You can install the simulation either inside a Docker container or directly on your host machine.

### Option 1: Docker Installation

```bash
# Enter the Ubuntu 22.04 Docker container
docker exec -it <container_name> /bin/bash

# Clone the repository
git clone https://github.com/Astribot-Dev/astribot_simulation.git

# Navigate to the project directory
cd astribot_simulation

# Initialize and update all submodules recursively
git submodule update --init --recursive

# Batch pull LFS objects for all submodules
git submodule foreach git lfs pull

# Install all simulators (MuJoCo, Genesis, ManiSkill, IsaacLab)
bash scripts/install.sh
```

### Option 2: Host Machine Installation

```bash
# Clone the repository
git clone https://github.com/Astribot-Dev/astribot_simulation.git

# Navigate to the project directory
cd astribot_simulation

# Initialize and update all submodules recursively
git submodule update --init --recursive

# Batch pull LFS objects for all submodules
git submodule foreach git lfs pull

# (ROS1 only) Install Miniconda (Ubuntu 20.04)
bash scripts/depends/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Install dependencies
# Choose one of the following:

# Install only MuJoCo simulator
bash scripts/lite_install/install_mujoco.sh

# OR install all simulators (MuJoCo, Genesis, ManiSkill, IsaacLab)
bash scripts/install.sh
```

---

## Quick Start

### Launch Astribot Simulation:

```bash
# (ROS1 only) activate conda env
conda activate astribot_simu

# run simulation
source env.sh && python3 astribot_simulation.py
```

If you want to reset the robot state, press Backspace while the simulation is running.

### Replay a Parquet dataset and extract events

Placement analysis has capture, candidate extraction and identification stages.
`scripts/replay_extract_event.py` runs this workflow, with `--experiment` (alias
`--task-config`) selecting a modular experiment. Offline success evaluation has
a separate explicit entrypoint.
Placement experiments live under `config/placement_distribution/` and reference
`config/robots/`, `config/placement_distribution/candidates/` and
`config/placement_distribution/jev/`. They do not require
scene, evaluation task or success profiles. Dataset identity lives in `source`;
startup filtering lives in `candidates.selection`. Evaluation experiments are
separate under `config/experiments/evaluation/`, with independently replaceable
`config/success/` profiles.
See [the placement architecture](docs/PLACEMENT_ARCHITECTURE.md) for configuration
composition, cache dependencies, result schemas and extension points.

Isaac Lab runs separately. Replay uses the existing SDK → ROS2 → Isaac Lab →
ROS2 feedback path. `--capture-only` records feedback without requiring CPU
extraction assets; `--resume` reuses matching completed captures. New experiments
declare world XY/yaw and geometry-based grounding through
`robot_placement`. Capture checks the separately launched simulator's actual
placement before replay; launch simulation with the same experiment. Robot TCP
and camera calibration remain local to the robot.

Launch the minimal replay environment from the same experiment:

```bash
python astribot_simulation.py \
  --experiment config/placement_distribution/press_cube.json --check-config
# Remove --check-config to launch; add --headless --no-cameras for state-only replay.
```

Replay/extraction no longer runs success evaluation automatically. The old
`--evaluate-only`, `--observations` and `--no-evaluate` placement options are removed;
use `python -m src.evaluation.placement` for explicit offline evaluation. This entry does
not implement policy rollout. Existing flat and modular v1/v2 input files remain
readable, but checked-in experiments use `schema_version: 3` and an explicit `kind`.
Legacy inputs preserve their original robot placement; only an explicit
`robot_placement` enables geometry-based grounding. Process historical captures
with their original configuration; new grounded declarations require matching
recorded runtime placement and cannot reinterpret an old capture.

For offline processing of existing captures:

```bash
python scripts/replay_extract_event.py /path/to/capture-batch \
  --extract-only --experiment config/placement_distribution/press_cube.json \
  --output /path/to/analysis/press_cube \
  --artifact-root /path/to/shared-artifacts --workers 2 --no-identify
```

`--resume-extraction` resumes matching episode checkpoints. `--incremental`
allows completed episodes in a partial batch; when used during replay it overlaps
CPU extraction with capture. Worker counts are bounded; keep them low when
sharing a machine with simulation. Use a new analysis directory after changing
candidate rules; compatible synchronization/FK caches remain reusable.

```bash
python scripts/replay_extract_event.py /path/to/analysis/press_cube \
  --identify-only --experiment config/placement_distribution/press_cube.json \
  --identify-workers 2
python -m src.evaluation.placement /path/to/analysis/press_cube \
  --experiment config/experiments/evaluation/press_cube.json
```

Identification uses original dataset videos and `DECISION_API_KEY` from the
environment. Frame/request caches and decision identities are separate: changing
timeout does not invalidate model results, and changing confidence thresholds
recomputes decisions from saved raw probabilities. Omit `--data-root` to reuse
verified archived frames without reopening the dataset; an explicit `--data-root`
always revalidates the requested source. `penholder.json` assigns the known object without a
model call. `cap_pen.json` remains `needs_annotation`. The three migrated evaluation
success profiles have deliberately unconfigured predicates, so task success remains
`unknown` until explicit predicates are supplied; motion candidates are not contact evidence.
`python -m src.evaluation.placement --observations /path/to/evidence.json` supplies explicit per-episode observations
to evaluation; its format is documented in the architecture guide.

The `table_penholder` evaluation example combines a white table, red pen and
yellow penholder with the shared S1 robot. Asset profiles under `config/assets/`
record source files, axis conversion, scale, collision and the tabletop frame.
Update those source paths for your asset installation. Validate the composition
without starting ROS or Isaac:

```bash
python astribot_simulation.py \
  --experiment config/experiments/evaluation/table_penholder.json --check-config
```

Launching this evaluation consumes measured object poses after reset and each
simulation step. Its success criterion checks the pen body origin in a tabletop target
region; it does not assert contact or insertion. The sample stops after 1,000
steps and writes `outputs/evaluation/table_penholder/evaluation.json`. It does not
provide a movement policy. Missing object/contact evidence remains `unknown`.

The penholder scene has a matte black enclosure with a 2 m × 2 m clear interior,
a 2.2 m ceiling and a black ground plane. Its only enabled scene light is a
downward white 0.5 m × 0.12 m ceiling strip above the robot at world XY (0, 0).
The background asset is `config/scenes/backgrounds/penholder_enclosure.usda`;
adjust `CeilingStripLight.inputs:intensity` (default 300, exposure 0) to tune
brightness for the camera exposure, then restart the simulation. The enclosure
panels are visual geometry; ground collision continues to use the ground plane.

Schema 5 `placement.json` is a manifest referencing shared immutable episode
shards, event snapshots and robot geometry. New extraction does not duplicate a
complete JSON/Parquet bundle or a robot GLB for every task. Web publication
registers a manifest in `.placement-catalog.json`; it does not copy result files
into `public/data`. `--no-web-publish` disables registration. Existing schema 1–4
bundles and legacy `public/data/<task>/placement.json` + `robot.glb` pairs remain
readable and are not rewritten or removed automatically.

Run `npm run build` after frontend code changes, then use Vite dev/preview. Its
read-only routes serve the registered artifact closure. The page checks the task
catalog every five seconds, reuses unchanged model resources and fetches full
robot snapshots only when selecting events. A generic server serving only
`dist` does not provide these runtime routes. Keep the referenced artifact store
available; recognition screenshots and raw responses remain local audit data.

Replay opens camera viewers by default. When simulator cameras are disabled,
add `--no-viewer`; these separate processes do not share CLI options. Neither
candidate extraction nor source-video identification requires simulator images.
CPU stages use the current Python or `--extract-python`. `--dry-run` validates
without replay or artifact writes (an explicit `--report` still writes its report).
Use `python scripts/replay_extract_event.py --help` for all options.

### Switch to Other Simulators

To switch simulators or robot setups, modify the YAML file path in `astribot_simulation.py` (line 16):

```python
from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

def main(robot_type):
    # Load param from yaml, create a simulation env using the Factory Pattern
    astribot_yaml_file = f'config/astribot_s1/simulation_mujoco_param.yaml'
    astribot_envs_factory = AstribotEnvsFactory()
    astribot_data = AstribotEnvsFactory.load_yaml_file(astribot_yaml_file)
    astribot_simulation_thread = astribot_envs_factory.create_simulation_env(astribot_data)
```

---

## Architecture

To achieve high modularity and support multiple simulation backends, `astribot_simulation` is designed with a factory-pattern architecture.

### Class Diagram
<p align="center">
  <img src="docs/class_diagram.png" alt="Astribot Simulation Class Diagram" width="900">
</p>

### Design Highlights
*   **Unified API**: All simulation environments (MuJoCo, Genesis, Isaac Lab, ManiSkill) inherit from `AstribotBaseEnv`, which is compatible with `gym.Env`.
*   **Factory Pattern**: Users can switch between different simulators by simply changing the `simulator_type` in the configuration via `AstribotEnvsFactory`.
*   **Sim-to-Real Bridge**: The `MultiRobotRosInterface` ensures that the internal simulation states are perfectly aligned with ROS messages, facilitating seamless deployment to real hardware.

---

## YAML Configuration Files

The `config/` folder contains YAML files for each robot variant (`astribot_s0`, `astribot_s1`, `astribot_t1`). Each file defines a combination of **simulator backend**, **robot variant**, and **hardware configuration**:

| YAML File (under `config/astribot_s1/`) | Description |
|-----------|-------------|
| `simulation_mujoco_param.yaml` | MuJoCo backend with default configuration. |
| `simulation_mujoco_param_chassis_fixed.yaml` | MuJoCo backend with fixed chassis. |
| `simulation_mujoco_param_with_camera.yaml` | MuJoCo backend with three-camera setup. |
| `simulation_mujoco_param_with_hand.yaml` | MuJoCo backend with BrainCo hand. |
| `simulation_genesis_param_chassis_fixed.yaml` | Genesis backend with fixed chassis. |
| `simulation_maniskill_param_chassis_fixed.yaml` | ManiSkill backend with fixed chassis. |
| `simulation_isaaclab_param.yaml` | Isaac Lab backend. |

**Tip:**
To switch simulators or robot setups, modify the `astribot_yaml_file` variable in `astribot_simulation.py` to point to the desired YAML file. No code change is needed beyond this.

---

## System Requirements

- **Operating System:** Ubuntu 20.04 LTS / Ubuntu 22.04 LTS
- **Middleware:** ROS Noetic (for 20.04) / ROS2 Humble (for 22.04)
- **Python Version:** Python == 3.10 (use Conda environment)

| Hardware / Software | Recommended Specifications            |
| ------------------ | ----------------------------------- |
| CPU                | Intel i5-14600F or higher           |
| GPU                | NVIDIA RTX 2080 Ti or higher        |
| GPU Driver         | NVIDIA driver >= 535                 |
| CUDA               | CUDA >= 12.0                         |
| Python             | Python == 3.10 (Conda recommended)   |

> **Note:** For GPU-accelerated simulation, ensure the above hardware and driver requirements are met.

---

## Joint Space Command

If you want to directly control the joints, note: The robot exposes joint-space command topic (e.g., `/astribot_arm_left/joint_space_command`) for controlling the arm. The interpretation of the command depends on the control mode:

| Control Mode | Command Dimension | Meaning | Notes |
|--------------|-----------------|--------|-------|
| Position / Velocity Control | 7-14 | Values 7-14:<br>**First 7 values**: target joint **positions**<br>**Last 7 values**: target joint **velocities** | Includes **velocity compensation** and **gravity compensation** for smoother motion |
| Force Control | 7 | Each value represents the **torque/force** applied to the corresponding joint | Note: This mode does **not guarantee sim-to-real accuracy**, mainly for simulation purposes |

**Tip:**
If using **Astribot SDK**, you generally do **not** need to worry about these details — refer to the Astribot SDK documentation for more information. When using joint-space commands, ensure the control mode matches the command dimension to avoid unexpected behavior.

Gripper position commands and feedback use the SDK/dataset **0–100 scale**,
not radians or a normalized 0–1 value. In Isaac Lab, the compensated command is
clamped to this range and mapped to the model's 0–0.93 rad travel (negative for
L1/L2/R11, positive for R1/R2/L11). Thus an L1 command of 50 produces −0.465 rad
and reads back as 50. Velocity and acceleration feedback use scale units/s and
scale units/s² with the same direction conversion. Feedback is not clipped, so
small physical excursions beyond the commanded range remain visible.

---

## Sensors

The robot supports cameras on hands and head. After loading the correct YAML configuration, you can receive image data on the corresponding ROS topics:

- **Raw color image**: `/<robot_name>/camera/<camera_name>/image_raw`
- **Depth image**: `/<robot_name>/camera/<camera_name>/depth`
- **Point cloud (in camera frame)**: `/<robot_name>/camera/<camera_name>/point_cloud`

| Backend     | RGB | Depth | Point Cloud | Force/Torque | IMU |
|-------------|-----|-------|-------------|--------------|-----|
| **MuJoCo**   | Yes | Yes | Yes | Yes | Yes |
| **ManiSkill** | Yes | No | No | No | No |
| **Genesis**   | No | No | No | No | No |

**Tip:**
 - To enable sensors in **MuJoCo**, use: `simulation_mujoco_param_with_camera.yaml`
 - To enable sensors in **ManiSkill**, use: `simulation_maniskill_param_chassis_fixed.yaml`

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/install.sh` | Full environment setup |
| `scripts/build.sh` | Build `astribot_msgs` for ROS1 or ROS2 |
| `scripts/depends/Miniconda3-py38_4.9.2-Linux-x86_64.sh` | Install Miniconda |
| `scripts/lite_install/install_mujoco.sh` | MuJoCo-only install |
| `scripts/bugfix/fix_GLIBCXX_3.4.30_bug.sh` | Patch for GLIBCXX bug |
| `scripts/bugfix/fix_LIBFFI_BASE_7.0_bug.sh` | Patch for libffi bug |

---

## Documentation

- Environment wrappers: [`src/astribot_envs/`](src/astribot_envs/)
- Config examples: [`config/astribot_s1/`](config/astribot_s1/)
- Utilities: [`src/simu_utils/`](src/simu_utils/)

---

## Contributing

Contributions are welcome!
If you'd like to add support for new simulators or robots, please contact me at [tonywang@astribot.com].

---

## License

[BSD 3-Clause License](LICENSE)

---

## Acknowledgements

Built on top of:
- [MuJoCo](https://mujoco.org/)
- [Genesis](https://genesis-world.readthedocs.io/en/latest/)
- [ManiSkill](https://maniskill.ai/)
- [Isaac Lab](https://developer.nvidia.com/isaac/lab)
