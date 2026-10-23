#!/usr/bin/env bash
# Isaac Sim / Isaac Lab only: Ubuntu 22.04, ROS2 Humble, Python 3.10.
# Run with bash. Re-running resumes downloads and reuses the Conda environment.
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IL_ENV_NAME="${IL_ENV_NAME:-astribot_simu_il}"
IL_LAB_ROOT="${IL_LAB_ROOT:-$(dirname "$SIM_ROOT")/IsaacLab-2.1.1}"
IL_CONDA_EXE="${IL_CONDA_EXE:-${CONDA_EXE:-$HOME/miniconda3/bin/conda}}"
IL_CONDA_CHANNEL="${IL_CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
IL_SKIP_SYSTEM_DEPS=0
IL_SKIP_SUBMODULES=0

for arg in "$@"; do
    case "$arg" in
        --skip-system-deps) IL_SKIP_SYSTEM_DEPS=1 ;;
        --skip-submodules) IL_SKIP_SUBMODULES=1 ;;
        -h|--help)
            cat <<'HELP'
Usage: bash scripts/lite_install/install_isaaclab.sh [--skip-system-deps] [--skip-submodules]
Defaults: astribot_simu_il, Python 3.10, Isaac Sim 4.5.0, Isaac Lab v2.1.1.
Override paths with IL_CONDA_EXE and IL_LAB_ROOT; override name with IL_ENV_NAME.
IL_CONDA_CHANNEL selects the Conda channel (default: Tsinghua main mirror).
--skip-system-deps: install apt dependencies separately (ROS imports are still checked).
--skip-submodules: use already initialized submodules or manually restored assets.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

trap 'echo "Installation failed at line $LINENO. Fix the reported error and rerun this script." >&2' ERR

[[ -x "$IL_CONDA_EXE" ]] || { echo "Conda not found: $IL_CONDA_EXE" >&2; exit 1; }
[[ -f /opt/ros/humble/setup.bash ]] || { echo "Install ROS2 Humble first." >&2; exit 1; }

mkdir -p "$SIM_ROOT/.tmp/isaaclab/tmp" "$SIM_ROOT/.tmp/isaaclab/pip-cache" "$SIM_ROOT/.tmp/isaaclab/wheels"
export TMPDIR="$SIM_ROOT/.tmp/isaaclab/tmp"
export PIP_CACHE_DIR="$SIM_ROOT/.tmp/isaaclab/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONNOUSERSITE=1

if [[ "$IL_SKIP_SYSTEM_DEPS" == 0 ]]; then
    apt_packages=(git-lfs cmake build-essential python3-colcon-common-extensions
        ros-humble-cv-bridge ros-humble-tf-transformations)
    missing_packages=()
    for package in "${apt_packages[@]}"; do
        if [[ "$package" == git-lfs ]] && command -v git-lfs >/dev/null; then
            continue
        fi
        if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
            missing_packages+=("$package")
        fi
    done
    if (( ${#missing_packages[@]} )); then
        echo "Installing system dependencies: ${missing_packages[*]}"
        sudo apt-get update
        sudo apt-get install -y "${missing_packages[@]}"
    fi
fi

if [[ "$IL_SKIP_SUBMODULES" == 0 ]]; then
    echo "Downloading Astribot submodules and LFS assets..."
    command -v git-lfs >/dev/null
    GIT_LFS_SKIP_SMUDGE=1 git -C "$SIM_ROOT" submodule update --init --recursive
    # Fetch the USD assets needed by Isaac Lab, leaving unrelated LFS meshes alone.
    git -C "$SIM_ROOT/astribot_descriptions" lfs pull --include='usd/**' --exclude=''
fi
[[ -f "$SIM_ROOT/astribot_descriptions/usd/astribot_s1_usd/astribot_s1.usda" ]] || {
    echo "S1 USD model is missing. Initialize astribot_descriptions, including Git LFS." >&2; exit 1;
}
[[ -f "$SIM_ROOT/astribot_msgs/CMakeLists_ros2.txt" ]] || {
    echo "astribot_msgs sources are missing." >&2; exit 1;
}

if [[ ! -f "$IL_LAB_ROOT/isaaclab.sh" ]]; then
    git clone --branch v2.1.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$IL_LAB_ROOT"
fi
[[ "$(cat "$IL_LAB_ROOT/VERSION")" == "2.1.1" ]] || {
    echo "Expected Isaac Lab 2.1.1 at $IL_LAB_ROOT." >&2; exit 1;
}

# Conda and ROS setup scripts are not safe with nounset enabled.
set +u
eval "$("$IL_CONDA_EXE" shell.bash hook)"
if ! "$IL_CONDA_EXE" env list --json | "$(dirname "$IL_CONDA_EXE")/python" -c \
    'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$IL_ENV_NAME"; then
    "$IL_CONDA_EXE" create -n "$IL_ENV_NAME" python=3.10 pip -y \
        --override-channels -c "$IL_CONDA_CHANNEL"
fi
conda activate "$IL_ENV_NAME"
source /opt/ros/humble/setup.bash
set -u
python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

echo "Installing Isaac Sim 4.5.0 into $CONDA_PREFIX..."
python -m pip install 'setuptools>=70,<81' 'wheel==0.45.1' 'numpy==1.26.4' \
    pyyaml open3d 'opencv-python==4.11.0.86' transforms3d \
    catkin_pkg lark 'empy==3.3.4' pytz
# Install Lab's required Torch first so Sim does not download an unrelated latest CUDA stack.
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' \
    --find-links "$SIM_ROOT/.tmp/isaaclab/wheels" --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==4.5.0' \
    --find-links "$SIM_ROOT/.tmp/isaaclab/wheels" --extra-index-url https://pypi.nvidia.com
# Show NVIDIA's first-use EULA prompt before later Python heredocs consume stdin.
# For unattended runs, set OMNI_KIT_ACCEPT_EULA=YES only after accepting the license.
python -c 'import isaacsim'

echo "Installing Isaac Lab 2.1.1 (without RL frameworks)..."
# flatdict still imports pkg_resources; its isolated build pulls incompatible setuptools.
python -m pip install --no-build-isolation 'flatdict==4.0.1'
(cd "$IL_LAB_ROOT" && TERM=xterm bash isaaclab.sh --install none)

echo "Building Astribot ROS2 messages with Python 3.10..."
(
    cd "$SIM_ROOT/astribot_msgs"
    ln -sfn CMakeLists_ros2.txt CMakeLists.txt
    ln -sfn package_ros2.xml package.xml
    /usr/bin/colcon build --executor sequential --cmake-args \
        -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
        -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
)

set +u
source "$SIM_ROOT/env.sh"
set -u
python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys
from packaging.version import Version
import cv_bridge
import rclpy
import tf_transformations
from astribot_msgs.msg import RobotJointController, RobotJointState
from astribot_msgs.srv import RawRequest
from rclpy.serialization import serialize_message, deserialize_message
command = RobotJointController()
command.mode = 1
command.command = [0.1, 0.0]
restored = deserialize_message(serialize_message(command), RobotJointController)
assert restored.mode == 1 and list(restored.command) == [0.1, 0.0]
assert importlib.util.find_spec("isaaclab.app") is not None
assert Version(metadata.version("isaacsim")) == Version("4.5.0")
for name in ("isaacsim", "isaaclab", "torch", "gymnasium", "numpy"):
    print(f"{name}: {metadata.version(name)}")
for name in ("mujoco", "genesis", "mani_skill", "sapien"):
    assert importlib.util.find_spec(name) is None, f"Unexpected simulator installed: {name}"
print(f"ROS2 and Astribot message imports OK: {sys.executable}")
PY
python -m pip check

echo "Installation complete. Start S1 with:"
echo "  bash $SIM_ROOT/scripts/lite_install/run_isaaclab.sh"
echo "GPU rendering must be verified separately with nvidia-smi and a simulation launch."
