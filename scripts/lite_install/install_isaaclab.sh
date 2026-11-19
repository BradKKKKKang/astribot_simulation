#!/usr/bin/env bash
# Isaac Sim / Isaac Lab only: Ubuntu 24.04, ROS 2 Jazzy, Python 3.12.
# Run with bash. Re-running resumes downloads and reuses the Conda environment.
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IL_ENV_NAME="${IL_ENV_NAME:-astribot_simu_il3}"
LAB_OVERRIDES="https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"
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
Defaults: astribot_simu_il3, Python 3.12, Isaac Lab 3.0.0, Isaac Sim 6.2.0.0.
Target: Ubuntu 24.04, ROS 2 Jazzy, CUDA 13.0 PyTorch wheels.
Override the Conda executable with IL_CONDA_EXE and the environment name with IL_ENV_NAME.
IL_CONDA_CHANNEL selects the Conda channel (default: Tsinghua main mirror).
--skip-system-deps: install apt dependencies separately (ROS imports are still checked).
--skip-submodules: use already initialized submodules or manually restored assets.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

trap 'echo "Installation failed at line $LINENO. Fix the reported error and rerun this script." >&2' ERR

[[ -r /opt/ros/jazzy/setup.bash && ${ROS_DISTRO:-jazzy} == jazzy ]] || {
    echo "Use an Ubuntu 24.04 / ROS 2 Jazzy shell. The Humble Python 3.10 binaries cannot load in Python 3.12." >&2
    exit 1
}
[[ -x "$IL_CONDA_EXE" ]] || { echo "Conda not found: $IL_CONDA_EXE" >&2; exit 1; }

export PYTHONNOUSERSITE=1
# Conda and ROS setup scripts are not safe with nounset enabled.
set +u
eval "$("$IL_CONDA_EXE" shell.bash hook)"
if ! "$IL_CONDA_EXE" env list --json | "$(dirname "$IL_CONDA_EXE")/python" -c \
    'import json,sys; from pathlib import Path; sys.exit(not any(Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$IL_ENV_NAME"; then
    "$IL_CONDA_EXE" create -n "$IL_ENV_NAME" python=3.12 pip -y \
        --override-channels -c "$IL_CONDA_CHANNEL"
fi
conda activate "$IL_ENV_NAME"
set -u
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Expected Python 3.12. Select a fresh IL_ENV_NAME.")'

mkdir -p "$SIM_ROOT/.tmp/isaaclab3/tmp"
export TMPDIR="$SIM_ROOT/.tmp/isaaclab3/tmp"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SIM_ROOT/.tmp/isaaclab3/uv-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

if [[ "$IL_SKIP_SYSTEM_DEPS" == 0 ]]; then
    apt_packages=(git-lfs cmake build-essential python3-colcon-common-extensions
        ros-jazzy-cv-bridge ros-jazzy-tf-transformations libopencv-dev libboost-python-dev)
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
        apt_cmd=(apt-get)
        if (( EUID != 0 )); then apt_cmd=(sudo apt-get); fi
        "${apt_cmd[@]}" update
        "${apt_cmd[@]}" install -y "${missing_packages[@]}"
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

echo "Installing the Isaac Lab 3.0 wheel and its Isaac Sim 6.2 extra into $CONDA_PREFIX..."
# The tagged NVIDIA overrides reconcile conflicting Lab and Sim package metadata.
python -m pip install --upgrade uv
python -m uv pip install --python "$CONDA_PREFIX/bin/python" \
    'isaaclab[isaacsim]==3.0.0' 'isaacsim==6.2.0.0' \
    --overrides "$LAB_OVERRIDES" --extra-index-url https://pypi.nvidia.com \
    --torch-backend cu130 --index-strategy unsafe-best-match \
    'transforms3d>=0.4.2' catkin_pkg lark 'empy==3.3.4' pytz pyyaml

# Allow the interactive first-use license prompt before running checks on stdin.
python -c 'from isaacsim import SimulationApp'
set +u
source /opt/ros/jazzy/setup.bash
set -u

# Build native Python extensions against this environment's NumPy 2 headers.
# Keep the Humble build and install directories usable from the old environment.
ros_workspace="$CONDA_PREFIX/astribot_ros"
mkdir -p "$ros_workspace"
if [[ ! -d "$ros_workspace/vision_opencv" ]]; then
    git clone --depth 1 --branch 4.1.0 https://github.com/ros-perception/vision_opencv.git "$ros_workspace/vision_opencv"
fi
(
    cd "$SIM_ROOT/astribot_msgs"
    ln -sfn CMakeLists_ros2.txt CMakeLists.txt
    ln -sfn package_ros2.xml package.xml
    cd "$ros_workspace"
    /usr/bin/colcon build --executor sequential \
        --paths "$SIM_ROOT/astribot_msgs" "$ros_workspace/vision_opencv/cv_bridge" \
        --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
)

set +u
export ASTRIBOT_MSGS_INSTALL="$ros_workspace/install"
source "$SIM_ROOT/env.sh"
set -u
python - <<'PYTHON'
import sys
from importlib.metadata import version

import numpy as np
import open3d
import torch
import tf_transformations
from cv_bridge import CvBridge
from isaaclab.app import AppLauncher
from packaging.version import Version
from astribot_msgs.msg import RobotJointController, RobotJointState
from astribot_msgs.srv import RawRequest
from rclpy.serialization import serialize_message, deserialize_message

expected_versions = {
    "isaaclab": "3.0.0",
    "isaacsim": "6.2.0.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "warp-lang": "1.17.0",
    "newton": "1.6.1",
}
for name, expected in expected_versions.items():
    installed = Version(version(name))
    assert Version(installed.public) == Version(expected), f"{name}: expected {expected}, found {installed}"
    print(f"{name}: {installed}")
assert torch.version.cuda == "13.0", torch.version.cuda
assert Version(np.__version__).major == 2, np.__version__

bridge = CvBridge()
rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
image = bridge.cv2_to_imgmsg(rgb, "rgb8")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(image, "bgr8"), rgb[:, :, ::-1])
command = RobotJointController()
command.mode = 1
command.command = [0.1, 0.0]
restored = deserialize_message(serialize_message(command), RobotJointController)
assert restored.mode == 1 and list(restored.command) == [0.1, 0.0]
print(f"ROS 2, camera conversion and Astribot messages OK: {sys.executable}")
PYTHON

echo "Installation complete. In a new Bash shell:"
printf '  source %q\n' "$(dirname "$(dirname "$IL_CONDA_EXE")")/etc/profile.d/conda.sh"
printf '  conda activate %q\n' "$IL_ENV_NAME"
echo '  source /opt/ros/jazzy/setup.bash'
echo '  export ASTRIBOT_MSGS_INSTALL="$CONDA_PREFIX/astribot_ros/install"'
printf '  source %q\n' "$SIM_ROOT/env.sh"
