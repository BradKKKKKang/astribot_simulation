#!/usr/bin/env bash
# Isaac Sim / Isaac Lab only: Ubuntu 22.04, ROS 2 Humble, Python 3.12.
# Run with bash. Re-running resumes downloads and reuses the Conda environment.
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IL_ENV_NAME="${IL_ENV_NAME:-astribot_simu_il3}"
source "$SIM_ROOT/scripts/lite_install/isaaclab_common.bash"
IL_CONDA_EXE="${IL_CONDA_EXE:-${CONDA_EXE:-$(type -P conda || echo "$HOME/miniconda3/bin/conda")}}"
IL_CONDA_CHANNEL="${IL_CONDA_CHANNEL:-conda-forge}"
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
Target: Ubuntu 22.04, ROS 2 Humble, CUDA 13.0 PyTorch wheels.
Override the Conda executable with IL_CONDA_EXE and the environment name with IL_ENV_NAME.
IL_CONDA_CHANNEL selects conda-forge or a mirror of it. ROS packages come from robostack-humble.
--skip-system-deps: install apt dependencies separately (ROS imports are still checked).
--skip-submodules: use already initialized submodules or manually restored assets.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

trap 'echo "Installation failed at line $LINENO. Fix the reported error and rerun this script." >&2' ERR

# System Humble on Jammy is built for Python 3.10. Keep its libraries out of
# the simulator process, including when the Docker image auto-sources ROS.
case ":${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:${PYTHONPATH:-}:${LD_LIBRARY_PATH:-}:" in
    *:/opt/ros/*)
        echo "System ROS paths are active. Open a fresh container shell (docker exec -it <name> bash --norc) before installing." >&2
        exit 1 ;;
esac
[[ -x "$IL_CONDA_EXE" ]] || { echo "Conda not found: $IL_CONDA_EXE" >&2; exit 1; }

export PYTHONNOUSERSITE=1
# Conda and ROS setup scripts are not safe with nounset enabled.
set +u
eval "$("$IL_CONDA_EXE" shell.bash hook)"
channels=(--override-channels --strict-channel-priority -c "$IL_CONDA_CHANNEL" -c robostack-humble)
ros_packages=(python=3.12 'numpy>=2,<3' pip uv
    ros-humble-ros-base ros-humble-rclpy ros-humble-cv-bridge ros-humble-tf-transformations
    ros-humble-rosidl-default-generators colcon-common-extensions 'cmake<4' empy=3.3.4
    make c-compiler cxx-compiler)
if conda activate "$IL_ENV_NAME" 2>/dev/null; then
    python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Expected Python 3.12. Select a fresh IL_ENV_NAME.")'
    "$IL_CONDA_EXE" install -n "$IL_ENV_NAME" -y "${channels[@]}" "${ros_packages[@]}"
else
    "$IL_CONDA_EXE" create -n "$IL_ENV_NAME" -y "${channels[@]}" "${ros_packages[@]}"
fi
# Run activation hooks from the newly installed Humble packages.
conda activate "$IL_ENV_NAME"
set -u
[[ ${ROS_DISTRO:-} == humble && ${ROS_VERSION:-} == 2 ]] || {
    echo "Conda activation did not load the RoboStack Humble environment." >&2; exit 1;
}
python - <<'PYROS'
import sys
from pathlib import Path
from rclpy.impl.implementation_singleton import rclpy_implementation

extension = Path(rclpy_implementation.__file__).resolve()
assert extension.is_relative_to(Path(sys.prefix).resolve()), f"Unexpected rclpy extension: {extension}"
PYROS

mkdir -p "$SIM_ROOT/.tmp/isaaclab3/tmp"
export TMPDIR="$SIM_ROOT/.tmp/isaaclab3/tmp"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SIM_ROOT/.tmp/isaaclab3/uv-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

prepare_isaaclab_assets
install_isaaclab_wheels "$CONDA_PREFIX/bin/uv" "$CONDA_PREFIX/bin/python"

# Allow the interactive first-use license prompt before running checks on stdin.
python -c 'from isaacsim import SimulationApp'

# Keep the Python 3.12 message overlay separate from native Humble builds.
ros_workspace="$CONDA_PREFIX/astribot_ros_humble"
mkdir -p "$ros_workspace"
(
    cd "$SIM_ROOT/astribot_msgs"
    ln -sfn CMakeLists_ros2.txt CMakeLists.txt
    ln -sfn package_ros2.xml package.xml
    cd "$ros_workspace"
    "$CONDA_PREFIX/bin/colcon" build --executor sequential --paths "$SIM_ROOT/astribot_msgs" \
        --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
)

set +u
export ASTRIBOT_MSGS_INSTALL="$ros_workspace/install"
source "$SIM_ROOT/env.sh"
set -u
python "$SIM_ROOT/scripts/lite_install/check_isaaclab.py"

echo "Installation complete. In a new Bash shell:"
printf '  source %q\n' "$("$IL_CONDA_EXE" info --base)/etc/profile.d/conda.sh"
printf '  conda activate %q\n' "$IL_ENV_NAME"
echo '  export ASTRIBOT_MSGS_INSTALL="$CONDA_PREFIX/astribot_ros_humble/install"'
printf '  source %q\n' "$SIM_ROOT/env.sh"
