#!/usr/bin/env bash
# Install the Isaac wheels in a dedicated uv environment.
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$SIM_ROOT/scripts/lite_install/isaaclab_common.bash"
IL_SKIP_SYSTEM_DEPS=0
IL_SKIP_SUBMODULES=0
for arg in "$@"; do
    case "$arg" in
        --skip-system-deps) IL_SKIP_SYSTEM_DEPS=1 ;;
        --skip-submodules) IL_SKIP_SUBMODULES=1 ;;
        -h|--help)
            cat <<'HELP'
Usage: bash scripts/lite_install/install_isaaclab_uv.sh [--skip-system-deps] [--skip-submodules]
Install Python 3.12, Isaac Lab 3.0.0, Isaac Sim 6.2.0.0 and common simulation dependencies.
IL_UV_ENV selects the virtualenv (default: .venv in the repository).
IL_KIT_ROS_PREFIX selects the compiled Python 3.12 Astribot Kit ROS overlay (default: /opt/astribot-kit-ros).
--skip-system-deps skips installing git-lfs. --skip-submodules uses existing assets.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 1; }
case ":${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:${PYTHONPATH:-}:${LD_LIBRARY_PATH:-}:" in
    *:/opt/ros/*) echo "Start in a fresh shell before sourcing system ROS." >&2; exit 1 ;;
esac
venv="$(realpath -m "${IL_UV_ENV:-$SIM_ROOT/.venv}")"
overlay="$(realpath -m "${IL_KIT_ROS_PREFIX:-/opt/astribot-kit-ros}")"
for package in astribot_msgs cv_bridge tf_transformations; do
    [[ -d "$overlay/python/$package" ]] || {
        echo "Missing $overlay/python/$package. Set IL_KIT_ROS_PREFIX to a Python 3.12 Humble overlay." >&2
        exit 1
    }
done

if [[ -e "$venv" ]]; then
    [[ -f "$venv/pyvenv.cfg" && -x "$venv/bin/python" ]] || {
        echo "Expected a virtualenv at $venv. Choose a new IL_UV_ENV." >&2; exit 1;
    }
    "$venv/bin/python" -I -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Expected a Python 3.12 virtualenv")'
else
    UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12 "$venv"
fi

prepare_isaaclab_assets
# Detect a stale prebuilt overlay before downloading the simulator.
for interface in "$SIM_ROOT"/astribot_msgs/{msg/*.msg,srv/*.srv,action/*.action}; do
    cmp -s "$interface" "$overlay/share/astribot_msgs/${interface#"$SIM_ROOT/astribot_msgs/"}" || {
        echo "Kit ROS overlay does not match the repository interface $interface" >&2; exit 1;
    }
done
install_isaaclab_wheels uv "$venv/bin/python" -r "$SIM_ROOT/pyproject.toml"

# Save the paths used for this install, including any custom overlay or venv.
{
    printf 'source %q || return 1\n' "$venv/bin/activate"
    printf 'export ASTRIBOT_MSGS_INSTALL=%q ROS_VERSION=2\n' "$overlay"
    printf 'source %q\n' "$SIM_ROOT/env.sh"
} > "$venv/activate-astribot.bash"
source "$venv/activate-astribot.bash"

# Give the NVIDIA EULA prompt access to the terminal.
python -c 'from isaacsim import SimulationApp'
python "$SIM_ROOT/scripts/lite_install/check_isaaclab.py"
printf 'Installation complete. In a fresh Bash shell run:\n  source %q\n' "$venv/activate-astribot.bash"
