# Get the directory of current script
if [ -n "${ZSH_VERSION:-}" ]; then
    # Zsh syntax
    export ASTRIBOT_SIMU_ROOT="${${(%):-%x}:A:h}"
    astribot_setup_shell=zsh
else
    # Bash syntax
    export ASTRIBOT_SIMU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    astribot_setup_shell=bash
fi

# Execute different commands based on ROS_VERSION
if [ "${ROS_VERSION:-}" = "1" ]; then
    astribot_msgs_setup="$ASTRIBOT_SIMU_ROOT/astribot_msgs/build/devel/setup.$astribot_setup_shell"
elif [ "${ROS_VERSION:-}" = "2" ]; then
    astribot_msgs_setup="$ASTRIBOT_SIMU_ROOT/astribot_msgs/install/setup.$astribot_setup_shell"
else
    echo "env.sh: ROS_VERSION must be 1 or 2 (current: ${ROS_VERSION:-unset})." >&2
    echo "Load ROS first; for ROS2 Humble: source /opt/ros/humble/setup.$astribot_setup_shell" >&2
    echo "Then run: source \"$ASTRIBOT_SIMU_ROOT/env.sh\"" >&2
    unset astribot_setup_shell
    # This script is sourced: return to the caller instead of exiting its shell.
    return 1
fi

if [ ! -f "$astribot_msgs_setup" ]; then
    echo "env.sh: Missing message workspace setup: $astribot_msgs_setup" >&2
    echo "Build the astribot_msgs workspace before sourcing env.sh." >&2
    unset astribot_setup_shell astribot_msgs_setup
    return 1
fi
if ! source "$astribot_msgs_setup"; then
    echo "env.sh: Failed to load the astribot_msgs workspace." >&2
    unset astribot_setup_shell astribot_msgs_setup
    return 1
fi
unset astribot_setup_shell astribot_msgs_setup

case ":${PYTHONPATH:-}:" in
    *":$ASTRIBOT_SIMU_ROOT:"*) ;;
    *) export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ASTRIBOT_SIMU_ROOT" ;;
esac
export PYTHONPATH
export PYTHONWARNINGS="ignore"
export ROS_DOMAIN_ID=25
