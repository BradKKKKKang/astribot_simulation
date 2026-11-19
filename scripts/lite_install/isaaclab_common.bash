# Shared by the Conda and uv lite installers.

prepare_isaaclab_assets() {
    if [[ "$IL_SKIP_SYSTEM_DEPS" == 0 ]] && ! command -v git-lfs >/dev/null; then
        local apt_cmd=(apt-get)
        if (( EUID != 0 )); then
            command -v sudo >/dev/null || {
                echo "Install git-lfs as root or supply the assets with --skip-system-deps --skip-submodules." >&2
                return 1
            }
            apt_cmd=(sudo apt-get)
        fi
        "${apt_cmd[@]}" update
        "${apt_cmd[@]}" install -y git-lfs
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
}

install_isaaclab_wheels() {
    local uv_exe="$1" interpreter="$2"
    shift 2
    local overrides="https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"
    "$uv_exe" pip install --python "$interpreter" \
        'isaaclab[isaacsim]==3.0.0' 'isaacsim==6.2.0.0' \
        --overrides "$overrides" --extra-index-url https://pypi.nvidia.com \
        --torch-backend cu130 --index-strategy unsafe-best-match \
        'transforms3d>=0.4.2' pytz pyyaml "$@"
}
