#!/usr/bin/env bash
# Run isolated, same-process 100 Hz joint tuning with the installed Isaac env.
set -eo pipefail
cd "$(dirname "$0")/.."
source "${ASTRIBOT_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ASTRIBOT_ISAAC_ENV:-astribot_simu_il}"
source /opt/ros/humble/setup.bash
source astribot_msgs/install/setup.bash
export ASTRIBOT_SIMU_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ROS_HOME=/tmp/astribot_roslog_torso
export ROS_VERSION=2 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=197
python scripts/replay_torso_tuning_100hz.py "$@"
# Kit may exit the interpreter during close and mask a Python exception.
# Completion requires the manifest written after all trials, not exit code 0.
python - "$@" <<'PY'
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--output-root', type=Path, required=True)
args, _ = p.parse_known_args()
manifest = args.output_root / 'run_manifest.json'
if not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'complete':
    raise SystemExit(f'Incomplete tuning run; inspect {args.output_root} and the simulation log')
PY
