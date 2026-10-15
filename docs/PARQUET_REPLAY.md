# Parquet replay through the ROS2 SDK

`scripts/replay_parquet.py` reads `joints_dict.joints_position_command` from an
Astribot episode and sends it through `Astribot.set_joints_position` in direct
joint control mode. The SDK publishes ROS2 joint commands consumed by the
simulator. The fixed chassis's first three values are excluded; torso, arms,
grippers and head retain dataset order and SDK units (grippers: 0–100).

The simulator and replay are separate processes and can use different Python
environments. Run replay with the Python that runs the SDK examples successfully;
activating the simulator's Conda environment does not install SDK dependencies.
Replay additionally requires PyArrow and a graphical desktop for three viewers.

## Setup and run (Zsh, ROS2 Humble)

Simulator terminal, from this repository:

```zsh
conda activate astribot_simu_il
source /opt/ros/humble/setup.zsh
source env.sh
export ROS_DOMAIN_ID=25 ROS_LOCALHOST_ONLY=1
python astribot_simulation.py
```

Replay terminal, using the working SDK environment. Replace the SDK and episode
paths below. Use an isolated simulation ROS domain; both terminals must have the
same domain and localhost setting. SDK `env.zsh` resets these variables, so set
them **after** sourcing it. The simulation must publish all three raw RGB cameras.

```zsh
source /opt/ros/humble/setup.zsh
source ../astribot_sdk_ros2/env.zsh
export ROS_DOMAIN_ID=25 ROS_LOCALHOST_ONLY=1
# Remain in this simulation repository, using the SDK's working Python.
python -m pip install -r scripts/requirements-replay.txt
python scripts/replay_parquet.py /path/to/episode_000000.parquet --dry-run
python scripts/replay_parquet.py /path/to/episode_000000.parquet
```

If the SDK normally uses system Python, use `/usr/bin/python` consistently for
both installation and replay. No SDK environment needs to be loaded into Isaac
Lab's process. If using the simulator's Python for replay instead, it must also
have all SDK dependencies; a missing `cryptography` can otherwise appear inside
the native SDK as a misleading `robotics_library_py ... is not a package` error.
Replay checks this dependency before waiting for the simulator or taking control.

## Timing and exit

`--start-frame` is inclusive and `--end-frame` exclusive. By default, timestamps
determine timing; `--fps` overrides timestamps and `--speed` scales elapsed time.
The initial pose transition uses a three-second smoothstep ramp, adjustable with
`--initial-duration`. Both phases send at 100 Hz, matching the SDK frequency.
Each dataset target is held until the next timestamp; delayed ticks are skipped
without sending a burst of old targets. The final dataset target is sent too.

Replay checks simulation feedback and all three camera streams before creating
the SDK client. It opens head and wrist camera viewers. Press `q` in a viewer or
Ctrl-C to stop; `--exit-after-replay` closes them automatically. Cleanup calls SDK
stop, shuts down middleware and terminates the viewer process. The separately
started simulator stays running.
