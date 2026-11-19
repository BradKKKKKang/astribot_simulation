"""Load the installed Astribot ROS environment for a standalone CLI process."""
from importlib.util import find_spec
import os
from pathlib import Path
import sys


def activate_cli_ros():
    if sys.flags.ignore_environment or os.environ.get("ROS_VERSION") not in (None, "", "2"):
        return
    if any(
        flag in sys.argv[1:] for flag in ("--help", "-h", "--check-config")
    ):
        return
    if os.environ.get("ROS_VERSION") == "2" and find_spec("cv_bridge") is not None:
        return
    activation = Path(sys.prefix) / "activate-astribot.bash"
    if activation.is_file():
        # The native loader reads LD_LIBRARY_PATH at process startup.
        # Reuse the installer's activation script before importing the backend.
        os.execlp(
            "bash", "bash", "-c",
            'source "$1" || exit $?; shift; '
            '"$1" -c "import cv_bridge" || exit $?; exec "$@"',
            "astribot", str(activation), sys.executable, *sys.orig_argv[1:],
        )
