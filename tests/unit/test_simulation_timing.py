import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "src/astribot_envs/simulation_timing.py"
SPEC = importlib.util.spec_from_file_location("simulation_timing", MODULE)
TIMING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)


def test_periodic_deadline_advances_from_absolute_schedule():
    assert TIMING.advance_periodic_deadline(100, 105, 10) == 110


def test_missed_deadline_resynchronizes_now_without_catchup():
    assert TIMING.advance_periodic_deadline(100, 127, 10) == 127


def test_periodic_deadline_requires_positive_period():
    with pytest.raises(ValueError, match="positive"):
        TIMING.advance_periodic_deadline(100, 100, 0)
