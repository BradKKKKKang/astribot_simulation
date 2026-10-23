import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/joint_align/replay_artifacts_sdk_100hz.py"
SPEC = importlib.util.spec_from_file_location("replay_artifacts_sdk_100hz", MODULE)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def write_episode(path, *, skew_joint=None):
    rows = []
    grid = [1.0, 1.01, 1.02]
    for joint in REPLAY.JOINTS:
        offset = 0.001 if joint == skew_joint else 0.0
        for frame, timestamp in enumerate(grid):
            rows.append({
                "kind": "command",
                "joint": joint,
                "time": timestamp + offset,
                "position": (50.0 if "gripper" in joint else timestamp) + frame * 0.01,
                "velocity": 0.0,
                "torque": 0.0,
                "acceleration": 0.0,
                "mode": 1,
            })
        for timestamp in [0.99, 1.0, 1.01, 1.02, 1.03]:
            rows.append({
                "kind": "state",
                "joint": joint,
                "time": timestamp,
                "position": 50.0 + timestamp if "gripper" in joint else timestamp,
                "velocity": 1.0,
                "torque": 0.0,
                "mode": 1,
            })
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_load_episode_requires_complete_common_100hz_grid(tmp_path):
    path = tmp_path / "episode.parquet"
    write_episode(path)

    loaded_path, grid, command, states = REPLAY.load_episode(path)

    assert loaded_path == path.resolve()
    assert command.shape == (3, 22)
    assert len(states) == 22
    np.testing.assert_allclose(np.diff(grid), 0.01)


def test_load_episode_rejects_skewed_joint_grid(tmp_path):
    path = tmp_path / "episode.parquet"
    write_episode(path, skew_joint=REPLAY.JOINTS[-1])

    with pytest.raises(ValueError, match="do not share a time grid"):
        REPLAY.load_episode(path)


def test_errors_use_simulator_header_and_sdk_receive_times_separately(tmp_path):
    path = tmp_path / "episode.parquet"
    write_episode(path)
    _, grid, _, states = REPLAY.load_episode(path)
    epoch_system_ns = 10_000_000_000
    epoch_mono_ns = 20_000_000_000
    rows = []

    for joint in REPLAY.JOINTS:
        for relative_time in (0.01, 0.02):
            position = (50.0 + grid[0] + relative_time
                        if "gripper" in joint else grid[0] + relative_time)
            rows.append({
                "joint": joint,
                "position": position,
                "velocity": 1.0,
                "header_stamp_ns": epoch_system_ns + int(relative_time * 1e9),
                "received_monotonic_ns": epoch_mono_ns + int((relative_time + 0.002) * 1e9),
            })

    by_header = REPLAY._metric_for_rows(
        states, rows, grid[0], grid[-1], epoch_system_ns, epoch_mono_ns,
        "simulator_header_time",
    )
    by_receive = REPLAY._metric_for_rows(
        states, rows, grid[0], grid[-1], epoch_system_ns, epoch_mono_ns,
        "sdk_receiver_time",
    )

    assert by_header["groups"]["angular_joints"]["position_rmse"] == pytest.approx(0.0)
    assert by_header["groups"]["grippers"]["position_rmse"] == pytest.approx(0.0)
    assert by_receive["groups"]["angular_joints"]["position_rmse"] == pytest.approx(0.002)
    assert by_receive["groups"]["grippers"]["position_rmse"] == pytest.approx(0.002)


def test_gripper_sim_names_map_to_artifact_joint_names():
    assert REPLAY._canonical_joint("astribot_gripper_left_joint_L1") == "astribot_gripper_left_joint_1"


def test_sdk_command_capture_compares_each_part_local_position_payload():
    position = np.arange(44, dtype=np.float64).reshape(2, 22)
    api_rows = [
        {"call_start_monotonic_ns": 1_000_000_000, "position": position[0].tolist()},
        {"call_start_monotonic_ns": 1_010_000_000, "position": position[1].tolist()},
    ]
    sdk_rows = []
    offset = 0
    for part, count in REPLAY.PART_CONFIGS:
        local = position[1, offset:offset + count]
        sdk_rows.append({
            "part": part,
            "mode": 1,
            "command": np.concatenate([local, np.zeros(count * 2)]).tolist(),
            "received_monotonic_ns": 1_011_000_000,
        })
        offset += count

    result = REPLAY._command_delivery_metrics(api_rows, sdk_rows)

    assert all(count == 1 for count in result["matched_frames_by_part"].values())
    assert result["groups"]["angular_joints"] == pytest.approx(0.0)
    assert result["groups"]["grippers"] == pytest.approx(0.0)
