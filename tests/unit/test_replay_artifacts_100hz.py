"""Offline replay input, timestamp, and artifact-format contracts (no Isaac/ROS)."""
import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/replay_artifacts_100hz.py'
spec = importlib.util.spec_from_file_location('replay_artifacts_100hz', MODULE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def write_episode(path, *, missing=None, skew=None):
    rows = []
    for joint in replay.JOINTS:
        if joint != missing:
            for i in range(3):
                rows.append(dict(kind='command', joint=joint, time=1.0 + i*.01 + (0.001 if joint == skew else 0),
                                 position=float(i), velocity=0., torque=0., mode=1))
        for i in range(4 if joint == replay.JOINTS[0] else 3):
            rows.append(dict(kind='state', joint=joint, time=.99 + i*.02, position=float(i),
                             velocity=0., torque=0., mode=1))
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_rejects_missing_joint_instead_of_zero_fill(tmp_path):
    path = tmp_path / 'samples.parquet'
    write_episode(path, missing=replay.JOINTS[0])
    with pytest.raises(ValueError, match='command joints mismatch'):
        replay.load_episode(path)


def test_rejects_individually_regular_but_skewed_grids(tmp_path):
    path = tmp_path / 'samples.parquet'
    write_episode(path, skew=replay.JOINTS[-1])
    with pytest.raises(ValueError, match='share a grid'):
        replay.load_episode(path)


def test_raw_real_arrays_preserve_unequal_lengths_without_pickle(tmp_path):
    path = tmp_path / 'samples.parquet'
    write_episode(path)
    grid, position, states, table = replay.load_episode(path)
    replay.save_real_states(tmp_path / 'real.npz', states)
    with np.load(tmp_path / 'real.npz', allow_pickle=False) as saved:
        assert saved['offsets'][-1] == table.num_rows
        for i, name in enumerate(saved['joint_names']):
            start, end = saved['offsets'][i:i+2]
            np.testing.assert_array_equal(saved['values'][start:end], states[name])
    assert position.shape == (3, 22)
    np.testing.assert_allclose(np.diff(grid), .01)


def test_post_step_state_uses_physics_time_and_separates_units():
    # q=t: labelling a post-step state t=.01 with command time t=0
    # would introduce an error of .01. The comparison must use state time.
    real = {name: np.array([[0., 0., 1., 0.], [.01, .01, 1., 0.], [.02, .02, 1., 0.]])
            for name in replay.JOINTS}
    time = np.array([.01, .02, .03])
    q = np.broadcast_to(time[:, None], (3, 22))
    v = np.ones_like(q)
    result = replay.tracking_errors(real, time, q, v, np.zeros_like(q), .02)
    assert all(v['position']['rmse'] == 0. for v in result['joints'].values())
    assert all(v['position']['samples'] == 2 for v in result['joints'].values())
    assert result['groups']['angular_joints']['position_unit'] == 'rad'
    assert result['groups']['grippers']['position_unit'] == 'percentage_points'
    assert result['torque_comparable'] is False
