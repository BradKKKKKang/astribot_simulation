from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.grasp_diagnostics import read_contact_view


def test_contacts_use_counts_copy_reused_buffers_and_convert_with_dt():
    # Garbage/unoccupied slots must not leak into output. Friction querying
    # overwrites the PhysX count/index buffers used by the preceding query.
    counts = np.array([[2, 0]])
    starts = np.array([[1, 0]])
    received_dt = []
    def normal(dt):
        received_dt.append(dt)
        return (np.array([[np.nan], [2.], [3.], [np.nan]]),
                np.zeros((4, 3)), np.ones((4, 3)), np.zeros((4, 1)), counts, starts)
    def friction(dt):
        received_dt.append(dt)
        counts[:] = [[1, 0]]
        starts[:] = [[0, 0]]
        return np.array([[.1, .2, .3]] * 4), np.zeros((4, 3)), counts, starts
    view = SimpleNamespace(sensor_count=1, filter_count=2, max_contact_data_count=4,
                           get_contact_data=normal, get_friction_data=friction)
    result = read_contact_view(view, .01)
    assert received_dt == [.01, .01]
    assert result['pairs'][0]['normal_force_sum_n'] == 5.
    assert len(result['pairs'][0]['contacts']) == 2
    assert len(result['pairs'][0]['friction_anchors']) == 1
    assert result['pairs'][1]['normal_force_sum_n'] == 0.
    assert result['pairs'][1]['contacts'] == []
    assert not result['possibly_truncated']


def test_contacts_reject_invalid_buffer_ranges():
    view = SimpleNamespace(sensor_count=1, filter_count=1, max_contact_data_count=1,
        get_contact_data=lambda dt: (np.zeros((1, 1)), np.zeros((1, 3)), np.zeros((1, 3)),
                                     np.zeros((1, 1)), np.array([[2]]), np.array([[0]])),
        get_friction_data=lambda dt: (np.zeros((1, 3)), np.zeros((1, 3)),
                                      np.array([[0]]), np.array([[0]])))
    with pytest.raises(RuntimeError, match='capacity'):
        read_contact_view(view, .01)
