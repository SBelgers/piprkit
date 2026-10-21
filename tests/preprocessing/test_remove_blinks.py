import numpy as np
import pytest

from piprkit import PupilMeasurement, PupilSeries


def _nan_indices(pm):
    return list(np.flatnonzero(np.isnan(pm.get_size())))


def test_remove_blinks_edges_are_inclusive():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(3, 5)])
    pm.filters.remove_blinks()
    assert _nan_indices(pm) == [3, 4, 5]


def test_remove_blinks_several_blinks():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    # No sample at t = 1.5, so the first blink only covers t = 1.
    pm.set_blinks([(1, 1.5), (6, 7)])
    pm.filters.remove_blinks()
    assert _nan_indices(pm) == [1, 6, 7]


def test_remove_blinks_without_blinks_changes_nothing():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.filters.remove_blinks()
    np.testing.assert_allclose(pm.get_size(), np.ones(10))


def test_remove_blinks_not_inplace():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(3, 5)])
    result = pm.filters.remove_blinks(inplace=False)
    assert _nan_indices(result) == [3, 4, 5]
    np.testing.assert_allclose(pm.get_size(), np.ones(10))


def test_remove_blinks_on_series():
    ps = PupilSeries(np.arange(10.0), np.ones(10), [(2.0, 3.0), (6.0, 7.0)])
    ps.set_blinks([(4, 5)])
    ps.filters.remove_blinks()
    assert _nan_indices(ps) == [4, 5]


def test_remove_blinks_buffer_seconds():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(4, 5)])
    # Widened to [3, 6].
    pm.filters.remove_blinks(buffer_seconds=1)
    assert _nan_indices(pm) == [3, 4, 5, 6]


def test_remove_blinks_does_not_change_earlier_get_size_result():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(3, 5)])
    size_before = pm.get_size()
    pm.filters.remove_blinks()
    np.testing.assert_allclose(size_before, np.ones(10))


def test_remove_blinks_keeps_blink_list():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(4, 5)])
    pm.filters.remove_blinks()
    assert pm.get_blinks() == [(4, 5)]


def test_remove_blinks_twice_is_harmless():
    pm = PupilMeasurement(np.arange(10.0), np.ones(10))
    pm.set_blinks([(4, 5)])
    pm.filters.remove_blinks()
    pm.filters.remove_blinks()
    assert _nan_indices(pm) == [4, 5]
