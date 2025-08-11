import numpy as np
import pytest

from pypipr.core.pupil_measurement import PupilMeasurement


def test_pupil_measurement_initialization():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    with pytest.raises(ValueError):
        pm = PupilMeasurement(time_data, size[:-1])
    with pytest.raises(ValueError):
        pm = PupilMeasurement(time_data[:-1], size)
    pm = PupilMeasurement(time_data, size)
    assert isinstance(pm, PupilMeasurement)



def test_set_time_offset():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    pm = PupilMeasurement(time_data, size)

    offset = 10
    pm.set_time_offset(offset)

    expected_time = np.array(time_data) + offset
    np.testing.assert_array_equal(pm.get_time(), expected_time)
    np.testing.assert_array_equal(pm.get_size(), size)


def test_interpolate():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    pm = PupilMeasurement(time_data, size)

    new_time = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    pm.interpolate(new_time)

    expected_size = np.array([6.05, 6.15, 6.25, 6.35, 6.45])
    np.testing.assert_allclose(pm.get_size(), expected_size)
    np.testing.assert_allclose(pm.get_time(), new_time)

    pm = PupilMeasurement(time_data, size)
    new_time = np.array([-1, 0, 1, 2, 3, 4, 5, 6])
    pm.interpolate(new_time)
    expected_size = np.array([np.nan, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, np.nan])
    np.testing.assert_allclose(pm.get_size(), expected_size)


def test_drop_nan():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, np.nan, 6.2, np.nan, 6.4, 6.5])
    pm = PupilMeasurement(time_data, size)
    pm.drop_nan()

    expected_time = np.array([0, 2, 4, 5])
    expected_size = np.array([6.0, 6.2, 6.4, 6.5])

    np.testing.assert_array_equal(pm.get_time(), expected_time)
    np.testing.assert_array_equal(pm.get_size(), expected_size)


def test_trim_time():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_time(2, np.inf)
    expected_time = np.array([2,3,4,5])
    expected_size = np.array([6.2,6.3,6.4,6.5])
    np.testing.assert_allclose(pm.get_time(), expected_time, equal_nan=True)
    np.testing.assert_allclose(pm.get_size(), expected_size, equal_nan=True)
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_time(-np.inf, 3)
    expected_time = np.array([0, 1, 2, 3])
    expected_size = np.array([6.0, 6.1, 6.2, 6.3])
    np.testing.assert_allclose(pm.get_time(), expected_time, equal_nan=True)
    np.testing.assert_allclose(pm.get_size(), expected_size, equal_nan=True)
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_time(1, 4, drop=False)
    expected_size = np.array([np.nan, 6.1, 6.2, 6.3, 6.4, np.nan])
    expected_time = np.array([0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(pm.get_size(), expected_size, equal_nan=True)
    np.testing.assert_allclose(pm.get_time(), expected_time, equal_nan=True)
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_time(10, 20, drop=False)
    expected_size = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(pm.get_size(), expected_size, equal_nan=True)


def test_trim_size():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.82, 6.3, 6.4, 6.5])
    pm = PupilMeasurement(time_data, size)
    pm.trim_size(6.1, 6.4)
    expected_size =[np.nan, 6.1, np.nan, 6.3, 6.4, np.nan] 
    np.testing.assert_allclose(pm.get_size(),expected_size,equal_nan=True)
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_size(10, 20)
    expected_size = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    np.testing.assert_allclose(pm.get_size(),expected_size,equal_nan=True)
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_size(-np.inf, 6.4)
    expected_size = [6.0, 6.1, np.nan, 6.3, 6.4, np.nan]
    np.testing.assert_allclose(
        pm.get_size(),
        expected_size,
        equal_nan=True,
    )
    pm = PupilMeasurement(time_data, size)
    pm.trim_size(-np.inf, np.inf)
    np.testing.assert_allclose(
        pm.get_size(),
        size,
        equal_nan=True,
    )

def test_pupil_measurement_inheritance():
    import pypipr.core.pupil_base

    time_data = np.array([0, 1, 2, 3])
    size = np.array([6.0, 6.1, 6.2, 6.3])
    pm = PupilMeasurement(time_data, size)
    assert isinstance(pm, pypipr.core.pupil_base.PupilBase)