from pypipr.core.pupil_measurement import PupilMeasurement
import pytest
import numpy as np

def test_rolling_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)

    def set_nan(window: np.ndarray) -> float:
        return np.nan

    pupil_measurement.rolling_filter(set_nan, 3)
    with pytest.warns(UserWarning):
        pupil_measurement.rolling_filter(set_nan, 2)
    pupil_measurement.rolling_filter(set_nan, 3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    assert np.isnan(pupil_measurement.get_size()).all()

    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.rolling_filter(np.sum, 3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    assert (
        pupil_measurement.get_size() == np.array([21, 30, 30, 26, 27, 28, 21])
    ).all()


def test_rolling_mean_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.rolling_mean(3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    assert (
        pupil_measurement.get_size()
        == np.array([10.5, 10, 10, 26 / 3, 9, 28 / 3, 10.5])
    ).all()


def test_rolling_median_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.rolling_median(3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    assert (
        pupil_measurement.get_size() == np.array([10.5, 10, 10, 9, 10, 10, 10.5])
    ).all()


def test_get_rate_of_change():
    time_data = np.array([0, 1, 2, 3, 4.5, 5])
    size = np.array([6.7, 6.2, 6.1, 6.8, 6.2, 6.1])
    pm = PupilMeasurement(time_data, size)
    rate_of_change = pm.get_rate_of_change()
    expected = np.array([np.nan, -0.5, -0.1, 0.7, -0.4, -0.2])
    np.testing.assert_allclose(rate_of_change, expected, equal_nan=True)


def test_limit_rate_of_change():
    time_data = np.array([0, 1, 2, 3, 4.5, 5])
    size = np.array([6.7, 6.2, 6.1, 6.8, 6.2, 6.1])
    pm = PupilMeasurement(time_data, size)
    pm.limit_rate_of_change(0.3)
    expected_size = np.array([6.7, np.nan, 6.1, np.nan, np.nan, 6.1])
    np.testing.assert_allclose(
        pm.get_size(), expected_size, equal_nan=True
    )


def test_get_average_size():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    pm = PupilMeasurement(time_data, size)

    average_size = pm.get_average_size(1, 4)
    assert average_size == 6.25
