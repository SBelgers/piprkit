import warnings

import numpy as np
import pytest

from piprkit.core.pupil_measurement import PupilMeasurement
from piprkit.utils.baseline import BaselineCorrection

def test_rolling_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)

    def set_nan(window: np.ndarray) -> float:
        return np.nan

    pupil_measurement.filters.rolling_filter(set_nan, 3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    assert np.isnan(pupil_measurement.get_size()).all()

    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.filters.rolling_filter(np.sum, 3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    np.testing.assert_allclose(
        pupil_measurement.get_size(), [21, 30, 30, 26, 27, 28, 21]
    )


def test_rolling_mean_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.filters.rolling_mean(3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    np.testing.assert_allclose(
        pupil_measurement.get_size(), [10.5, 10, 10, 26 / 3, 9, 28 / 3, 10.5]
    )


def test_rolling_median_filter():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pupil_measurement = PupilMeasurement(time_data, size)
    pupil_measurement.filters.rolling_median(3)
    assert len(pupil_measurement.get_time()) == len(time_data)
    assert len(pupil_measurement.get_size()) == len(size)
    np.testing.assert_allclose(
        pupil_measurement.get_size(), [10.5, 10, 10, 9, 10, 10, 10.5]
    )


def test_get_rate_of_change():
    time_data = np.array([0, 1, 2, 3, 4.5, 5])
    size = np.array([6.7, 6.2, 6.1, 6.8, 6.2, 6.1])
    pm = PupilMeasurement(time_data, size)
    rate_of_change = pm.filters.get_rate_of_change()
    expected = np.array([np.nan, -0.5, -0.1, 0.7, -0.4, -0.2])
    np.testing.assert_allclose(rate_of_change, expected, equal_nan=True)


def test_limit_rate_of_change():
    time_data = np.array([0, 1, 2, 3, 4.5, 5])
    size = np.array([6.7, 6.2, 6.1, 6.8, 6.2, 6.1])
    pm = PupilMeasurement(time_data, size)
    pm.filters.limit_rate_of_change(0.3)
    expected_size = np.array([6.7, np.nan, 6.1, np.nan, np.nan, 6.1])
    np.testing.assert_allclose(
        pm.get_size(), expected_size, equal_nan=True
    )


def test_limit_rate_of_change_default():
    pm = PupilMeasurement(np.arange(4.0), np.array([5.0, 11.0, 11.0, 4.0]))
    pm.filters.limit_rate_of_change()
    # Rates: 6 mm/s (kept, on the limit), 0 mm/s, -7 mm/s (removed).
    np.testing.assert_allclose(pm.get_size(), [5.0, 11.0, 11.0, np.nan], equal_nan=True)


@pytest.mark.parametrize("time_window", [0, -1])
def test_rolling_filter_window_must_be_positive(time_window):
    pm = PupilMeasurement(np.arange(7.0), np.ones(7))
    with pytest.raises(ValueError, match="positive"):
        pm.filters.rolling_filter(np.mean, time_window)


def test_rolling_filter_irregular_sampling():
    # The window used to be sized from the mean sample interval (here 1/3 s,
    # so a 0.5 s window became one sample). Now each window holds the samples
    # within +-0.25 s: the three clusters are averaged, the lone sample at t = 2 is kept.
    time_data = np.array([0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0])
    pm = PupilMeasurement(time_data, np.array([1, 2, 3, 10, 20, 30, 5.0]))
    pm.filters.rolling_mean(0.5)
    np.testing.assert_allclose(pm.get_size(), [2, 2, 2, 20, 20, 20, 5])


def test_rolling_filter_window_on_sample_boundary():
    # 0.1 s at 60 Hz: +-0.05 s holds 3 samples on each side, despite float rounding.
    pm = PupilMeasurement(np.arange(20) / 60, np.zeros(20))
    pm.filters.rolling_filter(lambda window: float(len(window)), 0.1)
    expected = [min(i, 3) + min(19 - i, 3) + 1 for i in range(20)]
    np.testing.assert_allclose(pm.get_size(), expected)


def test_rolling_filter_even_window_no_warning():
    # A 2 s window at 1 s sampling covers t-1 .. t+1; an even window used to warn.
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pm = PupilMeasurement(np.arange(7.0), size)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.filters.rolling_mean(2)
    np.testing.assert_allclose(pm.get_size(), [10.5, 10, 10, 26 / 3, 9, 28 / 3, 10.5])


def test_rolling_mean_not_inplace():
    size = np.array([10, 11, 9, 10, 7, 10, 11])
    pm = PupilMeasurement(np.arange(7.0), size)
    result = pm.filters.rolling_mean(3, inplace=False)
    np.testing.assert_allclose(result.get_size(), [10.5, 10, 10, 26 / 3, 9, 28 / 3, 10.5])
    np.testing.assert_allclose(pm.get_size(), size)


def test_get_rate_of_change_n_back_2():
    pm = PupilMeasurement(np.arange(6.0), np.array([0, 0.25, 0.5, 0.75, 2, 2]))
    # (size[i] - size[i - 2]) / 2
    expected = [np.nan, np.nan, 0.25, 0.25, 0.75, 0.625]
    np.testing.assert_allclose(pm.filters.get_rate_of_change(2), expected, equal_nan=True)


def test_get_rate_of_change_n_back_0_raises():
    # get_rate_of_change(0) used to return nothing (time_data[:-0] is empty); n_back must be at least 1.
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError, match="n_back"):
        pm.filters.get_rate_of_change(0)


def test_limit_rate_of_change_n_back_2():
    size = np.array([0, 0.25, 0.5, 0.75, 2, 2])
    pm = PupilMeasurement(np.arange(6.0), size)
    # n_back = 1 flags t = 4 (rate 1.25); n_back = 2 also flags t = 5 (rate 0.625).
    pm.filters.limit_rate_of_change(0.5, n_back=2)
    np.testing.assert_allclose(
        pm.get_size(), [0, 0.25, 0.5, 0.75, np.nan, np.nan], equal_nan=True
    )


def test_limit_rate_of_change_time_window():
    pm = PupilMeasurement(np.arange(6.0), np.array([0, 2, 2, 2, 4, 4]))
    # Jumps at t = 1 and t = 4; only t = 1 lies in [0, 3].
    pm.filters.limit_rate_of_change(0.5, start_time=0, end_time=3)
    np.testing.assert_allclose(pm.get_size(), [0, np.nan, 2, 2, 4, 4], equal_nan=True)


def test_limit_rate_of_change_buffer_seconds():
    size = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
    pm = PupilMeasurement(np.arange(10.0), size)
    # Jump at t = 4; t = 3 and t = 5 are exactly 1 s away.
    pm.filters.limit_rate_of_change(0.5, buffer_seconds=1)
    np.testing.assert_allclose(
        pm.get_size(), [0, 0, 0, np.nan, np.nan, np.nan, 2, 2, 2, 2], equal_nan=True
    )


def test_limit_rate_of_change_not_inplace():
    size = np.array([0, 2, 2, 2, 4, 4])
    pm = PupilMeasurement(np.arange(6.0), size)
    result = pm.filters.limit_rate_of_change(0.5, inplace=False)
    np.testing.assert_allclose(result.get_size(), [0, np.nan, 2, 2, np.nan, 4], equal_nan=True)
    np.testing.assert_allclose(pm.get_size(), size)


def test_limit_rate_of_change_buffer_ignores_samples_outside_time_window():
    size = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
    pm = PupilMeasurement(np.arange(10.0), size)
    # The jump at t = 4 lies outside [5, 9], so it is kept and does not remove t = 5.
    with pytest.warns(UserWarning, match="buffer_seconds"):
        pm.filters.limit_rate_of_change(0.5, buffer_seconds=1, start_time=5, end_time=9)
    np.testing.assert_allclose(pm.get_size(), size)


def test_remove_constant_signal():
    pm = PupilMeasurement(np.arange(7.0), np.array([5, 6, 6, 6, 7, 7, 8.0]))
    # Runs: 6 from t = 1 to 3 (2 s, removed), 7 from t = 4 to 5 (1 s, kept).
    pm.filters.remove_constant_signal(2)
    np.testing.assert_allclose(
        pm.get_size(), [5, np.nan, np.nan, np.nan, 7, 7, 8], equal_nan=True
    )


def test_remove_constant_signal_nan_ends_run():
    pm = PupilMeasurement(np.arange(5.0), np.array([6, 6, np.nan, 6, 6.0]))
    # The NaN splits the 6s into two 1 s runs.
    pm.filters.remove_constant_signal(2)
    np.testing.assert_allclose(pm.get_size(), [6, 6, np.nan, 6, 6], equal_nan=True)


def test_remove_constant_signal_on_sample_boundary():
    # 7 samples at 60 Hz span 0.1 s, despite float rounding.
    size = np.array([1.0] * 7 + [2.0, 3.0])
    pm = PupilMeasurement(np.arange(9) / 60, size)
    pm.filters.remove_constant_signal(0.1)
    np.testing.assert_allclose(pm.get_size(), [np.nan] * 7 + [2, 3], equal_nan=True)


@pytest.mark.parametrize("min_duration", [0, -1])
def test_remove_constant_signal_min_duration_must_be_positive(min_duration):
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError, match="min_duration"):
        pm.filters.remove_constant_signal(min_duration)


def test_remove_constant_signal_time_window_and_buffer():
    size = np.array([0, 1, 1, 1, 2, 3, 4, 4, 4, 5.0])
    pm = PupilMeasurement(np.arange(10.0), size)
    # Runs at t = 1-3 and t = 6-8; only the first lies in [0, 4]. The buffer adds t = 0 and t = 4.
    pm.filters.remove_constant_signal(2, buffer_seconds=1, start_time=0, end_time=4)
    np.testing.assert_allclose(
        pm.get_size(), [np.nan] * 5 + [3, 4, 4, 4, 5], equal_nan=True
    )


def test_remove_constant_signal_not_inplace():
    size = np.array([5, 6, 6, 6, 7.0])
    pm = PupilMeasurement(np.arange(5.0), size)
    result = pm.filters.remove_constant_signal(2, inplace=False)
    np.testing.assert_allclose(result.get_size(), [5, np.nan, np.nan, np.nan, 7], equal_nan=True)
    np.testing.assert_allclose(pm.get_size(), size)


def _baseline_pupil():
    # Size 4 up to and including light onset at t = 0, then 6.
    pm = PupilMeasurement(np.arange(-5.0, 5.0), np.array([4.0] * 6 + [6.0] * 4))
    pm.set_light_stimulus(0, 1)
    return pm


@pytest.mark.parametrize(
    "method, expected",
    [
        ("ratio", [1.0] * 6 + [1.5] * 4),  # size / 4
        ("subtractive", [0.0] * 6 + [2.0] * 4),  # size - 4
        ("percent", [0.0] * 6 + [50.0] * 4),  # (size - 4) / 4 * 100
    ],
)
def test_apply_baseline_correction(method, expected):
    pm = _baseline_pupil()
    # Baseline window [-5, 0] holds six samples of 4.
    pm.filters.apply_baseline_correction(duration=5, method=method)
    np.testing.assert_allclose(pm.get_size(), expected)
    assert pm.get_baseline_correction() == BaselineCorrection(method, 4.0, 5)


def test_apply_baseline_correction_unknown_method():
    pm = _baseline_pupil()
    with pytest.raises(ValueError, match="Unknown baseline correction method"):
        pm.filters.apply_baseline_correction(duration=5, method="log")


def test_absolute_baseline_correction_removed():
    # "absolute" was a subtractive correction; the old name is removed.
    pm = _baseline_pupil()
    with pytest.raises(ValueError, match="Unknown baseline correction method"):
        pm.filters.apply_baseline_correction(duration=5, method="absolute")
    assert not hasattr(pm.filters, "baseline_correction_absolute")
    pm.filters.baseline_correction_subtractive(duration=5)
    np.testing.assert_allclose(pm.get_size(), [0.0] * 6 + [2.0] * 4)


def test_apply_baseline_correction_twice_raises():
    pm = _baseline_pupil()
    pm.filters.apply_baseline_correction(duration=5)
    with pytest.raises(ValueError, match="Already baseline corrected"):
        pm.filters.apply_baseline_correction(duration=5)


def test_apply_baseline_correction_without_baseline_data_raises():
    pm = PupilMeasurement(np.arange(1.0, 5.0), np.ones(4))
    pm.set_light_stimulus(0, 1)
    # Baseline window [-5, 0] starts before the first sample and holds no samples.
    with pytest.warns(UserWarning, match="Start time is before the first data point"):
        with pytest.warns(UserWarning, match="No data points"):
            with pytest.raises(ValueError, match="Baseline is NaN"):
                pm.filters.apply_baseline_correction(duration=5)


def test_apply_baseline_correction_not_inplace():
    pm = _baseline_pupil()
    result = pm.filters.apply_baseline_correction(duration=5, inplace=False)
    np.testing.assert_allclose(result.get_size(), [1.0] * 6 + [1.5] * 4)
    np.testing.assert_allclose(pm.get_size(), [4.0] * 6 + [6.0] * 4)
    assert pm.get_baseline_correction() is None
