import warnings

import pytest
import numpy as np
from piprkit.core.pupil_base import PupilBase
from piprkit.core.pupil_measurement import PupilMeasurement
from piprkit.core.pupil_series import PupilSeries
import piprkit
from matplotlib.axes import Axes


def test_pupil_series_initialization():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    assert isinstance(ps, PupilSeries)

def test_light_stimulus_error():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    
    # The error messages used to name methods that don't exist.
    with pytest.raises(AttributeError, match="set_light_stimuli_from_list"):
        ps.set_light_stimulus()
    with pytest.raises(AttributeError, match="get_light_stimuli_series"):
        ps.get_light_stimulus()
    assert hasattr(ps, "set_light_stimuli_from_list")
    assert hasattr(ps, "get_light_stimuli_series")
    
def test_pupil_series_light_stimulus_list():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    for i, stim in enumerate(ps.get_light_stimuli_series().get_stimuli()):
        assert isinstance(stim, piprkit.LightStimulus)
        np.testing.assert_array_equal(stim.get_timing(), light_stimuli_series[i])

def test_pupil_series_split():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    pm_list = ps.split(1, 1)
    expected_times = [(-2, -1, 0, 1), (-2, -1, 0, 1)]
    expected_sizes = [(6.1,6.2,6.3,6.4), (6.3,6.4,6.5,6.6)]
    expected_light_stimuli_times = [(-1,0), (-1,0)]
    for i, pm in enumerate(pm_list):
        assert isinstance(pm, piprkit.PupilMeasurement)
        np.testing.assert_array_equal(pm.get_time(), expected_times[i])
        np.testing.assert_array_equal(pm.get_size(), expected_sizes[i])
        np.testing.assert_array_equal(pm.get_light_stimulus().get_timing(), expected_light_stimuli_times[i])  # type: ignore

def test_pupil_series_plot():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    ax = ps.plot()
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1


def test_pupil_series_plot_light_stimulus():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli_series = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli_series)
    ax = ps.plot_light_stimulus()
    assert isinstance(ax, Axes)
    # One shaded span per stimulus, from its start to its end.
    spans = [(patch.get_x(), patch.get_x() + patch.get_width()) for patch in ax.patches]
    np.testing.assert_allclose(spans, light_stimuli_series)


def _series():
    # Stimuli at 2-3 s and 6-7 s in a 10 s recording.
    return PupilSeries(np.arange(10.0), np.arange(10.0) + 1, [(2.0, 3.0), (6.0, 7.0)])


def test_pupil_series_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        PupilSeries(np.arange(5.0), np.ones(4), [(1.0, 2.0)])


def test_set_light_stimuli_from_list_replaces_stimuli():
    ps = _series()
    ps.set_light_stimuli_from_list([(1.0, 2.0)])
    assert ps.get_light_stimuli_series().get_timings() == [(1.0, 2.0)]


def test_pupil_series_set_time_offset():
    # set_time_offset on a series used to raise an AttributeError.
    ps = _series()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ps.set_time_offset(1)
    np.testing.assert_allclose(ps.get_time(), np.arange(10.0) + 1)
    np.testing.assert_allclose(ps.get_light_stimuli_series().get_timings(), [(3, 4), (7, 8)])


def test_split_without_blinks_does_not_warn():
    # split() used to warn several times per stimulus when no blinks were set.
    ps = _series()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm_list = ps.split(1, 1)
    assert [pm.get_blinks() for pm in pm_list] == [[], []]


def test_split_carries_blinks():
    ps = _series()
    ps.set_blinks([(2.5, 2.6), (4.5, 5.5), (8.5, 9.0)])
    first, second = ps.split(1, 1)
    # Window [1, 4], shifted by -3: only (2.5, 2.6) overlaps.
    np.testing.assert_allclose(first.get_blinks(), [(-0.5, -0.4)])
    # Window [5, 8], shifted by -7: (4.5, 5.5) overlaps, (8.5, 9.0) does not.
    np.testing.assert_allclose(second.get_blinks(), [(-2.5, -1.5)])
    # The series keeps its own blinks.
    assert ps.get_blinks() == [(2.5, 2.6), (4.5, 5.5), (8.5, 9.0)]


def test_split_measurements_are_independent():
    ps = _series()
    first, second = ps.split(1, 1)
    first.get_size()[0] = 99
    np.testing.assert_allclose(ps.get_size(), np.arange(10.0) + 1)
    np.testing.assert_allclose(second.get_size(), [6, 7, 8, 9])


@pytest.mark.parametrize(
    "pre_stimulus, post_stimulus, message",
    [(-1, 1, "pre_stimulus_duration"), (1, -1, "post_stimulus_duration")],
)
def test_split_warns_on_negative_duration(pre_stimulus, post_stimulus, message):
    ps = _series()
    with pytest.warns(UserWarning, match=message):
        ps.split(pre_stimulus, post_stimulus)


def test_pupil_series_filters():
    ps = _series()
    ps.filters.rolling_mean(3)
    # size = t + 1: a centred mean of a straight line is unchanged, except at the edges.
    np.testing.assert_allclose(ps.get_size(), [1.5, 2, 3, 4, 5, 6, 7, 8, 9, 9.5])


def test_pupil_series_filters_not_inplace():
    ps = _series()
    result = ps.filters.rolling_mean(3, inplace=False)
    assert isinstance(result, PupilSeries)
    np.testing.assert_allclose(ps.get_size(), np.arange(10.0) + 1)


def test_pupil_series_is_not_a_measurement():
    ps = _series()
    assert isinstance(ps, PupilBase)
    assert not isinstance(ps, PupilMeasurement)


def test_pupil_series_metrics_raise_helpful_error():
    ps = _series()
    with pytest.raises(AttributeError, match="split"):
        ps.metrics


def test_pupil_series_baseline_correction_raises():
    ps = _series()
    with pytest.raises(ValueError, match="split"):
        ps.filters.apply_baseline_correction()


@pytest.mark.parametrize("method", ["get_sections", "plot_sections"])
def test_sections_need_a_single_stimulus(method):
    with pytest.raises(AttributeError, match="split"):
        getattr(_series(), method)()
