import warnings

import numpy as np
import pytest

from piprkit.core.pupil_measurement import PupilMeasurement


def test_pupil_measurement_initialization():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    with pytest.raises(ValueError):
        pm = PupilMeasurement(time_data, size[:-1])
    with pytest.raises(ValueError):
        pm = PupilMeasurement(time_data[:-1], size)
    pm = PupilMeasurement(time_data, size)
    assert isinstance(pm, PupilMeasurement)



def test_get_samples_and_nan():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, np.nan, 3, np.nan]))
    assert pm.get_samples() == 4
    assert pm.get_nan() == 2


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
    
    pm = PupilMeasurement(time_data, size)
    pm.trim_size(0, 2, start_time=1, end_time=4)
    expected_size = [6.0, np.nan, np.nan, np.nan, np.nan, 6.5]

    np.testing.assert_allclose(
        pm.get_size(),
        expected_size,
        equal_nan=True,
    )

def test_pupil_measurement_inheritance():
    import piprkit.core.pupil_base

    time_data = np.array([0, 1, 2, 3])
    size = np.array([6.0, 6.1, 6.2, 6.3])
    pm = PupilMeasurement(time_data, size)
    assert isinstance(pm, piprkit.core.pupil_base.PupilBase)


def _ones_with(index, value, n=10):
    size = np.ones(n)
    size[index] = value
    return size


@pytest.mark.parametrize(
    "time_data, message",
    [
        ([0, np.nan, 2], "NaN"),
        ([0, 2, 1], "not sorted"),
        ([0, 1, 1], "duplicate"),
    ],
)
def test_pupil_measurement_rejects_invalid_time(time_data, message):
    with pytest.raises(ValueError, match=message):
        PupilMeasurement(np.array(time_data), np.ones(3))


def test_integer_input_is_stored_as_float():
    # Integer input used to stay integer, so NaN could not be stored in the size array.
    pm = PupilMeasurement(np.arange(5), np.array([1, 2, 3, 4, 5]))
    assert pm.get_time().dtype == np.float64
    assert pm.get_size().dtype == np.float64
    pm.trim_size(2, 4)
    np.testing.assert_allclose(pm.get_size(), [np.nan, 2, 3, 4, np.nan], equal_nan=True)


def test_input_arrays_are_copied():
    time_data = np.arange(5.0)
    size = np.ones(5)
    pm = PupilMeasurement(time_data, size)
    time_data[0] = -1
    size[0] = 99
    assert pm.get_time()[0] == 0
    assert pm.get_size()[0] == 1


def test_copy_is_deep():
    pm = PupilMeasurement(np.arange(5.0), np.ones(5))
    pm.set_light_stimulus(1, 2)
    pm.set_blinks([(3, 4)])
    pm_copy = pm.copy()
    pm_copy.get_size()[0] = 99
    pm_copy.get_light_stimulus().set_time_offset(10)
    pm_copy.blinks.set_time_offset(10)
    assert pm.get_size()[0] == 1
    assert pm.get_light_stimulus().get_timing() == (1, 2)
    assert pm.get_blinks() == [(3, 4)]


def test_drop_nan_not_inplace():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, np.nan, 3, 4]))
    result = pm.drop_nan(inplace=False)
    np.testing.assert_allclose(result.get_time(), [0, 2, 3])
    np.testing.assert_allclose(pm.get_size(), [1, np.nan, 3, 4], equal_nan=True)


def test_trim_size_not_inplace():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, 2, 3, 4]))
    result = pm.trim_size(2, 3, inplace=False)
    np.testing.assert_allclose(result.get_size(), [np.nan, 2, 3, np.nan], equal_nan=True)
    np.testing.assert_allclose(pm.get_size(), [1, 2, 3, 4])


def test_trim_time_not_inplace():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, 2, 3, 4]))
    result = pm.trim_time(1, 2, inplace=False)
    np.testing.assert_allclose(result.get_time(), [1, 2])
    np.testing.assert_allclose(pm.get_time(), [0, 1, 2, 3])


def test_interpolate_not_inplace():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, 2, 3, 4]))
    result = pm.interpolate(np.array([0.5, 1.5]), inplace=False)
    np.testing.assert_allclose(result.get_size(), [1.5, 2.5])
    np.testing.assert_allclose(pm.get_time(), [0, 1, 2, 3])


def test_inplace_methods_return_none():
    pm = PupilMeasurement(np.arange(4.0), np.array([1, 2, 3, 4]))
    assert pm.trim_size(0, 10) is None
    assert pm.trim_time(0, 10) is None
    assert pm.drop_nan() is None
    assert pm.interpolate(np.arange(4.0)) is None


def test_trim_size_buffer_seconds():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(4, 9))
    pm.trim_size(0, 2, buffer_seconds=1)
    # t = 4 is out of range; t = 3 and t = 5 are exactly 1 s away.
    np.testing.assert_allclose(pm.get_size(), _ones_with([3, 4, 5], np.nan), equal_nan=True)


def test_trim_size_buffer_seconds_inside_time_window():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(6, 9))
    pm.trim_size(0, 2, buffer_seconds=1, start_time=5, end_time=9)
    np.testing.assert_allclose(pm.get_size(), _ones_with([5, 6, 7], np.nan), equal_nan=True)


def test_trim_size_does_not_change_earlier_get_size_result():
    pm = PupilMeasurement(np.arange(5.0), np.array([1, 2, 3, 4, 5]))
    size_before = pm.get_size()
    pm.trim_size(2, 4)
    np.testing.assert_allclose(size_before, [1, 2, 3, 4, 5])


def test_trim_size_buffer_ignores_existing_nan():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(4, np.nan))
    pm.trim_size(0, 2, buffer_seconds=1)
    np.testing.assert_allclose(pm.get_size(), _ones_with(4, np.nan), equal_nan=True)


def test_trim_size_buffer_ignores_samples_outside_time_window():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(4, 9))
    # t = 4 is out of range but outside [5, 9], so it is kept and does not remove t = 5.
    with pytest.warns(UserWarning, match="buffer_seconds"):
        pm.trim_size(0, 2, buffer_seconds=1, start_time=5, end_time=9)
    np.testing.assert_allclose(pm.get_size(), _ones_with(4, 9))


def test_trim_size_buffer_stays_inside_time_window():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(5, 9))
    # t = 5 is removed; its buffer reaches t = 4, which lies outside [5, 9] and is kept.
    pm.trim_size(0, 2, buffer_seconds=1, start_time=5, end_time=9)
    np.testing.assert_allclose(pm.get_size(), _ones_with([5, 6], np.nan), equal_nan=True)


def test_trim_size_far_outside_time_window_does_not_warn():
    pm = PupilMeasurement(np.arange(10.0), _ones_with(1, 9))
    # t = 1 is more than 1 s from [5, 9], so its buffer would not reach the window.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.trim_size(0, 2, buffer_seconds=1, start_time=5, end_time=9)
    np.testing.assert_allclose(pm.get_size(), _ones_with(1, 9))


def test_trim_size_defaults():
    pm = PupilMeasurement(np.arange(5.0), np.array([1.5, 2.0, 5.0, 8.0, 8.5]))
    pm.trim_size()
    np.testing.assert_allclose(pm.get_size(), [np.nan, 2.0, 5.0, 8.0, np.nan], equal_nan=True)


def test_set_time_offset_shifts_stimulus_and_blinks():
    pm = PupilMeasurement(np.arange(-2.0, 3.0), np.ones(5))
    pm.set_light_stimulus(-1, 0)
    pm.set_blinks([(0.5, 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.set_time_offset(2)
    np.testing.assert_allclose(pm.get_time(), [0, 1, 2, 3, 4])
    assert pm.get_light_stimulus().get_timing() == (1, 2)
    assert pm.get_blinks() == [(2.5, 3)]


def test_set_time_offset_without_stimulus_does_not_warn():
    # set_time_offset used to warn "Light stimulus not set." when there is no stimulus.
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.set_time_offset(1)
    np.testing.assert_allclose(pm.get_time(), [1, 2, 3])


def test_get_light_stimulus_unset_warns():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        assert pm.get_light_stimulus() is None


def test_plot_light_stimulus_unset_raises():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError, match="Light stimulus not set"):
            pm.plot_light_stimulus()


def test_plot_light_stimulus():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    pm.set_light_stimulus(1, 2)
    ax = pm.plot_light_stimulus()
    patch = ax.patches[0]
    np.testing.assert_allclose((patch.get_x(), patch.get_x() + patch.get_width()), (1, 2))


def test_get_blinks_unset_is_empty():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    assert pm.get_blinks() == []


def test_set_blinks():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    pm.set_blinks([(0.5, 1)])
    assert pm.get_blinks() == [(0.5, 1)]


def test_add_blinks():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    pm.set_blinks([(0.5, 1)])
    pm.add_blinks([(1.5, 2)])
    assert pm.get_blinks() == [(0.5, 1), (1.5, 2)]


def test_add_blinks_unset():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    pm.add_blinks([(0.5, 1)])
    assert pm.get_blinks() == [(0.5, 1)]


def test_plot_blinks_unset_raises():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError, match="Blinks not set"):
        pm.plot_blinks()


def test_plot_blinks():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    pm.set_blinks([(0.5, 1)])
    ax = pm.plot_blinks()
    patch = ax.patches[0]
    np.testing.assert_allclose((patch.get_x(), patch.get_x() + patch.get_width()), (0.5, 1))


def test_repr():
    # The closing parenthesis used to come before blinks=; it comes last.
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    assert repr(pm) == (
        "PupilMeasurement(time_data.shape=(3,), size_data.shape=(3,), "
        "light_stimulus=None, blinks=[], baseline_correction=none)"
    )


def test_plot():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    ax = pm.plot()
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 1, 2])
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Pupil Diameter"


def test_plot_scatter():
    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    ax = pm.plot(scatter=True)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_keeps_labels_on_given_axes():
    # plot(ax=...) used to overwrite the axis labels already set on ax.
    import matplotlib.pyplot as plt

    pm = PupilMeasurement(np.arange(3.0), np.ones(3))
    _, ax = plt.subplots()
    ax.set_xlabel("my x")
    ax.set_ylabel("my y")
    assert pm.plot(ax=ax) is ax
    assert ax.get_xlabel() == "my x"
    assert ax.get_ylabel() == "my y"


@pytest.mark.parametrize(
    "method, label",
    [
        ("ratio", "Pupil Diameter (ratio to baseline)"),
        ("subtractive", "Pupil Diameter (change from baseline)"),
        ("percent", "Pupil Diameter (% change from baseline)"),
    ],
)
def test_plot_label_after_baseline_correction(method, label):
    pm = PupilMeasurement(np.arange(-5.0, 5.0), np.full(10, 4.0))
    pm.set_light_stimulus(0, 1)
    pm.filters.apply_baseline_correction(duration=5, method=method)
    assert pm.plot().get_ylabel() == label


def test_get_sections_span_the_data():
    pm = PupilMeasurement(np.arange(-5.0, 11.0), np.ones(16))
    pm.set_light_stimulus(-1, 0)
    assert pm.get_sections() == {
        "pre_stimulus": {"start": -5.0, "end": -1},
        "during_stimulus": {"start": -1, "end": 0},
        "post_stimulus": {"start": 0, "end": 10.0},
    }


def test_get_sections_without_stimulus_raises():
    pm = PupilMeasurement(np.arange(5.0), np.ones(5))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError, match="Light stimulus not set"):
            pm.get_sections()


def test_plot_sections_draws_the_three_sections():
    pm = PupilMeasurement(np.arange(-5.0, 11.0), np.ones(16))
    pm.set_light_stimulus(-1, 0)
    ax = pm.plot_sections()
    assert [text.get_text() for text in ax.texts] == ["pre-stimulus", "during stimulus", "post-stimulus"]
