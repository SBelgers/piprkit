import pytest
import numpy as np
from pypipr.core.pupil_series import PupilSeries
import pypipr
from matplotlib.axes import Axes


def test_pupil_series_initialization():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    assert isinstance(ps, PupilSeries)

def test_light_stimulus_error():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    
    with pytest.raises(AttributeError):
        ps.set_light_stimulus()
    with pytest.raises(AttributeError):
        ps.get_light_stimulus()
    
def test_pupil_series_light_stimulus_list():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    for i, stim in enumerate(ps.get_light_stimuli().get_stimuli()):
        assert isinstance(stim, pypipr.LightStimulus)
        np.testing.assert_array_equal(stim.get_time(), light_stimuli[i])

def test_pupil_series_split():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    pm_list = ps.split(1, 1)
    expected_times = [(-2, -1, 0, 1), (-2, -1, 0, 1)]
    expected_sizes = [(6.1,6.2,6.3,6.4), (6.3,6.4,6.5,6.6)]
    expected_light_stimuli_times = [(-1,0), (-1,0)]
    for i, pm in enumerate(pm_list):
        assert isinstance(pm, pypipr.PupilMeasurement)
        np.testing.assert_array_equal(pm.get_time(), expected_times[i])
        np.testing.assert_array_equal(pm.get_size(), expected_sizes[i])
        np.testing.assert_array_equal(pm.get_light_stimulus().get_time(), expected_light_stimuli_times[i])

def test_pupil_series_plot_light_stimuli():
    time_data = np.array([0, 1, 2, 3, 4, 5, 6])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])
    light_stimuli = [(2.0, 3.0), (4.0, 5.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    ax = ps.plot()
    assert isinstance(ax, Axes)

def test_pupil_series_inheritance():
    import pypipr.core.pupil_measurement
    time_data = np.array([0, 1, 2, 3])
    size = np.array([6.0, 6.1, 6.2, 6.3])
    light_stimuli = [(1.0, 2.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    assert isinstance(ps, pypipr.core.pupil_measurement.PupilMeasurement)