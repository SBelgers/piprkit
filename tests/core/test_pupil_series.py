import pytest
import numpy as np
from pypipr.core.pupil_series import PupilSeries


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
    # TODO: Implement
    raise NotImplementedError()    

def test_pupil_series_split():
    # TODO: Implement
    raise NotImplementedError

def test_pupil_series_plot_light_stimuli():
    # TODO: Implement
    raise NotImplementedError

def test_pupil_series_inheritance():
    import pypipr.core.pupil_measurement
    time_data = np.array([0, 1, 2, 3])
    size = np.array([6.0, 6.1, 6.2, 6.3])
    light_stimuli = [(1.0, 2.0)]
    ps = PupilSeries(time_data, size, light_stimuli)
    assert isinstance(ps, pypipr.core.pupil_measurement.PupilMeasurement)