import pytest
import numpy as np

from pypipr.data.loaders import load_simulated_pupil
from pypipr import PupilMeasurement
#TODO: There are missing tests. Implement them.

def test_load_sample():
    # Test loading blue trace
    blue_trace = load_simulated_pupil("blue")
    assert isinstance(blue_trace, PupilMeasurement)

    # Test loading red trace
    red_trace = load_simulated_pupil("red")
    assert isinstance(red_trace, PupilMeasurement)
    assert np.allclose(blue_trace.get_time(), red_trace.get_time(), atol=1e-8)
    assert len(blue_trace.get_size()) == 15999
    index_10 = np.where(blue_trace.get_time() == 8.952)[0][0]
    assert blue_trace.get_time()[index_10] == 8.952
    assert blue_trace.get_size()[index_10] == 7.1509962250612364



    # Test invalid input
    with pytest.raises(ValueError):
        load_simulated_pupil("green")
