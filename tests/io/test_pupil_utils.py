import pytest
import numpy as np

from pypipr.io.utils import load_simulated_trace
from pypipr.core.pupil_measurement import PupilMeasurement

#TODO: There are missing tests. Implement them.

def test_load_sample():
    # Test loading blue trace
    blue_trace = load_simulated_trace("blue")
    assert isinstance(blue_trace, PupilMeasurement)

    # Test loading red trace
    red_trace = load_simulated_trace("red")
    assert isinstance(red_trace, PupilMeasurement)

    assert np.array_equal(blue_trace.get_time(), red_trace.get_time())
    assert len(blue_trace.get_size()) == 15999
    assert blue_trace.get_size()[15932] == 90.10725602885903
    # Test invalid input
    with pytest.raises(ValueError):
        load_simulated_trace("green")
