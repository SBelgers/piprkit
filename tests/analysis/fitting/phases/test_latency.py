# Test for latency phase
import numpy as np
import pypipr
import pytest

@pytest.mark.xfail(reason="FitLatency is currently not implemented.")
def test_latency():
    time_data = np.array([-12, -10, -8, -6, -4, -2, 0, 2])
    size = np.array([8, 7, 6, 4, 5, 3, 2, 1])
    pypipr.FitLatency(time_data, size, -10, 0)
    assert False
