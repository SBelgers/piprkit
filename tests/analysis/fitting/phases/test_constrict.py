import numpy as np
import pypipr
import pytest


@pytest.mark.xfail(reason="FitConstrict is currently broken and returns NaNs")
def test_constrict():
    time_data = np.array([-12, -10, -8, -6, -4, -2, 0, 2])
    size = np.array([8,7,6,4,5,3,2,1])
    fit = pypipr.FitConstrict(time_data, size, -10, 0)
    fit.fit()
    params = fit.get_params()
    assert params == (-0.5, 2)
