# Test for Baseline phase
import numpy as np
import piprkit
import pytest

def test_baseline():
    time_data = np.array([-12, -10, -8, -6, -4, -2, 0, 2])
    size = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    with pytest.warns():
        piprkit.FitBaseline(time_data, size, -4, 0)

    time_data = np.array([-12, -10, -8, -6, -4, -2, 0, 2])
    size = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    fit = piprkit.FitBaseline(time_data, size, -10, 0)
    fit.fit()
    fit.get_params()
    assert fit.get_params() == (1,)
    
    time_data = np.array([-12, -10, -8, -6, -4, -2, 0, 2])
    size = np.array([1, 2, 0, 2, 0, 1, 1, 1])
    fit = piprkit.FitBaseline(time_data, size, -10, 0)
    fit.fit()
    fit.get_params()
    assert fit.get_params() == (1,)

    expected = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    predicted = fit.predict(time_data)
    np.testing.assert_array_equal(predicted, expected)
