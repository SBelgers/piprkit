# Test for redilation phase
import numpy as np
import pypipr


def test_redilation():
    time_data = np.array([0, 2, 4, 6, 8, 10, 12])
    s = 0.45
    k = -0.4
    p = 1
    size = np.asarray(-s * np.exp(k * time_data) + p)
    fit = pypipr.FitRedilation(time_data, size, 0, 12)
    fit.fit()
    fit.get_params()
    np.testing.assert_array_equal(fit.get_params(), (s, k, p))
    edges = np.array([-np.inf, 0, np.inf])
    predicted = fit.predict(edges)
    expected = np.array([-np.inf, 0.55, 1])
    np.testing.assert_allclose(predicted, expected, equal_nan=True)

    time_data = np.array([0, 2, 4, 6, 8, 10, 12])
    s = 0.45
    k = -0.4
    p = 1
    offset = 6
    size = np.asarray(-s * np.exp(k * (time_data + offset)) + p)
    fit = pypipr.FitRedilation(time_data, size, offset, 12)
    fit.fit()
    fit.get_params()
    np.testing.assert_array_equal(fit.get_params(), (s, k, p))
