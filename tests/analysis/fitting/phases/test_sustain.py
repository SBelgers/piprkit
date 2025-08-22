import numpy as np

import pypipr


def test_sustain():
    time_data = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    size = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    fit = pypipr.FitSustain(time_data, size, 0, 10)
    fit.fit()
    params = fit.get_params()
    np.testing.assert_array_equal(params, (0.5, 2))
