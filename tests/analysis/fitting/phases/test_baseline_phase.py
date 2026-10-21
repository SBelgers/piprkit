import warnings

import numpy as np

import piprkit

TIME = np.array([-12, -10, -8, -6, -4, -2, 0, 2], dtype=float)
SIZE = np.array([9, 2, 0, 2, 0, 1, 1, 5], dtype=float)


def test_baseline_model_function():
    model = piprkit.BaselinePhase.get_model_function()
    np.testing.assert_allclose(model(np.array([0.0, 1.0, 2.0]), 3.0), [3, 3, 3])


def test_baseline_keeps_only_the_window():
    phase = piprkit.BaselinePhase(TIME, SIZE, -10, 0)
    np.testing.assert_allclose(phase.get_time(), [-10, -8, -6, -4, -2, 0])
    np.testing.assert_allclose(phase.get_size(), [2, 0, 2, 0, 1, 1])


def test_baseline_fit_sets_c(plot_phase):
    phase = piprkit.BaselinePhase(TIME, SIZE, -10, 0)
    phase.fit(1.0)
    assert phase.get_params() == (1.0,)
    np.testing.assert_allclose(phase.get_predicted(), np.ones(6))
    plot_phase(phase, TIME, SIZE)


def test_baseline_does_not_warn_for_other_durations():
    # The old FitBaseline warned whenever the window was not 10 s.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        piprkit.BaselinePhase(TIME, np.ones(8), -4, 0).fit(1.0)


def test_baseline_formula_string():
    phase = piprkit.BaselinePhase(np.arange(-10.0, 1.0), np.ones(11), -10, 0)
    phase.fit(1.0)
    assert phase.get_formula_string() == "y = 1.0000 from -10.0000 to 0.0000"
