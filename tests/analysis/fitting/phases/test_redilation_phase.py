import numpy as np
import pytest

import piprkit

TIME = np.arange(0.0, 20.0, 2.0)
S, K, P = -0.5, -0.3, 1.2


def size_from(start):
    # x = 0 at the phase start
    return S * np.exp(K * (TIME - start)) + P


def test_redilation_model_function():
    model = piprkit.RedilationPhase.get_model_function()
    # S * exp(k * x) + P with S = -0.4, k = -ln 2, P = 1: exp(k * 1) = 0.5
    np.testing.assert_allclose(model(np.array([0.0, 1.0]), -0.4, -np.log(2), 1.0), [0.6, 0.8])


def test_redilation_fit_starting_at_zero(plot_phase):
    phase = piprkit.RedilationPhase(TIME, size_from(0), 0)
    phase.fit()
    np.testing.assert_allclose(phase.get_params(), (S, K, P), rtol=1e-6)
    plot_phase(phase, TIME, size_from(0))


def test_redilation_fit_with_offset_start(plot_phase):
    # The offset used to have the wrong sign. The model time starts at the phase start, so t = 6 is x = 0.
    phase = piprkit.RedilationPhase(TIME, size_from(6), 6)
    phase.fit()
    np.testing.assert_allclose(phase.get_params(), (S, K, P), rtol=1e-6)
    plot_phase(phase, TIME, size_from(6))


def test_redilation_end_defaults_to_last_sample():
    phase = piprkit.RedilationPhase(TIME, size_from(6), 6)
    assert phase.get_end_time() == 18


def test_redilation_formula_string():
    phase = piprkit.RedilationPhase(TIME, size_from(6), 6)
    phase.fit()
    assert phase.get_formula_string() == (
        "y = -0.5000 * exp(-0.3000 * (x - 6.0000)) + 1.2000 from 6.0000 to 18.0000"
    )


def test_redilation_default_bounds_keep_k_at_most_0():
    phase = piprkit.RedilationPhase(TIME, size_from(0), 0)
    phase.fit()
    assert phase.get_options()["bounds"] == ((-np.inf, -np.inf, -np.inf), (np.inf, 0.0, np.inf))


def test_redilation_growing_data_does_not_converge(plot_phase):
    # k = 0.2 lies above the default upper bound 0, and at k = 0 only S + P is defined
    size = 0.1 * np.exp(0.2 * TIME) + 1
    phase = piprkit.RedilationPhase(TIME, size, 0)
    with pytest.warns(UserWarning, match="Fit failed"):
        phase.fit()
    assert phase.get_flags()["not_converged"]
    plot_phase(phase, TIME, size)


def test_redilation_falling_curve_is_not_defined(plot_phase):
    # S > 0 with k < 0 falls towards P, so the pupil does not redilate
    size = 0.2 * np.exp(-0.5 * TIME) + 1
    phase = piprkit.RedilationPhase(TIME, size, 0)
    phase.fit()
    flags = phase.get_flags()
    assert flags["not_defined"]
    assert flags["reason"] == "S > 0: the fitted curve falls instead of redilating."
    assert np.isnan(phase.get_params()).all()
    plot_phase(phase, TIME, size)


def test_redilation_fit_respects_bounds(plot_phase):
    phase = piprkit.RedilationPhase(TIME, size_from(0), 0)
    lower, upper = [-2, -1, 0], [0, -0.35, 3]
    # The unbounded optimum k = -0.3 lies above the upper bound for k
    phase.fit((-0.5, -0.5, 1.2), bounds=(lower, upper))
    params = np.array(phase.get_params())
    assert np.all(params >= np.array(lower) - 1e-9)
    assert np.all(params <= np.array(upper) + 1e-9)
    assert params[1] == pytest.approx(-0.35)
    assert phase.get_flags()["at_bounds"] == ("k",)
    plot_phase(phase, TIME, size_from(0))


def test_redilation_delay(plot_phase):
    time_data = np.arange(0.0, 20.0, 0.5)
    size = S * np.exp(K * (time_data - 6)) + P
    phase = piprkit.RedilationPhase(time_data, size, 6, delay=1)
    # Fitting starts at 7, but x = 0 stays at 6, so S, k and P are unchanged
    assert phase.get_time()[0] == 7
    phase.fit()
    np.testing.assert_allclose(phase.get_params(), (S, K, P), rtol=1e-6)
    assert phase.get_window()["delay"] == 1
    assert phase.get_options()["delay"] == 1
    # The curve is NaN in the delay [6, 7)
    curve = phase.get_predicted(np.array([6.0, 6.5, 7.0, 8.0]), extrapolation="allow")
    assert np.isnan(curve[:2]).all()
    np.testing.assert_allclose(curve[2:], S * np.exp(K * np.array([1.0, 2.0])) + P, rtol=1e-6)
    plot_phase(phase, time_data, size)
