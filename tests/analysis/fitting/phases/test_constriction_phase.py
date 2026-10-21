import numpy as np
import pytest

import piprkit

TIME = np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=float)
M, C, START, END = -0.5, 6.0, 1, 4
# x = 0 at the phase start, so c is the size at t = 1
SIZE = M * (TIME - START) + C


def test_constriction_model_function():
    model = piprkit.ConstrictionPhase.get_model_function()
    # m * x + c with m = -2, c = 5
    np.testing.assert_allclose(model(np.array([0.0, 1.0, 2.0]), -2.0, 5.0), [5, 3, 1])


def test_constriction_fit_with_offset_start(plot_phase):
    # The constriction fit used to be forced to NaN.
    phase = piprkit.ConstrictionPhase(TIME, SIZE, START, END)
    phase.fit()
    np.testing.assert_allclose(phase.get_params(), (M, C), rtol=1e-6)
    assert not any(phase.get_flags()[flag] for flag in ("not_defined", "not_converged", "at_bounds", "too_few_samples"))
    plot_phase(phase, TIME, SIZE)


def test_constriction_formula_string():
    phase = piprkit.ConstrictionPhase(TIME, SIZE, START, END)
    phase.fit()
    assert phase.get_formula_string() == (
        "y = -0.5000 * (x - 1.0000) + 6.0000 from 1.0000 to 4.0000"
    )


def test_constriction_fit_respects_bounds(plot_phase):
    phase = piprkit.ConstrictionPhase(TIME, SIZE, START, END)
    # The unbounded optimum m = -0.5 lies below the lower bound for m
    phase.fit((-0.4, 5.0), bounds=([-0.45, 0], [0, 10]))
    assert phase.get_params()[0] == pytest.approx(-0.45)
    assert phase.get_flags()["at_bounds"] == ("m",)
    plot_phase(phase, TIME, SIZE)


def test_constriction_estimate_outside_bounds_is_moved_inside():
    phase = piprkit.ConstrictionPhase(TIME, SIZE, START, END)
    # The estimate has m = -3, below the lower bound -1
    with pytest.warns(UserWarning, match="Initial estimate for m is outside the bounds"):
        phase.fit((-3.0, 1.0), bounds=([-1, 0], [0, 10]))
    np.testing.assert_allclose(phase.get_params(), (M, C), rtol=1e-6)
    assert phase.get_options()["p0"] == (-1.0, 1.0)
