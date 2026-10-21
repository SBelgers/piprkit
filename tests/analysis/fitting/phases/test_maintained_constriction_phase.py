import numpy as np
import pytest

import piprkit

TIME = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14], dtype=float)
M, C, START, END = 0.5, 3.0, 2, 10
SIZE = M * (TIME - START) + C


def test_maintained_constriction_model_function():
    model = piprkit.MaintainedConstrictionPhase.get_model_function()
    # m * x + c with m = 0.5, c = 3
    np.testing.assert_allclose(model(np.array([0.0, 2.0]), 0.5, 3.0), [3, 4])


def test_maintained_constriction_fit_with_offset_start(plot_phase):
    # The old FitSustain was a stub that returned NaN.
    phase = piprkit.MaintainedConstrictionPhase(TIME, SIZE, START, END)
    phase.fit()
    np.testing.assert_allclose(phase.get_params(), (M, C), rtol=1e-6)
    plot_phase(phase, TIME, SIZE)


def test_maintained_constriction_formula_string():
    phase = piprkit.MaintainedConstrictionPhase(TIME, SIZE, START, END)
    phase.fit()
    assert phase.get_formula_string() == (
        "y = 0.5000 * (x - 2.0000) + 3.0000 from 2.0000 to 10.0000"
    )


def test_maintained_constriction_fit_respects_bounds(plot_phase):
    phase = piprkit.MaintainedConstrictionPhase(TIME, SIZE, START, END)
    # The unbounded optimum m = 0.5 lies above the upper bound for m
    phase.fit((0.3, 3.0), bounds=([0, 0], [0.4, 10]))
    assert phase.get_params()[0] == pytest.approx(0.4)
    assert phase.get_flags()["at_bounds"] == ("m",)
    plot_phase(phase, TIME, SIZE)
