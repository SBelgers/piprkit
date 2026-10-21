import numpy as np
import pytest

import piprkit

FITTED_PHASES = [
    piprkit.ConstrictionPhase,
    piprkit.MaintainedConstrictionPhase,
    piprkit.RedilationPhase,
]
ALL_PHASES = [piprkit.BaselinePhase, *FITTED_PHASES]


@pytest.mark.parametrize("phase_class", ALL_PHASES)
def test_phase_rejects_length_mismatch(phase_class):
    with pytest.raises(ValueError, match="same length"):
        phase_class(np.arange(5.0), np.ones(4), 0, 4)


@pytest.mark.parametrize("phase_class", ALL_PHASES)
def test_unfitted_params_are_nan(phase_class):
    phase = phase_class(np.arange(10.0), np.ones(10), 0, 9)
    assert len(phase.get_params()) == len(phase.get_param_names())
    assert np.isnan(phase.get_params()).all()
    assert np.isnan(phase.get_predicted()).all()


@pytest.mark.parametrize("phase_class", ALL_PHASES)
def test_window_counts_finite_samples(phase_class):
    size = np.ones(10)
    size[4] = np.nan
    phase = phase_class(np.arange(10.0), size, 2, 6)
    # t = 2..6 lie in the window, and t = 4 is NaN
    assert phase.get_window() == {"start": 2, "end": 6, "delay": 0.0, "n_samples": 4}


@pytest.mark.parametrize("phase_class", FITTED_PHASES)
def test_fit_with_fewer_than_3_samples(phase_class):
    # Only t = 0 and t = 1 lie in [0, 1]
    phase = phase_class(np.arange(10.0), np.ones(10), 0, 1)
    with pytest.warns(UserWarning, match="Not enough data points"):
        phase.fit()
    assert np.isnan(phase.get_params()).all()
    assert phase.get_flags()["too_few_samples"]


@pytest.mark.parametrize("phase_class", FITTED_PHASES)
def test_fit_that_does_not_converge(phase_class):
    time_data = np.arange(10.0)
    phase = phase_class(time_data, np.exp(-time_data) + 0.1 * time_data, 0, 9)
    with pytest.warns(UserWarning, match="Fit failed"):
        phase.fit(None, kwargs={"maxfev": 1})
    assert np.isnan(phase.get_params()).all()
    assert phase.get_flags()["not_converged"]


@pytest.mark.parametrize("phase_class", FITTED_PHASES)
def test_fit_resets_flags(phase_class):
    time_data = np.arange(10.0)
    phase = phase_class(time_data, np.exp(-time_data) + 0.1 * time_data, 0, 9)
    with pytest.warns(UserWarning, match="Fit failed"):
        phase.fit(None, kwargs={"maxfev": 1})
    phase.fit(None)
    assert not phase.get_flags()["not_converged"]


@pytest.mark.parametrize("phase_class", FITTED_PHASES)
def test_options_hold_estimate_bounds_and_delay(phase_class):
    phase = phase_class(np.arange(10.0), np.ones(10), 0, 9)
    estimate = tuple(-0.5 for _ in phase.get_param_names())
    bounds = ([-1] * len(estimate), [0] * len(estimate))
    phase.fit(estimate, bounds=bounds)
    assert phase.get_options() == {"p0": estimate, "bounds": bounds, "delay": 0.0, "min_samples": 3}


def test_min_samples_sets_too_few_samples():
    # t = 0..3 lie in [0, 3]: 4 samples, fewer than 5
    phase = piprkit.ConstrictionPhase(np.arange(10.0), np.ones(10), 0, 3, min_samples=5)
    with pytest.warns(UserWarning, match="at least 5"):
        phase.fit()
    assert phase.get_flags()["too_few_samples"]
    assert np.isnan(phase.get_params()).all()
