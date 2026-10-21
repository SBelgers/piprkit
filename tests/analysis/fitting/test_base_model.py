import numpy as np
import pytest

import piprkit
from piprkit.analysis.fitting.base_model import BaseModel, Phase

TIME = np.arange(0.0, 20.0, 2.0)
S, K, P = -0.5, -0.3, 1.2
SIZE = S * np.exp(K * TIME) + P


@pytest.fixture
def fitted():
    phase = piprkit.RedilationPhase(TIME, SIZE, 0)
    phase.fit()
    return phase


def baseline_fit(size, c):
    phase = piprkit.BaselinePhase(np.array([-9.0, -6.0, -3.0, 0.0]), np.asarray(size, dtype=float), -10, 0)
    phase.fit(c)
    return phase


@pytest.mark.parametrize(
    "phase_class",
    [
        piprkit.BaselinePhase,
        piprkit.ConstrictionPhase,
        piprkit.MaintainedConstrictionPhase,
        piprkit.RedilationPhase,
    ],
)
def test_phases_are_base_models(phase_class):
    assert issubclass(phase_class, Phase)
    assert issubclass(phase_class, BaseModel)


def test_get_time_and_size_are_the_fitted_data(fitted):
    np.testing.assert_allclose(fitted.get_time(), TIME)
    np.testing.assert_allclose(fitted.get_size(), SIZE)


def test_get_predicted_defaults_to_the_data_time(fitted):
    np.testing.assert_allclose(fitted.get_predicted(), SIZE, rtol=1e-6)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("MAE", 1.0),  # |residuals| = 1.5, 0.5, 0.5, 1.5
        ("RMSE", np.sqrt(1.25)),  # (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
        ("R2", 0.0),  # a constant at the mean explains nothing
    ],
)
def test_goodness_of_fit(method, expected):
    phase = baseline_fit([0, 1, 2, 3], 1.5)
    assert phase.get_goodness_of_fit(method) == pytest.approx(expected, abs=1e-12)


def test_goodness_of_fit_defaults_to_mae():
    phase = baseline_fit([0, 1, 2, 3], 1.5)
    assert phase.get_goodness_of_fit() == phase.get_goodness_of_fit("MAE")


def test_goodness_of_fit_ignores_nan():
    # Only 0, 2 and 3 count: |residuals| = 1, 1, 2
    phase = baseline_fit([0, np.nan, 2, 3], 1.0)
    assert phase.get_goodness_of_fit("MAE") == pytest.approx(4 / 3)


def test_goodness_of_fit_of_unfitted_model_is_nan():
    phase = piprkit.RedilationPhase(TIME, SIZE, 0)
    assert np.isnan(phase.get_goodness_of_fit())


def test_r2_of_constant_data_is_nan():
    assert np.isnan(baseline_fit([2, 2, 2, 2], 2.0).get_goodness_of_fit("R2"))


def test_goodness_of_fit_unknown_method(fitted):
    with pytest.raises(ValueError, match="Unknown method"):
        fitted.get_goodness_of_fit("MSE")


def test_get_predicted_at_new_time(fitted):
    time_data = np.linspace(0, 60, 7)
    np.testing.assert_allclose(fitted.get_predicted(time_data, extrapolation="allow"), S * np.exp(K * time_data) + P, rtol=1e-6)
    # The fitted data stay as they are
    np.testing.assert_allclose(fitted.get_time(), TIME)


def test_get_predicted_approaches_plateau(fitted, save_plot):
    time_data = np.linspace(0, 60, 601)
    assert fitted.get_predicted(time_data, extrapolation="allow")[-1] == pytest.approx(P, abs=1e-6)
    if save_plot:
        ax = fitted.plot(scatter=True, s=12, color="grey", label="data")
        fitted.plot_window(ax, label="window")
        fitted.plot_predicted(ax, time_data=time_data, extrapolation="allow", label="extrapolated")
        save_plot(ax, fitted.get_formula_string())


@pytest.mark.parametrize(
    "new_time, message",
    [
        (np.array([0.0, 2.0, 1.0]), "not sorted"),
        (np.array([0.0, 1.0, 1.0]), "duplicate"),
        (np.array([0.0, np.nan]), "NaN"),
    ],
)
def test_get_predicted_rejects_invalid_time(fitted, new_time, message):
    with pytest.raises(ValueError, match=message):
        fitted.get_predicted(new_time)


def test_get_predicted_raises_on_extrapolation_by_default(fitted):
    with pytest.raises(ValueError, match="extrapolated"):
        fitted.get_predicted(np.array([0.0, 60.0]))
    assert fitted.get_predicted(np.array([0.0, 60.0]), extrapolation="allow")[-1] == pytest.approx(P, abs=1e-6)


def test_to_measurement(fitted):
    measurement = fitted.to_measurement(np.array([0.0, 10.0]))
    assert type(measurement) is piprkit.PupilMeasurement
    np.testing.assert_allclose(measurement.get_time(), [0, 10])
    np.testing.assert_allclose(measurement.get_size(), S * np.exp(K * np.array([0, 10])) + P, rtol=1e-6)


def test_to_measurement_defaults_to_the_data_time(fitted):
    measurement = fitted.to_measurement()
    np.testing.assert_allclose(measurement.get_time(), TIME)
    np.testing.assert_allclose(measurement.get_size(), SIZE, rtol=1e-6)


def test_to_measurement_raises_on_extrapolation_by_default(fitted):
    with pytest.raises(ValueError, match="extrapolated"):
        fitted.to_measurement(np.array([0.0, 60.0]))


def test_to_measurement_warns_on_extrapolation(fitted):
    with pytest.warns(UserWarning, match="extrapolated"):
        measurement = fitted.to_measurement(np.array([-2.0, 0.0]), extrapolation="warn")
    np.testing.assert_allclose(measurement.get_time(), [-2, 0])


def test_to_measurement_allows_extrapolation(fitted, recwarn):
    measurement = fitted.to_measurement(np.array([0.0, 60.0]), extrapolation="allow")
    assert measurement.get_size()[-1] == pytest.approx(P, abs=1e-6)
    assert not [w for w in recwarn if "extrapolated" in str(w.message)]


def test_to_measurement_unknown_extrapolation(fitted):
    with pytest.raises(ValueError, match="Unknown extrapolation"):
        fitted.to_measurement(extrapolation="clip")


def test_metrics_raise_helpful_error(fitted):
    with pytest.raises(AttributeError, match="to_measurement"):
        fitted.metrics
