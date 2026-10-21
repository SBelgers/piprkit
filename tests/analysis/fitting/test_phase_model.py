import numpy as np
import pytest

import piprkit
from piprkit.analysis.fitting.base_model import BaseModel, Phase
from piprkit.analysis.fitting.phase_model import PHASES

# Parameters fitted to the default simulation (light from -1 to 0 s, constriction start at -0.9 s, peak at -0.75 s):
# constriction: the simulation falls from the baseline 1 at -0.9 s to 0.55 at the peak (m = -3), but the search ends
# the constriction one sample after the peak (-0.74 s), so the fitted line has m = -2.940686 and c = 0.9970343
# maintained constriction: c = 0.55, the size at the peak constriction
# redilation: S = -(1 - 0.55 - 0.025 * 0.75) = -0.43125 at light offset, where the redilation starts
SIMULATED = {
    "baseline": {"c": 1.0},
    "constriction": {"m": -2.940686, "c": 0.9970343},
    "maintained_constriction": {"m": 0.025, "c": 0.55},
    "redilation": {"S": -0.43125, "k": -0.4, "P": 1.0},
}


@pytest.fixture
def model(measurement):
    return piprkit.PhaseModel(measurement)


def test_recovers_simulation_parameters(model, measurement, plot_phase_model):
    params = model.get_all_params()
    for phase, expected in SIMULATED.items():
        assert params[phase] == pytest.approx(expected, rel=1e-6, abs=1e-9), phase
    plot_phase_model(model, measurement)


def test_phases_follow_each_other(model):
    # The constriction and the maintained constriction used to share the whole light window.
    window = model.get_window()
    # The baseline ends at light onset; the constriction starts where the simulated constriction starts
    assert window["baseline"]["end"] == -1
    assert window["constriction"]["start"] == pytest.approx(-0.9)
    # The constriction end lands one sample after the peak constriction, where the maintained constriction starts
    assert window["constriction"]["end"] == pytest.approx(-0.74)
    assert window["maintained_constriction"]["start"] == pytest.approx(-0.75)
    assert window["maintained_constriction"]["end"] == window["redilation"]["start"] == 0


def test_baseline_window_is_baseline_duration(measurement):
    model = piprkit.PhaseModel(measurement, baseline_duration=5)
    assert model.get_window()["baseline"]["start"] == -6


def test_selected_phases_only(measurement, plot_phase_model):
    model = piprkit.PhaseModel(measurement, phases=("baseline", "redilation"))
    assert list(model.get_all_params()) == ["baseline", "redilation"]
    time_data, curve = model.get_time(), model.get_predicted()
    # No phase covers the constriction, so the curve is NaN between the baseline end and the redilation start
    window = model.get_window()
    baseline_end, post_start = window["baseline"]["end"], window["redilation"]["start"]
    assert np.isnan(curve[(time_data >= baseline_end) & (time_data < post_start)]).all()
    assert np.isfinite(curve[(time_data >= -11) & (time_data < baseline_end)]).all()
    assert np.isfinite(curve[time_data >= post_start]).all()
    plot_phase_model(model, measurement)


def test_phases_keep_their_order(measurement):
    model = piprkit.PhaseModel(measurement, phases=("redilation", "baseline"))
    assert model.get_options()["phases"] == ("baseline", "redilation")


def test_unknown_phase_raises(measurement):
    with pytest.raises(ValueError, match="Unknown phases"):
        piprkit.PhaseModel(measurement, phases=("plateau",))


def test_without_stimulus_raises():
    measurement = piprkit.PupilMeasurement(np.arange(10.0), np.ones(10))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError, match="Light stimulus not set"):
            piprkit.PhaseModel(measurement)


def test_constriction_latency_starts_after_plr_latency(measurement):
    measurement.filters.apply_baseline_correction()
    latency = measurement.metrics.plr_latency()
    model = piprkit.PhaseModel(measurement, constriction_latency=True)
    onset = measurement.get_light_stimulus().get_timing()[0]
    assert model.get_window()["constriction"]["start"] >= onset + latency - 1e-9


def test_constriction_latency_without_correction_uses_baseline(measurement):
    with pytest.warns(UserWarning, match="not baseline corrected"):
        model = piprkit.PhaseModel(measurement, constriction_latency=True)
    # The simulated baseline is 1, so the latency is the same 0.11 s as on ratio-corrected data
    onset = measurement.get_light_stimulus().get_timing()[0]
    assert model.get_window()["constriction"]["start"] >= onset + 0.11 - 1e-9


def test_redilation_delay(measurement, plot_phase_model):
    model = piprkit.PhaseModel(measurement, redilation_delay=1)
    # The redilation starts at light offset + delay, so k and P are unchanged and S is the simulated S at light offset times exp(k * delay)
    expected = dict(SIMULATED["redilation"], S=-0.43125 * np.exp(-0.4 * 1))
    params = model.get_all_params()["redilation"]
    assert params == pytest.approx(expected, rel=1e-6)
    window = model.get_window()["redilation"]
    assert window["start"] == 1
    assert window["delay"] == 0
    time_data, curve = model.get_time(), model.get_predicted()
    # The maintained constriction runs up to the delayed redilation start
    assert model.get_window()["maintained_constriction"]["end"] == 1
    assert np.isfinite(curve[(time_data >= 0) & (time_data < 1)]).all()
    assert np.isfinite(curve[time_data >= 1]).all()
    plot_phase_model(model, measurement)


def test_redilation_end(measurement, plot_phase_model):
    model = piprkit.PhaseModel(measurement, redilation_end=6)
    assert model.get_window()["redilation"]["end"] == 6
    assert model.get_phases()["redilation"].get_time()[-1] <= 6
    # The simulation is the same exponential up to 6 s, so the parameters are unchanged
    assert model.get_all_params()["redilation"] == pytest.approx(SIMULATED["redilation"], rel=1e-6)
    # After the redilation end, the model continues the redilation
    time_data, curve = model.get_time(), model.get_predicted()
    assert np.isfinite(curve[time_data > 6]).all()
    plot_phase_model(model, measurement)


def test_redilation_matches_phase_fitted_on_its_own(model, measurement):
    phase = model.get_phases()["redilation"]
    exponential = piprkit.RedilationPhase(measurement.get_time(), measurement.get_size(), phase.get_start_time())
    exponential.fit()
    np.testing.assert_allclose(phase.get_params(), exponential.get_params(), rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_baseline_without_valid_samples_is_not_defined(measurement, plot_phase_model):
    time_data, size = measurement.get_time(), measurement.get_size().copy()
    size[time_data <= -1] = np.nan
    no_baseline = piprkit.PupilMeasurement(time_data, size)
    no_baseline.set_light_stimulus(-1, 0)
    model = piprkit.PhaseModel(no_baseline)
    flags = model.get_flags()["baseline"]
    assert flags["not_defined"]
    assert flags["reason"] == "No valid samples in the baseline window."
    assert np.isnan(model.get_param("baseline", "c"))
    plot_phase_model(model)


def test_peak_at_light_offset_leaves_maintained_constriction_not_defined(plot_phase_model):
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=1.0
    )
    model = piprkit.PhaseModel(measurement)
    flags = model.get_flags()["maintained_constriction"]
    assert flags["not_defined"]
    assert flags["reason"] == "The maintained constriction is shorter than breakpoint_min_duration (0.1 s)."
    plot_phase_model(model, measurement)


def constriction_after_offset():
    # Light from 0 to 1 s; the size falls at -0.3 per s until 2 s, then stays flat
    time_data = np.linspace(-5, 6, 1101)
    size = np.where(time_data < 0, 1.0, 1 - 0.3 * np.minimum(time_data, 2))
    measurement = piprkit.PupilMeasurement(time_data, size)
    measurement.set_light_stimulus(0, 1)
    return measurement


def test_constriction_continues_after_light_offset(plot_phase_model):
    measurement = constriction_after_offset()
    model = piprkit.PhaseModel(measurement)
    # The peak constriction is at 2 s, the end of the default window of 1 s, and the constriction ends there
    assert model.get_window()["constriction"]["end"] == pytest.approx(2)
    assert model.get_all_params()["constriction"] == pytest.approx({"m": -0.3, "c": 1.0}, rel=1e-6)
    assert model.get_flags()["maintained_constriction"]["not_defined"]
    # Redilation starts after the constriction
    window = model.get_window()
    assert window["redilation"]["start"] >= window["constriction"]["end"]
    plot_phase_model(model, measurement)


def test_peak_constriction_window_0_stops_at_light_offset(plot_phase_model):
    measurement = constriction_after_offset()
    model = piprkit.PhaseModel(measurement, peak_constriction_window=0)
    assert model.get_window()["constriction"]["end"] == pytest.approx(1)
    plot_phase_model(model, measurement)


def test_constriction_ends_at_a_peak_near_the_search_end(plot_phase_model):
    # The peak constriction lies 2 samples before light offset + peak_constriction_window
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=1.98, redilation_delay=5.0
    )
    model = piprkit.PhaseModel(measurement)
    assert model.get_window()["constriction"]["end"] == pytest.approx(0.98)
    plot_phase_model(model, measurement)


def test_get_param(model):
    assert model.get_param("redilation", "k") == pytest.approx(-0.4)


def test_get_param_unknown_phase(model):
    with pytest.raises(ValueError, match="Phase 'plateau' not found"):
        model.get_param("plateau", "c")


def test_get_param_unknown_param(model):
    with pytest.raises(ValueError, match="Parameter 'q' not found"):
        model.get_param("baseline", "q")


def test_records_have_one_entry_per_phase(model):
    assert list(model.get_flags()) == list(PHASES)
    assert list(model.get_window()) == list(PHASES)
    assert list(model.get_options()["phase_options"]) == list(PHASES)
    phases = model.get_phases()
    assert list(phases) == list(PHASES)
    assert all(isinstance(phase, Phase) for phase in phases.values())


def test_no_flags_on_the_simulation(model):
    for phase, flags in model.get_flags().items():
        assert not any(flags[flag] for flag in ("not_defined", "not_converged", "at_bounds", "too_few_samples")), phase


def test_options(model):
    options = model.get_options()
    assert {key: value for key, value in options.items() if key != "phase_options"} == {
        "phases": PHASES,
        "baseline_duration": 10.0,
        "constriction_latency": False,
        "redilation_delay": 0.0,
        "redilation_end": None,
        "peak_constriction_window": 1.0,
        "min_samples": 3,
        "breakpoint_resolution": 0.01,
        "breakpoint_min_duration": 0.1,
        "redilation_k_min": -10.0,
        "redilation_k_max": -0.01,
        "redilation_k_count": 40,
        "redilation_k_estimate": -0.4,
        "redilation_k_bounds": (-np.inf, 0.0),
    }


def test_options_are_passed_to_the_phases(measurement):
    model = piprkit.PhaseModel(measurement, min_samples=4, redilation_k_bounds=(-1.0, 0.0))
    phases = model.get_options()["phase_options"]
    for phase in ("constriction", "maintained_constriction", "redilation"):
        assert phases[phase]["min_samples"] == 4, phase
    assert phases["redilation"]["bounds"] == ((-np.inf, -1.0, -np.inf), (np.inf, 0.0, np.inf))


def test_get_phases_returns_a_new_dict(model):
    model.get_phases().pop("baseline")
    assert "baseline" in model.get_phases()


def test_goodness_of_fit_on_noise_free_data(model):
    # Only the constriction misses the data: its window ends one sample after the peak constriction
    assert model.get_goodness_of_fit("RMSE") == pytest.approx(2.563883e-4, rel=1e-6)


def test_get_predicted_approaches_plateau(model, plot_phase_model):
    curve = model.get_predicted(np.linspace(-12, 70, 821), extrapolation="allow")
    assert curve[-1] == pytest.approx(SIMULATED["redilation"]["P"], abs=1e-6)
    plot_phase_model(model)


def test_to_measurement(model):
    measurement = model.to_measurement()
    assert type(measurement) is piprkit.PupilMeasurement
    np.testing.assert_allclose(measurement.get_time(), model.get_time())
    np.testing.assert_allclose(measurement.get_size(), model.get_predicted(), equal_nan=True)
    assert measurement.get_light_stimulus().get_timing() == (-1.0, 0.0)
    assert measurement.get_baseline_correction() is None


def test_to_measurement_keeps_baseline_correction(measurement):
    measurement.filters.apply_baseline_correction(duration=5, method="subtractive")
    fitted = piprkit.PhaseModel(measurement).to_measurement()
    assert fitted.get_baseline_correction() == measurement.get_baseline_correction()
    assert fitted.metrics.pipr_6s().get_baseline_correction().get_method() == "subtractive"


def test_formula_string(model):
    lines = model.get_formula_string().splitlines()
    names = ["Baseline", "Constriction", "Maintained constriction", "Redilation"]
    assert lines[0] == "Formula:"
    for line, name, phase in zip(lines[1:], names, model.get_phases().values()):
        assert line == f"\t{name}: {phase.get_formula_string()}"


def test_metrics_raise_helpful_error(model):
    # Metrics computed from data are not on model.metrics; the error points to to_measurement()
    with pytest.raises(AttributeError, match="to_measurement"):
        model.metrics.pipr_6s
    with pytest.raises(AttributeError, match="has no attribute 'q'"):
        model.metrics.q


def test_phase_model_is_a_model_not_a_measurement(model):
    # PupilFit, the old PhaseModel, used to subclass PupilMeasurement.
    assert isinstance(model, BaseModel)
    assert not isinstance(model, piprkit.PupilMeasurement)


def test_phase_model_from_measurement(plot_phase_model):
    # Uses the public set_light_stimulus; the old test patched private methods.
    time_data = np.linspace(0, 20, 100)
    measurement = piprkit.PupilMeasurement(time_data, np.sin(time_data / 3) + 5)
    measurement.set_light_stimulus(5, 15)
    model = piprkit.PhaseModel(measurement)
    np.testing.assert_allclose(model.get_time(), time_data)
    assert model.get_light_stimulus().get_timing() == (5, 15)
    window = model.get_window()
    # The baseline ends at light onset
    assert window["baseline"]["end"] == 5
    assert 15 <= window["redilation"]["start"] <= 16
    assert np.isfinite(model.get_param("baseline", "c"))
    plot_phase_model(model)


def test_sections_match_the_measurement(model, measurement):
    assert model.get_sections() == measurement.get_sections()


def test_stretch_between_two_phases_is_interpolated(model):
    # The latency, from light onset to the constriction start, is filled with interpolate()
    phases = model.get_phases()
    end, start = phases["baseline"].get_end_time(), phases["constriction"].get_start_time()
    time_data = np.linspace(end, start, 12)[:-1]
    ends = piprkit.PupilMeasurement(
        np.array([end, start]),
        np.array([model.get_param("baseline", "c"), phases["constriction"].get_predicted(np.array([start]))[0]]),
    )
    expected = ends.interpolate(time_data, inplace=False).get_size()
    np.testing.assert_allclose(model.get_predicted(time_data), expected)


def test_before_the_baseline_the_model_continues_the_baseline(model):
    time_data, curve = model.get_time(), model.get_predicted()
    before = time_data < model.get_window()["baseline"]["start"]
    assert before.any()
    np.testing.assert_allclose(curve[before], model.get_param("baseline", "c"))


def test_no_interpolation_next_to_a_phase_without_a_fit(plot_phase_model):
    # The peak at light offset leaves the maintained constriction without a fit, so the model is NaN from
    # the constriction end to the redilation start
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=1.0
    )
    model = piprkit.PhaseModel(measurement)
    window = model.get_window()
    time_data, curve = model.get_time(), model.get_predicted()
    between = (time_data >= window["constriction"]["end"]) & (time_data < window["redilation"]["start"])
    assert between.any()
    assert np.isnan(curve[between]).all()
    plot_phase_model(model, measurement)


def trace(time_data, size, onset, offset):
    measurement = piprkit.PupilMeasurement(time_data, size)
    measurement.set_light_stimulus(onset, offset)
    return measurement


def test_plateau_after_light_offset(plot_phase_model):
    # Light from 0 to 1 s; the size falls until 1.5 s, stays at 0.5 until 6 s, then redilates with k = -0.3
    time_data = np.round(np.arange(-12, 40.001, 0.01), 10)
    size = np.where(
        time_data < 0.2,
        1.0,
        np.where(time_data < 1.5, 1 - 0.5 * (time_data - 0.2) / 1.3, np.where(time_data < 6, 0.5, 1 - 0.5 * np.exp(-0.3 * (time_data - 6)))),
    )
    measurement = trace(time_data, size, 0, 1)
    model = piprkit.PhaseModel(measurement)
    window = model.get_window()
    assert window["maintained_constriction"]["start"] == pytest.approx(1.5, abs=0.02)
    assert window["maintained_constriction"]["end"] == window["redilation"]["start"] == pytest.approx(6, abs=0.05)
    assert window["maintained_constriction"]["sections"] == ("post_stimulus",)
    assert window["constriction"]["sections"] == ("during_stimulus", "post_stimulus")
    assert model.get_all_params()["maintained_constriction"] == pytest.approx({"m": 0, "c": 0.5}, abs=1e-6)
    assert model.get_all_params()["redilation"] == pytest.approx({"S": -0.5, "k": -0.3, "P": 1}, rel=0.02)
    plot_phase_model(model, measurement)


def test_redilation_at_the_peak_constriction(plot_phase_model):
    # Light from 0 to 1 s; the size falls until 1.4 s and redilates right away, so there is no maintained constriction
    time_data = np.round(np.arange(-12, 30.001, 0.01), 10)
    size = np.where(
        time_data < 0.2, 1.0, np.where(time_data < 1.4, 1 - 0.5 * (time_data - 0.2) / 1.2, 1 - 0.5 * np.exp(-0.5 * (time_data - 1.4)))
    )
    measurement = trace(time_data, size, 0, 1)
    model = piprkit.PhaseModel(measurement)
    window = model.get_window()
    assert window["redilation"]["start"] == pytest.approx(1.4, abs=0.05)
    assert model.get_flags()["maintained_constriction"]["not_defined"]
    assert model.get_all_params()["redilation"] == pytest.approx({"S": -0.5, "k": -0.5, "P": 1}, rel=0.02)
    plot_phase_model(model, measurement)


def test_peak_during_the_light(plot_phase_model):
    # Light from 0 to 10 s; the size falls until 1 s, rises slowly while the light stays on, and redilates after light offset
    time_data = np.round(np.arange(-12, 40.001, 0.01), 10)
    size = np.where(
        time_data < 0.2,
        1.0,
        np.where(
            time_data < 1,
            1 - 0.5 * (time_data - 0.2) / 0.8,
            np.where(time_data < 10, 0.5 + 0.005 * (time_data - 1), 1 - 0.455 * np.exp(-0.4 * (time_data - 10))),
        ),
    )
    measurement = trace(time_data, size, 0, 10)
    model = piprkit.PhaseModel(measurement)
    window = model.get_window()
    assert window["maintained_constriction"]["sections"] == ("during_stimulus",)
    assert window["redilation"]["start"] == pytest.approx(10, abs=0.05)
    assert model.get_param("maintained_constriction", "m") == pytest.approx(0.005, rel=0.05)
    plot_phase_model(model, measurement)
