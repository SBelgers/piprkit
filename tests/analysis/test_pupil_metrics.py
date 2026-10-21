import contextlib
import warnings

import numpy as np
import pytest

import piprkit
from piprkit.analysis.pupil_metrics import MetricResult
from piprkit.utils.baseline import BaselineCorrection


@pytest.fixture
def ramp():
    # size = 6 + 0.4 * t for t = -10, -9, ..., 10
    pupil = piprkit.PupilMeasurement(np.linspace(-10, 10, 21), np.linspace(2, 10, 21))
    pupil.set_light_stimulus(-1, 0)
    return pupil


def test_peak_constriction():
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25])
    size = np.array([7, 7, 6, 5, 6, 7, 7])
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pupil.metrics.peak_constriction() == 5


def test_time_to_peak():
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25])
    size = np.array([7, 7, 6, 5, 6, 7, 7])
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    # Minimum at t = -0.5, light onset at t = -1.
    assert pupil.metrics.time_to_peak() == pytest.approx(0.5)


def test_time_to_peak_with_later_end():
    # The pupil keeps constricting after light offset (t = 0) until t = 0.5
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    size = np.array([7, 7, 6, 5, 4.5, 4, 3.5, 3, 4])
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    # The default search runs until offset + 1 s and finds 3 at t = 0.5, 1.5 s after onset
    assert pupil.metrics.time_to_peak() == pytest.approx(1.5)
    # Stopping at light offset finds 4 at t = 0, the last sample of that window, so it warns
    with pytest.warns(UserWarning, match="may still be constricting"):
        assert pupil.metrics.time_to_peak(peak_constriction_window=0) == pytest.approx(1.0)


def test_pipr_xs(ramp):
    # Window [5.5, 6.5] after light offset holds only t = 6: 6 + 0.4 * 6 = 8.4.
    assert ramp.metrics.pipr_xs(5.5, 6.5) == pytest.approx(8.4)
    # Window [1, 5] holds t = 1..5, mean t = 3: 6 + 0.4 * 3 = 7.2.
    assert ramp.metrics.pipr_xs(1, 5) == pytest.approx(7.2)


def test_pipr_6s(ramp):
    assert ramp.metrics.pipr_6s() == pytest.approx(8.4)


def test_get_average_size(ramp):
    with pytest.warns(UserWarning, match="End time is after the last data point"):
        ramp.metrics.get_average_size(5, 20)
    # t = -6..-1, mean t = -3.5: 6 + 0.4 * -3.5 = 4.6
    assert ramp.metrics.get_average_size(-6, -1) == pytest.approx(4.6)
    # t = 0..5, mean t = 2.5: 6 + 0.4 * 2.5 = 7
    assert ramp.metrics.get_average_size(0, 5) == pytest.approx(7)
    # t = -10..3, mean t = -3.5: 4.6
    assert ramp.metrics.get_average_size(-10, 3) == pytest.approx(4.6)


def test_get_average_size_includes_window_edges():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    pm = piprkit.PupilMeasurement(time_data, size)
    # t = 1 and t = 4 are included: (6.1 + 6.2 + 6.3 + 6.4) / 4 = 6.25
    assert pm.metrics.get_average_size(1, 4) == pytest.approx(6.25)


def test_baseline_without_stimulus_raises():
    pupil = piprkit.PupilMeasurement(np.linspace(-10, 10, 21), np.linspace(2, 10, 21))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError):
            pupil.metrics.baseline()


def test_baseline(ramp):
    # Window [-21, -1] starts before the first sample at t = -10.
    with pytest.warns(UserWarning, match="Start time is before the first data point"):
        ramp.metrics.baseline(20)
    # Window [-6, -1]: 4.6 (see test_get_average_size)
    assert ramp.metrics.baseline(5) == pytest.approx(4.6)


@pytest.mark.parametrize(
    "metric, args",
    [("peak_constriction", ()), ("time_to_peak", ()), ("pipr_xs", (1, 5)), ("pipr_6s", ()), ("plr_latency", ()), ("transient_plr", ()), ("auc", (0, 10)), ("pipr_duration", ())],
)
def test_metric_without_stimulus_raises(metric, args):
    pupil = piprkit.PupilMeasurement(np.linspace(-10, 10, 21), np.linspace(2, 10, 21))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError, match="Light stimulus not set"):
            getattr(pupil.metrics, metric)(*args)


def test_pipr_xs_without_stimulus_names_pipr():
    # The error message used to say "time to peak" instead of naming pipr_xs.
    pupil = piprkit.PupilMeasurement(np.linspace(-10, 10, 21), np.linspace(2, 10, 21))
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        with pytest.raises(ValueError, match="Cannot calculate pipr_xs"):
            pupil.metrics.pipr_xs(1, 5)


def test_peak_only_looks_inside_stimulus_and_ignores_nan():
    time_data = np.array([-2, -1, -0.5, -0.25, 0, 1])
    size = np.array([1, 7, np.nan, 5, 6, 0])
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    # The 1 and 0 lie outside [-1, 0]; the minimum inside is 5 at t = -0.25.
    assert pupil.metrics.peak_constriction(peak_constriction_window=0) == 5
    assert pupil.metrics.time_to_peak(peak_constriction_window=0) == pytest.approx(0.75)


def test_metric_result_without_baseline_correction(ramp):
    result = ramp.metrics.pipr_6s()
    assert isinstance(result, MetricResult)
    assert isinstance(result, float)
    assert result.get_baseline_correction() is None
    assert str(result) == "6s-PIPR_5.95-6.05^mm = 8.4 [start, end: 5.95, 6.05 s] [baseline correction: none]"
    assert repr(result) == "MetricResult(6s-PIPR_5.95-6.05^mm = 8.4 [start, end: 5.95, 6.05 s] [baseline correction: none])"


def test_metric_result_carries_baseline_correction(ramp):
    # Baseline over [-6, -1] is 4.6 (see test_get_average_size).
    ramp.filters.apply_baseline_correction(duration=5, method="subtractive")
    result = ramp.metrics.pipr_6s()
    # 8.4 - 4.6
    assert result == pytest.approx(3.8)
    assert result.get_baseline_correction().get_method() == "subtractive"
    assert result.get_baseline_correction().get_baseline() == pytest.approx(4.6)
    assert result.get_baseline_correction().get_duration() == 5
    assert str(result) == "6s-PIPR_5.95-6.05^S = 3.8 [start, end: 5.95, 6.05 s] [baseline correction: subtractive (baseline=4.6, 5 s)]"


def test_baseline_does_not_warn(ramp):
    # baseline() used to warn "double implementation" on every call.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ramp.metrics.baseline(5)


def test_calculate_baseline_removed(ramp):
    # There is one baseline function: calculate_baseline() is removed.
    assert not hasattr(ramp.metrics, "calculate_baseline")


def test_get_average_size_empty_window(ramp):
    # No sample between t = 0.2 and t = 0.8.
    with pytest.warns(UserWarning, match="No data points"):
        assert np.isnan(ramp.metrics.get_average_size(0.2, 0.8))


def test_get_average_size_ignores_nan():
    pupil = piprkit.PupilMeasurement(np.arange(3.0), np.array([1, np.nan, 3]))
    assert pupil.metrics.get_average_size(0, 2) == pytest.approx(2)


@pytest.mark.parametrize(
    "metric, args, time_data, size_data",
    [
        # Window [-6, -1]: 4.6 (see test_get_average_size)
        ("baseline", (5,), [-6, -1], [4.6, 4.6]),
        # Window [5.95, 6.05] holds only t = 6: 8.4 (see test_pipr_xs)
        ("pipr_6s", (), [5.95, 6.05], [8.4, 8.4]),
    ],
)
def test_metric_result_plot_window(ramp, metric, args, time_data, size_data):
    result = getattr(ramp.metrics, metric)(*args)
    assert isinstance(result, MetricResult)
    (line,) = result.plot().get_lines()
    assert line.get_label() == result.get_label()
    np.testing.assert_allclose(line.get_xdata(), time_data)
    np.testing.assert_allclose(line.get_ydata(), size_data)


def test_metric_result_plot_peak():
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25])
    size = np.array([7, 7, 6, 5, 6, 7, 7])
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    # Minimum 5 at t = -0.5; time to peak runs from light onset (t = -1) to the minimum
    (peak,) = pupil.metrics.peak_constriction().plot().get_lines()
    np.testing.assert_allclose(peak.get_xdata(), [-0.5])
    np.testing.assert_allclose(peak.get_ydata(), [5])
    (time_to_peak,) = pupil.metrics.time_to_peak().plot().get_lines()
    np.testing.assert_allclose(time_to_peak.get_xdata(), [-1, -0.5])
    np.testing.assert_allclose(time_to_peak.get_ydata(), [5, 5])


@pytest.fixture
def simulation():
    # Noise-free default simulation with dt = 0.01 s. Its baseline is exactly 1, so the ratio
    # correction keeps the values of the simulation.
    pm = piprkit.simulate_pupil_measurement(t_start=-12, t_end=10, num_samples=2201)
    pm.filters.apply_baseline_correction()
    return pm


@pytest.fixture
def model(simulation):
    return piprkit.PhaseModel(simulation)


# The parameters of the default simulation (see SIMULATED in tests/analysis/fitting/test_phase_model.py)
@pytest.mark.parametrize(
    "metric, expected, result_name",
    [
        ("constriction_v", -2.940686, "ConstrictionVelocity^R"),
        ("pupil_escape", 0.025, "PupilEscape^R"),
        ("redilation_v", -0.4, "RedilationVelocity^R"),
        ("plateau_pipr", 1.0, "PlateauPIPR^R"),
    ],
)
def test_model_metrics_recover_simulation(simulation, model, metric, expected, result_name, plot_metrics):
    result = getattr(model.metrics, metric)()
    assert isinstance(result, MetricResult)
    assert result == pytest.approx(expected, rel=1e-6)
    assert result.get_name() == result_name
    assert result.get_baseline_correction() == simulation.get_baseline_correction()
    assert result.get_flags() == {
        "not_defined": False,
        "reason": None,
        "not_converged": False,
        "at_bounds": (),
        "too_few_samples": False,
    }
    plot_metrics(simulation, [result], name=metric)


@pytest.mark.parametrize("metric", ["constriction_v", "pupil_escape", "redilation_v", "plateau_pipr"])
def test_fit_metric_on_measurement_points_to_model(simulation, model, metric):
    message = rf"model\.metrics\.{metric}\(\)"
    with pytest.raises(AttributeError, match=message):
        getattr(simulation.metrics, metric)()
    # Passing the model doesn't help: the metric is read from model.metrics
    with pytest.raises(AttributeError, match=message):
        getattr(simulation.metrics, metric)(model)


def test_model_metric_needs_its_phase(simulation):
    model = piprkit.PhaseModel(simulation, phases=("baseline",))
    with pytest.raises(ValueError, match="needs the constriction phase"):
        model.metrics.constriction_v()


def test_model_metric_without_fit_is_nan_and_keeps_flags():
    # The peak constriction at light offset leaves a maintained constriction shorter than breakpoint_min_duration
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=1.0
    )
    model = piprkit.PhaseModel(measurement)
    with pytest.warns(UserWarning, match="pupil_escape is NaN: maintained_constriction not_defined"):
        result = model.metrics.pupil_escape()
    assert np.isnan(result)
    assert result.get_name() == "PupilEscape^mm"
    assert result.get_flags()["not_defined"]
    assert result.get_flags()["reason"] == "The maintained constriction is shorter than breakpoint_min_duration (0.1 s)."


def test_model_metric_plots(model):
    # constriction_v: the fitted line from the constriction start (-0.9 s, c = 0.9970343) to the constriction end (-0.74 s, 0.5265245), with markers
    (line,) = model.metrics.constriction_v().plot().get_lines()
    np.testing.assert_allclose(line.get_xdata(), [-0.9, -0.74])
    np.testing.assert_allclose(line.get_ydata(), [0.9970343, 0.5265245], rtol=1e-6)
    assert line.get_marker() == "o"
    # plateau_pipr: P = 1 over the redilation window, from the redilation start (light offset, 0 s) to the last sample
    (level,) = model.metrics.plateau_pipr().plot().get_lines()
    np.testing.assert_allclose(level.get_xdata(), [0, 10])
    np.testing.assert_allclose(level.get_ydata(), [1, 1], rtol=1e-6)
    # redilation_v: the fitted curve, drawn as a line without markers
    (curve,) = model.metrics.redilation_v().plot().get_lines()
    assert len(curve.get_xdata()) > 2
    assert curve.get_marker() == "None"


def constriction_past_offset():
    # Light from -1 to 0 s; the size falls from 7 to 4 until t = 0.5, then rises again
    time_data = np.arange(-2, 3.01, 0.25)
    size = np.where(time_data < -1, 7.0, np.where(time_data <= 0.5, 7 - 2 * (time_data + 1), 4 + (time_data - 0.5)))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    return pupil


def test_peak_after_light_offset():
    pupil = constriction_past_offset()
    # The default search runs until offset + 1 s, past the peak at t = 0.5; inside the window, so no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pupil.metrics.peak_constriction() == pytest.approx(4)
        assert pupil.metrics.time_to_peak() == pytest.approx(1.5)


def test_peak_at_window_end_warns():
    pupil = constriction_past_offset()
    # Until light offset the size is still falling, so the peak is the last sample of the window
    with pytest.warns(UserWarning, match="may still be constricting"):
        assert pupil.metrics.peak_constriction(peak_constriction_window=0) == pytest.approx(5)


def test_peak_constriction_and_time_to_peak_share_the_peak():
    pupil = constriction_past_offset()
    peak = pupil.metrics.peak_constriction()
    time_to_peak = pupil.metrics.time_to_peak()
    onset = pupil.get_light_stimulus().get_timing()[0]
    np.testing.assert_allclose(peak.get_time(), [onset + time_to_peak])
    np.testing.assert_allclose(time_to_peak.get_size(), [peak, peak])


@pytest.mark.parametrize("metric", ["peak_constriction", "time_to_peak"])
def test_peak_without_valid_samples_is_nan(metric):
    time_data = np.arange(-2, 3.01, 0.25)
    size = np.where((time_data >= -1) & (time_data <= 1), np.nan, 7.0)
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    with pytest.warns(UserWarning, match="No valid samples"):
        assert np.isnan(getattr(pupil.metrics, metric)())


def test_plr_latency(simulation, plot_metrics):
    # The simulation leaves the baseline at -0.9 s; the next sample (-0.89 s) is 0.97, already below 0.99
    result = simulation.metrics.plr_latency()
    assert result == pytest.approx(0.11)
    assert result.get_name() == "1%-PLRLatency^R"
    # 10%: the size first drops to 0.90 or less at -0.86 s (0.88)
    assert simulation.metrics.plr_latency(constriction_percent=10) == pytest.approx(0.14)
    plot_metrics(simulation, [result], name="plr_latency")


def five_mm_constriction():
    # A 5 mm baseline; light from 0 to 1 s; the size falls at 2 mm/s from t = 0.2
    time_data = np.round(np.arange(-5, 3.001, 0.1), 10)
    size = np.where(time_data < 0.2, 5.0, 5 - 2 * (time_data - 0.2))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(0, 1)
    return pupil


@contextlib.contextmanager
def expect_warning(match):
    if match is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield
    else:
        with pytest.warns(UserWarning, match=match):
            yield


@pytest.mark.parametrize(
    "method, level, warning",
    [("ratio", 0.99, None), ("percent", -1, "percentage point"), ("subtractive", -0.05, "subtractive corrected")],
)
def test_plr_latency_is_the_same_for_each_correction_method(method, level, warning):
    pupil = five_mm_constriction()
    pupil.filters.apply_baseline_correction(duration=5, method=method)
    with expect_warning(warning):
        result = pupil.metrics.plr_latency()
    # 1% of 5 mm is 0.05 mm, first reached at t = 0.3 (4.8 mm)
    assert result == pytest.approx(0.3)
    # The level of 99% of the baseline, in the units of the correction
    assert result.get_size()[0] == pytest.approx(level)


def test_plr_latency_without_correction_uses_baseline():
    with pytest.warns(UserWarning, match="not baseline corrected"):
        result = five_mm_constriction().metrics.plr_latency()
    # 99% of the 5 mm baseline is 4.95 mm, first reached at t = 0.3 (4.8 mm)
    assert result == pytest.approx(0.3)
    assert result.get_size()[0] == pytest.approx(4.95)


def test_plr_latency_without_constriction_is_nan():
    pupil = piprkit.PupilMeasurement(np.arange(-5, 3.001, 0.5), np.full(17, 5.0))
    pupil.set_light_stimulus(0, 1)
    pupil.filters.apply_baseline_correction(duration=5)
    with pytest.warns(UserWarning, match="does not constrict by 1%"):
        assert np.isnan(pupil.metrics.plr_latency())


def test_plr_latency_warns_when_implausibly_short():
    # The pupil drifts to 4.9 mm in the last second before onset, below 99% of the 5 s baseline mean (4.98 mm)
    time_data = np.round(np.arange(-5, 3.001, 0.1), 10)
    size = np.where(time_data < -1, 5.0, np.where(time_data < 0.2, 4.9, 4.9 - 2 * (time_data - 0.2)))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(0, 1)
    pupil.filters.apply_baseline_correction(duration=5)
    with pytest.warns(UserWarning, match="shorter than 0.1 s"):
        assert pupil.metrics.plr_latency() == pytest.approx(0)


def test_transient_plr(simulation, plot_metrics):
    # The simulated constriction reaches 0.55 at -0.75 s, inside [-1 + 0.18, -1 + 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = simulation.metrics.transient_plr()
    assert result == pytest.approx(0.55)
    np.testing.assert_allclose(result.get_time(), [-0.75])
    plot_metrics(simulation, [result], name="transient_plr")


def transient_pupil(size_after_onset):
    # 7 mm before onset; light from 0 to 1 s; samples every 0.02 s
    time_data = np.round(np.arange(-1, 1.001, 0.02), 10)
    size = np.where(time_data < 0, 7.0, size_after_onset(time_data))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(0, 1)
    return pupil


def test_transient_plr_ignores_minimum_outside_window():
    # A dip to 4 mm at 0.3 s, and a deeper one to 2 mm at 0.8 s, outside [0.18, 0.5]; uncorrected data
    pupil = transient_pupil(lambda t: np.minimum(4 + 10 * np.abs(t - 0.3), 2 + 10 * np.abs(t - 0.8)))
    # The deeper dip warns, but the value stays the minimum inside the window
    with pytest.warns(UserWarning, match="smaller after the transient_plr window"):
        assert pupil.metrics.transient_plr() == pytest.approx(4)


@pytest.mark.parametrize("edge, size_after_onset", [(0.18, lambda t: 4 + t), (0.5, lambda t: 4.5 + 5 * np.abs(t - 0.5))])
def test_transient_plr_minimum_at_window_edge_warns(edge, size_after_onset):
    # Rising through the window puts the minimum on its first sample; a dip with its bottom at 0.5 s puts it on its last
    pupil = transient_pupil(size_after_onset)
    with pytest.warns(UserWarning, match="edge of its window"):
        result = pupil.metrics.transient_plr()
    np.testing.assert_allclose(result.get_time(), [edge])


def test_transient_plr_warns_when_smaller_after_window():
    # The size falls until 0.8 s, past the window [0.18, 0.5], so the window minimum is not the peak
    pupil = transient_pupil(lambda t: 3 + 5 * np.abs(t - 0.8))
    with pytest.warns(UserWarning, match="smaller after the transient_plr window"):
        result = pupil.metrics.transient_plr()
    # 3 + 5 * 0.3 at the end of the window
    assert result == pytest.approx(4.5)
    # A window up to 1 s holds the bottom of the dip, so it does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pupil.metrics.transient_plr(end=1) == pytest.approx(3)


def test_transient_plr_without_valid_samples_is_nan():
    pupil = transient_pupil(lambda t: np.where(t <= 0.5, np.nan, 5.0))
    with pytest.warns(UserWarning, match="No valid samples"):
        assert np.isnan(pupil.metrics.transient_plr())


def auc_ramp(scale=1.0):
    # Baseline `scale` until light onset (-1 s); after light offset (0 s) the size falls linearly to
    # half the baseline at 10 s, and stays there
    time_data = np.round(np.arange(-12, 12.001, 0.5), 10)
    size = scale * np.where(time_data < 0, 1.0, 1 - 0.05 * np.minimum(time_data, 10))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    return pupil


def test_auc():
    # BPD - APD rises linearly from 0 to 0.5 over 0-10 s: an area of 0.5 * 10 / 2 = 2.5
    result = auc_ramp().metrics.auc(0, 10)
    assert result == pytest.approx(2.5)
    assert result.get_name() == "AUC_0-10^mm"
    # The location is the samples in the window, so the plot can shade the area
    np.testing.assert_allclose(result.get_time(), np.arange(0, 10.001, 0.5))
    np.testing.assert_allclose(result.get_size(), 1 - 0.05 * result.get_time())


def test_auc_plot_shades_the_area():
    result = auc_ramp().metrics.auc(0, 10)
    ax = result.plot()
    assert not ax.get_lines()
    (area,) = ax.collections
    assert area.get_label() == result.get_label()
    # From the baseline (1) down to the pupil size at 10 s (0.5): x from 0 to 10, y from 0.5 to 1
    np.testing.assert_allclose(area.get_paths()[0].get_extents().bounds, [0, 0.5, 10, 0.5])


def test_auc_bridges_gaps():
    pupil = auc_ramp()
    size = pupil.get_size().copy()
    size[(pupil.get_time() > 3) & (pupil.get_time() < 6)] = np.nan
    pupil.set_time_and_size(pupil.get_time(), size)
    # The gap lies on a straight stretch, so the straight bridge gives the same area
    result = pupil.metrics.auc(0, 10)
    assert result == pytest.approx(2.5)
    # The samples in the gap are left out of the location, as in the integral
    assert not np.isnan(result.get_size()).any()


def auc_above_baseline():
    # As auc_ramp, but after light offset the size rises from 0.8 and crosses the baseline (1) at 4 s
    pupil = auc_ramp()
    time_data = pupil.get_time()
    pupil.set_time_and_size(time_data, np.where(time_data < 0, 1.0, 0.8 + 0.05 * np.minimum(time_data, 10)))
    return pupil


def test_auc_warns_on_negative_part():
    # Above the baseline from 4 to 10 s: -0.3 * 6 / 2 = -0.9, against 0.2 * 4 / 2 = 0.4 below it
    with pytest.warns(UserWarning, match="that part adds -0.9 to auc"):
        result = auc_above_baseline().metrics.auc(0, 10)
    assert result == pytest.approx(-0.5)


def test_auc_at_baseline_does_not_warn():
    # At light offset the size equals the baseline, which is not a negative part
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auc_ramp().metrics.auc(0, 10)


@pytest.mark.parametrize("method, factor", [(None, 5), ("ratio", 1), ("percent", 100), ("subtractive", 5)])
def test_auc_per_correction_method(method, factor):
    # A 5 mm baseline: the ratio area is 2.5, percent is 100 times that, mm and subtractive 5 times
    pupil = auc_ramp(scale=5)
    if method is not None:
        pupil.filters.apply_baseline_correction(method=method)
    assert pupil.metrics.auc(0, 10) == pytest.approx(2.5 * factor)


def test_auc_early_and_late(plot_metrics):
    # The redilation of the default simulation is 1 - 0.43125 * exp(-0.4 t) after light offset (0 s)
    simulation = piprkit.simulate_pupil_measurement(t_start=-12, t_end=31, num_samples=4301)
    early, late = simulation.metrics.auc_early(), simulation.metrics.auc_late()
    assert early == pytest.approx(0.43125 / 0.4 * (1 - np.exp(-4)), abs=5e-3)
    assert late == pytest.approx(0.43125 / 0.4 * (np.exp(-4) - np.exp(-12)), abs=1e-4)
    plot_metrics(simulation, [early, late], name="auc")


def test_auc_window_past_data_warns():
    with pytest.warns(UserWarning, match="End time is after the last data point"):
        auc_ramp().metrics.auc(10, 30)


def test_auc_without_samples_is_nan():
    with pytest.warns(UserWarning, match="Fewer than 2 valid samples"):
        assert np.isnan(auc_ramp().metrics.auc(0.1, 0.2))


def test_auc_baseline_duration():
    # The data start 11 s before light onset, so a 12 s baseline starts before the first sample
    with pytest.warns(UserWarning, match="Start time is before the first data point"):
        auc_ramp().metrics.auc(0, 10, baseline_duration=12)
    # A 5 s baseline fits in the data and has the same size (1), so the area is unchanged
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert auc_ramp().metrics.auc(0, 10, baseline_duration=5) == pytest.approx(2.5)


def test_pipr_duration(simulation, plot_metrics):
    # The redilation 1 - 0.43125 * exp(-0.4 t) reaches 0.99 at t = 9.410 s; the next sample is 9.42 s
    result = simulation.metrics.pipr_duration()
    assert result == pytest.approx(9.42)
    assert result.get_name() == "99%-PIPRDuration^R"
    # 95%: reached at t = 5.387 s; the next sample is 5.39 s
    assert simulation.metrics.pipr_duration(baseline_percent=95) == pytest.approx(5.39)
    plot_metrics(simulation, [result], name="pipr_duration")


def test_pipr_duration_never_reached_is_nan(simulation):
    # The simulated redilation only approaches the baseline
    with pytest.warns(UserWarning, match="does not return to 100% of baseline"):
        assert np.isnan(simulation.metrics.pipr_duration(baseline_percent=100))


def five_mm_redilation():
    # A 5 mm baseline; light from -1 to 0 s at 3 mm; after offset the size rises 0.45 mm/s back to 5 mm
    time_data = np.round(np.arange(-12, 8.001, 0.1), 10)
    size = np.where(time_data <= -1, 5.0, np.where(time_data < 0, 3.0, np.minimum(3 + 0.45 * time_data, 5)))
    pupil = piprkit.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    return pupil


@pytest.mark.parametrize(
    "method, warning",
    [("ratio", None), ("percent", "percentage point"), ("subtractive", "subtractive corrected")],
)
def test_pipr_duration_is_the_same_for_each_correction_method(method, warning):
    pupil = five_mm_redilation()
    pupil.filters.apply_baseline_correction(method=method)
    # 99% of 5 mm is 4.95 mm, reached at t = 4.33 s; the next sample is 4.4 s
    with expect_warning(warning):
        assert pupil.metrics.pipr_duration() == pytest.approx(4.4)


def test_pipr_duration_without_correction_uses_baseline():
    with pytest.warns(UserWarning, match="not baseline corrected"):
        result = five_mm_redilation().metrics.pipr_duration()
    # 99% of the 5 mm baseline is 4.95 mm, reached at t = 4.33 s; the next sample is 4.4 s
    assert result == pytest.approx(4.4)


@pytest.mark.parametrize(
    "correction, name",
    [
        (None, "6s-PIPR_5.95-6.05^mm"),
        (BaselineCorrection("ratio", 5.0, 10), "6s-PIPR_5.95-6.05^R"),
        (BaselineCorrection("percent", 5.0, 10), "6s-PIPR_5.95-6.05^%"),
        (BaselineCorrection("subtractive", 5.0, 10), "6s-PIPR_5.95-6.05^S"),
    ],
)
def test_metric_result_name_ends_with_the_correction(correction, name):
    result = MetricResult(0.8, correction, "6s-PIPR_5.95-6.05")
    assert result.get_name() == name
    assert result.get_name(with_correction=False) == "6s-PIPR_5.95-6.05"


def test_metric_result_without_name():
    assert MetricResult(0.8).get_name() is None
    assert MetricResult(0.8).get_label() is None


@pytest.mark.parametrize(
    "correction, name, label",
    [
        (BaselineCorrection("ratio", 5.0, 10), "6s-PIPR_5.95-6.05", r"6s-PIPR${}_{\mathrm{5.95\text{-}6.05}}^{\mathrm{R}}$"),
        (None, "Baseline_-10-0", r"Baseline${}_{\mathrm{-10\text{-}0}}^{\mathrm{mm}}$"),
        (BaselineCorrection("percent", 5.0, 10), "1%-PLRLatency", r"1%-PLRLatency${}^{\%}$"),
        (
            BaselineCorrection("subtractive", 5.0, 10),
            "net_pipr(PIPR_10-30, PIPR_10-30)",
            r"net_pipr(PIPR${}_{\mathrm{10\text{-}30}}$, PIPR${}_{\mathrm{10\text{-}30}}$)${}^{\mathrm{S}}$",
        ),
    ],
)
def test_metric_result_label_is_typeset(correction, name, label):
    result = MetricResult(0.8, correction, name)
    assert result.get_label() == label
    # Drawing the legend fails if matplotlib cannot parse the label
    ax = result.plot()
    ax.legend()
    ax.figure.canvas.draw()


def corrected_simulation(method):
    # Noise-free simulation from 12 s before to 31 s after light offset, long enough for every window
    pupil = piprkit.simulate_pupil_measurement(t_start=-12, t_end=31, num_samples=4301)
    if method is not None:
        pupil.filters.apply_baseline_correction(method=method)
    return pupil


@pytest.mark.parametrize("method, suffix", [(None, "^mm"), ("ratio", "^R"), ("percent", "^%"), ("subtractive", "^S")])
@pytest.mark.parametrize(
    "metric, args, name",
    [
        ("baseline", (), "Baseline_-10-0"),
        ("transient_plr", (), "TransientPLR_0.18-0.5"),
        ("plr_latency", (), "1%-PLRLatency"),
        ("plr_latency", (10,), "10%-PLRLatency"),
        ("peak_constriction", (), "PeakConstriction"),
        ("time_to_peak", (), "TimeToPeak"),
        ("pipr_xs", (10, 30), "PIPR_10-30"),
        ("pipr_6s", (), "6s-PIPR_5.95-6.05"),
        ("auc", (0, 5), "AUC_0-5"),
        ("auc_early", (), "AUC_0-10"),
        ("auc_late", (), "AUC_10-30"),
        ("pipr_duration", (), "99%-PIPRDuration"),
    ],
)
def test_metric_name_states_window_and_correction(metric, args, name, method, suffix):
    pupil = corrected_simulation(method)
    # Except on ratio data, plr_latency and pipr_duration warn about their level (tested above)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = getattr(pupil.metrics, metric)(*args)
    assert result.get_name() == name + suffix
