import warnings

import numpy as np
import pytest

import piprkit
from tests.pipeline import metric_results, preprocess_measurements, preprocess_series

REAL_SIZE_RANGE = (1.5, 9)
REAL_IDS = [f"real-{i}" for i in range(6)]
# Measurements 0, 2 and 4 keep constricting for seconds after light offset, so the lowest point of their peak
# search window can be its last sample, depending on the noise
STILL_CONSTRICTING = (0, 2, 4)


def test_public_api_resolves():
    for name in piprkit.__all__:
        assert getattr(piprkit, name) is not None


def test_simulated_pipeline_recovers_parameters():
    # The default response scaled by 5, with noise. After smoothing and a ratio baseline
    # correction it should match the default simulation: peak = 2.75 / 5 = 0.55,
    # PIPR at 6 s = 1 - 0.43125 * exp(-0.4 * 6).
    pm = piprkit.simulate_pupil_measurement(
        baseline_c=5.0,
        maintained_constriction_c=2.75,
        maintained_constriction_m=0.125,
        redilation_P=5.0,
        noise_std=0.1,
        rng=0,
    )
    pm.filters.rolling_mean(0.2)
    pm.filters.apply_baseline_correction(duration=5)
    assert pm.metrics.baseline(5) == pytest.approx(1)
    assert pm.metrics.peak_constriction() == pytest.approx(0.55, abs=0.01)
    assert pm.metrics.pipr_6s() == pytest.approx(1 - 0.43125 * np.exp(-2.4), abs=0.01)


@pytest.fixture(scope="module")
def real_series_stages():
    return preprocess_series(piprkit.load_real_series(), REAL_SIZE_RANGE)


@pytest.fixture(scope="module")
def real_measurement_stages(real_series_stages):
    return preprocess_measurements(real_series_stages["3_size_trimmed"])


@pytest.fixture(scope="module")
def real_measurements(real_measurement_stages):
    return [stages["6_baseline_corrected"] for stages in real_measurement_stages]


def test_real_series_stages(real_series_stages, plot_stage):
    # The packaged series has no blinks
    assert list(real_series_stages) == ["1_raw", "3_size_trimmed"]
    size = real_series_stages["3_size_trimmed"].get_size()
    size = size[~np.isnan(size)]
    assert ((size >= REAL_SIZE_RANGE[0]) & (size <= REAL_SIZE_RANGE[1])).all()
    previous = None
    for name, stage in real_series_stages.items():
        plot_stage(stage, previous, name=name)
        previous = stage


def test_real_series_pipeline(real_measurement_stages):
    assert len(real_measurement_stages) == 6


@pytest.mark.parametrize("index", range(6), ids=REAL_IDS)
def test_real_series_measurement_stages(real_measurement_stages, index, plot_stage):
    stages = real_measurement_stages[index]
    split, median = stages["4_split"], stages["5_median"]
    assert np.isfinite(split.get_size()).all()
    assert np.isfinite(median.get_size()).all()
    plot_stage(split, name="4_split")
    plot_stage(median, split, name="5_median")
    # Baseline corrected data has other units, so it is drawn without the stage before it
    plot_stage(stages["6_baseline_corrected"], name="6_baseline_corrected")


@pytest.mark.parametrize("index", range(6), ids=REAL_IDS)
def test_real_series_metrics(real_measurements, index, plot_metrics):
    pm = real_measurements[index]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = metric_results(pm)
    # Measurements 1, 3 and 5 redilate soon after the pulse, so their peak lies inside the search window
    if index not in STILL_CONSTRICTING:
        assert not [w for w in caught if "may still be constricting" in str(w.message)]
    baseline, peak, time_to_peak, pipr = results
    onset, offset = pm.get_light_stimulus().get_timing()
    assert baseline == pytest.approx(1)
    assert 0 < peak < 1
    assert 0 <= time_to_peak <= offset - onset + 1.0
    assert np.isfinite(pipr)
    plot_metrics(pm, results, name="7_metrics")


@pytest.mark.parametrize("index", range(6), ids=REAL_IDS)
def test_real_series_transient_plr(real_measurements, index, plot_metrics):
    # The pupil is still constricting 0.5 s after light onset, so the default window misses the peak and warns
    pm = real_measurements[index]
    onset, offset = pm.get_light_stimulus().get_timing()
    with pytest.warns(UserWarning, match="smaller after the transient_plr window"):
        default = pm.metrics.transient_plr()
    # A window up to the end of the peak search finds the peak of peak_constriction. Only the measurements that keep
    # constricting may warn that this peak is the last sample of the window
    with warnings.catch_warnings():
        if index not in STILL_CONSTRICTING:
            warnings.simplefilter("error")
        peak = pm.metrics.peak_constriction()
        assert pm.metrics.transient_plr(end=offset - onset + 1.0) == pytest.approx(peak)
    plot_metrics(pm, [default, peak], name="transient_plr")


@pytest.mark.parametrize("index", range(6), ids=REAL_IDS)
def test_real_series_phase_model(real_measurements, index, plot_phase_model):
    model = piprkit.PhaseModel(real_measurements[index], baseline_duration=5)
    # c is metrics.baseline over the same 5 s as the ratio correction, so it is 1
    assert model.get_param("baseline", "c") == pytest.approx(1)
    flags = model.get_flags()
    for phase, params in model.get_all_params().items():
        flagged = any(flags[phase][flag] for flag in ("not_defined", "not_converged", "too_few_samples"))
        # A phase has finite parameters, or a flag that says why not
        assert flagged or np.isfinite(list(params.values())).all(), phase
    plot_phase_model(model, name="8_phase_model")


def test_real_series_measurement_0(real_measurements):
    # The pupil keeps constricting after light offset, and redilates only slowly afterwards
    measurement = real_measurements[0]
    model = piprkit.PhaseModel(measurement, baseline_duration=5)
    offset = measurement.get_light_stimulus().get_timing()[1]
    assert model.get_window()["constriction"]["end"] > offset
    flags = model.get_flags()["redilation"]
    assert not flags["not_defined"]
    params = model.get_all_params()["redilation"]
    assert params["S"] < 0
    assert -0.2 < params["k"] < 0


# Measurements 0, 2 and 4 stay constricted for several seconds after light offset; 1, 3 and 5 redilate right away
@pytest.mark.parametrize("index, plateau", [(0, True), (1, False), (2, True), (3, False), (4, True), (5, False)], ids=REAL_IDS)
def test_real_series_maintained_constriction_and_redilation_start(real_measurements, index, plateau):
    measurement = real_measurements[index]
    model = piprkit.PhaseModel(measurement, baseline_duration=5)
    offset = measurement.get_light_stimulus().get_timing()[1]
    window = model.get_window()
    maintained = model.get_flags()["maintained_constriction"]
    assert window["maintained_constriction"]["end"] == window["redilation"]["start"]
    assert window["maintained_constriction"]["sections"] == ("post_stimulus",)
    if plateau:
        assert window["redilation"]["start"] - offset > 4
        assert not maintained["not_defined"]
    else:
        assert window["redilation"]["start"] - offset < 1
        assert maintained["not_defined"]
