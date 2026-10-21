from functools import cache
from pathlib import Path

import numpy as np
import pytest

import piprkit
from tests.pipeline import BLINK_BUFFER, metric_results, preprocess_measurements, preprocess_series

# Smoke tests on recordings that cannot be shared: pytest tests/test_private_data.py --data=<recording.csv>
# Without --data every test here is skipped. Assertions use generic bounds only, never values from the data.
# --data-size-range MIN MAX adds the trim_size stage. With --plots each stage is saved per recording.


def _options(config):
    # (size column, size range), the arguments after path of the cached functions below
    size_range = config.getoption("--data-size-range")
    return config.getoption("--data-column") or 1, None if size_range is None else tuple(size_range)


@cache
def _load(path, size_col):
    recording = piprkit.load_csv(path, size_col=size_col)
    if isinstance(recording, piprkit.PupilMeasurement):
        # One stimulus: wrap it in a series so split() handles it like the rest
        stimulus = recording.get_light_stimulus()
        series = piprkit.PupilSeries(recording.get_time(), recording.get_size(), [] if stimulus is None else [stimulus.get_timing()])
        series.set_blinks(recording.get_blinks())
        return series
    return recording


@cache
def _series_stages(path, size_col, size_range):
    return preprocess_series(_load(path, size_col), size_range)


@cache
def _measurement_stages(path, size_col, size_range):
    # Split the last series stage
    return preprocess_measurements(list(_series_stages(path, size_col, size_range).values())[-1])


def pytest_generate_tests(metafunc):
    paths = metafunc.config.getoption("--data")
    options = _options(metafunc.config)
    if "index" in metafunc.fixturenames:
        params = [(path, i) for path in paths for i in range(len(_measurement_stages(path, *options)))]
        metafunc.parametrize("path, index", params, ids=[f"{Path(path).stem}-{i}" for path, i in params])
    elif "path" in metafunc.fixturenames:
        metafunc.parametrize("path", paths, ids=[Path(path).stem for path in paths])


@pytest.fixture
def options(request):
    return _options(request.config)


def test_series_stages(path, options, plot_stage):
    stages = _series_stages(path, *options)
    raw = stages["1_raw"]
    time = raw.get_time()
    stimuli = raw.get_light_stimuli_series().get_timings()
    assert stimuli
    for start, end in stimuli + raw.get_blinks():
        assert time[0] <= start <= end <= time[-1]
    if "2_blinks_removed" in stages:
        size = stages["2_blinks_removed"].get_size()
        for start, end in raw.get_blinks():
            in_blink = (time >= start - BLINK_BUFFER) & (time <= end + BLINK_BUFFER)
            assert np.isnan(size[in_blink]).all()
    if "3_size_trimmed" in stages:
        min_size, max_size = options[1]
        size = stages["3_size_trimmed"].get_size()
        size = size[~np.isnan(size)]
        assert ((size >= min_size) & (size <= max_size)).all()
    previous = None
    for name, stage in stages.items():
        plot_stage(stage, previous, name=name)
        previous = stage


def test_measurement_stages(path, index, options, plot_stage):
    stages = _measurement_stages(path, *options)[index]
    split, median = stages["4_split"], stages["5_median"]
    assert np.isfinite(split.get_size()).all()
    assert np.isfinite(median.get_size()).all()
    plot_stage(split, name="4_split")
    plot_stage(median, split, name="5_median")
    # Baseline corrected data has other units, so it is drawn without the stage before it
    plot_stage(stages["6_baseline_corrected"], name="6_baseline_corrected")


def test_metrics(path, index, options, plot_metrics):
    pm = _measurement_stages(path, *options)[index]["6_baseline_corrected"]
    results = metric_results(pm)
    baseline, peak, time_to_peak, pipr = results
    onset, offset = pm.get_light_stimulus().get_timing()
    assert baseline == pytest.approx(1)
    assert 0 < peak < 1
    assert 0 <= time_to_peak <= offset - onset + 1.0
    assert np.isfinite(pipr)
    plot_metrics(pm, results, name="7_metrics")


def test_phase_model(path, index, options, plot_phase_model):
    model = piprkit.PhaseModel(_measurement_stages(path, *options)[index]["6_baseline_corrected"], baseline_duration=5)
    flags = model.get_flags()
    for phase, params in model.get_all_params().items():
        flagged = any(flags[phase][flag] for flag in ("not_defined", "not_converged", "too_few_samples"))
        # A phase has finite parameters, or a flag that says why not
        assert flagged or np.isfinite(list(params.values())).all(), phase
    plot_phase_model(model, name="8_phase_model")
