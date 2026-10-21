# One pipeline for the packaged real series (test_integration.py) and private recordings (test_private_data.py).
# Each function keeps a copy after every stage, keyed by the name its figure is saved under.

BLINK_BUFFER = 0.1
PRE_STIMULUS = 5
POST_STIMULUS = 60
MEDIAN_WINDOW = 0.05
BASELINE_DURATION = 5


def preprocess_series(series, size_range=None):
    stages = {"1_raw": series.copy()}
    series = series.copy()
    if series.get_blinks():
        series.filters.remove_blinks(BLINK_BUFFER)
        stages["2_blinks_removed"] = series.copy()
    if size_range is not None:
        series.trim_size(*size_range)
        stages["3_size_trimmed"] = series.copy()
    return stages


def preprocess_measurements(series):
    # split() drops NaN samples (trim_time with drop=True), so no separate drop_nan()
    measurements = []
    for pm in series.split(PRE_STIMULUS, POST_STIMULUS):
        stages = {"4_split": pm.copy()}
        pm.filters.rolling_median(MEDIAN_WINDOW)
        stages["5_median"] = pm.copy()
        pm.filters.apply_baseline_correction(duration=BASELINE_DURATION)
        stages["6_baseline_corrected"] = pm.copy()
        measurements.append(stages)
    return measurements


def metric_results(pm):
    return [
        pm.metrics.baseline(BASELINE_DURATION),
        pm.metrics.peak_constriction(),
        pm.metrics.time_to_peak(),
        pm.metrics.pipr_6s(),
    ]
