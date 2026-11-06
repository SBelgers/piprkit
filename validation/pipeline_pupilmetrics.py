# Pipeline B, PIPRkit side: PupilMetrics' cleaning and measures, matched as closely as PIPRkit allows.
# Writes its outputs to validation/output/pupilmetrics_<rate>hz/, including the cleaned series that
# pupilmetrics_reference.py also measures; report.py compares them with PupilMetrics' outputs.
import argparse

import numpy as np
import pandas as pd

from common import EYES, OUTPUT, POST_ONSET, PRE_STIMULUS, call, load_series

MEASURES = ("baseline", "MCA", "RT", "6s constriction")


def piprkit_steps(series):
    # B1-B3 with the options nearest to PupilMetrics', and the warnings of B1 and B2
    b1 = series.copy()
    _, clip_warnings = call(b1.trim_size, 0.2, 10)
    b2 = b1.copy()
    _, rate_warnings = call(b2.filters.limit_rate_of_change, 6.0)
    # To match PupilMetrics' straight-line rule: more than f // 8 samples, 0.125 s at 200 Hz
    _, constant_warnings = call(b2.filters.remove_constant_signal, 0.125)
    b3 = b2.drop_nan(inplace=False)
    b3.interpolate(series.get_time())
    return b1, b2, b3, {"B1": clip_warnings, "B2": join(rate_warnings, constant_warnings)}


def piprkit_metrics(epoch, matched):
    # PIPRkit's values in PupilMetrics' units; matched=False uses PIPRkit's default arguments instead
    onset, offset = epoch.get_light_stimulus().get_timing()
    duration = offset - onset
    # To match PupilMetrics: baseline 0.25 s before onset, peak search to 2.025 s and "6s constriction" 7.0-7.1 s after onset
    # PIPRkit counts the last two from light offset, the stimulus duration later
    ratio, ratio_warnings = call(epoch.filters.baseline_correction_ratio, 0.25, inplace=False)
    if matched:
        baseline, baseline_warnings = call(epoch.metrics.baseline, 0.25)
        peak, peak_warnings = call(ratio.metrics.peak_constriction, 2.025 - duration)
        time_to_peak, time_warnings = call(ratio.metrics.time_to_peak, 2.025 - duration)
        pipr, pipr_warnings = call(ratio.metrics.pipr_xs, 7.0 - duration, 7.1 - duration)
    else:
        baseline, baseline_warnings = call(epoch.metrics.baseline)
        peak, peak_warnings = call(ratio.metrics.peak_constriction)
        time_to_peak, time_warnings = call(ratio.metrics.time_to_peak)
        pipr, pipr_warnings = call(ratio.metrics.pipr_6s)
    values = {
        "baseline": float(baseline),
        "MCA": 100 * (1 - peak),
        "RT": float(time_to_peak),
        "6s constriction": 100 * (1 - pipr),
    }
    texts = {
        "baseline": baseline_warnings,
        "MCA": join(ratio_warnings, peak_warnings),
        "RT": join(ratio_warnings, time_warnings),
        "6s constriction": join(ratio_warnings, pipr_warnings),
    }
    return values, texts


def join(*texts):
    return " | ".join(text for text in texts if text)


def write_options(folder):
    rows = [
        ("B1", "range clip in PupilData (< 0.2 or > 10 mm)", "trim_size(0.2, 10)", "", ""),
        ("B2", "PupilData.no_artefact(), thresholds 3 ('medium'), 'Remove only steep artifacts' on",
         "filters.limit_rate_of_change(6.0), then filters.remove_constant_signal(0.125)", "", ""),
        ("B3", "the interpolation at the end of no_artefact()", "drop_nan(), then interpolate(grid)", "", ""),
        ("B4", "per flash in the whole recording", f"split({PRE_STIMULUS:g}, {POST_ONSET:g} - D)", "", ""),
        ("baseline", "mean of the int(0.25 f) samples before the flash", "metrics.baseline(0.25)",
         "metrics.baseline()", ""),
        ("MCA", "100 - smallest % of baseline in the round(2.025 f) samples from the flash",
         "metrics.peak_constriction(2.025 - D) after filters.baseline_correction_ratio(0.25)",
         "metrics.peak_constriction() on the same data", "100 (1 - peak)"),
        ("RT", "(n + 2.42) / f, n samples from the flash to that smallest value",
         "metrics.time_to_peak(2.025 - D) on the same data", "metrics.time_to_peak() on the same data", ""),
        ("6s constriction", "100 - mean % of baseline over the samples 7 f to int(7.1 f) after the flash",
         "metrics.pipr_xs(7.0 - D, 7.1 - D) on the same data", "metrics.pipr_6s() on the same data",
         "100 (1 - value)"),
    ]
    columns = ["step", "pupilmetrics", "piprkit", "piprkit_default", "conversion"]
    pd.DataFrame(rows, columns=columns).to_csv(folder / "options.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Pipeline B, PIPRkit side, on the packaged real recording")
    parser.add_argument("--rate", type=int, default=200, help="rate of the even time grid in Hz (default 200)")
    rate = parser.parse_args().rate
    folder = OUTPUT / f"pupilmetrics_{rate}hz"
    folder.mkdir(parents=True, exist_ok=True)
    write_options(folder)
    metric_rows, step_rows = [], []
    for eye in EYES:
        series, _, _ = load_series(eye, rate)
        b1, b2, b3, step_warnings = piprkit_steps(series)
        np.savez_compressed(folder / f"piprkit_{eye}.npz", time=series.get_time(), b1=b1.get_size(),
                            b2=b2.get_size(), b3=b3.get_size())
        for step, text in step_warnings.items():
            step_rows.append({"eye": eye, "step": step, "piprkit_warnings": text})
        durations = [end - start for start, end in b3.get_light_stimuli_series().get_timings()]
        for trial, epoch in enumerate(b3.split(PRE_STIMULUS, POST_ONSET - np.mean(durations))):
            matched, matched_warnings = piprkit_metrics(epoch, matched=True)
            default, default_warnings = piprkit_metrics(epoch, matched=False)
            for measure in MEASURES:
                metric_rows.append({
                    "eye": eye,
                    "trial": trial + 1,
                    "metric": measure,
                    "piprkit": matched[measure],
                    "piprkit_default": default[measure],
                    "piprkit_warnings": matched_warnings[measure],
                    "default_warnings": default_warnings[measure],
                })
    pd.DataFrame(metric_rows).to_csv(folder / "piprkit_metrics.csv", index=False)
    pd.DataFrame(step_rows).to_csv(folder / "piprkit_step_warnings.csv", index=False)


if __name__ == "__main__":
    main()
