# Pipeline A: PyPlr's preprocessing and PLR metrics next to PIPRkit in a similar pipeline; small differences between
# the tools are not corrected. Writes both tools' outputs to validation/output/pyplr_<rate>hz/; report.py compares them.
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyplr import preproc, utils
from pyplr.plr import PLR

import piprkit
from common import BLINKS, EYES, OUTPUT, POST_ONSET, PRE_STIMULUS, call, load_series, write_versions

# pandas 2.2 on Python 3.14 warns about chained assignment on plain column assignments, also inside PyPlr; the values
# are set all the same, so this warning is not shown.
warnings.filterwarnings("ignore", message="ChainedAssignmentError", category=FutureWarning)


def pyplr_steps(series, blinks, stimuli, rate, half):
    # A1-A5 in PyPlr
    samples = pd.DataFrame({"diameter": series.get_size()}, index=series.get_time())
    events = pd.DataFrame({"start_timestamp": blinks[:, 0], "end_timestamp": blinks[:, 1]})
    a1 = preproc.mask_blinks(samples, events)
    a2 = preproc.interpolate_pupil(a1)
    a3 = preproc.rolling_mean_series(a2, 2 * half + 1)
    pre, duration = round(PRE_STIMULUS * rate), round((PRE_STIMULUS + POST_ONSET) * rate)
    onsets = pd.DataFrame(index=stimuli[:, 0])
    a4 = utils.extract(a3.copy(), onsets, offset=-pre, duration=duration)
    baselines = a4.loc[(slice(None), slice(0, pre - 1)), ["diameter"]].groupby(level="event").mean()
    a5 = preproc.percent_signal_change(a4.copy(), baselines, ["diameter"])
    return a1, a2, a3, a4, a5


def piprkit_steps(series, blinks):
    # A1-A5 in PIPRkit, with the options that match PyPlr's, and the warnings of A5 per trial
    a1 = series.copy()
    a1.set_blinks([tuple(blink) for blink in blinks])
    a1.filters.remove_blinks()
    a2 = a1.drop_nan(inplace=False)
    a2.interpolate(series.get_time())
    # 0.1 s, to match PyPlr's rolling mean of 21 samples at 200 Hz
    a3 = a2.filters.rolling_mean(0.1, inplace=False)
    durations = [end - start for start, end in a3.get_light_stimuli_series().get_timings()]
    a4 = a3.split(PRE_STIMULUS, POST_ONSET - np.mean(durations))
    a5 = [call(epoch.filters.baseline_correction_percent, PRE_STIMULUS, inplace=False) for epoch in a4]
    return a1, a2, a3, a4, [epoch for epoch, _ in a5], [text for _, text in a5]


def pyplr_metrics(values, rate, duration):
    # PyPlr's metrics of one epoch, and the time of its minimum after the onset sample
    plr = PLR(values, rate, round(PRE_STIMULUS * rate), duration)
    metrics = {
        "Baseline": plr.baseline(),
        "PeakCon": plr.peak_constriction(),
        "ConAmplitude": plr.constriction_amplitude(),
        "T2MaxCon": plr.time_to_max_constriction(),
        "Latency_a": plr.latency_to_constriction_a(),
        "ConTime": plr.constriction_time(),
    }
    return metrics, (plr.peak_constriction_idx() - plr.onset_idx) / rate


def piprkit_metrics(epoch, matched):
    # PIPRkit's values, with the options that match PyPlr's; matched=False uses PIPRkit's default arguments instead
    ratio, ratio_warnings = call(epoch.filters.baseline_correction_ratio, PRE_STIMULUS, inplace=False)
    # Epoch time 0 is light offset, so a window up to the last sample time searches to the end of the epoch
    window = (epoch.get_time()[-1],) if matched else ()
    baseline, baseline_warnings = call(epoch.metrics.baseline, *((PRE_STIMULUS,) if matched else ()))
    peak, peak_warnings = call(epoch.metrics.peak_constriction, *window)
    time_to_peak, time_warnings = call(epoch.metrics.time_to_peak, *window)
    # 1% of the 5 s baseline, to match the 1% of PyPlr's Latency_a
    latency, latency_warnings = call(ratio.metrics.plr_latency, *((1.0,) if matched else ()))
    values = {
        "Baseline": float(baseline),
        "PeakCon": float(peak),
        "ConAmplitude": baseline - peak,
        "T2MaxCon": float(time_to_peak),
        "Latency_a": float(latency),
        "ConTime": time_to_peak - latency,
    }
    texts = {
        "Baseline": baseline_warnings,
        "PeakCon": peak_warnings,
        "ConAmplitude": join(baseline_warnings, peak_warnings),
        "T2MaxCon": time_warnings,
        "Latency_a": join(ratio_warnings, latency_warnings),
        "ConTime": join(time_warnings, ratio_warnings, latency_warnings),
    }
    return values, texts, [baseline, peak, time_to_peak]


def join(*texts):
    return " | ".join(text for text in texts if text)


def write_options(folder, rate, half):
    pre, duration = round(PRE_STIMULUS * rate), round((PRE_STIMULUS + POST_ONSET) * rate)
    rows = [
        ("A1", "preproc.mask_blinks(samples, blinks)", "set_blinks(blinks), then filters.remove_blinks()", "", ""),
        ("A2", "preproc.interpolate_pupil(samples)", "drop_nan(), then interpolate(grid)", "", ""),
        ("A3", f"preproc.rolling_mean_series(samples, {2 * half + 1})", "filters.rolling_mean(0.1)", "", ""),
        ("A4", f"utils.extract(samples, events, offset=-{pre}, duration={duration})",
         f"split({PRE_STIMULUS:g}, {POST_ONSET:g} - D)", "", ""),
        ("A5", f"preproc.percent_signal_change(ranges, baselines), baselines: mean of the {pre} samples before the "
         "onset sample", f"filters.baseline_correction_percent({PRE_STIMULUS:g})", "", ""),
        ("Baseline", "PLR.baseline()", f"metrics.baseline({PRE_STIMULUS:g})", "metrics.baseline()", ""),
        ("PeakCon", "PLR.peak_constriction()", "metrics.peak_constriction(w), w = epoch end",
         "metrics.peak_constriction()", ""),
        ("ConAmplitude", "PLR.constriction_amplitude()",
         f"metrics.baseline({PRE_STIMULUS:g}) - metrics.peak_constriction(w)",
         "metrics.baseline() - metrics.peak_constriction()", ""),
        ("T2MaxCon", "PLR.time_to_max_constriction()", "metrics.time_to_peak(w)", "metrics.time_to_peak()", ""),
        ("Latency_a", "PLR.latency_to_constriction_a()",
         f"metrics.plr_latency(1.0) after filters.baseline_correction_ratio({PRE_STIMULUS:g})",
         "metrics.plr_latency() on the same data", ""),
        ("ConTime", "PLR.constriction_time()", "T2MaxCon - Latency_a, both as above", "the same, with the defaults",
         ""),
    ]
    columns = ["step", "pyplr", "piprkit", "piprkit_default", "conversion"]
    pd.DataFrame(rows, columns=columns).to_csv(folder / "options.csv", index=False)


def plot_trial(folder, eye, trial, epoch, pyplr_time, pyplr_values, results):
    ax = epoch.plot(color="black", label="PIPRkit")
    piprkit.PupilMeasurement(pyplr_time, pyplr_values).plot(ax, color="tab:orange", linestyle="--", label="PyPlr")
    epoch.plot_light_stimulus(ax)
    for result in results:
        result.plot(ax)
    ax.set_title(f"Pipeline A, {eye} eye, trial {trial + 1}", fontsize=9)
    ax.legend(fontsize=7)
    ax.figure.savefig(folder / "figures" / f"{eye}_trial{trial + 1}.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    parser = argparse.ArgumentParser(description="Pipeline A: PIPRkit next to PyPlr on the packaged real recording")
    parser.add_argument("--rate", type=int, default=200, help="rate of the even time grid in Hz (default 200)")
    rate = parser.parse_args().rate
    half = round(0.05 * rate)
    folder = OUTPUT / f"pyplr_{rate}hz"
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    write_versions(folder / "versions.csv")
    write_options(folder, rate, half)
    blink_table = pd.read_csv(BLINKS)
    metric_rows, step_rows, context_rows = [], [], []
    for eye in EYES:
        series, stimuli, _ = load_series(eye, rate)
        blinks = blink_table.loc[blink_table["eye"] == eye, ["start", "end"]].to_numpy()
        p1, p2, p3, p4, p5 = pyplr_steps(series, blinks, stimuli, rate, half)
        k1, k2, k3, k4, k5, k5_warnings = piprkit_steps(series, blinks)
        arrays = {"time": series.get_time()}
        for step, pyplr_step, piprkit_step in (("a1", p1, k1), ("a2", p2, k2), ("a3", p3, k3)):
            arrays[f"{step}_pyplr"] = pyplr_step["diameter"].to_numpy()
            arrays[f"{step}_piprkit"] = piprkit_step.get_size()
        offsets = [end for _, end in series.get_light_stimuli_series().get_timings()]
        for trial, (epoch, corrected, offset) in enumerate(zip(k4, k5, offsets)):
            pyplr_time = p4.loc[trial, "orig_idx"].to_numpy()
            pyplr_values = p4.loc[trial, "diameter"].to_numpy()
            arrays[f"a4_pyplr_time_{trial}"] = pyplr_time
            arrays[f"a4_pyplr_{trial}"] = pyplr_values
            arrays[f"a4_piprkit_time_{trial}"] = epoch.get_time() + offset
            arrays[f"a4_piprkit_{trial}"] = epoch.get_size()
            arrays[f"a5_pyplr_{trial}"] = p5.loc[trial, "diameter_pc"].to_numpy()
            arrays[f"a5_piprkit_{trial}"] = corrected.get_size()
            step_rows.append({"eye": eye, "trial": trial + 1, "step": "A5", "piprkit_warnings": k5_warnings[trial]})
            reference, minimum_time = pyplr_metrics(pyplr_values, rate, stimuli[trial, 1] - stimuli[trial, 0])
            matched, matched_warnings, results = piprkit_metrics(epoch, matched=True)
            default, default_warnings, _ = piprkit_metrics(epoch, matched=False)
            for metric, value in reference.items():
                metric_rows.append({
                    "eye": eye,
                    "trial": trial + 1,
                    "metric": metric,
                    "pyplr": value,
                    "piprkit": matched[metric],
                    "piprkit_default": default[metric],
                    "piprkit_warnings": matched_warnings[metric],
                    "default_warnings": default_warnings[metric],
                })
            context_rows.append({"eye": eye, "trial": trial + 1, "pyplr_minimum_after_onset_sample": minimum_time})
            plot_trial(folder, eye, trial, epoch, pyplr_time - offset, pyplr_values, results)
        np.savez_compressed(folder / f"{eye}.npz", **arrays)
    pd.DataFrame(metric_rows).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(step_rows).to_csv(folder / "step_warnings.csv", index=False)
    pd.DataFrame(context_rows).to_csv(folder / "context.csv", index=False)


if __name__ == "__main__":
    main()
