# Builds validation/output/report.log from the outputs of the pipeline scripts and draws its plots: the Pipeline B
# figures per trial and, per comparison, the processed traces, the processed samples and the metrics against each other.
# The log holds the numbers only.
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import piprkit
from common import EYES, OUTPUT, POST_ONSET, PRE_STIMULUS, load_series

# pandas 2.2 on Python 3.14 warns about chained assignment on plain column assignments; the values are set all the
# same, so this warning is not shown.
warnings.filterwarnings("ignore", message="ChainedAssignmentError", category=FutureWarning)
MEASURES = ("baseline", "MCA", "RT", "6s constriction")
UNITS = {"Baseline": "mm", "PeakCon": "mm", "ConAmplitude": "mm", "T2MaxCon": "s", "Latency_a": "s", "ConTime": "s",
         "baseline": "mm", "MCA": "%", "RT": "s", "6s constriction": "%"}
LABELS = {"pyplr": "PyPlr", "pupilmetrics": "PupilMetrics", "piprkit": "PIPRkit", "piprkit_default": "PIPRkit default",
          "conversion": "conversion"}
KNOWN_A = (
    "Onset: PyPlr counts from the first sample at or after the recorded onset; PIPRkit from the recorded onset, up "
    "to one sample (5 ms) earlier.",
    "Rolling mean: PyPlr's window trails (a sample and the 20 before it); PIPRkit's is centred (10 samples on each "
    "side). PyPlr's smoothed trace lags 50 ms.",
    "Latency: PyPlr takes the first sample more than 1% below the size at the onset sample; PIPRkit the first sample "
    "at or below 99% of the 5 s baseline.",
    "Peak: PyPlr searches the whole epoch, including the 5 s before onset; PIPRkit searches from onset to the end of "
    "the epoch.",
    "Windows: PyPlr counts whole samples and leaves out the last; PIPRkit's windows are in seconds and include both "
    "ends.",
)
KNOWN_B = (
    "Onset: PupilMetrics counts from the first sample at or after the recorded onset; PIPRkit from the recorded "
    "onset, up to one sample earlier.",
    "Cleaning: PupilMetrics also limits acceleration (90 mm/s²), takes speed as a central difference, removes any "
    "straight stretch (acceleration exactly 0), removes samples 1.2–3.3 s after a flash by its own rule, and merges "
    "gaps less than 0.25 s apart. PIPRkit limits speed with the difference to the previous sample and removes "
    "stretches of equal values only.",
    "Straight stretches: PupilMetrics needs more than f // 8 samples (0.125 s at 200 Hz, 0.12 s at 100 Hz); PIPRkit "
    "0.125 s.",
    "RT: PupilMetrics adds 2.42 samples (12 ms at 200 Hz, 24 ms at 100 Hz) to the time of the minimum.",
    "Windows: PupilMetrics counts whole samples from the flash sample and leaves out the last (baseline int(0.25 f), "
    "peak round(2.025 f), 6s constriction 7 f to int(7.1 f)); PIPRkit's windows are in seconds from the recorded onset "
    "and include both ends.",
    "Gaps at the start or end: PupilMetrics fills them with the first or last value; PIPRkit leaves them empty.",
    "Units: MCA and 6s constriction are in % constriction; PIPRkit's ratio values are converted with 100 (1 − value).",
)
SKIPPED_A = (
    "the derivative mask (mask_pupil_first_derivative): taken out",
    "T2Rec75pc: taken out; it counts from the minimum, pipr_duration from light offset",
    "zero masking: the recording has no zeros",
    "the confidence mask: the recording has no confidence values",
    "the Butterworth filter (butterworth_series): not in PIPRkit",
    "the velocity and acceleration metrics (Latency_b, T2MaxVel, VelConMax, VelConAve, AccConMax, VelRedAve, "
    "VelRedMax, AccRedMax): not in PIPRkit",
)
SKIPPED_B = (
    "the drop detection (fit_signal, no_drop): switched off with PupilMetrics' 'Remove only steep artifacts'; "
    "not in PIPRkit",
    "dAMP, dLAT and dAUC (velocity measures): not in PIPRkit",
)
DEFINITIONS = (
    "r: Pearson's correlation; 1 when the values lie on a rising straight line.",
    "95% CI and p: for r, p tests r = 0. Both assume 12 independent values; these are 6 trials of each eye of one "
    "recording.",
    "r²: the share of the variance that this line explains.",
    "CCC: Lin's concordance correlation coefficient; 1 only when the values lie on y = x.",
    "Mean difference: PIPRkit minus the reference.",
    "The plots draw y = x as perfect correlation.",
)
# Plot colours on the light surface: the eyes (blue and aqua, which also differ in marker shape), the reference tool,
# and the ink, axis, grid and surface tones
EYE_STYLES = {"left": ("#2a78d6", "o"), "right": ("#1baf7a", "s")}
REFERENCE_COLOUR = "#eb6834"
INK, SECONDARY, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#fcfcfb"


def table(frame):
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6g}")


def heading(text, underline):
    return ["", text, underline * len(text), ""]


def options(path):
    # Per step or metric, the reference's call, PIPRkit's matched and default calls, and the conversion
    lines = []
    for _, row in pd.read_csv(path).fillna("").iterrows():
        lines.append(row["step"])
        lines += [f"    {LABELS[column]}: {row[column]}" for column in row.index[1:] if row[column]]
    return lines


def rate_of(folder):
    return int(folder.name.split("_")[1][:-2])


def agreement(reference, piprkit_values):
    # n, Pearson's r with its 95% CI and p, r², Lin's CCC and the mean difference, over the pairs that have both values
    reference, piprkit_values = np.asarray(reference, dtype=np.float64), np.asarray(piprkit_values, dtype=np.float64)
    both = np.isfinite(reference) & np.isfinite(piprkit_values)
    x, y = reference[both], piprkit_values[both]
    r = p = low = high = ccc = difference = np.nan
    if x.size >= 4 and np.ptp(x) > 0 and np.ptp(y) > 0:
        result = stats.pearsonr(x, y)
        r, p = float(result.statistic), float(result.pvalue)
        low, high = result.confidence_interval(confidence_level=0.95)
    if x.size >= 2:
        spread = x.var() + y.var() + (x.mean() - y.mean()) ** 2
        ccc = 2 * np.cov(x, y, bias=True)[0, 1] / spread if spread > 0 else np.nan
        difference = float(np.mean(y - x))
    interval = "undefined" if np.isnan(low) else f"[{low:.3f}, {high:.3f}]"
    return {"n": int(x.size), "r": r, "95% CI": interval, "p": p, "r²": r**2, "CCC": ccc, "mean difference": difference}


def compare(reference, piprkit_values):
    # The largest absolute difference and Pearson's r, over the samples where both tools have a value
    both = np.isfinite(reference) & np.isfinite(piprkit_values)
    largest = np.abs(reference[both] - piprkit_values[both]).max() if both.any() else np.nan
    return largest, agreement(reference, piprkit_values)["r"]


def pair(reference_time, reference_values, piprkit_time, piprkit_values, rate):
    # The values of both tools at the times they share on the even grid
    reference_index = np.round(reference_time * rate).astype(np.int64)
    piprkit_index = np.round(piprkit_time * rate).astype(np.int64)
    _, i, j = np.intersect1d(reference_index, piprkit_index, return_indices=True)
    return reference_values[i], piprkit_values[j]


def removed(reference_mask, piprkit_mask, reference_name):
    return {
        "both": int((reference_mask & piprkit_mask).sum()),
        f"only {reference_name}": int((reference_mask & ~piprkit_mask).sum()),
        "only PIPRkit": int((~reference_mask & piprkit_mask).sum()),
        "samples": reference_mask.size,
    }


def warnings_table(frame, columns):
    # Each warning text once per metric or step and call, with how often it was raised
    rows = []
    for column, call in columns:
        for _, row in frame.iterrows():
            text = row[column] if isinstance(row[column], str) else ""
            for warning in filter(None, text.split(" | ")):
                rows.append({"where": row.get("metric", row.get("step")), "call": call, "warning": warning})
    if not rows:
        return "none"
    counts = pd.DataFrame(rows).value_counts(sort=False).reset_index(name="times")
    return table(counts)


def figures(folder):
    return [path.relative_to(OUTPUT).as_posix() for path in sorted((folder / "figures").glob("*.png"))]


def epochs_of(eye, rate):
    # The input series of an eye and, per trial, its epoch window and its recorded light onset and offset
    series, _, _ = load_series(eye, rate)
    time = series.get_time()
    windows = [((time >= onset - PRE_STIMULUS) & (time <= onset + POST_ONSET), onset, offset)
               for onset, offset in series.get_light_stimuli_series().get_timings()]
    return series, windows


def pipeline_a(folder):
    # The section of Pipeline A in the log; its metrics, processed samples and epochs for the correlation and plots
    rate = rate_of(folder)
    lines = heading(f"Pipeline A, PyPlr and PIPRkit, at {rate} Hz", "=")
    lines += ["Options"] + options(folder / "options.csv")
    lines += ["", "Known differences, not corrected"] + [f"- {text}" for text in KNOWN_A]
    blinks, values, epochs, samples, panels = [], [], [], {}, {}
    for eye in EYES:
        data = np.load(folder / f"{eye}.npz")
        blinks.append({"eye": eye, **removed(np.isnan(data["a1_pyplr"]), np.isnan(data["a1_piprkit"]), "PyPlr")})
        for step in ("a2", "a3"):
            largest, r = compare(data[f"{step}_pyplr"], data[f"{step}_piprkit"])
            values.append({"eye": eye, "step": step.upper(), "largest |difference|": largest, "r": r,
                           "samples": data["time"].size})
        series, windows = epochs_of(eye, rate)
        pyplr_samples, piprkit_samples, panels[eye] = [], [], []
        for trial, (window, onset, offset) in enumerate(windows):
            pyplr_time, piprkit_time = data[f"a4_pyplr_time_{trial}"], data[f"a4_piprkit_time_{trial}"]
            x4, y4 = pair(pyplr_time, data[f"a4_pyplr_{trial}"], piprkit_time, data[f"a4_piprkit_{trial}"], rate)
            x5, y5 = pair(pyplr_time, data[f"a5_pyplr_{trial}"], piprkit_time, data[f"a5_piprkit_{trial}"], rate)
            largest4, r4 = compare(x4, y4)
            largest5, r5 = compare(x5, y5)
            epochs.append({"eye": eye, "trial": trial + 1, "samples PyPlr": pyplr_time.size,
                           "samples PIPRkit": piprkit_time.size, "shared samples": x4.size,
                           "A4 largest |difference|": largest4, "A4 r": r4, "A5 largest |difference|": largest5,
                           "A5 r": r5})
            pyplr_samples.append(x4)
            piprkit_samples.append(y4)
            panels[eye].append({
                "time": series.get_time()[window] - offset,
                "input": series.get_size()[window],
                "onset": onset - offset,
                "reference_time": pyplr_time - offset,
                "reference": data[f"a4_pyplr_{trial}"],
                "piprkit_time": piprkit_time - offset,
                "piprkit": data[f"a4_piprkit_{trial}"],
            })
        samples[eye] = (np.concatenate(pyplr_samples), np.concatenate(piprkit_samples))
    lines += ["", "A1: removed samples (blinks)", table(pd.DataFrame(blinks))]
    lines += ["", "A2 and A3: values per sample", table(pd.DataFrame(values))]
    lines += ["", "A4 and A5: per epoch", table(pd.DataFrame(epochs))]
    metrics = pd.read_csv(folder / "metrics.csv")
    metrics["PIPRkit - PyPlr"] = metrics["piprkit"] - metrics["pyplr"]
    columns = ["eye", "trial", "metric", "pyplr", "piprkit", "PIPRkit - PyPlr", "piprkit_default"]
    shown = metrics[columns].rename(
        columns={"pyplr": "PyPlr", "piprkit": "PIPRkit", "piprkit_default": "PIPRkit default"})
    lines += ["", "A6: metrics", table(shown)]
    lines += ["", "Time of PyPlr's minimum after the onset sample (s)", table(pd.read_csv(folder / "context.csv"))]
    lines += ["", "PIPRkit warnings in A5", warnings_table(pd.read_csv(folder / "step_warnings.csv"),
                                                          [("piprkit_warnings", "matched")])]
    lines += ["", "PIPRkit warnings in A6", warnings_table(metrics, [("piprkit_warnings", "matched"),
                                                                     ("default_warnings", "default")])]
    lines += ["", "Skipped"] + [f"- {text}" for text in SKIPPED_A]
    lines += ["", "Figures"] + figures(folder)
    correlation = metrics.rename(columns={"pyplr": "reference"})[["eye", "trial", "metric", "reference", "piprkit"]]
    return lines, correlation, samples, panels


def plot_b(folder, rate):
    (folder / "figures").mkdir(exist_ok=True)
    for eye in EYES:
        series, _, _ = load_series(eye, rate)
        cleaned = np.load(folder / f"piprkit_{eye}.npz")["b3"]
        reference = np.load(folder / f"reference_{eye}.npz")["cleaned"]
        time = series.get_time()
        for trial, (onset, offset) in enumerate(series.get_light_stimuli_series().get_timings()):
            window = (time >= onset - PRE_STIMULUS) & (time <= onset + POST_ONSET)
            epoch_time = time[window] - offset
            data = piprkit.PupilMeasurement(epoch_time, series.get_size()[window])
            data.set_light_stimulus(onset - offset, 0.0)
            ax = data.plot(color="lightgrey", label="input")
            piprkit.PupilMeasurement(epoch_time, cleaned[window]).plot(ax, color="black", label="PIPRkit")
            piprkit.PupilMeasurement(epoch_time, reference[window]).plot(
                ax, color="tab:orange", linestyle="--", label="PupilMetrics")
            data.plot_light_stimulus(ax)
            ax.set_title(f"Pipeline B at {rate} Hz, {eye} eye, trial {trial + 1}", fontsize=9)
            ax.legend(fontsize=7)
            ax.figure.savefig(folder / "figures" / f"{eye}_trial{trial + 1}.png", dpi=120, bbox_inches="tight")
            plt.close(ax.figure)


def pipeline_b(folder):
    # The section of Pipeline B in the log; its measures (own and same series), processed samples and epochs
    rate = rate_of(folder)
    plot_b(folder, rate)
    lines = heading(f"Pipeline B, PupilMetrics and PIPRkit, at {rate} Hz", "=")
    source = pd.read_csv(folder / "reference_source.csv").iloc[0]
    lines += [f"PupilMetrics: {source['pupilmetrics_folder']}, settings {source['settings']}"]
    lines += ["", "Options"] + options(folder / "options.csv")
    lines += ["", "Known differences, not corrected"] + [f"- {text}" for text in KNOWN_B]
    removals, values, metric_rows, samples, panels = [], [], [], {}, {}
    piprkit_metrics = pd.read_csv(folder / "piprkit_metrics.csv")
    for eye in EYES:
        own = np.load(folder / f"piprkit_{eye}.npz")
        reference = np.load(folder / f"reference_{eye}.npz")
        removals.append({"eye": eye, "step": "B1", **removed(np.isnan(reference["clipped"]), np.isnan(own["b1"]),
                                                            "PupilMetrics")})
        removals.append({"eye": eye, "step": "B2", **removed(reference["removed"], np.isnan(own["b2"]),
                                                            "PupilMetrics")})
        largest, r = compare(reference["cleaned"], own["b3"])
        values.append({"eye": eye, "step": "B3", "largest |difference|": largest, "r": r, "samples": own["time"].size})
        for index, measure in enumerate(MEASURES):
            rows = piprkit_metrics[(piprkit_metrics["eye"] == eye) & (piprkit_metrics["metric"] == measure)]
            for trial, (_, row) in enumerate(rows.iterrows()):
                pupilmetrics = reference["measures"][trial, index]
                on_piprkit = reference["measures_on_piprkit"][trial, index]
                metric_rows.append({
                    "eye": eye,
                    "trial": trial + 1,
                    "measure": measure,
                    "PupilMetrics": pupilmetrics,
                    "piprkit": row["piprkit"],
                    "PIPRkit - PupilMetrics": row["piprkit"] - pupilmetrics,
                    "PupilMetrics on PIPRkit's series": on_piprkit,
                    "PIPRkit - that": row["piprkit"] - on_piprkit,
                    "piprkit_default": row["piprkit_default"],
                })
        series, windows = epochs_of(eye, rate)
        in_epochs = np.any([window for window, _, _ in windows], axis=0)
        samples[eye] = (reference["cleaned"][in_epochs], own["b3"][in_epochs])
        panels[eye] = [{
            "time": series.get_time()[window] - offset,
            "input": series.get_size()[window],
            "onset": onset - offset,
            "reference_time": series.get_time()[window] - offset,
            "reference": reference["cleaned"][window],
            "piprkit_time": series.get_time()[window] - offset,
            "piprkit": own["b3"][window],
        } for window, onset, offset in windows]
    lines += ["", "B1 and B2: removed samples (B2 includes B1)", table(pd.DataFrame(removals))]
    lines += ["", "B3: values per sample", table(pd.DataFrame(values))]
    metrics = pd.DataFrame(metric_rows).sort_values(["eye", "trial"], kind="stable")
    lines += ["", "B4 and B5: measures",
              table(metrics.rename(columns={"piprkit": "PIPRkit", "piprkit_default": "PIPRkit default"}))]
    lines += ["", "PIPRkit warnings in B1 and B2",
              warnings_table(pd.read_csv(folder / "piprkit_step_warnings.csv"), [("piprkit_warnings", "matched")])]
    lines += ["", "PIPRkit warnings in B4 and B5",
              warnings_table(piprkit_metrics, [("piprkit_warnings", "matched"), ("default_warnings", "default")])]
    lines += ["", "Skipped"] + [f"- {text}" for text in SKIPPED_B]
    lines += ["", "Figures"] + figures(folder)
    keep = ["eye", "trial", "metric", "reference", "piprkit"]
    own_values = metrics.rename(columns={"measure": "metric", "PupilMetrics": "reference"})[keep]
    same_values = metrics.rename(columns={"measure": "metric", "PupilMetrics on PIPRkit's series": "reference"})[keep]
    return lines, own_values, same_values, samples, panels


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=7)


def identity(ax, values):
    # The same range on both axes and the dashed line y = x of perfect correlation, labelled at its end. The margin
    # keeps the end of the line, where the label goes, free of data.
    finite = values[np.isfinite(values)]
    low, high = finite.min(), finite.max()
    margin = 0.14 * (high - low) if high > low else 0.14 * max(abs(high), 1.0)
    low, high = low - margin, high + margin
    ax.plot([low, high], [low, high], color=SECONDARY, linestyle="--", linewidth=1.2, zorder=2.5)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    end = low + 0.975 * (high - low)
    ax.annotate("y = x", (end, end), xytext=(-2, 3), textcoords="offset points", rotation=45, rotation_mode="anchor",
                ha="right", va="bottom", color=SECONDARY, fontsize=7)


def stats_title(ax, x, y):
    # r and CCC on the right of the panel's title line, clear of the data
    result = agreement(x, y)
    r = "r undefined" if np.isnan(result["r"]) else f"r = {result['r']:.3f}"
    ax.set_title(f"{r}, CCC = {result['CCC']:.3f}", loc="right", fontsize=7.5, color=SECONDARY)


def finish(fig, title, handles, path):
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path.relative_to(OUTPUT).as_posix()


def identity_handle():
    return Line2D([], [], color=SECONDARY, linestyle="--", linewidth=1.2, label="Perfect correlation (y = x)")


def plot_metrics(path, title, reference_name, values):
    # One square panel per metric: the reference's value against PIPRkit's, per eye and trial
    metrics = list(dict.fromkeys(values["metric"]))
    columns = 3 if len(metrics) > 4 else len(metrics)
    rows = -(-len(metrics) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.3 * rows + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric in zip(axes.flat, metrics):
        subset = values[values["metric"] == metric]
        style(ax)
        identity(ax, subset[["reference", "piprkit"]].to_numpy(dtype=np.float64).ravel())
        for eye in EYES:
            colour, marker = EYE_STYLES[eye]
            points = subset[subset["eye"] == eye]
            ax.scatter(points["reference"], points["piprkit"], s=46, color=colour, marker=marker, edgecolors=SURFACE,
                       linewidths=1.2, zorder=3)
        stats_title(ax, subset["reference"].to_numpy(), subset["piprkit"].to_numpy())
        unit = UNITS[metric]
        ax.set_title(f"{metric} ({unit})", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"{reference_name} ({unit})", fontsize=8, color=SECONDARY)
        ax.set_ylabel(f"PIPRkit ({unit})", fontsize=8, color=SECONDARY)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=EYE_STYLES[eye][0], marker=EYE_STYLES[eye][1], linestyle="", markersize=7,
                      markeredgecolor=SURFACE, label=f"{eye.capitalize()} eye") for eye in EYES]
    return finish(fig, title, handles + [identity_handle()], path)


def plot_samples(path, title, reference_name, samples):
    # One square panel per eye: every processed sample of the epochs, the reference's value against PIPRkit's
    fig, axes = plt.subplots(1, len(samples), figsize=(3.6 * len(samples), 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (eye, (x, y)) in zip(axes.flat, samples.items()):
        style(ax)
        identity(ax, np.concatenate([x, y]))
        ax.scatter(x, y, s=2, color=EYE_STYLES[eye][0], alpha=0.15, linewidths=0, rasterized=True, zorder=2)
        stats_title(ax, x, y)
        ax.set_title(f"{eye.capitalize()} eye", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"{reference_name} (mm)", fontsize=8, color=SECONDARY)
        ax.set_ylabel("PIPRkit (mm)", fontsize=8, color=SECONDARY)
    return finish(fig, title, [identity_handle()], path)


def plot_traces(path, title, reference_name, panels):
    # One panel per eye and trial: the input and both tools' processed epoch, drawn with PIPRkit's plot methods
    trials = len(panels[EYES[0]])
    fig, axes = plt.subplots(len(EYES), trials, figsize=(2.5 * trials, 2.6 * len(EYES) + 0.6), sharex=True,
                             sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, eye in enumerate(EYES):
        for column, panel in enumerate(panels[eye]):
            ax = axes[row, column]
            style(ax)
            data = piprkit.PupilMeasurement(panel["time"], panel["input"])
            data.set_light_stimulus(panel["onset"], 0.0)
            data.plot(ax, color=AXIS, linewidth=0.8, label="input")
            piprkit.PupilMeasurement(panel["piprkit_time"], panel["piprkit"]).plot(
                ax, color=INK, linewidth=0.8, label="PIPRkit")
            piprkit.PupilMeasurement(panel["reference_time"], panel["reference"]).plot(
                ax, color=REFERENCE_COLOUR, linestyle="--", linewidth=0.9, label=reference_name)
            data.plot_light_stimulus(ax)
            if row == 0:
                ax.set_title(f"Trial {column + 1}", fontsize=9, color=INK, loc="left")
            if column == 0:
                ax.set_ylabel(f"{eye.capitalize()} eye\npupil size (mm)", fontsize=8, color=SECONDARY)
            if row == len(EYES) - 1:
                ax.set_xlabel("time from light offset (s)", fontsize=8, color=SECONDARY)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    return finish(fig, title, handles, path)


def correlation(comparisons, sample_comparisons, plots):
    lines = heading("Correlation", "=") + list(DEFINITIONS)
    rows = []
    for name, values in comparisons:
        for metric in dict.fromkeys(values["metric"]):
            subset = values[values["metric"] == metric]
            rows.append({"comparison": name, "metric": metric,
                         **agreement(subset["reference"].to_numpy(), subset["piprkit"].to_numpy())})
    lines += ["", "Metrics (6 trials of each eye)", table(pd.DataFrame(rows))]
    rows = []
    for name, samples in sample_comparisons:
        for eye, (x, y) in samples.items():
            result = agreement(x, y)
            rows.append({"comparison": name, "eye": eye, "n": result["n"], "r": result["r"], "CCC": result["CCC"],
                         "mean difference": result["mean difference"]})
    lines += ["", "Samples after processing (the epochs, sample by sample)", table(pd.DataFrame(rows))]
    lines += ["", "Plots"] + plots
    return lines


def main():
    lines = ["PIPRkit validation against PyPlr and PupilMetrics",
             "Generated by validation/report.py from the files in validation/output/."]
    a_folders = sorted(OUTPUT.glob("pyplr_*hz"))
    b_folders = sorted(OUTPUT.glob("pupilmetrics_*hz"))
    if a_folders:
        lines += heading("Versions", "=") + [table(pd.read_csv(a_folders[0] / "versions.csv"))]
    comparisons, sample_comparisons, plots = [], [], []
    for folder in a_folders:
        section, values, samples, panels = pipeline_a(folder)
        lines += section
        name, rate = f"A {rate_of(folder)} Hz: PyPlr", rate_of(folder)
        comparisons.append((name, values))
        sample_comparisons.append((name, samples))
        plots.append(plot_metrics(OUTPUT / f"correlation_{folder.name}.png",
                                  f"Pipeline A at {rate} Hz: metrics", "PyPlr", values))
        plots.append(plot_traces(OUTPUT / f"traces_{folder.name}.png",
                                 f"Pipeline A at {rate} Hz: epochs after processing", "PyPlr", panels))
        plots.append(plot_samples(OUTPUT / f"samples_{folder.name}.png",
                                  f"Pipeline A at {rate} Hz: samples after processing", "PyPlr",
                                  samples))
    for folder in b_folders:
        section, own_values, same_values, samples, panels = pipeline_b(folder)
        lines += section
        rate = rate_of(folder)
        comparisons.append((f"B {rate} Hz: PupilMetrics", own_values))
        comparisons.append((f"B {rate} Hz: same series", same_values))
        sample_comparisons.append((f"B {rate} Hz: PupilMetrics", samples))
        plots.append(plot_metrics(OUTPUT / f"correlation_{folder.name}.png",
                                  f"Pipeline B at {rate} Hz: measures", "PupilMetrics",
                                  own_values))
        plots.append(plot_traces(OUTPUT / f"traces_{folder.name}.png",
                                 f"Pipeline B at {rate} Hz: epochs after processing", "PupilMetrics", panels))
        plots.append(plot_samples(OUTPUT / f"samples_{folder.name}.png",
                                  f"Pipeline B at {rate} Hz: samples after processing",
                                  "PupilMetrics", samples))
    lines += correlation(comparisons, sample_comparisons, plots)
    (OUTPUT / "report.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
