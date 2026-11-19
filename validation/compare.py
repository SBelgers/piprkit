# Compares the three tools' outputs of the pipeline: per pair of tools, the statistics of the metrics and of the
# processed samples, with plots of both and of the traces. Run it after prepare.py and the three pipeline scripts.
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import piprkit

INTERMEDIATE = Path(__file__).parent / "output" / "intermediate"
FINAL = Path(__file__).parent / "output" / "final"
EYES = ("left",)
# The window around each stimulus that is compared and shown, as PIPRkit's epochs: 10 s before light onset to 45 s
# after light offset
PRE_STIMULUS = 10.0
POST_STIMULUS = 45.0
TOOLS = {"piprkit": "PIPRkit", "pupilmetrics": "PupilMetrics", "pyplr": "PyPlr"}
# Each pair as (x, y); the mean difference is y - x
PAIRS = (("piprkit", "pupilmetrics"), ("piprkit", "pyplr"), ("pupilmetrics", "pyplr"))
# The metrics in plot order, with their unit after the conversions in load_metrics
UNITS = {"baseline": "mm", "peak constriction": "% change", "time to peak": "s", "6 s PIPR": "% change"}
# The colours of the trials in trials.svg and of their light stimulus in the traces: blue for trials 1, 3, 5 and red
# for 2, 4, 6
TRIAL_COLOURS = (("#2563c9", (1, 3, 5)), ("#d1342f", (2, 4, 6)))
# Plot colours on the light surface: the tools, and the ink, axis, grid and surface tones
TOOL_COLOURS = {"piprkit": "#0b0b0b", "pupilmetrics": "#eb6834", "pyplr": "#1baf7a"}
TOOL_LINESTYLES = {"piprkit": "-", "pupilmetrics": "-", "pyplr": "--"}
INK, SECONDARY, AXIS, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9", "#fcfcfb"
# The figures are for a paper: full page width (174 mm) and all text at 10 pt, saved at that size, so the text is
# printed at 10 pt without scaling
WIDTH = 174 / 25.4
FONT_SIZE = 10


def mean_ci(values):
    # The mean and its 95% confidence interval (t distribution)
    mean = float(np.mean(values))
    low, high = stats.t.interval(0.95, values.size - 1, loc=mean, scale=stats.sem(values))
    return mean, float(low), float(high)


def agreement(x, y):
    # Over the pairs that have both values: n; the mean of x and of y with their 95% CIs; the mean difference (y - x)
    # with its 95% CI and the p of a paired t-test; and Pearson's r with its 95% CI and p, and r²
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    x, y = x[both], y[both]
    x_mean = x_low = x_high = y_mean = y_low = y_high = np.nan
    difference = difference_low = difference_high = t = t_p = r = r_p = r_low = r_high = np.nan
    if x.size >= 2:
        x_mean, x_low, x_high = mean_ci(x)
        y_mean, y_low, y_high = mean_ci(y)
        difference = float(np.mean(y - x))
        test = stats.ttest_rel(y, x)
        t, t_p = float(test.statistic), float(test.pvalue)
        difference_low, difference_high = test.confidence_interval(confidence_level=0.95)
    if x.size >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        result = stats.pearsonr(x, y)
        r, r_p = float(result.statistic), float(result.pvalue)
        r_low, r_high = result.confidence_interval(confidence_level=0.95)
    return {"n": int(x.size), "x_mean": x_mean, "x_ci95_low": x_low, "x_ci95_high": x_high, "y_mean": y_mean,
            "y_ci95_low": y_low, "y_ci95_high": y_high, "mean_difference": difference,
            "difference_ci95_low": float(difference_low), "difference_ci95_high": float(difference_high), "t": t,
            "t_p": t_p, "r": r, "r_ci95_low": float(r_low), "r_ci95_high": float(r_high), "r_p": r_p, "r2": r**2}


def load_metrics(tool):
    # One tool's metrics in the units of UNITS: PupilMetrics' % constriction becomes % change by a change of sign
    metrics = pd.read_csv(FINAL / f"{tool}_metrics.csv")
    flip = (metrics["unit"] == "% constriction").to_numpy()
    return metrics.assign(value=np.where(flip, -metrics["value"], metrics["value"]))


def metric_pairs(x_tool, y_tool):
    # Per eye, trial and metric that both tools have: the x tool's value and the y tool's
    x = load_metrics(x_tool)[["eye", "trial", "metric", "value"]].rename(columns={"value": "x"})
    y = load_metrics(y_tool)[["eye", "trial", "metric", "value"]].rename(columns={"value": "y"})
    return x.merge(y, on=["eye", "trial", "metric"])


def trial_metrics(metrics, eye, trial):
    # One tool's metrics of one trial, by name
    rows = metrics[(metrics["eye"] == eye) & (metrics["trial"] == trial)]
    return dict(zip(rows["metric"], rows["value"]))


def windows(time, stimuli):
    # Per trial, the samples from PRE_STIMULUS before light onset to POST_STIMULUS after light offset
    return [(time >= onset - PRE_STIMULUS) & (time <= offset + POST_STIMULUS) for onset, offset in stimuli]


def points(tool, metrics, duration):
    # Where a tool's metrics sit in time from light onset and their values on the tool's trace: the baseline and the
    # 6 s PIPR in the middle of their windows, the peak at the time to peak
    if tool == "piprkit":
        return [("baseline", "s", -2.5, 0.0),
                ("peak", "o", metrics["time to peak"], metrics["peak constriction"]),
                ("6 s PIPR", "D", duration + 6.0, metrics["6 s PIPR"])]
    if tool == "pyplr":
        return [("baseline", "s", -2.5, 0.0),
                ("peak", "o", metrics["time to peak"], metrics["peak constriction"])]
    # PupilMetrics' trace is in mm, so its points are too
    baseline = metrics["baseline"]
    return [("baseline", "s", -0.125, baseline),
            ("peak", "o", metrics["time to peak"], baseline * (1 + metrics["peak constriction"] / 100)),
            ("6 s PIPR", "D", 7.05, baseline * (1 + metrics["6 s PIPR"] / 100))]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=FONT_SIZE)


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
    # A square panel, so the equal ranges have equal scales; this also holds for a twin axis
    ax.set_box_aspect(1)
    end = low + 0.975 * (high - low)
    ax.annotate("y = x", (end, end), xytext=(-2, 3), textcoords="offset points", rotation=45, rotation_mode="anchor",
                ha="right", va="bottom", color=SECONDARY, fontsize=FONT_SIZE)


def r_text(x, y):
    r = agreement(x, y)["r"]
    return "r undefined" if np.isnan(r) else f"r = {r:.3f}"


def twin_axis(ax):
    # A y axis on the right of the panel, styled like the left one
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(AXIS)
    for side in ("top", "left", "bottom"):
        twin.spines[side].set_visible(False)
    twin.tick_params(colors=SECONDARY, labelsize=FONT_SIZE)
    return twin


def same_range_twin(ax):
    # A twin axis with the range and square box of the left axis, so y = x holds on both
    twin = twin_axis(ax)
    twin.set_box_aspect(1)
    twin.set_yticks(ax.get_yticks())
    twin.set_ylim(ax.get_ylim())
    return twin


def tool_handles():
    return [Line2D([], [], color=TOOL_COLOURS["pyplr"], marker="o", linestyle="", markersize=7,
                   markeredgecolor=SURFACE, label="PyPlr (left axis)"),
            Line2D([], [], color=TOOL_COLOURS["pupilmetrics"], marker="s", linestyle="", markersize=7,
                   markeredgecolor=SURFACE, label="PupilMetrics (right axis)")]


def finish(fig, title, handles, path):
    # The title on top and the legend below the panels, at most four entries per row. The figure keeps its size, so
    # the text keeps its size; the dpi is for the rasterized points of samples.svg.
    fig.suptitle(title, x=0.01, ha="left", fontsize=FONT_SIZE, color=INK)
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), frameon=False,
               fontsize=FONT_SIZE)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def identity_handle():
    return Line2D([], [], color=SECONDARY, linestyle="--", linewidth=1.2, label="Perfect correlation (y = x)")


def plot_metrics(path, title, values):
    # A panel per metric, per trial: PIPRkit's value on the x axis against PyPlr's on the left axis and PupilMetrics'
    # on a twin axis on the right. Both y axes have the range of the x axis, so y = x holds for both tools.
    columns = 2
    rows = -(-len(UNITS) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(WIDTH, 3.5 * rows + 0.8), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax, (metric, unit) in zip(axes.flat, UNITS.items()):
        pyplr = values[("piprkit", "pyplr")]
        pyplr = pyplr[pyplr["metric"] == metric]
        pupilmetrics = values[("piprkit", "pupilmetrics")]
        pupilmetrics = pupilmetrics[pupilmetrics["metric"] == metric]
        style(ax)
        identity(ax, pd.concat([pyplr, pupilmetrics])[["x", "y"]].to_numpy(dtype=np.float64).ravel())
        ax.locator_params(nbins=5)
        lines = [f"{metric} ({unit})"]
        if not pyplr.empty:
            ax.scatter(pyplr["x"], pyplr["y"], s=46, marker="o", color=TOOL_COLOURS["pyplr"], edgecolors=SURFACE,
                       linewidths=1.2, zorder=3)
            ax.set_ylabel(f"PyPlr ({unit})", fontsize=FONT_SIZE, color=SECONDARY)
            lines.append(f"PyPlr: {r_text(pyplr['x'], pyplr['y'])}")
        if not pupilmetrics.empty:
            twin = same_range_twin(ax)
            twin.scatter(pupilmetrics["x"], pupilmetrics["y"], s=46, marker="s", color=TOOL_COLOURS["pupilmetrics"],
                         edgecolors=SURFACE, linewidths=1.2, zorder=3)
            twin.set_ylabel(f"PupilMetrics ({unit})", fontsize=FONT_SIZE, color=SECONDARY)
            lines.append(f"PupilMetrics: {r_text(pupilmetrics['x'], pupilmetrics['y'])}")
        ax.set_title("\n".join(lines), loc="left", fontsize=FONT_SIZE, color=INK)
        ax.set_xlabel(f"PIPRkit ({unit})", fontsize=FONT_SIZE, color=SECONDARY)
    for ax in axes.flat[len(UNITS):]:
        ax.set_visible(False)
    finish(fig, title, tool_handles() + [identity_handle()], path)


def trials_text(trials):
    return "trials " + ", ".join(str(trial) for trial in trials)


def plot_trials(path, title, metrics, eye):
    # A panel per metric, in its own unit, with a column per tool: each trial's value as a dot and a line joining one
    # trial's values across the tools, both in the trial's colour. A tool without the metric keeps its column, empty.
    rows = -(-len(UNITS) // 2)
    fig, axes = plt.subplots(rows, 2, figsize=(WIDTH, 3.0 * rows + 0.8), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    columns = np.arange(len(TOOLS))
    for ax, (metric, unit) in zip(axes.flat, UNITS.items()):
        style(ax)
        for colour, trials in TRIAL_COLOURS:
            for trial in trials:
                values = np.array([trial_metrics(metrics[tool], eye, trial).get(metric, np.nan) for tool in TOOLS])
                present = np.isfinite(values)
                ax.plot(columns[present], values[present], color=colour, linewidth=1.0, alpha=0.6, zorder=2)
                ax.scatter(columns[present], values[present], s=46, color=colour, edgecolors=SURFACE,
                           linewidths=1.2, zorder=3)
        ax.set_xticks(columns, list(TOOLS.values()), fontsize=FONT_SIZE, color=SECONDARY)
        ax.set_xlim(-0.4, len(TOOLS) - 0.6)
        ax.grid(axis="x", visible=False)
        # The y axis runs from 0, with a line at 0, and leaves 5% of its range free beyond the data
        low, high = ax.get_ylim()
        low, high = min(low, 0.0), max(high, 0.0)
        pad = 0.05 * (high - low)
        ax.set_ylim(low - pad if low < 0 else low, high + pad if high > 0 else high)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.locator_params(axis="y", nbins=6)
        ax.set_title(f"{metric} ({unit})", loc="left", fontsize=FONT_SIZE, color=INK)
        ax.set_ylabel(unit, fontsize=FONT_SIZE, color=SECONDARY)
    handles = [Line2D([], [], color=colour, marker="o", markersize=7, markeredgecolor=SURFACE, linewidth=1.0,
                      label=trials_text(trials).capitalize()) for colour, trials in TRIAL_COLOURS]
    finish(fig, title, handles, path)


def plot_samples(path, title, samples, piprkit_baselines):
    # A panel per eye: every sample of the epochs, PIPRkit's (% change) on the x axis against PyPlr's (% change) on the
    # left axis, which has the range of the x axis so y = x holds, and PupilMetrics' (mm, it has no baseline
    # correction) on a twin axis on the right. The mm axis is scaled with PIPRkit's mean baseline,
    # mm = baseline (1 + % / 100); its ticks are true mm.
    fig, axes = plt.subplots(1, len(EYES), figsize=(WIDTH, 6.4), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax, eye in zip(axes.flat, EYES):
        piprkit_x, pyplr = samples[eye][("piprkit", "pyplr")]
        piprkit_x_pm, pupilmetrics = samples[eye][("piprkit", "pupilmetrics")]
        style(ax)
        identity(ax, np.concatenate([piprkit_x, pyplr]))
        ax.locator_params(nbins=5)
        ax.scatter(piprkit_x, pyplr, s=2, color=TOOL_COLOURS["pyplr"], alpha=0.15, linewidths=0, rasterized=True,
                   zorder=2)
        twin = twin_axis(ax)
        twin.set_box_aspect(1)
        low, high = ax.get_ylim()
        twin.set_ylim(piprkit_baselines[eye] * (1 + low / 100), piprkit_baselines[eye] * (1 + high / 100))
        twin.scatter(piprkit_x_pm, pupilmetrics, s=2, color=TOOL_COLOURS["pupilmetrics"], alpha=0.15, linewidths=0,
                     rasterized=True, zorder=2)
        lines = [f"{eye.capitalize()} eye",
                 f"PyPlr: {r_text(piprkit_x, pyplr)}",
                 f"PupilMetrics: {r_text(piprkit_x_pm, pupilmetrics)}",
                 f"PyPlr vs PupilMetrics: {r_text(*samples[eye][('pupilmetrics', 'pyplr')])}"]
        ax.set_title("\n".join(lines), loc="left", fontsize=FONT_SIZE, color=INK)
        ax.set_xlabel("PIPRkit (% change)", fontsize=FONT_SIZE, color=SECONDARY)
        ax.set_ylabel("PyPlr (% change)", fontsize=FONT_SIZE, color=SECONDARY)
        twin.set_ylabel("PupilMetrics (mm)", fontsize=FONT_SIZE, color=SECONDARY)
    finish(fig, title, tool_handles() + [identity_handle()], path)


def plot_points(ax, tool, tool_points, onset):
    # A tool's metric points; their times count from light onset, the panel's from light offset
    for _, marker, at, value in tool_points:
        ax.scatter(onset + at, value, marker=marker, s=60, color=TOOL_COLOURS[tool], edgecolors=SURFACE,
                   linewidths=1.2, zorder=4)


def draw_traces(ax, panel):
    # The three tools' traces of one trial: PIPRkit's and PyPlr's corrected traces (% change) on the left axis, the
    # raw data and PupilMetrics' trace (mm) on a twin axis on the right, drawn with PIPRkit's plot methods, with each
    # tool's metric points
    style(ax)
    twin = twin_axis(ax)
    piprkit.PupilMeasurement(panel["raw_time"], panel["raw"]).plot(twin, color=AXIS, linewidth=1.0, label="raw data")
    piprkit.PupilMeasurement(panel["pupilmetrics_time"], panel["pupilmetrics"]).plot(
        twin, color=TOOL_COLOURS["pupilmetrics"], linewidth=0.9, label="PupilMetrics")
    own = piprkit.PupilMeasurement(panel["time"], panel["piprkit"])
    own.set_light_stimulus(panel["onset"], 0.0)
    own.plot(ax, color=TOOL_COLOURS["piprkit"], linewidth=0.9, label="PIPRkit")
    piprkit.PupilMeasurement(panel["time"], panel["pyplr"]).plot(
        ax, color=TOOL_COLOURS["pyplr"], linestyle=TOOL_LINESTYLES["pyplr"], linewidth=0.9, label="PyPlr")
    # Its legend entry is made in plot_traces
    own.plot_light_stimulus(ax, color=panel["light_colour"], label="_light stimulus")
    for tool, target in (("piprkit", ax), ("pyplr", ax), ("pupilmetrics", twin)):
        plot_points(target, tool, panel[f"{tool}_points"], panel["onset"])
    # The lines above the twin's raw data
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("PIPRkit, PyPlr\nchange from baseline (%)", fontsize=FONT_SIZE, color=SECONDARY)
    twin.set_ylabel("raw data, PupilMetrics\npupil size (mm)", fontsize=FONT_SIZE, color=SECONDARY)
    ax.set_xlabel("time from light offset (s)", fontsize=FONT_SIZE, color=SECONDARY)
    return twin


def plot_traces(path, title, panels):
    # A row per trial with its full epoch, the rows sharing their axes; the legend tells the tools' lines and the
    # metrics' markers apart
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, 3.0 * len(panels) + 1.0), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    axes = axes[:, 0]
    twins = [draw_traces(ax, panel) for ax, panel in zip(axes, panels)]
    for ax, twin, panel in zip(axes, twins, panels):
        # Once all rows are drawn: the mm axis is scaled so that PIPRkit's baseline in mm sits at 0% on the left axis,
        # mm = baseline (1 + % / 100); the traces themselves are not changed
        low, high = ax.get_ylim()
        twin.set_ylim(panel["piprkit_baseline"] * (1 + low / 100), panel["piprkit_baseline"] * (1 + high / 100))
        if len(panels) > 1:
            ax.set_title(f"Trial {panel['trial']}", fontsize=FONT_SIZE, color=INK, loc="left")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    # The legend: the light stimulus of each row first, so their long labels share the first column, then the tools'
    # lines and the metrics' markers
    lights = [Patch(color=panel["light_colour"], alpha=0.3, label=panel["light_label"]) for panel in panels]
    labelled = {}
    for target in (axes[0], twins[0]):
        handles, labels = target.get_legend_handles_labels()
        labelled.update(zip(labels, handles))
    shapes = [Line2D([], [], color=SECONDARY, marker=marker, linestyle="", markersize=7, markeredgecolor=SURFACE,
                     label=label) for label, marker, _, _ in panels[0]["piprkit_points"]]
    finish(fig, title, lights + list(labelled.values()) + shapes, path)


def main():
    recording = pd.read_csv(INTERMEDIATE / "input.csv")
    stimuli = pd.read_csv(INTERMEDIATE / "stimuli.csv").to_numpy()
    time = recording["time"].to_numpy()
    in_epochs = np.any(windows(time, stimuli), axis=0)
    series = {tool: pd.read_csv(INTERMEDIATE / f"{tool}_series.csv") for tool in TOOLS}
    corrected = {tool: pd.read_csv(INTERMEDIATE / f"{tool}_corrected.csv") for tool in ("piprkit", "pyplr")}
    grid = series["pupilmetrics"]["time"].to_numpy()
    metrics = {tool: load_metrics(tool) for tool in TOOLS}

    # Metrics, per pair of tools
    values = {pair: metric_pairs(*pair) for pair in PAIRS}
    metric_rows = []
    for (x_tool, y_tool), pair_values in values.items():
        for metric in UNITS:
            subset = pair_values[pair_values["metric"] == metric]
            if not subset.empty:
                metric_rows.append({"x_tool": x_tool, "y_tool": y_tool, "metric": metric,
                                    **agreement(subset["x"], subset["y"])})
    plot_metrics(FINAL / "correlation.svg", "Metrics per trial", values)

    plot_trials(FINAL / "trials.svg", f"{EYES[0].capitalize()} eye, metrics per trial and tool", metrics, EYES[0])

    # Samples of the epochs after baseline correction: PIPRkit's and PyPlr's in % change. PupilMetrics has no
    # baseline correction, so its series stays in mm; it is on the 67 Hz grid, so it is interpolated onto the raw
    # sample times.
    samples, sample_rows, piprkit_baselines = {}, [], {}
    for eye in EYES:
        on_raw = {"piprkit": corrected["piprkit"][eye].to_numpy(), "pyplr": corrected["pyplr"][eye].to_numpy(),
                  "pupilmetrics": np.interp(time, grid, series["pupilmetrics"][eye].to_numpy())}
        samples[eye] = {(x_tool, y_tool): (on_raw[x_tool][in_epochs], on_raw[y_tool][in_epochs])
                        for x_tool, y_tool in PAIRS}
        for (x_tool, y_tool), (x, y) in samples[eye].items():
            result = agreement(x, y)
            # The mean difference needs one unit, which pairs with PupilMetrics (mm) don't have
            one_unit = "pupilmetrics" not in (x_tool, y_tool)
            sample_rows.append({"x_tool": x_tool, "y_tool": y_tool, "eye": eye, "n": result["n"], "r": result["r"],
                                "mean_difference": result["mean_difference"] if one_unit else np.nan})
        piprkit = metrics["piprkit"]
        piprkit_baselines[eye] = piprkit[(piprkit["eye"] == eye) & (piprkit["metric"] == "baseline")]["value"].mean()
    plot_samples(FINAL / "samples.svg", "Samples after baseline correction (PupilMetrics has none, in mm)", samples,
                 piprkit_baselines)

    # Traces per trial, of the first eye: each trial in an SVG of its own, and trial 6 above trial 5
    eye = EYES[0]
    panels = {}
    for trial, ((onset, offset), window, grid_window) in enumerate(
            zip(stimuli, windows(time, stimuli), windows(grid, stimuli)), start=1):
        panel = {
            "trial": trial,
            "onset": onset - offset,
            "time": time[window] - offset,
            "raw_time": time[window] - offset,
            "raw": recording[eye].to_numpy()[window],
            "piprkit": corrected["piprkit"][eye].to_numpy()[window],
            "pyplr": corrected["pyplr"][eye].to_numpy()[window],
            "pupilmetrics_time": grid[grid_window] - offset,
            "pupilmetrics": series["pupilmetrics"][eye].to_numpy()[grid_window],
        }
        for tool in TOOLS:
            panel[f"{tool}_points"] = points(tool, trial_metrics(metrics[tool], eye, trial), offset - onset)
        panel["piprkit_baseline"] = trial_metrics(metrics["piprkit"], eye, trial)["baseline"]
        for colour, trials in TRIAL_COLOURS:
            if trial in trials:
                panel["light_colour"] = colour
                panel["light_label"] = f"light stimulus, {trials_text(trials)}"
        panels[trial] = panel
        plot_traces(FINAL / f"traces_trial{trial}.svg", f"{eye.capitalize()} eye, trial {trial}, after processing",
                    [panel])
    plot_traces(FINAL / "traces_trials5_6.svg", f"{eye.capitalize()} eye, trials 5 and 6, after processing",
                [panels[6], panels[5]])

    pd.DataFrame(metric_rows).to_csv(FINAL / "statistics_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(FINAL / "statistics_samples.csv", index=False)


if __name__ == "__main__":
    main()
