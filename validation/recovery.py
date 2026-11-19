# Parameter recovery of PIPRkit's PhaseModel, part 1: the known values. The Fityk fits of each trial (fityk/fits.csv)
# give the arguments of a simulation that follows those fits. The figures show the real trace, PIPRkit's fit of it and
# the Fityk fits, and compare the Fityk values with PIPRkit's. Run it after pipeline_piprkit.py and the Fityk fits.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq

import piprkit
from compare import (AXIS, FONT_SIZE, INK, SECONDARY, SURFACE, TRIAL_COLOURS, WIDTH, agreement, finish, identity,
                     identity_handle, r_text, style, trials_text)
from pipeline_piprkit import BASELINE, POST_STIMULUS, PRE_STIMULUS, metrics

OUTPUT = Path(__file__).parent / "output"
# The fits made by hand in Fityk: per trial and phase the fitted stretch, the function and its values. A line is
# m * x + c and the redilation s * exp(k * (x - x0)) + p, with x in seconds from light offset. A phase with NaN values
# was not fitted.
FITS = Path(__file__).parent / "fityk" / "fits.csv"
EYE = "left"
# The values compared between the known simulation and PIPRkit's fits, named as the simulation arguments
UNITS = {"constriction_delay": "s", "time_to_peak": "s", "redilation_delay": "s", "baseline_c": "mm",
         "constriction_m": "mm/s", "constriction_c": "mm", "maintained_constriction_m": "mm/s",
         "maintained_constriction_c": "mm", "redilation_S": "mm", "redilation_k": "1/s", "redilation_P": "mm"}


def read_fits():
    # Per trial, the fitted values by phase and parameter, and the stretch each phase was fitted on
    rows = pd.read_csv(FITS, dtype=str, keep_default_na=False)
    fits = {}
    for row in rows.itertuples():
        value = float(row.value)
        if not np.isnan(value):
            values, stretches = fits.setdefault(int(row.trial), ({}, {}))
            values[(row.phase.strip(), row.parameter.strip())] = value
            stretches[row.phase.strip()] = (float(row.start), float(row.end))
    return fits


def fityk_functions(values):
    # The functions fitted in Fityk, by phase
    functions = {
        "baseline": lambda x: np.full(np.shape(x), values[("baseline", "c")]),
        "constriction": lambda x: values[("constriction", "m")] * x + values[("constriction", "c")],
        "redilation": lambda x: values[("redilation", "p")]
                                + values[("redilation", "s")]
                                * np.exp(values[("redilation", "k")] * (x - values[("redilation", "x0")])),
    }
    if ("maintained_constriction", "m") in values:
        functions["maintained_constriction"] = lambda x: (values[("maintained_constriction", "m")] * x
                                                          + values[("maintained_constriction", "c")])
    return functions


def crossing(line, curve, start, time):
    # The first time after start at which the curve reaches the line, refined between the two samples around it
    after = time[time > start]
    reached = np.flatnonzero(curve(after) >= line(after))
    if reached.size == 0:
        raise ValueError("The redilation curve never reaches the line before it.")
    low = start if reached[0] == 0 else after[reached[0] - 1]
    return brentq(lambda x: line(x) - curve(x), low, after[reached[0]])


def simulation_args(values, epoch):
    # The arguments of simulate_pupil_measurement from one trial's fits. Each phase starts where its fit meets the fit
    # of the phase before, so the simulation follows the fits: its constriction slope is the fitted m, and its S gives
    # the fitted redilation curve
    onset, offset = epoch.get_light_stimulus().get_timing()
    time = epoch.get_time()
    functions = fityk_functions(values)
    baseline_c = values[("baseline", "c")]
    m, c = values[("constriction", "m")], values[("constriction", "c")]
    constriction_start = (baseline_c - c) / m
    if "maintained_constriction" in functions:
        maintained_m, maintained_c = values[("maintained_constriction", "m")], values[("maintained_constriction", "c")]
        peak = (maintained_c - c) / (m - maintained_m)
        redilation_start = crossing(functions["maintained_constriction"], functions["redilation"], peak, time)
    else:
        # Without a maintained constriction the redilation starts at the peak
        maintained_m = 0.0
        peak = redilation_start = crossing(functions["constriction"], functions["redilation"], constriction_start,
                                           time)
    args = {"time_data": time, "t_light_onset": onset, "t_light_offset": offset,
            "constriction_delay": constriction_start - onset, "time_to_peak": peak - onset,
            "redilation_delay": redilation_start - offset, "baseline_c": baseline_c,
            "maintained_constriction_c": functions["constriction"](peak), "maintained_constriction_m": maintained_m,
            "redilation_k": values[("redilation", "k")], "redilation_P": values[("redilation", "p")]}
    # When the peak and the redilation start are the same time, rounding must not put the peak after it
    while onset + args["time_to_peak"] > offset + args["redilation_delay"]:
        args["time_to_peak"] = np.nextafter(args["time_to_peak"], -np.inf)
    return args


def known_values(values, args):
    # The values PIPRkit's fit should find in the simulation. The constriction slope and S follow from the arguments,
    # as in simulate_pupil_measurement. Without a maintained constriction its m and c are not defined.
    maintained = ("maintained_constriction", "m") in values
    peak = args["t_light_onset"] + args["time_to_peak"]
    redilation_start = args["t_light_offset"] + args["redilation_delay"]
    return {
        "constriction_delay": args["constriction_delay"],
        "time_to_peak": args["time_to_peak"],
        "redilation_delay": args["redilation_delay"],
        "baseline_c": args["baseline_c"],
        "constriction_m": (args["maintained_constriction_c"] - args["baseline_c"])
                          / (args["time_to_peak"] - args["constriction_delay"]),
        "constriction_c": args["baseline_c"],
        "maintained_constriction_m": args["maintained_constriction_m"] if maintained else np.nan,
        "maintained_constriction_c": args["maintained_constriction_c"] if maintained else np.nan,
        "redilation_S": args["maintained_constriction_c"]
                        + args["maintained_constriction_m"] * (redilation_start - peak) - args["redilation_P"],
        "redilation_k": args["redilation_k"],
        "redilation_P": args["redilation_P"],
    }


def phase_model_values(model):
    # The same values from a PhaseModel fit, NaN where a phase has no fit. time_to_peak is the start of the maintained
    # constriction, which the model takes from the data.
    onset, offset = model.get_light_stimulus().get_timing()
    window, params = model.get_window(), model.get_all_params()
    constriction_fitted = not np.isnan(params["constriction"]["m"])
    redilation_fitted = not np.isnan(params["redilation"]["k"])
    return {
        "constriction_delay": window["constriction"]["start"] - onset if constriction_fitted else np.nan,
        "time_to_peak": window["maintained_constriction"]["start"] - onset,
        "redilation_delay": window["redilation"]["start"] - offset if redilation_fitted else np.nan,
        "baseline_c": params["baseline"]["c"],
        "constriction_m": params["constriction"]["m"],
        "constriction_c": params["constriction"]["c"],
        "maintained_constriction_m": params["maintained_constriction"]["m"],
        "maintained_constriction_c": params["maintained_constriction"]["c"],
        "redilation_S": params["redilation"]["S"],
        "redilation_k": params["redilation"]["k"],
        "redilation_P": params["redilation"]["P"],
    }


def outcome_rows(trial, traces, models, known):
    # The four data metrics of pipeline_piprkit.py and the PhaseModel's values, on the real trace and on the simulation
    data = {name: metrics(trace, trace.filters.baseline_correction_percent(BASELINE, inplace=False))
            for name, trace in traces.items()}
    rows = [{"trial": trial, "outcome": metric, "unit": unit, "fityk": np.nan, "real": real_value,
             "simulation": simulated_value}
            for (metric, real_value, unit), (_, simulated_value, _) in zip(data["real"], data["simulation"])]
    fitted = {name: phase_model_values(model) for name, model in models.items()}
    rows += [{"trial": trial, "outcome": name, "unit": unit, "fityk": known[name], "real": fitted["real"][name],
              "simulation": fitted["simulation"][name]} for name, unit in UNITS.items()]
    return rows


def light_colour(trial):
    for colour, trials in TRIAL_COLOURS:
        if trial in trials:
            return colour


def plot_trial(ax, trial, real, model, values, stretches):
    # One trial in one panel: the real trace, PIPRkit's fit of it, and each Fityk fit over the stretch it was fitted on,
    # drawn with PIPRkit's plot methods
    style(ax)
    real.plot(ax, color=AXIS, linewidth=0.9, label="real trace")
    model.plot_phases(ax)
    time = real.get_time()
    for phase, function in fityk_functions(values).items():
        start, end = stretches[phase]
        inside = (time >= start) & (time <= end)
        label = "Fityk fits, each over its fitted stretch" if phase == "baseline" else "_Fityk fit"
        piprkit.PupilMeasurement(time[inside], function(time[inside])).plot(ax, color=INK, linewidth=1.0,
                                                                            linestyle="--", label=label)
    real.plot_light_stimulus(ax, color=light_colour(trial), label="_light stimulus")
    ax.set_title(f"Trial {trial}", fontsize=FONT_SIZE, color=INK, loc="left")
    ax.set_xlabel("time from light offset (s)", fontsize=FONT_SIZE, color=SECONDARY)
    ax.set_ylabel("pupil size (mm)", fontsize=FONT_SIZE, color=SECONDARY)


def plot_trials(path, title, trials, traces, models, fits):
    # A panel per trial; the panels share their axes
    fig, axes = plt.subplots(len(trials), 1, figsize=(WIDTH, 3.2 * len(trials) + 1.2), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    axes = axes[:, 0]
    for ax, trial in zip(axes, trials):
        plot_trial(ax, trial, traces[trial]["real"], models[trial]["real"], *fits[trial])
    for ax in axes[:-1]:
        ax.set_xlabel("")
    # The legend below the panels: the light stimulus of each trial, the real trace, the Fityk fits and the phases of
    # PIPRkit's fit as its plot labels them. Two columns, because those labels are long; otherwise as finish() in
    # compare.py
    handles = {f"light stimulus, trial {trial}": Patch(color=light_colour(trial), alpha=0.3,
                                                       label=f"light stimulus, trial {trial}") for trial in trials}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    title += "\nIn colour: PIPRkit's fit of the real trace; dashed: the Fityk fits"
    fig.suptitle(title, x=0.01, ha="left", fontsize=FONT_SIZE, color=INK)
    fig.legend(handles=list(handles.values()), loc="outside lower center", ncol=2, frameon=False, fontsize=FONT_SIZE)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation(path, title, outcomes):
    # A panel per value: each trial's Fityk value against PIPRkit's fit of the real trace, coloured by trial group, with
    # the same range on both axes and the line y = x
    columns = 3
    rows = -(-len(UNITS) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(WIDTH, 2.4 * rows + 0.8), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, unit) in zip(axes.flat, UNITS.items()):
        values = outcomes[outcomes["outcome"] == name]
        style(ax)
        identity(ax, values[["fityk", "real"]].to_numpy(dtype=np.float64).ravel())
        ax.locator_params(nbins=4)
        for colour, trials in TRIAL_COLOURS:
            group = values[values["trial"].isin(trials)]
            ax.scatter(group["fityk"], group["real"], s=40, color=colour, edgecolors=SURFACE, linewidths=1.0, zorder=3)
        ax.set_title(f"{name.replace('_', ' ')}\n({unit}), {r_text(values['fityk'], values['real'])}", loc="left",
                     fontsize=FONT_SIZE, color=INK)
        ax.set_xlabel("Fityk", fontsize=FONT_SIZE, color=SECONDARY)
        ax.set_ylabel("PIPRkit", fontsize=FONT_SIZE, color=SECONDARY)
    for ax in axes.flat[len(UNITS):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=colour, marker="o", linestyle="", markersize=7, markeredgecolor=SURFACE,
                      label=trials_text(trials).capitalize()) for colour, trials in TRIAL_COLOURS]
    finish(fig, title, handles + [identity_handle()], path)


def main():
    final = OUTPUT / "final"
    final.mkdir(parents=True, exist_ok=True)
    processed = pd.read_csv(OUTPUT / "intermediate" / "piprkit_series.csv")
    stimuli = pd.read_csv(OUTPUT / "intermediate" / "stimuli.csv").to_numpy()
    series = piprkit.PupilSeries(processed["time"].to_numpy(), processed[EYE].to_numpy(),
                                 [tuple(stimulus) for stimulus in stimuli])
    epochs = series.split(PRE_STIMULUS, POST_STIMULUS)
    fits = read_fits()
    value_rows, outcomes, traces, models = [], [], {}, {}
    for trial, (values, _) in sorted(fits.items()):
        real = epochs[trial - 1]
        # The samples of prepare_fityk.py's files
        real.drop_nan()
        args = simulation_args(values, real)
        try:
            simulation = piprkit.simulate_pupil_measurement(**args)
        except ValueError as error:
            raise ValueError(f"Trial {trial}: the Fityk fits meet in the wrong order. {error}") from error
        known = known_values(values, args)
        value_rows += [{"trial": trial, "value": name, "unit": unit, "known": known[name]}
                       for name, unit in UNITS.items()]
        traces[trial] = {"real": real, "simulation": simulation}
        models[trial] = {name: piprkit.PhaseModel(trace) for name, trace in traces[trial].items()}
        outcomes += outcome_rows(trial, traces[trial], models[trial], known)
        plot_trials(final / f"simulation_trial{trial}.svg", f"{EYE.capitalize()} eye, trial {trial}", [trial],
                    traces, models, fits)
    plot_trials(final / "simulation_trials5_6.svg", f"{EYE.capitalize()} eye, trials 5 and 6", [6, 5], traces, models,
                fits)
    outcomes = pd.DataFrame(outcomes)
    plot_correlation(final / "correlation_fits.svg", f"{EYE.capitalize()} eye, the Fityk fits against PIPRkit's fits "
                     "of the real traces, per trial", outcomes)
    # Per value, Fityk (x) against PIPRkit's fit of the real trace (y): the statistics of agreement() in compare.py
    statistics = [{"value": name, "unit": unit, "x_tool": "fityk", "y_tool": "piprkit",
                   **agreement(*outcomes.loc[outcomes["outcome"] == name, ["fityk", "real"]].T.to_numpy())}
                  for name, unit in UNITS.items()]
    pd.DataFrame(value_rows).to_csv(final / "simulation_values.csv", index=False)
    outcomes.to_csv(final / "simulation_outcomes.csv", index=False)
    pd.DataFrame(statistics).to_csv(final / "statistics_fits.csv", index=False)


if __name__ == "__main__":
    main()
