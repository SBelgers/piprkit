# PIPRkit's pipeline on the raw samples: blink removal, straight-line filter, smoothing, speed filter, interpolation,
# epochs, a percent baseline correction, and PIPRkit's metrics with their default arguments except the baseline.
from pathlib import Path

import numpy as np
import pandas as pd

import piprkit

OUTPUT = Path(__file__).parent / "output"
# The blinks in the packaged recording, marked by hand: eye, start and end in seconds
BLINKS = Path(__file__).parent / "blinks.csv"
EYES = ("left",)
# Epochs from 10 s before light onset, the length of PIPRkit's default baseline, to 45 s after light offset
PRE_STIMULUS = 10.0
POST_STIMULUS = 45.0
# The baseline over the 5 s before light onset, as in PyPlr's documented pipeline, for the baseline and the correction
BASELINE = 5.0


def metrics(epoch, corrected):
    # The metrics of one epoch: epoch in mm, corrected in % change from the baseline
    return [
        ("baseline", float(epoch.metrics.baseline(BASELINE)), "mm"),
        ("peak constriction", float(corrected.metrics.peak_constriction()), "% change"),
        ("time to peak", float(corrected.metrics.time_to_peak()), "s"),
        ("6 s PIPR", float(corrected.metrics.pipr_6s()), "% change"),
    ]


def main():
    (OUTPUT / "final").mkdir(parents=True, exist_ok=True)
    recording = pd.read_csv(OUTPUT / "intermediate" / "input.csv")
    stimuli = pd.read_csv(OUTPUT / "intermediate" / "stimuli.csv").to_numpy()
    blinks = pd.read_csv(BLINKS)
    time = recording["time"].to_numpy()
    processed, corrected_epochs = {"time": time}, {"time": time}
    rows = []
    for eye in EYES:
        series = piprkit.PupilSeries(time, recording[eye].to_numpy(), [tuple(stimulus) for stimulus in stimuli])
        series.set_blinks([tuple(blink) for blink in blinks.loc[blinks["eye"] == eye, ["start", "end"]].to_numpy()])
        series.filters.remove_blinks()
        # Before smoothing, which would shorten the straight stretches
        series.filters.remove_constant_signal(0.125)
        # Smoothing before the speed filter, as its documentation advises at high sampling rates
        series.filters.rolling_mean(0.2)
        series.filters.limit_rate_of_change()
        series.drop_nan()
        series.interpolate(time)
        processed[eye] = series.get_size()
        values = np.full(time.size, np.nan)
        for trial, epoch in enumerate(series.split(PRE_STIMULUS, POST_STIMULUS)):
            onset, offset = stimuli[trial]
            corrected = epoch.filters.baseline_correction_percent(BASELINE, inplace=False)
            # The samples split() keeps for this epoch
            kept = (time >= onset - PRE_STIMULUS) & (time <= offset + POST_STIMULUS) & np.isfinite(processed[eye])
            values[kept] = corrected.get_size()
            for metric, value, unit in metrics(epoch, corrected):
                rows.append({"eye": eye, "trial": trial + 1, "metric": metric, "value": value, "unit": unit})
        corrected_epochs[eye] = values
    pd.DataFrame(processed).to_csv(OUTPUT / "intermediate" / "piprkit_series.csv", index=False)
    pd.DataFrame(corrected_epochs).to_csv(OUTPUT / "intermediate" / "piprkit_corrected.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / "final" / "piprkit_metrics.csv", index=False)


if __name__ == "__main__":
    main()
