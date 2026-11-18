# PyPlr's pipeline with the raw samples read as 200 Hz: first-derivative and blink masking, interpolation, smoothing,
# epochs, percent signal change and PLR metrics.
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyplr import preproc, utils
from pyplr.plr import PLR

# pandas 2.2 on Python 3.14 warns about chained assignment on plain column assignments, also inside PyPlr; the values
# are set all the same, so this warning is not shown.
warnings.filterwarnings("ignore", message="ChainedAssignmentError", category=FutureWarning)
OUTPUT = Path(__file__).parent / "output"
# The blinks in the packaged recording, marked by hand: eye, start and end in seconds
BLINKS = Path(__file__).parent / "blinks.csv"
EYES = ("left",)
# PyPlr counts samples at 200 Hz; missed samples shift its times
RATE = 200
# 5 s before and 45 s after onset
BEFORE = 5 * RATE
DURATION = 50 * RATE


def main():
    (OUTPUT / "final").mkdir(parents=True, exist_ok=True)
    recording = pd.read_csv(OUTPUT / "intermediate" / "input.csv")
    stimuli = pd.read_csv(OUTPUT / "intermediate" / "stimuli.csv")
    blinks = pd.read_csv(BLINKS)
    time = recording["time"].to_numpy()
    onsets = pd.DataFrame(index=stimuli["onset"].to_numpy())
    processed, corrected = {"time": time}, {"time": time}
    rows = []
    for eye in EYES:
        samples = pd.DataFrame({"diameter": recording[eye].to_numpy()}, index=time)
        eye_blinks = blinks.loc[blinks["eye"] == eye]
        events = pd.DataFrame({"start_timestamp": eye_blinks["start"].to_numpy(),
                               "end_timestamp": eye_blinks["end"].to_numpy()})
        # As in PyPlr's documented pipeline, first: samples whose change from the previous sample is more than 3 SD from
        # the mean change; the blinks then take the place of its confidence mask
        samples = preproc.mask_pupil_first_derivative(samples, threshold=3.0)
        samples = preproc.mask_blinks(samples, events)
        # Linear interpolation over sample counts
        samples = preproc.interpolate_pupil(samples)
        # As in PyPlr's documented pipeline: a 3rd-order Butterworth filter with a 4 Hz cut-off, run forwards and
        # backwards
        samples = preproc.butterworth_series(samples, fields=["diameter"], filt_order=3, cutoff_freq=4 / (RATE / 2))
        processed[eye] = samples["diameter"].to_numpy()
        # Counts from the first sample at or after onset
        ranges = utils.extract(samples, onsets, offset=-BEFORE, duration=DURATION)
        baselines = ranges.loc[(slice(None), slice(0, BEFORE - 1)), ["diameter"]].groupby(level="event").mean()
        ranges = preproc.percent_signal_change(ranges, baselines, ["diameter"])
        values = np.full(time.size, np.nan)
        for trial, (onset, offset) in enumerate(stimuli.to_numpy()):
            epoch = ranges.loc[trial]
            values[np.searchsorted(time, epoch["orig_idx"].to_numpy())] = epoch["diameter_pc"].to_numpy()
            # The peak and the time to peak from the % epoch; the peak is searched in the whole epoch, including the 5 s
            # before onset. The baseline in mm is the one percent_signal_change used.
            plr = PLR(epoch["diameter_pc"].to_numpy(), RATE, BEFORE, offset - onset)
            metrics = [
                ("baseline", baselines.loc[trial, "diameter"], "mm"),
                ("peak constriction", plr.peak_constriction(), "% change"),
                ("time to peak", plr.time_to_max_constriction(), "s"),
            ]
            for metric, value, unit in metrics:
                rows.append({"eye": eye, "trial": trial + 1, "metric": metric, "value": value, "unit": unit})
        corrected[eye] = values
    pd.DataFrame(processed).to_csv(OUTPUT / "intermediate" / "pyplr_series.csv", index=False)
    pd.DataFrame(corrected).to_csv(OUTPUT / "intermediate" / "pyplr_corrected.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / "final" / "pyplr_metrics.csv", index=False)


if __name__ == "__main__":
    main()
