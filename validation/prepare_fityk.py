# Writes the six trials after PIPRkit's pipeline as text files for the manual fits in Fityk: time in seconds from light
# offset and pupil size in mm, with the light onset and offset in the header. Run it after pipeline_piprkit.py.
from pathlib import Path

import numpy as np
import pandas as pd

import piprkit

INTERMEDIATE = Path(__file__).parent / "output" / "intermediate"
EYE = "left"
# The epochs of pipeline_piprkit.py: from 10 s before light onset to 45 s after light offset
PRE_STIMULUS = 10.0
POST_STIMULUS = 45.0


def main():
    processed = pd.read_csv(INTERMEDIATE / "piprkit_series.csv")
    stimuli = pd.read_csv(INTERMEDIATE / "stimuli.csv").to_numpy()
    series = piprkit.PupilSeries(processed["time"].to_numpy(), processed[EYE].to_numpy(),
                                 [tuple(stimulus) for stimulus in stimuli])
    for trial, epoch in enumerate(series.split(PRE_STIMULUS, POST_STIMULUS), start=1):
        # Fityk does not read NaN well
        epoch.drop_nan()
        onset, offset = epoch.get_light_stimulus().get_timing()
        # Fityk skips the lines that start with #
        header = (f"Trial {trial}, {EYE} eye, after PIPRkit's pipeline. Light onset at {onset:.5f} s, light offset at "
                  f"{offset:.5f} s.\ntime (s from light offset), pupil size (mm)")
        np.savetxt(INTERMEDIATE / f"fityk_trial{trial}.txt", np.column_stack([epoch.get_time(), epoch.get_size()]),
                   fmt="%.6f", header=header)


if __name__ == "__main__":
    main()
