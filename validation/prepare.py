# Prepares the packaged real recording for the pipeline scripts: the raw samples and stimuli, and for PupilMetrics a
# 67 Hz grid written as NeuroLight files.
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

import piprkit

INTERMEDIATE = Path(__file__).parent / "output" / "intermediate"
EYES = ("left",)
# PupilMetrics was made for the NeuroLight pupillometer, which samples at 67 Hz. It checks its speed and acceleration
# limits per sample, so at the recording's 200 Hz the acceleration limit also removes the constriction.
RATE = 67


def write_neurolight(folder, eye, size, onset_samples):
    # The two files PupilMetrics' NeuroLight reader expects: DATA (diameter in 0.01 mm and an event per sample) and
    # PARAM (the eye on line 14, the sampling rate on line 15)
    folder.mkdir(parents=True, exist_ok=True)
    events = np.full(size.size, " ", dtype=object)
    # A placeholder label: the number in brackets (100, not below 60) makes is_strong False
    events[onset_samples] = "Blue(100)"
    data = pd.DataFrame({"DIAM": size * 100, "EVENT": events})
    data.to_csv(folder / f"{eye} DATA.txt", sep="\t", index=False)
    params = ["NAME=PIPRkit validation", "DATE=unknown", "HOUR=unknown"] + ["UNUSED=0"] * 11
    params += [f"EYE={EYES.index(eye)}", f"FREQ={RATE}"]
    (folder / f"{eye} PARAM.txt").write_text("[PARAM]\n" + "\n".join(params) + "\n")


def main():
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    folder = files("piprkit.data.examples.real_series")
    recording = pd.read_csv(folder.joinpath("pupil_trace.csv"))
    recording.to_csv(INTERMEDIATE / "input.csv", index=False)
    packaged = pd.read_csv(folder.joinpath("stimuli.csv"))
    stimuli = pd.DataFrame({"onset": packaged["start_time"], "offset": packaged["end_time"]})
    stimuli.to_csv(INTERMEDIATE / "stimuli.csv", index=False)

    time = recording["time"].to_numpy()
    grid = time[0] + np.arange(int((time[-1] - time[0]) * RATE) + 1) / RATE
    # PupilMetrics' flash: the first grid sample at or after the recorded onset
    onset_samples = np.searchsorted(grid, stimuli["onset"], side="left")
    resampled = {"time": grid}
    for eye in EYES:
        measurement = piprkit.PupilMeasurement(time, recording[eye].to_numpy())
        measurement.interpolate(grid)
        resampled[eye] = measurement.get_size()
        write_neurolight(INTERMEDIATE / f"neurolight_{eye}", eye, resampled[eye], onset_samples)
    pd.DataFrame(resampled).to_csv(INTERMEDIATE / "grid_67hz.csv", index=False)


if __name__ == "__main__":
    main()
