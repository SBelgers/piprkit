# Pipeline B, PupilMetrics side: PupilMetrics' own reader, range clip, cleaning and measures on the packaged real
# recording, written as NeuroLight files. PupilMetrics (GPL-3.0) is imported from the folder given as argument; none
# of its code is part of this repository. Run this after pipeline_pupilmetrics.py, whose cleaned series it also
# measures. PupilMetrics' pupildata.py silences all warnings at import, so it runs in this script of its own.
import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from common import EYES, OUTPUT, load_series

# PupilMetrics' default thresholds (3, "medium") with "Remove only steep artifacts" on; the two values before it are
# its zoom and fit display options, off
SETTINGS = [3, 3, 3, 3, 0, 0, 1]


def write_neurolight(folder, eye, series, onset_samples, rate):
    # The two files PupilMetrics' NeuroLight reader expects: DATA (diameter in 0.01 mm and an event per sample) and
    # PARAM (the eye on line 14, the sampling rate on line 15)
    events = np.full(series.get_size().size, " ", dtype=object)
    # A placeholder label: the number in brackets (100, not below 60) makes is_strong False
    events[onset_samples] = "Blue(100)"
    data = pd.DataFrame({"DIAM": series.get_size() * 100, "EVENT": events})
    data.to_csv(folder / f"{eye} DATA.txt", sep="\t", index=False)
    params = ["NAME=PIPRkit validation", "DATE=unknown", "HOUR=unknown"] + ["UNUSED=0"] * 11
    params += [f"EYE={EYES.index(eye)}", f"FREQ={rate}"]
    (folder / f"{eye} PARAM.txt").write_text("[PARAM]\n" + "\n".join(params) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline B, PupilMetrics side, on the packaged real recording")
    parser.add_argument("pupilmetrics", type=Path, help="folder with PupilMetrics' code (pupildata.py, updater.py)")
    parser.add_argument("--rate", type=int, default=200, help="rate of the even time grid in Hz (default 200)")
    arguments = parser.parse_args()
    rate = arguments.rate
    # At the end of sys.path, so PupilMetrics' module names (read, plot, settings) can't shadow other modules
    sys.path.append(str(arguments.pupilmetrics.resolve()))
    import pupildata
    import updater

    class RecordingPupilData(pupildata.PupilData):
        # PupilData.fill extends the NaN stretches of the signal in place before it interpolates; this keeps a copy
        # of that NaN mask so the removed samples can be compared. The computation is unchanged.
        def fill(self, signal):
            filled = super().fill(signal)
            self.masks.append(np.isnan(signal))
            return filled

    class MeasuresOnly(updater.Updater):
        # Updater.compute_measures without what Updater.__init__ also does: the averaged recording, plots and files
        def __init__(self, data):
            self.data = data
            self.settings = data.settings
            self.m = len(data.flash_times[0])
            self.flash_forgot = np.zeros((1, self.m, 2), dtype=np.int64)

        def measures(self, size):
            self.data.drop_free = [np.asarray(size, dtype=np.float64)]
            self.data.measures = np.zeros((1, self.m, 7))
            self.data.grad = [[]]
            self.data.deriv_smooth = [[]]
            for k in range(self.m):
                self.compute_measures(0, k)
            return self.data.measures[0].copy()

    folder = OUTPUT / f"pupilmetrics_{rate}hz"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"pupilmetrics_folder": str(arguments.pupilmetrics.resolve()), "settings": str(SETTINGS)}]).to_csv(
        folder / "reference_source.csv", index=False)
    for eye in EYES:
        series, _, onset_samples = load_series(eye, rate)
        with tempfile.TemporaryDirectory() as temporary:
            write_neurolight(Path(temporary), eye, series, onset_samples, rate)
            data = RecordingPupilData(Path(temporary).as_posix() + "/", "NeuroLight", ["", ""], np.array(SETTINGS))
        if data.errors[0] or data.f != rate or list(data.flash_times[0]) != list(onset_samples):
            raise RuntimeError(f"PupilMetrics did not read the {eye} eye as written: {data.errors[1]}")
        clipped = np.array(data.raw[0], dtype=np.float64)
        data.masks = []
        data.no_artefact()
        cleaned = np.array(data.artefact_free[0], dtype=np.float64)
        measures = MeasuresOnly(data)
        own = measures.measures(cleaned)
        on_piprkit = measures.measures(np.load(folder / f"piprkit_{eye}.npz")["b3"])
        np.savez_compressed(folder / f"reference_{eye}.npz", time=series.get_time(), clipped=clipped,
                            removed=data.masks[-1], cleaned=cleaned, measures=own, measures_on_piprkit=on_piprkit,
                            flash_samples=np.array(data.flash_times[0]))


if __name__ == "__main__":
    main()
