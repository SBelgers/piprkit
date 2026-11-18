# PupilMetrics' pipeline on the 67 Hz grid: its NeuroLight reader, range clip, cleaning, drop detection and measures.
# PupilMetrics (GPL-3.0) is imported from the folder given as argument; none of its code is part of this repository.
# Its pupildata.py silences all warnings at import, so it runs in a script of its own. Its drop detection needs numpy
# 1.23 or older, so this script runs in a venv of its own.
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).parent / "output"
EYES = ("left",)
RATE = 67
# PupilMetrics' default settings from its README and help: delete threshold, drop detection and derivative smoothing 3
# ("medium"), fit precision 5 ("high"; its shipped files/settings.npy has 3), zoom and fit display off, and "Remove only
# steep artifacts" off, so its drop detection runs
SETTINGS = [3, 3, 5, 3, 0, 0, 0]


def main():
    parser = argparse.ArgumentParser(description="PupilMetrics' pipeline")
    parser.add_argument("pupilmetrics", type=Path, help="folder with PupilMetrics' code (pupildata.py, updater.py)")
    # At the end of sys.path, so PupilMetrics' module names (read, plot, settings) can't shadow other modules
    sys.path.append(str(parser.parse_args().pupilmetrics.resolve()))
    import pupildata
    import updater

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

    (OUTPUT / "final").mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(OUTPUT / "intermediate" / "grid_67hz.csv")
    stimuli = pd.read_csv(OUTPUT / "intermediate" / "stimuli.csv")
    onset_samples = np.searchsorted(grid["time"], stimuli["onset"], side="left")
    cleaned = {"time": grid["time"].to_numpy()}
    rows = []
    for eye in EYES:
        folder = (OUTPUT / "intermediate" / f"neurolight_{eye}").resolve().as_posix() + "/"
        data = pupildata.PupilData(folder, "NeuroLight", ["", ""], np.array(SETTINGS))
        if data.errors[0] or data.f != RATE or list(data.flash_times[0]) != list(onset_samples):
            raise RuntimeError(f"PupilMetrics did not read the {eye} eye as written: {data.errors[1]}")
        # Removes samples faster than 6 mm/s or 90 mm/s² (central differences), straight stretches over 0.125 s and
        # samples 1.2-3.3 s after a flash by its own rule; merges gaps less than 0.25 s apart and interpolates
        data.no_artefact()
        # Its drop detection: a curve fitted per flash, and the drops below it removed and interpolated. Without the
        # GUI's review every detected drop stays removed, as the GUI first shows it.
        data.fit_signal(["", ""])
        data.no_drop()
        cleaned[eye] = np.array(data.drop_free[0], dtype=np.float64)
        # Its windows count whole samples from the flash sample: baseline 0.25 s before, peak search to 2.025 s after
        # (RT adds 2.42 samples), 6s constriction 7.0-7.1 s after
        values = MeasuresOnly(data).measures(cleaned[eye])
        for trial in range(len(onset_samples)):
            baseline, mca, rt, six_s = values[trial, :4]
            metrics = [
                ("baseline", baseline, "mm"),
                ("peak constriction", mca, "% constriction"),
                ("time to peak", rt, "s"),
                ("6 s PIPR", six_s, "% constriction"),
            ]
            for metric, value, unit in metrics:
                rows.append({"eye": eye, "trial": trial + 1, "metric": metric, "value": value, "unit": unit})
    pd.DataFrame(cleaned).to_csv(OUTPUT / "intermediate" / "pupilmetrics_series.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / "final" / "pupilmetrics_metrics.csv", index=False)


if __name__ == "__main__":
    main()
