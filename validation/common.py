# Shared by the validation scripts: the packaged real recording on an even time grid, and small helpers.
import sys
import warnings
from importlib import metadata
from importlib.resources import files
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyplr
import scipy

import piprkit

OUTPUT = Path(__file__).parent / "output"
# The blinks in the packaged recording, marked by hand: eye, start and end in seconds
BLINKS = Path(__file__).parent / "blinks.csv"
EYES = ("left", "right")
# Each epoch runs from PRE_STIMULUS seconds before to POST_ONSET seconds after light onset
PRE_STIMULUS = 5.0
POST_ONSET = 50.0


def load_series(eye, rate):
    # The eye's recording, resampled to an even grid at rate Hz, with its recorded light stimuli. PyPlr and
    # PupilMetrics count from the onset sample: the first sample at or after the recorded onset.
    folder = files("piprkit.data.examples.real_series")
    data = np.loadtxt(folder.joinpath("pupil_trace.csv"), delimiter=",", skiprows=1)
    stimuli = np.loadtxt(folder.joinpath("stimuli.csv"), delimiter=",", skiprows=1)[:, 1:3]
    recording = piprkit.PupilMeasurement(data[:, 0], data[:, EYES.index(eye) + 1])
    grid = data[0, 0] + np.arange(int((data[-1, 0] - data[0, 0]) * rate) + 1) / rate
    recording.interpolate(grid)
    onset_samples = np.searchsorted(grid, stimuli[:, 0], side="left")
    series = piprkit.PupilSeries(grid, recording.get_size(), [tuple(stimulus) for stimulus in stimuli])
    return series, stimuli, onset_samples


def call(function, *args, **kwargs):
    # The result of function(*args, **kwargs) and the text of the warnings it raised
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return result, " | ".join(str(warning.message) for warning in caught)


def write_versions(path):
    modules = {"piprkit": piprkit, "pyplr": pyplr, "numpy": np, "scipy": scipy, "pandas": pd, "matplotlib": matplotlib}
    rows = [{"package": "python", "version": sys.version.split()[0], "location": sys.executable}]
    for name, module in modules.items():
        rows.append({"package": name, "version": metadata.version(name), "location": str(Path(module.__file__).parent)})
    pd.DataFrame(rows).to_csv(path, index=False)
