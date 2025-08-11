"""
Data loading functions for example datasets included with pypipr.

This module provides convenient functions to load example pupillometry datasets
for testing, tutorials, and demonstrations.
"""
from __future__ import annotations

from importlib.resources import files
import numpy as np

from ..core.pupil_measurement import PupilMeasurement
from ..core.pupil_series import PupilSeries


def load_simulated_trace(blue_or_red: str = "blue") -> PupilMeasurement:
    """
    Load a simulated pupil light response trace.
    
    Parameters
    ----------
    blue_or_red : str, default "blue"
        Which simulated trace to load: "blue" or "red"
        
    Returns
    -------
    PupilMeasurement
        A PupilMeasurement object with simulated pupil response data
        
    Raises
    ------
    ValueError
        If blue_or_red is not "blue" or "red"
    """
    if blue_or_red not in ["blue", "red"]:
        raise ValueError("blue_or_red must be either 'blue' or 'red'")
    if blue_or_red == "blue":
        data_path = files("pypipr.data.examples.simulated_measures").joinpath("simulated_trace_blue.csv")
    else:
        data_path = files("pypipr.data.examples.simulated_measures").joinpath("simulated_trace_red.csv")
    raw_data = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    time_data = raw_data[:, 0]  # type: ignore
    size = raw_data[:, 1]  # type: ignore
    measurement = PupilMeasurement(time_data, size)  # type: ignore
    measurement.set_light_stimulus(-1.0, 0.0)  # type: ignore
    return measurement


def load_real_series() -> PupilSeries:
    """
    Load a real pupillometry recording with multiple light stimuli.
    
    Returns
    -------
    PupilSeries
        A PupilSeries object with real pupillometry data containing multiple stimuli
    """
    pupil_path = files("pypipr.data.examples.real_series").joinpath("pupil_trace.csv")
    raw_data = np.loadtxt(pupil_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    stimuli_path = files("pypipr.data.examples.real_series").joinpath("stimuli.csv")
    stimuli_data = np.loadtxt(stimuli_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    stimuli_data = stimuli_data[:, 1:3] # type: ignore
    time_data = raw_data[:, 0]  # type: ignore
    size = raw_data[:, 1]  # type: ignore
    return PupilSeries(time_data, size, stimuli_data)  # type: ignore
