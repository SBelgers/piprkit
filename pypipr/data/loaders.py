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


def load_simulated_pupil(blue_or_red: str = "blue") -> PupilMeasurement:
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
        data_path = files("pypipr.data.examples.simulated_measures").joinpath(
            "simulated_trace_blue.csv"
        )
    else:
        data_path = files("pypipr.data.examples.simulated_measures").joinpath(
            "simulated_trace_red.csv"
        )
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
    stimuli_data = stimuli_data[:, 1:3]  # type: ignore
    time_data = raw_data[:, 0]  # type: ignore
    size = raw_data[:, 1]  # type: ignore
    return PupilSeries(time_data, size, stimuli_data)  # type: ignore


def simulate_pupil_measurement(
    t_start: float = -10.0,
    t_end: float = 10.0,
    num_samples: int = 1000,
    t_light_onset: float = -1.0,
    t_light_offset: float = 0.0,
    plr_latency: float = 0.1,
    t_peak_constrict_rel_to_light_onset: float = 0.25,
    baseline_param_y0: float = 1.0,
    constrict_param_cv: float = -3.0,
    sustained_param_m: float = 0.025,
    redilation_param_k: float = -0.4,
    redilation_param_p: float = 1.0,
    noise_std: float = 0.0,
    trend_std: float = 0.0,
) -> PupilMeasurement:
    # TODO: Include docstring

    t_plr_latency = t_light_onset + plr_latency
    t_peak_constrict = t_light_onset + t_peak_constrict_rel_to_light_onset
    
    # Calculate missing params
    sustained_param_c = baseline_param_y0 + (
        constrict_param_cv * (t_peak_constrict_rel_to_light_onset - plr_latency)
    )

    redilation_param_s = (
        baseline_param_y0
        - sustained_param_c
        - (sustained_param_m * (t_light_offset - t_peak_constrict))
        - (baseline_param_y0 - redilation_param_p)
    )


    from ..analysis.fitting.phase_fits import (
        FitBaseline,
        FitLatency,
        FitConstrict,
        FitSustain,
        FitRedilation,
    )

    time_data = np.linspace(t_start, t_end, num_samples)

    baseline_t = time_data[time_data < t_light_onset]
    baseline_t_start_at_0 = baseline_t - min(baseline_t)
    baseline_size = FitBaseline._model_function(
        baseline_t_start_at_0, baseline_param_y0
    )
    latency_t = time_data[(time_data >= t_light_onset) & (time_data < t_plr_latency)]
    latency_t_start_at_0 = latency_t - min(latency_t)
    latency_size = FitLatency._model_function(latency_t_start_at_0)

    constrict_t = time_data[
        (time_data >= t_plr_latency) & (time_data < t_peak_constrict)
    ]
    constrict_t_start_at_0 = constrict_t - min(constrict_t)
    constrict_size = FitConstrict._model_function(
        constrict_t_start_at_0, constrict_param_cv, baseline_param_y0
    )

    sustain_t = time_data[
        (time_data >= t_peak_constrict) & (time_data < t_light_offset)
    ]
    sustain_t_start_at_0 = sustain_t - min(sustain_t)
    sustain_size = FitSustain._model_function(
        sustain_t_start_at_0, sustained_param_m, sustained_param_c
    )

    redilation_t = time_data[time_data >= t_light_offset]
    redilation_t_start_at_0 = redilation_t - min(redilation_t)
    redilation_size = FitRedilation._model_function(
        redilation_t_start_at_0,
        redilation_param_s,
        redilation_param_k,
        redilation_param_p,
    )

    size = np.concatenate(
        [baseline_size, latency_size, constrict_size, sustain_size, redilation_size]
    )
    size += np.random.normal(0, noise_std, size.shape)  # Add noise
    size += np.cumsum(np.random.normal(0, trend_std, size.shape)) # add trend
    
    measurement = PupilMeasurement(time_data, size)
    measurement.set_light_stimulus(t_light_onset, t_light_offset)
    return measurement
