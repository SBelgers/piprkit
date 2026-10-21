from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np

from ..core.pupil_measurement import PupilMeasurement
from ..core.pupil_series import PupilSeries


def load_simulated_pupil(blue_or_red: str = "blue") -> PupilMeasurement:
    """Load one of the two simulated pupil traces that come with piprkit.

    Each trace runs from -6 to 10 s with a sample every 1 ms, and has a light stimulus from -1 to
    0 s, so time 0 is light offset. The pupil size starts at about 8. ``"blue"`` and ``"red"``
    select the two traces.

    Args:
        blue_or_red (str, optional): Which simulated trace to load: "blue" or "red". Defaults to "blue".

    Raises:
        ValueError: If blue_or_red is not "blue" or "red"

    Returns:
        PupilMeasurement: The trace, with its light stimulus set. No blinks or baseline correction.
    """

    if blue_or_red not in ["blue", "red"]:
        raise ValueError("blue_or_red must be either 'blue' or 'red'")
    if blue_or_red == "blue":
        data_path = files("piprkit.data.examples.simulated_measures").joinpath(
            "simulated_trace_blue.csv"
        )
    else:
        data_path = files("piprkit.data.examples.simulated_measures").joinpath(
            "simulated_trace_red.csv"
        )
    raw_data = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    time_data = raw_data[:, 0]  # type: ignore
    size = raw_data[:, 1]  # type: ignore
    measurement = PupilMeasurement(time_data, size)  # type: ignore
    measurement.set_light_stimulus(-1.0, 0.0)  # type: ignore
    return measurement


def load_real_series() -> PupilSeries:
    """Load an example of a real pupil recording that comes with piprkit.

    The recording is 360 s of the left pupil, sampled about every 5 ms (200 Hz), with six light
    stimuli of about 1.17 s each, about 60 s apart. The pupil size data were smoothed with a 0.2 s
    median before packaging. The packaged file also holds the right pupil and an intensity per
    stimulus; these are not loaded. Use :meth:`PupilSeries.split` to cut it into one measurement
    per stimulus.

    Returns:
        PupilSeries: The recording, with its six light stimuli set. No blinks are set.
    """

    pupil_path = files("piprkit.data.examples.real_series").joinpath("pupil_trace.csv")
    raw_data = np.loadtxt(pupil_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    stimuli_path = files("piprkit.data.examples.real_series").joinpath("stimuli.csv")
    stimuli_data = np.loadtxt(stimuli_path, delimiter=",", skiprows=1, dtype=np.float64)  # type: ignore
    stimuli_data = stimuli_data[:, 1:3]  # type: ignore
    time_data = raw_data[:, 0]  # type: ignore
    size = raw_data[:, 1]  # type: ignore
    return PupilSeries(time_data, size, stimuli_data)  # type: ignore


def load_csv(
    path: str | Path,
    time_col: str | int = 0,
    size_col: str | int = 1,
    stimuli: Sequence[tuple[float, float]] | None = None,
    blinks: Sequence[tuple[float, float]] | None = None,
    delimiter: str = ",",
    sidecar: bool = True,
) -> PupilMeasurement | PupilSeries:
    """Load a pupil measurement or series from a CSV file.

    The file needs a header row with column names. One column holds the time in seconds, another
    the pupil size; other columns are ignored. Empty cells become NaN. For example::

        time,size
        0.000,6.41
        0.005,6.40

    The light stimuli and blinks come from the ``stimuli`` and ``blinks`` arguments, or from a JSON
    sidecar file next to the CSV file, with the same name and a .json suffix (``data.csv`` ->
    ``data.json``). :func:`save_csv` writes this file. It gives the onset and duration of each
    light stimulus and blink, in seconds::

        {
          "light_stimuli": [{"onset": 8.06, "duration": 1.17}],
          "blinks": [{"onset": 12.3, "duration": 0.15}]
        }

    With more than one light stimulus the result is a :class:`PupilSeries`. Otherwise it is a
    :class:`PupilMeasurement`, with its light stimulus set if there is one.

    Args:
        path (str | Path): Path to the CSV file.
        time_col (str | int, optional): Name or index of the time column. Defaults to 0 (the
            first column).
        size_col (str | int, optional): Name or index of the pupil size column. Defaults to 1
            (the second column).
        stimuli (Sequence[tuple[float, float]] | None, optional): One ``(start, end)`` tuple per
            light stimulus, in seconds. Replaces the stimuli of the sidecar. Defaults to None.
        blinks (Sequence[tuple[float, float]] | None, optional): One ``(start, end)`` tuple per
            blink, in seconds. Replaces the blinks of the sidecar. Defaults to None.
        delimiter (str, optional): Column separator. Defaults to ",".
        sidecar (bool, optional): Whether to read the JSON sidecar (light stimuli and blinks). A
            missing sidecar is ignored. Defaults to True.

    Raises:
        ValueError: If a column name is not in the file, or if the time data is not strictly
            increasing or contains NaN.

    Returns:
        PupilMeasurement | PupilSeries: A PupilMeasurement or PupilSeries object containing the loaded data.
    """
    data = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=delimiter,
            names=True,
            dtype=np.float64,
            deletechars="",
            replace_space=" ",
        )
    )
    time_data = _csv_column(data, time_col)
    size = _csv_column(data, size_col)
    if sidecar:
        # Arguments passed explicitly win over the sidecar
        sidecar_stimuli, sidecar_blinks = _read_sidecar(Path(path).with_suffix(".json"))
        if stimuli is None:
            stimuli = sidecar_stimuli
        if blinks is None:
            blinks = sidecar_blinks
    stimuli = list(stimuli) if stimuli is not None else []
    if len(stimuli) > 1:
        measurement = PupilSeries(time_data, size, stimuli)
    else:
        measurement = PupilMeasurement(time_data, size)
        if stimuli:
            measurement.set_light_stimulus(*stimuli[0])
    if blinks:
        measurement.set_blinks(list(blinks))
    return measurement


def save_csv(
    measurement: PupilMeasurement | PupilSeries,
    path: str | Path,
    size_col: str = "size",
) -> None:
    """Save a pupil measurement or series to a CSV file, with a JSON sidecar.

    The CSV file has two columns, time and size. The sidecar is written next to it with the
    same name and a .json suffix. It holds the light stimuli and blinks as lists of
    {"onset", "duration"} entries. Existing files are overwritten.

    Args:
        measurement (PupilMeasurement | PupilSeries): The measurement or series to save.
        path (str | Path): Path to the CSV file.
        size_col (str, optional): Header of the size column. Defaults to "size".
    """
    path = Path(path)
    np.savetxt(
        path,
        np.column_stack([measurement.get_time(), measurement.get_size()]),
        fmt="%.17g",
        delimiter=",",
        header=f"time,{size_col}",
        comments="",
    )
    metadata = {
        "light_stimuli": _onset_durations(_light_stimulus_times(measurement)),
        "blinks": _onset_durations(measurement.get_blinks()),
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _csv_column(data: np.ndarray, column: str | int) -> np.ndarray:
    names = data.dtype.names
    if isinstance(column, int):
        return np.asarray(data[names[column]], dtype=np.float64)
    if column not in names:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(names)}"
        )
    return np.asarray(data[column], dtype=np.float64)


def _read_sidecar(
    path: Path,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as file:
        metadata = json.load(file)
    return (
        _intervals(metadata.get("light_stimuli", [])),
        _intervals(metadata.get("blinks", [])),
    )


def _intervals(entries: list[dict]) -> list[tuple[float, float]]:
    return [(entry["onset"], entry["onset"] + entry["duration"]) for entry in entries]


def _onset_durations(intervals: Sequence[tuple[float, float]]) -> list[dict]:
    return [{"onset": float(start), "duration": float(end - start)} for start, end in intervals]


def _light_stimulus_times(measurement: PupilMeasurement | PupilSeries) -> list[tuple[float, float]]:
    if isinstance(measurement, PupilSeries):
        try:
            return measurement.get_light_stimuli_series().get_timings()
        except ValueError:
            return []
    stimulus = getattr(measurement, "light_stimulus", None)
    return [] if stimulus is None else [stimulus.get_timing()]


def simulate_pupil_measurement(
    t_start: float = -10.0,
    t_end: float = 10.0,
    num_samples: int = 1000,
    time_data: np.ndarray | None = None,
    t_light_onset: float = -1.0,
    t_light_offset: float = 0.0,
    constriction_delay: float = 0.1,
    time_to_peak: float = 0.25,
    redilation_delay: float = 0.0,
    baseline_c: float = 1.0,
    maintained_constriction_c: float = 0.55,
    maintained_constriction_m: float = 0.025,
    redilation_k: float = -0.4,
    redilation_P: float = 1.0,
    noise_std: float = 0.0,
    trend_std: float = 0.0,
    rng: int | np.random.Generator | None = None,
) -> PupilMeasurement:
    """Simulate a pupil measurement with a light stimulus and a pupillary light response.

    The response is built from the model functions of the four phases of a PhaseModel. Each phase
    starts where the previous one ends, so the curve has no jumps:

    - Baseline: the constant size ``baseline_c``, up to the constriction start at light onset +
      ``constriction_delay`` (BaselinePhase).
    - Constriction: a line from ``baseline_c`` down to ``maintained_constriction_c`` at the peak
      constriction, at light onset + ``time_to_peak`` (ConstrictionPhase).
    - Maintained constriction: a line with slope ``maintained_constriction_m`` that starts at
      ``maintained_constriction_c``, from the peak constriction to the redilation start at light
      offset + ``redilation_delay`` (MaintainedConstrictionPhase).
    - Redilation: the exponential S · e^(k · x) + P from the redilation start on, with x the time
      since the redilation start, k = ``redilation_k`` and P = ``redilation_P`` (RedilationPhase).

    The phases don't have to line up with the sections of the light stimulus: the peak constriction
    can come after light offset, and the redilation start later still. The slope of the constriction
    and the amplitude S are not arguments: they follow from the others, so that the phases join up.
    The other arguments have the names of the fitted parameters, phase and parameter as in
    :meth:`PhaseModel.get_all_params`, so a fitted model can be simulated with
    ``simulate_pupil_measurement(**model.get_simulation_args())``.

    The defaults give an illustrative response; they are not taken from measurements. Times are in
    seconds. Sizes have no fixed unit; with the default ``baseline_c=1.0`` they read as a fraction of
    the baseline.

    Args:
        t_start (float, optional): Time of the first sample, in seconds. Defaults to -10.0.
        t_end (float, optional): Time of the last sample, in seconds. Defaults to 10.0.
        num_samples (int, optional): Number of samples, evenly spaced from t_start to t_end.
            Defaults to 1000.
        time_data (np.ndarray | None, optional): Times of the samples, in seconds. If given, it replaces
            t_start, t_end and num_samples. Defaults to None.
        t_light_onset (float, optional): Light onset, in seconds. Defaults to -1.0.
        t_light_offset (float, optional): Light offset, in seconds. Defaults to 0.0.
        constriction_delay (float, optional): Time from light onset to the constriction start, in
            seconds. Until then the size stays at the baseline. Defaults to 0.1.
        time_to_peak (float, optional): Time from light onset to the peak constriction, in seconds.
            Defaults to 0.25.
        redilation_delay (float, optional): Time from light offset to the redilation start, in
            seconds. Defaults to 0.0.
        baseline_c (float, optional): Baseline size. Defaults to 1.0.
        maintained_constriction_c (float, optional): Size at the peak constriction. Defaults to 0.55.
        maintained_constriction_m (float, optional): Slope of the maintained constriction, in size
            units per second. Defaults to 0.025.
        redilation_k (float, optional): Rate constant k of the redilation exponential, in 1/s.
            Defaults to -0.4.
        redilation_P (float, optional): Plateau P of the redilation exponential: the size it
            approaches. Defaults to 1.0.
        noise_std (float, optional): Standard deviation of independent Gaussian noise added to each
            sample. Defaults to 0.0.
        trend_std (float, optional): Standard deviation of the steps of a random walk (the
            cumulative sum of Gaussian steps) added to the trace, so the trace drifts. Defaults to 0.0.
        rng (int | np.random.Generator | None, optional): Seed or ``numpy.random.Generator`` for the
            noise and trend; the same seed gives the same trace. Defaults to None.

    Raises:
        ValueError: Unless t_start <= light onset <= constriction start < peak constriction <=
            redilation start <= t_end, and light offset <= redilation start.

    Returns:
        PupilMeasurement: The simulated measurement, with its light stimulus set.
    """

    if time_data is None:
        time_data = np.linspace(t_start, t_end, num_samples)
    else:
        time_data = np.array(time_data, dtype=np.float64)
        t_start, t_end = float(time_data[0]), float(time_data[-1])
    constriction_start = t_light_onset + constriction_delay
    peak_constriction = t_light_onset + time_to_peak
    redilation_start = t_light_offset + redilation_delay
    if not (
        t_start <= t_light_onset <= constriction_start < peak_constriction <= redilation_start <= t_end
        and t_light_offset <= redilation_start
    ):
        raise ValueError(
            "Expected t_start <= light onset <= constriction start < peak constriction <= redilation start <= t_end, "
            "and light offset <= redilation start."
        )

    # The constriction slope and S follow from the other values, so that the phases join up
    constriction_m = (maintained_constriction_c - baseline_c) / (time_to_peak - constriction_delay)
    redilation_S = (
        maintained_constriction_c
        + maintained_constriction_m * (redilation_start - peak_constriction)
        - redilation_P
    )

    from ..analysis.fitting.phases import (
        BaselinePhase,
        ConstrictionPhase,
        MaintainedConstrictionPhase,
        RedilationPhase,
    )

    size = np.empty_like(time_data)
    baseline = time_data < constriction_start
    constriction = (time_data >= constriction_start) & (time_data < peak_constriction)
    maintained_constriction = (time_data >= peak_constriction) & (time_data < redilation_start)
    redilation = time_data >= redilation_start
    size[baseline] = BaselinePhase.get_model_function()(time_data[baseline] - t_start, baseline_c)
    size[constriction] = ConstrictionPhase.get_model_function()(
        time_data[constriction] - constriction_start, constriction_m, baseline_c
    )
    size[maintained_constriction] = MaintainedConstrictionPhase.get_model_function()(
        time_data[maintained_constriction] - peak_constriction,
        maintained_constriction_m,
        maintained_constriction_c,
    )
    size[redilation] = RedilationPhase.get_model_function()(
        time_data[redilation] - redilation_start, redilation_S, redilation_k, redilation_P
    )
    rng = np.random.default_rng(rng)
    size += rng.normal(0, noise_std, size.shape)  # Add noise
    size += np.cumsum(rng.normal(0, trend_std, size.shape))  # add trend

    measurement = PupilMeasurement(time_data, size)
    measurement.set_light_stimulus(t_light_onset, t_light_offset)
    return measurement
