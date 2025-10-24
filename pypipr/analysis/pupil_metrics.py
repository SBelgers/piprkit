# TODO: Update to standards in pupillometry
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..analysis.fitting.pupil_fit import PupilFit
    from ..core.pupil_measurement import PupilMeasurement

"""
Metrics as described by Adhikari et al. (2015), table 2:


| Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
|--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
| baseline           | Baseline Pupil Diameter             | Average over 10 s prestimulus period                              | mm               |

# PLR (Pupillary Light Reflex) Metrics
| transient_plr      | Transient PLR                       | Peak change from 180 to 500 ms after light onset                  | unitless         |
| plr_latency        | PLR Latency                         | Time for 1% constriction                                          | s                |
| constriction_v     | Constriction Velocity               | Stimulus gradient of linear model at light onset                  | mm/s             |
| peak_constriction  | Peak Constriction Amplitude         | Minimum pupil size during light presentation (% of baseline)      | unitless         |
| time_to_peak       | Time to Peak                        | Time to peak pupil constriction                                   | s                |
| pupil_escape       | Pupil Escape                        | Stimulus gradient of linear model during light stimulation        | mm/s             |

# PIPR (Post-Illumination Pupil Response) Metrics
| redilation_v       | Redilation Velocity                 | Global rate constant (k) of exponential model                     | mm/s             |
| pipr_6s            | 6 s PIPR Amplitude                  | Pupil size at 6 s after light offset (% of baseline)              | unitless         |
| plateau            | Plateau PIPR                        | Plateau of exponential model                                      | unitless         |
| auc_early          | AUC Early                           | sum(BPD APD) over 0-10 s after light offset                       | unitless         |
| auc_late           | AUC Late                            | sum(BPD APD) over 10-30 s after light offset                      | unitless         |
| pipr_duration      | PIPR Duration                       | Time to return to baseline after light offset                     | s                |

# Net PIPR Metrics
| net_pipr           | Net PIPR                            | Diff. between 465 nm and 637 nm PIPR in corresponding metric*     | varies by metric |

"""


def baseline(pupil_measurement: PupilMeasurement, duration: float = 10) -> float:
    """
    Metrics as described by Adhikari et al. (2015), table 2:


    | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
    |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
    | baseline           | Baseline Pupil Diameter             | Average over 10 s prestimulus period                              | mm               |


    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.
        duration (float, optional): The duration of the baseline period in seconds. Defaults to 10.

    Raises:
        ValueError: If the light stimulus is not set.
        ValueError: If the baseline period is outside the time range of the measurement.
        ValueError: If there are no data points in the specified baseline period.

    Returns:
        float: The average pupil size during the baseline period.
    """
    warnings.warn("This is a double implementation and will be deprecated.")
    return calculate_baseline(pupil_measurement, duration)


def transient_plr(pupil_measurement: PupilMeasurement) -> float:
    """

        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | transient_plr      | Transient PLR                       | Peak change from 180 to 500 ms after light onset                  | unitless         |

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()
    # TODO: Implement


def plr_latency(pupil_measurement: PupilMeasurement) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | plr_latency        | PLR Latency                         | Time for 1% constriction                                          | s                |

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def constriction_v(PupilFit: PupilFit) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | constriction_v     | Constriction Velocity               | Stimulus gradient of linear model at light onset                  | mm/s             |

    Args:
        PupilFit (PupilFit): The pupil fit data.

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def peak_constriction(pupil_measurement: PupilMeasurement) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | peak_constriction  | Peak Constriction Amplitude         | Minimum pupil size during light presentation (% of baseline)      | unitless         |

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.

    Raises:
        ValueError: If the light stimulus is not set.

    Returns:
        float: The peak constriction amplitude.
    """
    warnings.warn(
        "The peak constriction function assumes the size is already relative to the baseline."
    )
    pupil_measurement = pupil_measurement.copy()
    stimulus = pupil_measurement.get_light_stimulus()
    if stimulus is None:
        raise ValueError("Light stimulus not set. Cannot calculate peak constriction.")
    light_start, light_end = stimulus.get_time()
    pupil_measurement.trim_time(light_start, light_end)
    size = pupil_measurement.get_size()
    minimum_size = np.nanmin(size)
    return float(minimum_size)


def time_to_peak(pupil_measurement: PupilMeasurement) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | time_to_peak       | Time to Peak                        | Time to peak pupil constriction                                   | s                |

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.

    Raises:
        ValueError: If the light stimulus is not set.

    Returns:
        float: The time to peak pupil constriction.
    """
    pupil_measurement = pupil_measurement.copy()
    stimulus = pupil_measurement.get_light_stimulus()
    if stimulus is None:
        raise ValueError("Light stimulus not set. Cannot calculate time to peak.")

    stimulus_start, stimulus_end = stimulus.get_time()
    pupil_measurement.trim_time(stimulus_start, stimulus_end)
    size = pupil_measurement.get_size()
    time_data = pupil_measurement.get_time()
    peak_index = np.nanargmin(size)
    warnings.warn("This does not correctly account for the pupil escape.")
    return float(time_data[peak_index] - stimulus_start)


def pupil_escape(PupilFit: PupilFit) -> float:
    """_summary_
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | pupil_escape       | Pupil Escape                        | Stimulus gradient of linear model during light stimulation        | mm/s             |

    Args:
        PupilFit (PupilFit): The pupil fit data.

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def redilation_v(PupilFit: PupilFit) -> float:
    """_summary_
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | redilation_v       | Redilation Velocity                 | Global rate constant (k) of exponential model                     | mm/s             |

    Args:
        PupilFit (PupilFit): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def pipr_xs(pupil_measurement: PupilMeasurement, start: float, end: float) -> float:
    """The x second PIPR amplitude between start and end time after light offset.

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.
        start (float): The start time (in seconds) for the PIPR calculation.
        end (float): The end time (in seconds) for the PIPR calculation.

    Raises:
        ValueError: If the light stimulus is not set.

    Returns:
        float: The x second PIPR amplitude.
    """
    pupil_measurement = pupil_measurement.copy()
    stimulus = pupil_measurement.get_light_stimulus()
    if stimulus is None:
        raise ValueError("Light stimulus not set. Cannot calculate time to peak.")

    _, stimulus_end = stimulus.get_time()
    # pupil_measurement.trim_time(stimulus_end + start, stimulus_end + end)
    return get_average_size(pupil_measurement, stimulus_end + start, stimulus_end + end)


def pipr_6s(pupil_measurement: PupilMeasurement) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | pipr_6s            | 6 s PIPR Amplitude                  | Pupil size at 6 s after light offset (% of baseline)              | unitless         |

    Args:
        pupil_measurement (PupilMeasurement): The pupil measurement data.

    Returns:
        float: The 6 second PIPR amplitude.
    """
    return pipr_xs(pupil_measurement, 5.5, 6.5)


def plateau(PupilFit: PupilFit) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | plateau            | Plateau PIPR                        | Plateau of exponential model                                      | unitless         |

    Args:
        PupilFit (PupilFit): _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def auc_early(pupil_measurement: PupilMeasurement) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | auc_early          | AUC Early                           | sum(BPD APD) over 0-10 s after light offset                       | unitless         |

    Args:
        pupil_measurement (PupilMeasurement): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def auc_late(pupil_measurement: PupilMeasurement) -> float:
    """_summary_
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | auc_late           | AUC Late                            | sum(BPD APD) over 10-30 s after light offset                      | unitless         |

    Args:
        pupil_measurement (PupilMeasurement): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def pipr_duration(pupil_measurement: PupilMeasurement) -> float:
    """_summary_
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | pipr_duration      | PIPR Duration                       | Time to return to baseline after light offset                     | s                |

    Args:
        pupil_measurement (PupilMeasurement): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def net_pipr(
    pupil_measurement1: PupilMeasurement, pupil_measurement2: PupilMeasurement
) -> float:
    """
        Metrics as described by Adhikari et al. (2015), table 2:


        | Shorthand Name     | Full Metric Name                    | Description                                                       | Unit             |
        |--------------------|-------------------------------------|-------------------------------------------------------------------|------------------|
        | net_pipr           | Net PIPR                            | Diff. between 465 nm and 637 nm PIPR in corresponding metric*     | varies by metric |


    Args:
        pupil_measurement1 (PupilMeasurement): _description_
        pupil_measurement2 (PupilMeasurement): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        float: _description_
    """
    raise NotImplementedError()


def get_average_size(
    pupil: PupilMeasurement, start_time: float, end_time: float
) -> float:
    """Calculate average pupil size within a time range.

    Args:
        pupil (PupilMeasurement): The pupil measurement object
        start_time (float): Start time for averaging window
        end_time (float): End time for averaging window

    Raises:
        ValueError: If no data points are found in the specified time range.

    Returns:
        float: Average pupil size in the specified time range
    """
    time_data = pupil.get_time()
    size = pupil.get_size()
    if min(time_data) > start_time:
        warnings.warn("Start time is before the first data point.")
    if max(time_data) < end_time:
        warnings.warn("End time is after the last data point.")

    mask = (time_data >= start_time) & (time_data <= end_time)
    if not np.any(mask):
        raise ValueError("No data points in the specified time range.")
    return np.nanmean(size[mask])  # type: ignore


def calculate_baseline(pupil: PupilMeasurement, duration: float = 10) -> float:
    """Calculate baseline pupil diameter from prestimulus period.

    Args:
        pupil (PupilMeasurement): The pupil measurement object
        duration (float, optional): Duration of prestimulus period to use for baseline calculation. Defaults to 10.

    Raises:
        ValueError: If the light stimulus is not set.

    Returns:
        float: Average baseline pupil diameter
    """
    light_stimulus = pupil.get_light_stimulus()
    if light_stimulus is None:
        raise ValueError("Light stimulus not set. Cannot determine baseline period.")
    start_time = light_stimulus.get_time()[0] - duration
    end_time = light_stimulus.get_time()[0]
    return get_average_size(pupil, start_time, end_time)


# def apply_baseline_correction(pupil: PupilMeasurement, baseline: float) -> None:
#     """Apply baseline correction to pupil size data.

#     Args:
#         pupil (PupilMeasurement): The pupil measurement object to correct
#         baseline (float): Baseline value for normalization
#     """
#     size_relative = pupil.get_size() / baseline
#     pupil.set_time_and_size(pupil.get_time(), size_relative)
