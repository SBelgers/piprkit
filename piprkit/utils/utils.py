import warnings

import numpy as np
from numpy.typing import NDArray

from ..analysis.pupil_metrics import MetricResult


def check_time_series(time_data: NDArray[np.number]) -> None:
    """
    Check if the time series is sorted and does not contain duplicates.

    Args:
        time_data (NDArray[np.number]): The time series data.

    Raises:
        ValueError: If the time series contains NaNs, is not sorted, or contains duplicates.
    """
    if np.any(np.isnan(time_data)):
        raise ValueError("Time series contains NaN values.")
    if np.any(np.diff(time_data) < 0):
        raise ValueError("Time array is not sorted in ascending order.")
    if np.any(np.diff(time_data) == 0):
        raise ValueError("Time array contains duplicate values.")

def _add_buffer_seconds(time_data: NDArray[np.number], mask: NDArray[np.bool_], buffer_seconds: float) -> NDArray[np.bool_]:
    """
    Add a buffer to a boolean mask based on time data.

    Args:
        time_data (NDArray[np.number]): The time series data.
        mask (NDArray[np.bool_]): The boolean mask to be buffered.
        buffer_seconds (float): The amount of time to buffer in seconds.

    Returns:
        NDArray[np.bool_]: The buffered boolean mask.
    """
    if buffer_seconds <= 0:
        return mask

    removed_times = np.sort(time_data[~mask])
    if removed_times.size == 0:
        return mask

    left = np.searchsorted(removed_times, time_data - buffer_seconds, side="left")
    right = np.searchsorted(removed_times, time_data + buffer_seconds, side="right")
    buffer_mask = right > left

    return mask & ~buffer_mask


def _removal_mask_in_window(
    time_data: NDArray[np.number],
    remove_mask: NDArray[np.bool_],
    start_time: float,
    end_time: float,
    buffer_seconds: float,
) -> NDArray[np.bool_]:
    """
    Restricts a removal mask to a time window and adds a buffer around removed samples.

    Args:
        time_data (NDArray[np.number]): The time series data.
        remove_mask (NDArray[np.bool_]): True for samples that should be removed.
        start_time (float): Start of the window in which samples may be removed.
        end_time (float): End of the window in which samples may be removed.
        buffer_seconds (float): Time in seconds around each removed sample that is also removed.

    Returns:
        NDArray[np.bool_]: True for samples that should be removed.
    """
    time_mask = (time_data >= start_time) & (time_data <= end_time)
    in_window = remove_mask & time_mask
    if buffer_seconds <= 0:
        return in_window
    near_window = (time_data >= start_time - buffer_seconds) & (
        time_data <= end_time + buffer_seconds
    )
    if np.any(remove_mask & ~time_mask & near_window):
        warnings.warn(
            "Samples just outside [start_time, end_time] would be removed but are kept, "
            "so no buffer_seconds buffer is applied around them."
        )
    buffered = ~_add_buffer_seconds(time_data, ~in_window, buffer_seconds)
    return buffered & time_mask


def net_pipr(a: MetricResult, b: MetricResult) -> MetricResult:
    """Net PIPR: the difference a - b between the same metric of two measurements.

    Adhikari et al. (2015), Table 2: "Diff. between 465 nm and 637 nm PIPR in corresponding metric".
    Pass the metric of the 465 nm (blue) measurement as a and that of the 637 nm (red) measurement
    as b, e.g. ``net_pipr(blue.metrics.pipr_6s(), red.metrics.pipr_6s())``. Metrics read from fitted
    models work the same way: ``net_pipr(blue_model.metrics.plateau_pipr(), red_model.metrics.plateau_pipr())``.

    Args:
        a (MetricResult): The metric of the first measurement.
        b (MetricResult): The same metric of the second measurement.

    Raises:
        TypeError: If a or b is not a MetricResult.
        ValueError: If a and b were computed with different baseline correction methods (or one with
            and one without a correction).

    Returns:
        MetricResult: a - b, with the baseline correction of a.

    References:
        - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
    """
    if not isinstance(a, MetricResult) or not isinstance(b, MetricResult):
        raise TypeError(
            "net_pipr needs two MetricResults, e.g. net_pipr(blue.metrics.pipr_6s(), red.metrics.pipr_6s())."
        )
    correction_a, correction_b = a.get_baseline_correction(), b.get_baseline_correction()
    method_a = None if correction_a is None else correction_a.get_method()
    method_b = None if correction_b is None else correction_b.get_method()
    if method_a != method_b:
        raise ValueError(
            f"The metrics have different baseline corrections ({method_a or 'none'} and {method_b or 'none'}), "
            "so their difference has no meaning."
        )
    name = f"net_pipr({a.get_name(with_correction=False)}, {b.get_name(with_correction=False)})"
    return MetricResult(float(a) - float(b), correction_a, name)
