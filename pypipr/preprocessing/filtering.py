"""
Signal filtering and preprocessing functions for pupil data.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.pupil_base import PupilBase


def rolling_filter(
    pupil: PupilBase,
    func: Callable[[NDArray[np.number]], float],
    time_window: float,
) -> None:
    """Applies a rolling filter to the pupil data.

    Args:
        pupil (PupilBase): The pupil measurement object to filter.
        func (Callable[[NDArray[np.number]], float]): The filtering function to apply. Should take an array and return a single value.
        time_window (float): The time window (in seconds) over which to apply the filter.

    Raises:
        ValueError: If the time window is too small.
    """
    time_data = pupil.get_time()
    size = pupil.get_size()
    dt = np.diff(time_data)
    window_size = int(time_window / np.mean(dt))

    if window_size < 1:
        raise ValueError("Time window is too small. Must be at least 1 sample.")

    if window_size % 2 == 0:
        warnings.warn(
            f"Window size is even. To ensure alignment, time window is adjusted to {(window_size + 1) * np.mean(dt)} ."
        )
        window_size += 1

    half_window = window_size // 2
    new_size = np.full_like(size, np.nan, dtype=np.float64)

    for i in range(len(size)):
        start = max(0, i - half_window)
        end = min(len(size), i + half_window + 1)
        window = size[start:end]
        new_size[i] = func(window)

    pupil.set_time_and_size(time_data, new_size)


def rolling_mean(pupil: PupilBase, time_window: float) -> None:
    """Applies a rolling mean filter to the pupil data.

    Args:
        pupil (PupilBase): The pupil measurement object to filter.
        time_window (float): The time window (in seconds) over which to apply the mean filter.
    """
    rolling_filter(pupil, np.nanmean, time_window)


def rolling_median(pupil: PupilBase, time_window: float) -> None:
    """Applies a rolling median filter to the pupil data.

    Args:
        pupil (PupilBase): The pupil measurement object to filter.
        time_window (float): The time window (in seconds) over which to apply the median filter.
    """
    rolling_filter(pupil, np.nanmedian, time_window)


def get_rate_of_change(pupil: PupilBase, n_back: int = 1) -> NDArray[np.number]:
    """Calculates the rate of change of the pupil size.

    Args:
        pupil (PupilBase): The pupil measurement object to analyze.
        n_back (int, optional): The number of samples to look back for the rate of change calculation. Defaults to 1.

    Returns:
        NDArray[np.number]: The rate of change of the pupil size.
    """
    time_data = pupil.get_time()
    size = pupil.get_size()
    
    dt = np.diff(time_data, n=n_back)
    ds = np.diff(size, n=n_back)
    rate_of_change = np.full_like(size, np.nan, dtype=np.float64)
    rate_of_change[n_back:] = ds / dt
    return rate_of_change


def limit_rate_of_change(
    pupil: PupilBase,
    max_rate_of_change: float,
    n_back: int = 1,
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> None:
    """Limits the rate of change of the pupil size.

    Args:
        pupil (PupilBase): The pupil measurement object to filter.
        max_rate_of_change (float): The maximum allowed rate of change.
        n_back (int, optional): The number of samples to look back for the rate of change calculation. Defaults to 1.
        start_time (float, optional): The start time for the filtering. Defaults to None.
        end_time (float, optional): The end time for the filtering. Defaults to None.

    Raises:
        ValueError: If the method is not recognized.
    """
    if start_time is None:
        start_time = -np.inf
    if end_time is None:
        end_time = np.inf
    time_data = pupil.get_time()
    time_mask = (time_data >= start_time) & (time_data <= end_time)

    roc_mask = np.full_like(pupil.get_size(), True, dtype=bool)

    for i in range(1, n_back + 1):
        rate_of_change = get_rate_of_change(pupil, n_back=i)
        # If rate_of_change is nan or zero, do nothing (keep True in mask)
        iteration_roc_mask = np.where(
            np.isnan(rate_of_change),
            True,
            np.abs(rate_of_change) <= max_rate_of_change
        )
        roc_mask = roc_mask & iteration_roc_mask
    combined_mask = roc_mask & time_mask
    size = pupil.get_size().copy()
    size[~combined_mask] = np.nan
    pupil.set_time_and_size(time_data, size)
