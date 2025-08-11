from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..analysis import pupil_metrics
from ..preprocessing import filtering
from ..utils.light_stimuli import LightStimulus
from .pupil_base import PupilBase


class PupilMeasurement(PupilBase):
    """A class representing a single pupil measurement with time series data."""

    # ============================================================================
    # INITIALIZATION AND CORE DATA METHODS
    # ============================================================================

    def __init__(self, time_data: NDArray[np.number], size: NDArray[np.number]) -> None:
        """Initialize PupilMeasurement with time and size data.

        Args:
            time_data (NDArray[np.number]): Array of time data.
            size (NDArray[np.number]): Array of size data.

        Raises:
            ValueError: If the time and size arrays do not have the same length.
        """
        if len(size) != len(time_data):
            raise ValueError("Size and time_data must have the same length")
        self.set_time_and_size(time_data, size)

    # ============================================================================
    # BASIC DATA MANIPULATION
    # ============================================================================

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for the measurement.
        Can be used to shift the time data by a certain amount.

        Args:
            offset (float): The offset to add to the time data.
        """
        self.set_time_and_size(self.get_time() + offset, self.get_size())
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is not None:
            light_stimulus.set_time_offset(offset)

    def interpolate(self, new_time: NDArray[np.number]) -> None:
        """Interpolate pupil size data to a new time array using np.interp.

        Args:
            new_time (NDArray[np.number]): Array of new time points for interpolation.
        """
        time_data = self.get_time()
        size = self.get_size()
        new_size = np.interp(  # type: ignore
            x=new_time,  # type: ignore
            xp=time_data,
            fp=size,
            left=np.nan,
            right=np.nan,
        )
        self.set_time_and_size(new_time, new_size)  # type: ignore

    # ============================================================================
    # DATA CLEANING AND TRIMMING
    # ============================================================================

    def drop_nan(self) -> None:
        """Remove NaN values from the time and size data."""
        mask = ~np.isnan(self.get_size())
        self.set_time_and_size(
            self.get_time()[mask],
            self.get_size()[mask],
        )

    def trim_time(self, start: float, end: float, drop: bool = True) -> None:
        """Trim the time and size data to a specific interval.

        Args:
            start (float): The start time of the interval to keep.
            end (float): The end time of the interval to keep.
            drop (bool, optional): Whether to drop NaN values after trimming. Defaults to True.
        """
        time_data = self.get_time()
        mask = (time_data >= start) & (time_data <= end)
        new_size = np.full_like(self.get_size(), np.nan, dtype=np.float64)
        new_size[mask] = self.get_size()[mask]
        self.set_time_and_size(
            time_data,
            new_size,
        )
        if drop:
            self.drop_nan()

    def trim_size(self, min_size: float, max_size: float, drop: bool = False) -> None:
        """Trim the size data to a specific range.

        Args:
            min_size (float): The minimum size of the interval to keep.
            max_size (float): The maximum size of the interval to keep.
            drop (bool, optional): Whether to drop NaN values after trimming. Defaults to False.
        """
        size = self.get_size()
        mask = (size >= min_size) & (size <= max_size)
        new_size = np.full_like(size, np.nan, dtype=np.float64)
        new_size[mask] = size[mask]
        self.set_time_and_size(
            self.get_time(),
            new_size,
        )
        if drop:
            self.drop_nan()

    # ============================================================================
    # PREPROCESSING AND FILTERING (WRAPPER METHODS)
    # ============================================================================

    def rolling_filter(
        self, func: Callable[[NDArray[np.number]], float], time_window: float
    ) -> None:
        """ Wrapper for rolling filter method. See `filtering.rolling_filter` for details."""
        filtering.rolling_filter(self, func, time_window)

    def rolling_mean(self, time_window: float) -> None:
        """ Wrapper for rolling mean method. See `filtering.rolling_mean` for details."""
        filtering.rolling_mean(self, time_window)

    def rolling_median(self, time_window: float) -> None:
        """ Wrapper for rolling median method. See `filtering.rolling_median` for details."""
        filtering.rolling_median(self, time_window)

    def get_rate_of_change(self) -> NDArray[np.number]:
        """Wrapper for get rate of change method. See `filtering.get_rate_of_change` for details."""
        return filtering.get_rate_of_change(self)

    def limit_rate_of_change(
        self, max_rate_of_change: float
    ) -> None:
        """Wrapper for limit rate of change method. See `filtering.limit_rate_of_change` for details."""
        filtering.limit_rate_of_change(self, max_rate_of_change)

    # ============================================================================
    # ANALYSIS METHODS (WRAPPER METHODS)
    # ============================================================================
    def get_average_size(self, start_time: float, end_time: float) -> float:
        """Wrapper for get average size method. See `pupil_metrics.get_average_size` for details."""
        return pupil_metrics.get_average_size(self, start_time, end_time)

    def calculate_baseline(self, duration: float = 10) -> float:
        """Wrapper for calculate baseline method. See `pupil_metrics.calculate_baseline` for details."""
        return pupil_metrics.calculate_baseline(self, duration)

    def apply_baseline_correction(self, baseline: float) -> None:
        """Wrapper for apply baseline correction method. See `pupil_metrics.apply_baseline_correction` for details."""
        pupil_metrics.apply_baseline_correction(self, baseline)

    # ============================================================================
    # LIGHT STIMULUS MANAGEMENT
    # ============================================================================

    def set_light_stimulus(self, start: float, end: float) -> None:
        """Set the light stimulus period.

        Args:
            start (float): The start time of the light stimulus.
            end (float): The end time of the light stimulus.
        """
        self.light_stimulus = LightStimulus(start, end)

    def get_light_stimulus(self) -> Optional[LightStimulus]:
        """Get the light stimulus object.

        Returns:
            Optional[LightStimulus]: The light stimulus object or None if not set.
        """
        if not hasattr(self, "light_stimulus"):
            warnings.warn("Light stimulus not set.")
            return None
        return self.light_stimulus

    def plot_light_stimulus(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the light stimulus period.

        Args:
            ax (Optional[Axes], optional): The axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            Axes: The axes with the light stimulus plotted.
        """
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set.")
        return light_stimulus.plot(ax, show, **kwargs)

    # ============================================================================
    # BLINK MANAGEMENT
    # ============================================================================

    def find_blinks(self):  # type: ignore
        """Find blinks in the pupil size data.
        Raises:
            NotImplementedError: Blink detection is not implemented yet.
        """
        warnings.warn(
            "Finding blink based on pupil size traces is not recommended. Use a more reliable method, such as video analysis."
        )
        raise NotImplementedError("Blink detection is not implemented yet.")

    def set_blinks(self, blink_list: tuple[tuple[float, float]]) -> None:
        # TODO update docstring to explain why this is useful.
        """Set the blink periods.

        Args:
            blink_list (tuple[tuple[float, float]]): A list of blink periods, where each period is defined by a start and end time.
        """
        self.blink_list = blink_list

    def get_blinks(self) -> tuple[tuple[float, float]]:
        """Get the blink periods.

        Raises:
            ValueError: Blink list not set.

        Returns:
            tuple[tuple[float, float]]: The list of blink periods.
        """
        if not hasattr(self, "blink_list"):
            raise ValueError("Blink list not set.")
        return self.blink_list
