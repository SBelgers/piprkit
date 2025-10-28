import warnings
from typing import Any, Optional, override

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..utils.light_stimuli import LightStimuliSeries
from .pupil_measurement import PupilMeasurement
from .pupil_base import PupilBase
from ..preprocessing.filtering import FilterMixin


class PupilSeries(FilterMixin, PupilBase):
    """
    A class representing a series of pupil measurements with multiple light stimuli.

    This class extends PupilMeasurement to handle multiple light stimuli events
    and provides functionality to split the series into individual measurements.
    """

    # ============================================================================
    # INITIALIZATION
    # ============================================================================

    def __init__(
        self,
        time_data: NDArray[np.number],
        size_data: NDArray[np.number],
        stimuli_list: list[tuple[float, float]],
    ) -> None:
        """Initialize PupilSeries with time, size, and light stimuli data.

        Args:
            time_data (NDArray[np.number]): Array of time data.
            size (NDArray[np.number]): Array of size data.
            stimuli_list (list[tuple[float, float]]): List of (start, end) tuples for light stimuli.
        """
        if len(size_data) != len(time_data):
            raise ValueError("Size and time_data must have the same length")
        self.set_time_and_size(time_data, size_data)
        self.set_light_stimuli_from_list(stimuli_list)

    # ============================================================================
    # LIGHT STIMULI MANAGEMENT (OVERRIDES)
    # ============================================================================

    @override
    def set_light_stimulus(self, *args: Any, **kwargs: Any) -> None:
        """Override single stimulus method - use set_light_stimuli_list instead."""
        raise AttributeError("Use set_light_stimuli_list instead.")

    @override
    def get_light_stimulus(self, **kwargs: Any):
        """Override single stimulus method - use get_light_stimuli_list instead."""
        raise AttributeError("Use get_light_stimuli_list instead.")

    def set_light_stimuli_from_list(
        self, light_stimuli_list: list[tuple[float, float]]
    ) -> None:
        """Set multiple light stimuli from a list of (start, end) tuples.

        Args:
            light_stimuli_list (list[tuple[float, float]]): List of (start, end) tuples for light stimuli.
        """
        self.light_stimuli = LightStimuliSeries()
        for start_time, end_time in light_stimuli_list:
            self.light_stimuli.add_stimulus(start_time=start_time, end_time=end_time)

    def get_light_stimuli(self) -> LightStimuliSeries:
        """Get the light stimuli series object.

        Raises:
            ValueError: If light stimuli list is not set.

        Returns:
            LightStimuliSeries: The light stimuli series object.
        """
        if not hasattr(self, "light_stimuli"):
            raise ValueError("Light stimuli list is not set.")
        return self.light_stimuli

    # ============================================================================
    # SERIES PROCESSING METHODS
    # ============================================================================
    def split(
        self, prepulse_duration: float, postpulse_duration: float
    ) -> list[PupilMeasurement]:
        """Split the pupil series into individual measurements around each stimulus.

        Args:
            prepulse_duration (float): Duration before stimulus onset to include.
            postpulse_duration (float): Duration after stimulus offset to include.

        Raises:
            ValueError: If light stimuli list is not set.

        Returns:
            list[PupilMeasurement]: List of individual PupilMeasurement objects, one per stimulus.
        """
        if not hasattr(self, "light_stimuli"):
            raise ValueError("Light stimuli list is not set.")
        if prepulse_duration < 0:
            warnings.warn(
                f"prepulse_duration should be a positive number if time before the pulse is included. Is {prepulse_duration} correct?"
            )
        if postpulse_duration < 0:
            warnings.warn(
                f"postpulse_duration should be a positive number if time after the pulse is included. Is {postpulse_duration} correct?"
            )
        series: list[PupilMeasurement] = []
        light_stimulus_list = self.light_stimuli.get_times()
        time, size = self.get_time(), self.get_size()

        for light_start, light_end in light_stimulus_list:
            start = light_start - prepulse_duration
            end = light_end + postpulse_duration

            series.append(PupilMeasurement(time.copy(), size.copy()))
            series[-1].trim_time(start, end, drop=True)
            series[-1].set_light_stimulus(light_start, light_end)
            series[-1].set_time_offset(-light_end)

        return series

    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================

    def plot_light_stimulus(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all light stimuli in the series.

        Args:
            ax (Optional[Axes], optional): Axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.

        Returns:
            Axes: The axes with the plotted light stimuli.
        """
        light_stimuli = self.get_light_stimuli()
        return light_stimuli.plot(ax, show, **kwargs)
