import warnings
from typing import Any, NoReturn, override

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..utils.light_stim import LightStimuliSeries
from .pupil_base import PupilBase
from .pupil_measurement import PupilMeasurement


class PupilSeries(PupilBase):
    """
    A class representing a series of pupil measurements with multiple light stimuli.

    This class extends PupilBase to handle multiple light stimuli events
    and provides functionality to split the series into individual measurements.
    """

    def __init__(
        self,
        time_data: NDArray[np.number],
        size_data: NDArray[np.number],
        stimuli_list: list[tuple[float, float]],
    ) -> None:
        """Initialize PupilSeries with time, pupil size, and light stimuli data.

        Args:
            time_data (NDArray[np.number]): 1-D array with the time of each sample in seconds,
                strictly increasing and without NaN.
            size_data (NDArray[np.number]): 1-D array with the pupil size of each sample (usually
                the diameter in mm), same length as ``time_data``. NaN marks missing samples.
            stimuli_list (list[tuple[float, float]]): One ``(start, end)`` tuple per light
                stimulus: light onset and offset in seconds, on the same time axis as ``time_data``.

        Raises:
            ValueError: If the arrays differ in length, or if time_data is not strictly
                increasing or contains NaN.
        """
        if len(size_data) != len(time_data):
            raise ValueError("Size and time_data must have the same length")
        self.set_time_and_size(time_data, size_data)
        self.set_light_stimuli_from_list(stimuli_list)

    @property
    def metrics(self):
        """Not available on a series.

        Metrics need a single light stimulus. Split the series with :meth:`split` and
        use ``metrics`` on each measurement.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(
            "Metrics need a single light stimulus. Use split() first and compute metrics per measurement."
        )

    @override
    def set_light_stimulus(self, *args: Any, **kwargs: Any) -> None:
        """Not available on a series, which has several light stimuli.

        Use :meth:`set_light_stimuli_from_list` instead.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError("Use set_light_stimuli_from_list instead.")

    @override
    def get_light_stimulus(self, **kwargs: Any):
        """Not available on a series, which has several light stimuli.

        Use :meth:`get_light_stimuli_series` instead.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError("Use get_light_stimuli_series instead.")

    @override
    def get_sections(self) -> NoReturn:
        """Not available on a series, which has several light stimuli.

        Split the series with :meth:`split` and use ``get_sections`` on each measurement.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError("Sections need a single light stimulus. Use split() first.")

    @override
    def plot_sections(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Not available on a series, which has several light stimuli.

        Split the series with :meth:`split` and use ``plot_sections`` on each measurement.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError("Sections need a single light stimulus. Use split() first.")

    def set_light_stimuli_from_list(
        self, light_stimuli_list: list[tuple[float, float]]
    ) -> None:
        """Set the light stimuli of the series, replacing any set before.

        A light stimulus is a period in which the light that evokes a pupil response is on.
        :meth:`split` cuts the series into one measurement per stimulus.

        Args:
            light_stimuli_list (list[tuple[float, float]]): One ``(start, end)`` tuple per
                light stimulus: light onset and offset in seconds, on the same time axis as
                :meth:`get_time`.
        """
        self.light_stimuli_series = LightStimuliSeries()
        for start_time, end_time in light_stimuli_list:
            self.light_stimuli_series.add_stimulus(start_time=start_time, end_time=end_time)

    def get_light_stimuli_series(self) -> LightStimuliSeries:
        """Get the light stimuli series object.

        Raises:
            ValueError: If light stimuli list is not set.

        Returns:
            LightStimuliSeries: The light stimuli; ``get_timings()`` gives their
            ``(start, end)`` times in seconds.
        """
        if not hasattr(self, "light_stimuli_series"):
            raise ValueError("Light stimuli list is not set.")
        return self.light_stimuli_series

    def split(
        self, pre_stimulus_duration: float, post_stimulus_duration: float
    ) -> list[PupilMeasurement]:
        """Split the pupil series into individual measurements around each stimulus.

        Args:
            pre_stimulus_duration (float): Time in seconds before each light onset to include; this
                becomes the pre-stimulus section of each measurement.
            post_stimulus_duration (float): Time in seconds after each light offset to include; this
                becomes the post-stimulus section of each measurement.

        Raises:
            ValueError: If light stimuli list is not set.

        Returns:
            list[PupilMeasurement]: List of individual PupilMeasurement objects, one per stimulus.
            Time 0 of each measurement is at its light offset.
        """
        if not hasattr(self, "light_stimuli_series"):
            raise ValueError("Light stimuli list is not set.")
        if pre_stimulus_duration < 0:
            warnings.warn(
                f"pre_stimulus_duration should be a positive number if time before the stimulus is included. Is {pre_stimulus_duration} correct?"
            )
        if post_stimulus_duration < 0:
            warnings.warn(
                f"post_stimulus_duration should be a positive number if time after the stimulus is included. Is {post_stimulus_duration} correct?"
            )
        series: list[PupilMeasurement] = []
        light_stimulus_list = self.light_stimuli_series.get_timings()
        time, size = self.get_time(), self.get_size()

        for light_start, light_end in light_stimulus_list:
            start = light_start - pre_stimulus_duration
            end = light_end + post_stimulus_duration

            series.append(PupilMeasurement(time.copy(), size.copy()))
            series[-1].trim_time(start, end, drop=True)
            series[-1].set_light_stimulus(light_start, light_end)
            if self.get_blinks():
                series[-1].set_blinks(self.get_blinks())
                series[-1].blinks.drop_blinks_out_of_time_range(start, end)
            series[-1].set_time_offset(-light_end)

        return series

    def plot_light_stimulus(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all light stimuli in the series.

        Args:
            ax (Axes | None, optional): Axes to plot on. If None, a new figure is created.
                Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to :meth:`LightStimulus.plot` for each stimulus. Only the first
                stimulus gets a legend label.

        Returns:
            Axes: The axes with the plotted light stimuli.
        """
        light_stimuli_series = self.get_light_stimuli_series()
        return light_stimuli_series.plot(ax, show, **kwargs)
