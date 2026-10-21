import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..preprocessing.filtering import PupilFilters
from ..utils.baseline import BaselineCorrection
from ..utils.blinks import PupilBlinks
from ..utils.light_stim import LightStimulus
from ..utils.utils import check_time_series, _removal_mask_in_window


class TimeSeriesBase(ABC):
    """Base class for anything with time and size data that can be plotted."""

    @abstractmethod
    def get_time(self) -> np.ndarray:
        """Get the time data.

        Returns:
            np.ndarray: Time points in seconds.
        """

    @abstractmethod
    def get_size(self) -> np.ndarray:
        """Get the pupil size data.

        Returns:
            np.ndarray: Pupil size at each time point.
        """

    def copy(self) -> Self:
        """Create a deep copy of this object.

        Returns:
            Self: A deep copy of the object.
        """
        return copy.deepcopy(self)

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = False,
        scatter: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the pupil size over time.

        Args:
            ax (Axes | None, optional): Matplotlib axes to plot on. If None, a new figure is
                created, with time and pupil size axis labels. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            scatter (bool, optional): Whether to draw a dot per sample instead of a line.
                Defaults to False.
            **kwargs: Passed to ``ax.plot``, or ``ax.scatter`` when ``scatter=True``, for
                example ``color`` or ``label``.

        Returns:
            Axes: The matplotlib axes object with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
            ax.set_xlabel("Time (s)")  # type: ignore
            ax.set_ylabel(self._y_label())  # type: ignore
        if scatter:
            ax.scatter(self.get_time(), self.get_size(), **kwargs)  # type: ignore
        else:
            ax.plot(self.get_time(), self.get_size(), **kwargs)  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax

    def _y_label(self) -> str:
        """Get the y-axis label for the plot.

        Returns:
            str: The y-axis label.
        """
        return "Pupil Diameter"


class PupilBase(TimeSeriesBase):
    """Base class for storing and accessing pupil time-series data."""

    def set_time_and_size(self, time_data: np.ndarray, size_data: np.ndarray) -> None:
        """Set time and size arrays with validation.

        Args:
            time_data (np.ndarray): Array of time data.
            size_data (np.ndarray): Array of size data.

        Raises:
            ValueError: If the time and size arrays do not have the same length.
            ValueError: If the time array contains NaNs, is not sorted, or contains duplicates.
        """
        if len(time_data) != len(size_data):
            raise ValueError("Time and size must have the same length")
        check_time_series(time_data)
        self.time_data = np.array(time_data, dtype=np.float64)
        self.size_data = np.array(size_data, dtype=np.float64)

    def get_time(self) -> np.ndarray:
        """Return the time of each sample.

        Returns:
            np.ndarray: 1-D float array of times in seconds, strictly increasing and without
            NaN. This is the stored array, not a copy.
        """
        return self.time_data

    def get_size(self) -> np.ndarray:
        """Return the pupil size of each sample.

        Returns:
            np.ndarray: 1-D float array with the same length as :meth:`get_time`. The unit is
            that of the loaded data (usually mm) or, after a baseline correction, relative
            to the baseline (see :meth:`get_baseline_correction`). Removed samples are NaN.
            This is the stored array, not a copy.
        """
        return self.size_data

    def get_samples(self) -> int:
        """Return the number of samples, NaN samples included.

        Returns:
            int: The number of samples.
        """
        return len(self.get_size())

    def get_nan(self) -> int:
        """Return the number of NaN samples in the pupil size data.

        Returns:
            int: The number of NaN samples.
        """
        return int(np.count_nonzero(np.isnan(self.get_size())))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"time_data.shape={getattr(self, 'time_data', np.array([])).shape}, "
            f"size_data.shape={getattr(self, 'size_data', np.array([])).shape}, "
            f"light_stimulus={getattr(self, 'light_stimulus', None)}, "
            f"blinks={self.get_blinks()}, "
            f"baseline_correction={self.get_baseline_correction() or 'none'})"
        )

    @property
    def filters(self) -> PupilFilters:
        """The preprocessing filters for this measurement.

        Each filter is a method, for example ``measurement.filters.rolling_mean(1)``.
        See :class:`~piprkit.preprocessing.filtering.PupilFilters` for all filters.
        """
        return PupilFilters(self)

    def get_baseline_correction(self) -> BaselineCorrection | None:
        """Get the baseline correction applied to the pupil measurement.

        Returns:
            BaselineCorrection | None: The baseline correction, or None if no correction has been applied.
        """
        return getattr(self, "baseline_correction", None)

    def _set_baseline_correction(self, correction: BaselineCorrection) -> None:
        self.baseline_correction = correction

    def _y_label(self) -> str:
        correction = self.get_baseline_correction()
        if correction is None:
            return super()._y_label()
        return {
            "ratio": "Pupil Diameter (ratio to baseline)",
            "subtractive": "Pupil Diameter (change from baseline)",
            "percent": "Pupil Diameter (% change from baseline)",
        }[correction.get_method()]

    def drop_nan(self, *, inplace: bool = True) -> Self | None:
        """Drop NaN values from the pupil measurement.

        Args:
            inplace (bool, optional): Whether to drop NaN values in place or return a new instance. Defaults to True.

        Returns:
            Self | None: The updated pupil measurement, or None if inplace=True.
        """
        if not inplace:
            measurement = self.copy()
            measurement.drop_nan()
            return measurement
        mask = ~np.isnan(self.get_size())
        self.set_time_and_size(
            self.get_time()[mask],
            self.get_size()[mask],
        )

    def trim_size(
        self,
        min_size: float = 2.0,
        max_size: float = 8.0,
        buffer_seconds: float = 0.0,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        drop: bool = False,
        inplace: bool = True,
    ) -> Self | None:
        """Remove samples with a pupil size outside ``[min_size, max_size]``.

        Out-of-range samples are set to NaN, together with the samples within
        ``buffer_seconds`` of them. Only samples between ``start_time`` and ``end_time`` are
        checked and removed. With ``drop=True`` all NaN samples are removed afterwards.
        The default limits assume a pupil diameter in mm; they don't fit baseline-corrected data.
        Kret & Sjak-Shie (2019) use 1.5 to 9 mm.

        Args:
            min_size (float, optional): Smallest pupil size to keep, in the unit of the pupil
                size data. Defaults to 2.0 (Watson & Yellott, 2012).
            max_size (float, optional): Largest pupil size to keep, in the unit of the pupil
                size data. Defaults to 8.0 (Watson & Yellott, 2012).
            buffer_seconds (float, optional): Time in seconds around each out-of-range sample
                that is also removed. Defaults to 0.0.
            start_time (float | None, optional): Start of the window in which samples are
                checked, in seconds. None means from the first sample. Defaults to None.
            end_time (float | None, optional): End of the window in which samples are checked,
                in seconds. None means up to the last sample. Defaults to None.
            drop (bool, optional): Whether to remove all NaN samples afterwards, including
                ones that were NaN before. Defaults to False.
            inplace (bool, optional): Whether to trim the data in place or return a new instance. Defaults to True.

        Returns:
            Self | None: The updated pupil measurement, or None if inplace=True.

        References:
            - Mathôt, S., Fabius, J., Van Heusden, E., & Van der Stigchel, S. (2018). Safe and sensible preprocessing and baseline correction of pupil-size data. Behavior Research Methods, 50(1), 94–106. https://doi.org/10.3758/s13428-017-1007-2
            - Watson, A. B., & Yellott, J. I. (2012). A unified formula for light-adapted pupil size. Journal of Vision, 12(10), 12. https://doi.org/10.1167/12.10.12
            - Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data: Guidelines and code. Behavior Research Methods, 51(3), 1336–1342. https://doi.org/10.3758/s13428-018-1075-y
        """
        if not inplace:
            measurement = self.copy()
            measurement.trim_size(
                min_size,
                max_size,
                buffer_seconds,
                start_time=start_time,
                end_time=end_time,
                drop=drop,
            )
            return measurement
        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        time_data = self.get_time()
        size = self.get_size()
        # NaN compares False, so samples that are already NaN are not "out of range".
        out_of_range = (size < min_size) | (size > max_size)
        remove_mask = _removal_mask_in_window(
            time_data, out_of_range, start_time, end_time, buffer_seconds
        )
        trimmed_size = size.copy()
        trimmed_size[remove_mask] = np.nan

        self.set_time_and_size(
            time_data,
            trimmed_size,
        )
        if drop:
            self.drop_nan()

    def trim_time(
        self, start: float, end: float, drop: bool = True, *, inplace: bool = True
    ) -> Self | None:
        """Keep only the samples between ``start`` and ``end``, inclusive.

        Samples outside the interval are set to NaN. With ``drop=True`` all NaN samples are
        then removed, so the time data and pupil size data get shorter. With ``drop=False``
        they keep their length. Blinks and the light stimulus are not changed.

        Args:
            start (float): Start of the interval to keep, in seconds.
            end (float): End of the interval to keep, in seconds.
            drop (bool, optional): Whether to remove all NaN samples afterwards, including
                ones that were NaN before. Defaults to True.
            inplace (bool, optional): Whether to trim the data in place or return a new instance. Defaults to True.

        Returns:
            Self | None: The updated pupil measurement, or None if inplace=True.
        """
        if not inplace:
            measurement = self.copy()
            measurement.trim_time(start, end, drop)
            return measurement
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

    def set_light_stimulus(self, start: float, end: float) -> None:
        """Set the light stimulus, replacing one set before.

        The light stimulus is the period in which the light that evokes the pupil response
        is on. The metrics and the baseline correction use it as their time reference.

        Args:
            start (float): Light onset, in seconds on the same time axis as :meth:`get_time`.
            end (float): Light offset, in seconds on the same time axis as :meth:`get_time`.
        """
        self.light_stimulus = LightStimulus(start_time=start, end_time=end)

    def get_light_stimulus(self) -> LightStimulus | None:
        """Get the light stimulus object.

        Returns:
            LightStimulus | None: The light stimulus object or None if not set.
        """
        if not hasattr(self, "light_stimulus"):
            warnings.warn("Light stimulus not set.")
            return None
        return self.light_stimulus

    def plot_light_stimulus(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the light stimulus period.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created.
                Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for the shaded area, see :meth:`LightStimulus.plot`.
                Defaults: label "Light Stimulus", the stimulus colour, alpha 0.3.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            Axes: The axes with the light stimulus plotted.
        """
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set.")
        return light_stimulus.plot(ax, show, **kwargs)

    def get_sections(self) -> dict[str, dict[str, float]]:
        """Get the pre-stimulus, during-stimulus and post-stimulus sections of the measurement.

        The light stimulus divides the measurement into three sections: from the first sample to light
        onset, from light onset to light offset, and from light offset to the last sample. See
        :meth:`LightStimulus.get_sections`.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            dict[str, dict[str, float]]: For each section, its ``start`` and ``end`` in seconds.
        """
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set.")
        time_data = self.get_time()
        return light_stimulus.get_sections(float(time_data[0]), float(time_data[-1]))

    def plot_sections(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the sections as a labelled band along the top of the axes.

        See :meth:`LightStimulus.plot_sections`.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created.
                Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for each section, see :meth:`LightStimulus.plot_sections`.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            Axes: The axes with the plotted sections.
        """
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set.")
        time_data = self.get_time()
        return light_stimulus.plot_sections(float(time_data[0]), float(time_data[-1]), ax, show, **kwargs)

    def set_blinks(self, blink_list: list[tuple[float, float]] | None) -> None:
        """Set the blinks, replacing any blinks set before.

        The blinks are only stored; the pupil size data is not changed. Use
        ``filters.remove_blinks()`` to set the samples during the blinks to NaN, and
        :meth:`plot_blinks` to show them.

        Args:
            blink_list (list[tuple[float, float]] | None): One ``(start, end)`` tuple per blink,
                in seconds on the same time axis as :meth:`get_time`. None removes all blinks.
        """
        self.blinks = PupilBlinks()
        self.blinks.set_blinks(blink_list)

    def get_blinks(self) -> list[tuple[float, float]]:
        """Return the blinks.

        Returns:
            list[tuple[float, float]]: One ``(start, end)`` tuple per blink, in seconds on the
            same time axis as :meth:`get_time`. Empty if no blinks are set.
        """
        if not hasattr(self, "blinks"):
            return []
        return self.blinks.get_blinks()

    def add_blinks(self, blink_list: list[tuple[float, float]]) -> None:
        """Add blinks to the blinks set before.

        Args:
            blink_list (list[tuple[float, float]]): One ``(start, end)`` tuple per blink,
                in seconds on the same time axis as :meth:`get_time`.
        """
        self.set_blinks(self.get_blinks() + list(blink_list))

    def plot_blinks(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all blinks.

        Args:
            ax (Axes | None, optional): Axes to plot on. If None, a new figure is created.
                Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for each blink. Defaults: color "red", alpha 0.3.

        Raises:
            ValueError: If no blinks are set.

        Returns:
            Axes: The axes with the plotted blinks.
        """
        if not hasattr(self, "blinks"):
            raise ValueError("Blinks not set.")
        return self.blinks.plot(ax, show, **kwargs)

    def set_time_offset(self, offset: float) -> None:
        """Shift the measurement in time by adding ``offset`` to every time point.

        The blinks and the light stimulus are shifted too, so they stay aligned with the
        pupil size data. Shifts add up: two calls with ``offset=1.0`` shift by 2 s in total.
        ``PupilSeries.split()`` uses this to put time 0 at each light offset.

        Args:
            offset (float): Time in seconds to add. A negative value shifts to earlier times.
        """
        self.set_time_and_size(self.get_time() + offset, self.get_size())
        if hasattr(self, "blinks"):
            self.blinks.set_time_offset(offset)
        self._offset_light_stim(offset)

    def _offset_light_stim(self, offset: float) -> None:
        """Offset the light stimulus (measurement) or light stimuli series (series), if set.

        Args:
            offset (float): The offset to add to the light stimulus times.
        """
        for attr in ("light_stimulus", "light_stimuli_series"):
            stimulus = getattr(self, attr, None)
            if stimulus is not None:
                stimulus.set_time_offset(offset)

    def interpolate(
        self, new_time: NDArray[np.number], *, inplace: bool = True
    ) -> Self | None:
        """Resample the pupil size at new time points by linear interpolation (``np.interp``).

        New time points outside the current time range get NaN. New points between a NaN
        sample and its neighbours are NaN too, so call :meth:`drop_nan` first to fill gaps.
        Blinks and the light stimulus are not changed.

        Args:
            new_time (NDArray[np.number]): 1-D array of the new time points in seconds,
                strictly increasing and without NaN.
            inplace (bool, optional): Whether to perform interpolation in place or return a new instance. Defaults to True.

        Returns:
            Self | None: The updated pupil measurement, or None if inplace=True.
        """
        if not inplace:
            measurement = self.copy()
            measurement.interpolate(new_time)
            return measurement
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
