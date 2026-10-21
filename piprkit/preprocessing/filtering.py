from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from piprkit.utils.baseline import BaselineCorrection
from piprkit.utils.utils import _removal_mask_in_window

if TYPE_CHECKING:
    from ..core.pupil_base import PupilBase


class PupilFilters:
    """The preprocessing filters of a pupil measurement or series.

    Use the ``filters`` property of a measurement or series rather than creating this class::

        measurement.filters.rolling_mean(0.1)
        measurement.filters.limit_rate_of_change(5.0)

    The filters change the pupil size data; the time data stays the same.

    - Removal filters set samples to NaN: :meth:`limit_rate_of_change`,
      :meth:`remove_constant_signal` and :meth:`remove_blinks`. Call ``drop_nan()`` on the
      measurement to remove the NaN samples afterwards.
    - Smoothing filters replace each sample by a value computed over a time window around it:
      :meth:`rolling_mean`, :meth:`rolling_median` and :meth:`rolling_filter`.
    - :meth:`apply_baseline_correction` expresses the size relative to the baseline.

    Each filter changes the measurement in place and returns None. Pass ``inplace=False`` to get
    a changed copy instead.
    """

    def __init__(self, measurement: PupilBase) -> None:
        """Use the ``filters`` property of a measurement instead of calling this.

        Args:
            measurement (PupilBase): The measurement or series to filter.
        """
        self._m = measurement

    def rolling_filter(
        self,
        func: Callable[[NDArray[np.number]], float],
        time_window: float,
        *,
        inplace: bool = True,
    ) -> PupilBase | None:
        """Replace each sample by ``func`` applied to the samples in a time window around it.

        The window is centred on each sample and ``time_window`` seconds wide: it holds the samples
        from t - time_window/2 to t + time_window/2. It is defined in time, not in number of samples,
        so uneven sampling works too. Near the start and end of the data the window is cut off and
        holds fewer samples. Every window uses the original pupil size data. The window can contain
        NaN samples, so ``func`` should handle NaN, like ``np.nanmean``. The time data is not changed.

        For example, ``measurement.filters.rolling_filter(np.nanmax, 0.2)`` replaces each sample by
        the largest pupil size within 0.1 s before or after it.

        Args:
            func (Callable[[NDArray[np.number]], float]): Takes the pupil sizes in the window as a 1-D
                array and returns one value.
            time_window (float): Width of the window, in seconds.
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If time_window is less than or equal to 0.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.rolling_filter(func, time_window)
            return measurement
        if time_window <= 0:
            raise ValueError(f"Time window must be positive, got {time_window}.")
        time_data = self._m.get_time()
        size = self._m.get_size()
        # The 1e-9 s keeps samples exactly on the window edge despite float rounding.
        half_window = time_window / 2 + 1e-9
        starts = np.searchsorted(time_data, time_data - half_window, side="left")
        ends = np.searchsorted(time_data, time_data + half_window, side="right")
        new_size = np.full_like(size, np.nan, dtype=np.float64)
        for i, (start, end) in enumerate(zip(starts, ends)):
            new_size[i] = func(size[start:end])

        self._m.set_time_and_size(time_data, new_size)

    def rolling_mean(
        self, time_window: float, *, inplace: bool = True
    ) -> PupilBase | None:
        """Applies a rolling mean filter to the pupil data.

        Uses :meth:`rolling_filter` with ``np.nanmean``, so NaN samples in the window are ignored.

        Args:
            time_window (float): The time window (in seconds) over which to apply the mean filter.
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.rolling_mean(time_window)
            return measurement
        self.rolling_filter(np.nanmean, time_window)

    def rolling_median(
        self, time_window: float, *, inplace: bool = True
    ) -> PupilBase | None:
        """Applies a rolling median filter to the pupil data.

        Uses :meth:`rolling_filter` with ``np.nanmedian``, so NaN samples in the window are ignored.

        Args:
            time_window (float): The time window (in seconds) over which to apply the median filter.
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.rolling_median(time_window)
            return measurement
        self.rolling_filter(np.nanmedian, time_window)

    def get_rate_of_change(self, n_back: int = 1) -> NDArray[np.number]:
        """Calculates the rate of change of the pupil size.

        Args:
            n_back (int, optional): The number of samples to look back for the rate of change calculation. Defaults to 1.

        Returns:
            NDArray[np.number]: The rate of change of the pupil size.
        """
        if n_back < 1:
            raise ValueError(f"n_back must be at least 1, got {n_back}.")
        time_data = self._m.get_time()
        size = self._m.get_size()

        dt = time_data[n_back:] - time_data[:-n_back]
        ds = size[n_back:] - size[:-n_back]
        rate_of_change = np.full_like(size, np.nan, dtype=np.float64)
        rate_of_change[n_back:] = ds / dt
        return rate_of_change

    def limit_rate_of_change(
        self,
        max_rate_of_change: float = 6.0,
        n_back: int = 1,
        buffer_seconds: float = 0.0,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        inplace: bool = True,
    ) -> PupilBase | None:
        """Removes samples where the pupil size changes faster than a given rate.

        At high sampling rates, smooth the data first, e.g. with ``rolling_mean(0.1)``.

        Args:
            max_rate_of_change (float, optional): The maximum allowed absolute rate of change per second. The size unit depends on the baseline correction. Defaults to 6.0 (Amiot et al., 2024).
            n_back (int, optional): The largest lag (in samples) to check. All lags from 1 to n_back are checked. Defaults to 1.
            buffer_seconds (float, optional): Additionally removes samples within this many seconds before and after any removed sample. The buffer does not extend outside [start_time, end_time]. Defaults to 0.0.
            start_time (float | None, optional): Only samples at or after this time are removed. Defaults to None (no lower limit).
            end_time (float | None, optional): Only samples at or before this time are removed. Defaults to None (no upper limit).
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.

        References:
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.limit_rate_of_change(
                max_rate_of_change,
                n_back,
                buffer_seconds,
                start_time=start_time,
                end_time=end_time,
            )
            return measurement
        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        time_data = self._m.get_time()

        roc_mask = np.full_like(self._m.get_size(), True, dtype=bool)

        for i in range(1, n_back + 1):
            rate_of_change = self.get_rate_of_change(n_back=i)
            # If rate_of_change is nan or zero, do nothing (keep True in mask)
            iteration_roc_mask = np.where(
                (np.isnan(rate_of_change) | np.isinf(rate_of_change)),
                True,
                np.abs(rate_of_change) <= max_rate_of_change,
            )
            roc_mask = roc_mask & iteration_roc_mask

        remove_mask = _removal_mask_in_window(
            time_data, ~roc_mask, start_time, end_time, buffer_seconds
        )
        size = self._m.get_size().copy()
        size[remove_mask] = np.nan
        self._m.set_time_and_size(time_data, size)

    def remove_constant_signal(
        self,
        min_duration: float,
        buffer_seconds: float = 0.0,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        inplace: bool = True,
    ) -> PupilBase | None:
        """Remove stretches where the pupil size stays exactly the same for at least ``min_duration`` seconds.

        A run of samples with equal pupil size that lasts at least ``min_duration`` (from its first to
        its last sample) is set to NaN, together with the samples within ``buffer_seconds`` of it. A
        NaN sample ends a run. Only samples between ``start_time`` and ``end_time`` are removed.

        Args:
            min_duration (float): Shortest run that is removed, in seconds. Must be positive.
            buffer_seconds (float, optional): Also removes samples within this many seconds before and
                after a removed run. The buffer does not extend outside [start_time, end_time].
                Defaults to 0.0.
            start_time (float | None, optional): Only samples at or after this time are removed.
                Defaults to None (no lower limit).
            end_time (float | None, optional): Only samples at or before this time are removed.
                Defaults to None (no upper limit).
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If min_duration is not positive.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.

        References:
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.remove_constant_signal(
                min_duration,
                buffer_seconds,
                start_time=start_time,
                end_time=end_time,
            )
            return measurement
        if min_duration <= 0:
            raise ValueError(f"min_duration must be positive, got {min_duration}.")
        if start_time is None:
            start_time = -np.inf
        if end_time is None:
            end_time = np.inf
        time_data = self._m.get_time()
        size = self._m.get_size().copy()

        # A run of equal values starts wherever the size differs from the previous sample.
        # NaN never equals anything, so NaNs end a run.
        run_starts = np.flatnonzero(np.concatenate(([True], size[1:] != size[:-1])))
        run_ends = np.append(run_starts[1:], len(size))
        constant_mask = np.zeros(len(size), dtype=bool)
        for start, end in zip(run_starts, run_ends):
            # The 1e-9 s keeps runs of exactly min_duration despite float rounding.
            if time_data[end - 1] - time_data[start] >= min_duration - 1e-9:
                constant_mask[start:end] = True

        remove_mask = _removal_mask_in_window(
            time_data, constant_mask, start_time, end_time, buffer_seconds
        )
        size[remove_mask] = np.nan
        self._m.set_time_and_size(time_data, size)

    def remove_blinks(
        self, buffer_seconds: float = 0.0, *, inplace: bool = True
    ) -> PupilBase | None:
        """Removes blinks from the pupil size data by setting them to NaN.

        Args:
            buffer_seconds (float, optional): Extra time in seconds removed before the start and after the end of each blink. Defaults to 0.0.
            inplace (bool, optional): If True, the filter is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.remove_blinks(buffer_seconds)
            return measurement
        blink_list = self._m.get_blinks()
        if blink_list is None:
            raise ValueError("No blinks have been set")
        size = self._m.get_size().copy()
        time_data = self._m.get_time()
        for blink_start, blink_end in blink_list:
            mask = (time_data >= blink_start - buffer_seconds) & (
                time_data <= blink_end + buffer_seconds
            )
            size[mask] = np.nan

        self._m.set_time_and_size(time_data, size)

    def apply_baseline_correction(
        self, duration: float = 10, method: str = "ratio", *, inplace: bool = True
    ) -> PupilBase | None:
        """Express the pupil size relative to the baseline before light onset.

        The baseline is ``metrics.baseline(duration)``: the average pupil size over the ``duration``
        seconds before light onset. ``method`` picks one of three corrections, each also available as
        its own method:

        - ``"ratio"``: size / baseline, see :meth:`baseline_correction_ratio`.
        - ``"subtractive"``: size - baseline, see :meth:`baseline_correction_subtractive`.
        - ``"percent"``: (size - baseline) / baseline * 100, see :meth:`baseline_correction_percent`.

        The correction is stored on the measurement and can be retrieved with
        ``get_baseline_correction()``. A measurement can be corrected only once.

        Args:
            duration (float, optional): The length of the baseline period before stimulus onset, in seconds. Defaults to 10.
            method (str, optional): "ratio", "subtractive" or "percent", see above. Defaults to "ratio".
            inplace (bool, optional): If True, the correction is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If an unknown method is provided, if the measurement is a series (use split() first), if it is already baseline corrected, or if the baseline period contains no valid data.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        corrections = {
            "ratio": self.baseline_correction_ratio,
            "subtractive": self.baseline_correction_subtractive,
            "percent": self.baseline_correction_percent,
        }
        if method not in corrections:
            raise ValueError(
                f"Unknown baseline correction method '{method}'. Options: {list(corrections)}"
            )
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.apply_baseline_correction(duration, method)
            return measurement
        corrections[method](duration)

    def baseline_correction_ratio(
        self, duration: float = 10, *, inplace: bool = True
    ) -> PupilBase | None:
        """Divides the pupil size by the baseline (size / baseline).

        Args:
            duration (float, optional): The length of the baseline period before stimulus onset, in seconds. Defaults to 10.
            inplace (bool, optional): If True, the correction is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If the measurement is a series (use split() first), if it is already baseline corrected, or if the baseline period contains no valid data.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.baseline_correction_ratio(duration)
            return measurement
        baseline = self._baseline_for_correction(duration)
        self._m.set_time_and_size(self._m.get_time(), self._m.get_size() / baseline)
        self._m._set_baseline_correction(
            BaselineCorrection("ratio", baseline, duration)
        )

    def baseline_correction_subtractive(
        self, duration: float = 10, *, inplace: bool = True
    ) -> PupilBase | None:
        """Subtracts the baseline from the pupil size (size - baseline).

        Args:
            duration (float, optional): The length of the baseline period before stimulus onset, in seconds. Defaults to 10.
            inplace (bool, optional): If True, the correction is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If the measurement is a series (use split() first), if it is already baseline corrected, or if the baseline period contains no valid data.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.baseline_correction_subtractive(duration)
            return measurement
        baseline = self._baseline_for_correction(duration)
        self._m.set_time_and_size(self._m.get_time(), self._m.get_size() - baseline)
        self._m._set_baseline_correction(
            BaselineCorrection("subtractive", baseline, duration)
        )

    def baseline_correction_percent(
        self, duration: float = 10, *, inplace: bool = True
    ) -> PupilBase | None:
        """Expresses the pupil size as percent change from baseline ((size - baseline) / baseline * 100).

        Args:
            duration (float, optional): The length of the baseline period before stimulus onset, in seconds. Defaults to 10.
            inplace (bool, optional): If True, the correction is applied in-place. If False, a new PupilBase object is returned. Defaults to True.

        Raises:
            ValueError: If the measurement is a series (use split() first), if it is already baseline corrected, or if the baseline period contains no valid data.

        Returns:
            PupilBase | None: The updated pupil measurement if inplace is False, otherwise None.
        """
        if not inplace:
            measurement = self._m.copy()
            measurement.filters.baseline_correction_percent(duration)
            return measurement
        baseline = self._baseline_for_correction(duration)
        self._m.set_time_and_size(
            self._m.get_time(), (self._m.get_size() - baseline) / baseline * 100
        )
        self._m._set_baseline_correction(
            BaselineCorrection("percent", baseline, duration)
        )

    def _baseline_for_correction(self, duration: float) -> float:
        if hasattr(self._m, "light_stimuli_series"):
            raise ValueError(
                "Baseline correction needs a single light stimulus. Use split() first."
            )
        existing = self._m.get_baseline_correction()
        if existing is not None:
            raise ValueError(f"Already baseline corrected ({existing.get_method()}).")
        baseline = float(self._m.metrics.baseline(duration))
        if np.isnan(baseline):
            raise ValueError("Baseline is NaN: no valid data in the baseline window.")
        return baseline
