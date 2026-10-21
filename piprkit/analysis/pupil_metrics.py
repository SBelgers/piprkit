from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn, Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ..utils.baseline import BaselineCorrection

if TYPE_CHECKING:
    from ..core.pupil_measurement import PupilMeasurement
    from .fitting.base_model import Phase
    from .fitting.phase_model import PhaseModel

# The superscript of a metric name per baseline correction method; mm when the data are not corrected
_CORRECTION_SYMBOLS = {None: "mm", "ratio": "R", "subtractive": "S", "percent": "%"}
# The time window in a metric name, such as "_5.95-6.05" or "_-10-0"
_WINDOW = re.compile(r"_(-?[\d.]+)-(-?[\d.]+)")


class MetricResult(float):
    """A metric value that also records how it was computed.

    A MetricResult can be used as a float: ``float(result)``, arithmetic and comparisons work as
    usual, and arithmetic gives a plain float. It also records the name of the metric, the baseline
    correction of the data, the flags of a fitted phase, and where on the trace the metric was
    measured; read them with the get methods, and draw the location with :meth:`plot`.

    Args:
        value (float): The metric value.
        baseline_correction (BaselineCorrection | None, optional): See :meth:`get_baseline_correction`.
            Defaults to None.
        name (str | None, optional): The name of the metric without its superscript, such as
            "6s-PIPR_5.95-6.05"; see :meth:`get_name`. Defaults to None.
        time_data (ArrayLike, optional): Times in seconds where the metric was measured. Defaults to ().
        size_data (ArrayLike, optional): Pupil size at each of those times. Defaults to ().
        flags (dict[str, Any] | None, optional): See :meth:`get_flags`. Defaults to None.
        area_level (float | None, optional): The level an area is measured from, such as the baseline
            of an AUC. If given, :meth:`plot` shades the area between this level and the pupil size.
            Defaults to None.
    """

    _baseline_correction: BaselineCorrection | None
    _base_name: str | None
    _name: str | None
    _flags: dict[str, Any] | None
    _time_data: np.ndarray
    _size_data: np.ndarray
    _area_level: float | None

    def __new__(
        cls,
        value: float,
        baseline_correction: BaselineCorrection | None = None,
        name: str | None = None,
        time_data: ArrayLike = (),
        size_data: ArrayLike = (),
        flags: dict[str, Any] | None = None,
        area_level: float | None = None,
    ) -> Self:
        result = super().__new__(cls, value)
        result._baseline_correction = baseline_correction
        result._base_name = name
        method = None if baseline_correction is None else baseline_correction.get_method()
        result._name = None if name is None else f"{name}^{_CORRECTION_SYMBOLS[method]}"
        result._flags = flags
        result._time_data = np.asarray(time_data, dtype=np.float64)
        result._size_data = np.asarray(size_data, dtype=np.float64)
        result._area_level = area_level
        return result

    def get_name(self, with_correction: bool = True) -> str | None:
        """Return the name of the metric, such as ``6s-PIPR_5.95-6.05^R`` or ``AUC_0-10^mm``.

        The name has the form <metric>_<window>^<correction> (Belgers et al., 2026). The subscript
        after "_" is the time window of the metric in seconds from light onset or light offset, e.g.
        ``PIPR_10-30`` or ``Baseline_-10-0``; a metric without a window has none, e.g. ``PeakConstriction``.
        A time point or level goes in front of the metric, e.g. ``6s-PIPR`` or ``1%-PLRLatency``. The
        superscript after "^" is the baseline correction of the data: "R" (ratio), "S" (subtractive),
        "%" (percent), or "mm" when the data are not corrected. It is added when the result is created.
        :meth:`get_label` gives the name typeset for plots.

        Args:
            with_correction (bool, optional): Whether to include the superscript. Defaults to True.

        Returns:
            str | None: The name of the metric, or None if it has none.

        References:
            - Belgers, S., Cuijpers, R. H., & Heynderickx, I. (2026). Review of methodology, preprocessing, and metrics for the post-illumination pupil response to measure non-image forming effects [Manuscript submitted for publication].
        """
        return self._name if with_correction else self._base_name

    def get_label(self) -> str | None:
        """Return the name of the metric typeset for plots, with the window as a subscript and the baseline correction as a superscript.

        :meth:`plot` uses it as the default label. Use it in other labels and titles, e.g.
        ``label=f"{result.get_label()} (data)"``.

        Returns:
            str | None: The name in matplotlib mathtext, or None if the metric has no name.
        """
        if self._name is None:
            return None
        name, _, symbol = self._name.rpartition("^")
        superscript = r"\%" if symbol == "%" else rf"\mathrm{{{symbol}}}"
        label = _WINDOW.sub(r"${}_{\\mathrm{\1\\text{-}\2}}$", name)
        if label.endswith("}$"):
            return f"{label[:-1]}^{{{superscript}}}$"
        return f"{label}${{}}^{{{superscript}}}$"

    def get_baseline_correction(self) -> BaselineCorrection | None:
        """Return the baseline correction of the data the metric was computed on.

        It tells the unit of the value, e.g. a fraction of the baseline on ratio-corrected data.

        Returns:
            BaselineCorrection | None: The baseline correction, or None if the data were not corrected.
        """
        return self._baseline_correction

    def get_flags(self) -> dict[str, Any] | None:
        """Return the flags of the fitted phase a metric comes from, see ``PhaseModel.get_flags``.

        Returns:
            dict[str, Any] | None: A copy of the flags, or None for a metric computed from data.
        """
        return None if self._flags is None else dict(self._flags)

    def get_time(self) -> np.ndarray:
        """Return the times that show where the metric was measured.

        Returns:
            np.ndarray: 1-D array of times in seconds: one point (e.g. the peak), two (the ends of a
            window or a level), the valid samples in a window (an AUC), or a whole curve (a fitted
            phase). Empty if there is no location.
        """
        return self._time_data

    def get_size(self) -> np.ndarray:
        """Return the pupil size at each time of :meth:`get_time`.

        Returns:
            np.ndarray: 1-D array with the same length as get_time, in the units of the data: the
            measured size, or the level the metric refers to, such as the baseline.
        """
        return self._size_data

    def __repr__(self) -> str:
        return f"MetricResult({self.__str__()})"

    def __str__(self) -> str:
        baseline_str = (
            f"baseline correction: {self._baseline_correction}"
            if self._baseline_correction is not None
            else "baseline correction: none"
        )
        timing_str = (
            f"[start, end: {self._time_data[0]:.4g}, {self._time_data[-1]:.4g} s]"
            if self._time_data.size > 0
            else "[no timing data]"
        )
        name_str = "" if self._name is None else f"{self._name} = "
        return f"{name_str}{float(self):.4g} {timing_str} [{baseline_str}]"

    def plot(self, ax: Axes | None = None, show: bool = False, **kwargs: Any) -> Axes:
        """Plot where on the trace the metric was measured.

        A result with one or two points (a point, a window or a segment) is drawn with markers by
        default. A longer result, such as a fitted curve, is drawn as a line. An area, such as an
        AUC, is shaded between its level and the pupil size.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.plot``, or to ``ax.fill_between`` for an area. Defaults: the
                typeset metric name (:meth:`get_label`) as label, marker "o" for at most two points,
                and alpha 0.3 for an area.

        Returns:
            Axes: The axes with the plotted metric.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        kwargs.setdefault("label", self.get_label())
        if self._area_level is not None:
            kwargs.setdefault("alpha", 0.3)
            ax.fill_between(self.get_time(), self._area_level, self.get_size(), **kwargs)  # type: ignore
        else:
            if self.get_time().size <= 2:
                kwargs.setdefault("marker", "o")
            ax.plot(self.get_time(), self.get_size(), **kwargs)  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax


class PupilMetrics:
    """
    Class for pupil metrics on a PupilMeasurement.

    Use the ``metrics`` property of a measurement rather than creating this class::

        measurement.metrics.baseline()
        measurement.metrics.peak_constriction()
        measurement.metrics.pipr_6s()
    """

    def __init__(self, measurement: PupilMeasurement) -> None:
        """Use the ``metrics`` property of a measurement instead of calling this.

        Args:
            measurement (PupilMeasurement): The measurement to compute the metrics on.
        """
        self._m = measurement

    def baseline(self, duration: float = 10) -> MetricResult:
        """Baseline pupil size: the average pupil size before light onset.

        Adhikari, Zele & Feigl (2015), Table 2: "Average over 10 s prestimulus period". The value is the
        mean of the valid samples in [light onset - duration, light onset], in the units of the data.

        Args:
            duration (float, optional): Length of the window, in seconds before light onset.
                Defaults to 10 (Adhikari et al., 2015, Table 2).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The average pupil size in the window. NaN, with a warning, if the window
            has no samples. Warns when the window starts before the first sample.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
        """
        onset, _ = self._stimulus_timing("baseline")
        start_time, end_time = onset - duration, onset
        value = float(self.get_average_size(start_time, end_time))
        return MetricResult(
            value,
            self._m.get_baseline_correction(),
            f"Baseline_{-duration:g}-0",
            (start_time, end_time),
            (value, value),
        )

    def transient_plr(self, start: float = 0.18, end: float = 0.5, peak_constriction_window: float = 1.0) -> MetricResult:
        """Transient PLR: the smallest pupil size shortly after light onset.

        Adhikari, Zele & Feigl (2015), Table 2: "Peak change from 180 to 500 ms after light onset". The
        value is the smallest size in [onset + start, onset + end], on the data as they are: on
        percent-corrected data it is the percent change from baseline (negative for a constriction).

        The default window is based on the 180-500 ms window of Adhikari et al. (2015, Table 2). This is an error-prone estimate.

        Args:
            start (float, optional): Start of the window, in seconds after light onset. Defaults to 0.18
                (Adhikari et al., 2015, Table 2).
            end (float, optional): End of the window, in seconds after light onset. Defaults to 0.5
                (Adhikari et al., 2015, Table 2).
            peak_constriction_window (float, optional): End of the search for a smaller size after the
                window, in seconds after light offset; the same end as the search of
                ``peak_constriction``. Defaults to 1.0 (Adhikari et al., 2015; Amiot et al., 2024).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The smallest pupil size in the window. NaN, with a warning, if the window has
            no valid samples. Warns when the pupil is smaller after the window than in it, or else when
            the minimum is the first or the last valid sample of the window, since the peak may then lie
            outside it.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        onset, offset = self._stimulus_timing("transient_plr")
        name = f"TransientPLR_{start:g}-{end:g}"
        time_data, size = self._m.get_time(), self._m.get_size()
        measured = ~np.isnan(size)
        valid = (time_data >= onset + start) & (time_data <= onset + end) & measured
        later = (time_data > onset + end) & (time_data <= offset + peak_constriction_window) & measured
        if not np.any(valid):
            warnings.warn(
                "No valid samples in the transient_plr window, so transient_plr is NaN."
            )
            return MetricResult(np.nan, self._m.get_baseline_correction(), name)
        later_time, later_size = time_data[later], size[later]
        time_data, size = time_data[valid], size[valid]
        minimum = int(np.argmin(size))
        if later_size.size and later_size.min() < size[minimum]:
            smallest = int(np.argmin(later_size))
            warnings.warn(
                f"The pupil is smaller after the transient_plr window ({later_size[smallest]:.4g} at "
                f"{later_time[smallest] - onset:.3g} s after light onset) than in it ({size[minimum]:.4g} at "
                f"{time_data[minimum] - onset:.3g} s), so the constriction may peak after the window. "
                "Pass a later end to include the peak."
            )
        elif minimum in (0, size.size - 1):
            warnings.warn(
                f"The transient_plr minimum is at the edge of its window (t = {time_data[minimum]:.4g} s), "
                "so the peak may lie outside it."
            )
        return MetricResult(
            float(size[minimum]),
            self._m.get_baseline_correction(),
            name,
            (time_data[minimum],),
            (size[minimum],),
        )

    def plr_latency(self, constriction_percent: float = 1.0, warn_latency: float = 0.1) -> MetricResult:
        """PLR latency: the time from light onset until the pupil has constricted by a given percentage.

        Adhikari, Zele & Feigl (2015), Table 2: "Time for 1% constriction". The latency is taken at the
        first sample at or after light onset whose size is at most (100 - constriction_percent)% of the
        baseline. The baseline is the one stored with the baseline correction, or ``baseline()`` when the
        data are not corrected. On percent-corrected data the level is in percentage points, and on
        subtractive-corrected data in the units of the data. These three cases warn.

        In noisy data, the noise can cross the 1% level before the response starts. A larger
        ``constriction_percent`` makes this less likely, and ``warn_latency`` flags very short latencies.

        Args:
            constriction_percent (float, optional): The constriction, in percent of the baseline, that
                ends the latency. Defaults to 1.0 (Adhikari et al., 2015, Table 2).
            warn_latency (float, optional): Warn if the latency is shorter than this many seconds as
                a sign of potential noise or baseline drift. Defaults to 0.1 (Lobato-Rincón et al., 2014).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The latency, in seconds. NaN, with a warning, if the pupil never constricts
            that far after light onset. Warns when the latency is shorter than ``warn_latency``.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Lobato-Rincón, L.-L., Cabanillas-Campos, M. del C., Bonnin-Arias, C., Chamorro-Gutiérrez, E., Murciano-Cespedosa, A., & Sánchez-Ramos Roda, C. (2014). Pupillary behavior in relation to wavelength and age. Frontiers in Human Neuroscience, 8, 221. https://doi.org/10.3389/fnhum.2014.00221
        """
        onset, _ = self._stimulus_timing("plr_latency")
        level = self._percent_of_baseline(100 - constriction_percent, "plr_latency")
        time_data, size = self._m.get_time(), self._m.get_size()
        reached = np.flatnonzero((time_data >= onset) & (size <= level))
        crossing = float(time_data[reached[0]]) if reached.size else np.nan
        if np.isnan(crossing):
            warnings.warn(
                f"The pupil does not constrict by {constriction_percent:g}% after light onset, so plr_latency is NaN."
            )
        elif crossing - onset < warn_latency:
            warnings.warn(
                f"plr_latency is {crossing - onset:.3g} s, shorter than {warn_latency:g} s: "
                "noise or a baseline drift may have crossed the level before the response."
            )
        return MetricResult(
            crossing - onset,
            self._m.get_baseline_correction(),
            f"{constriction_percent:g}%-PLRLatency",
            (onset, crossing),
            (level, level),
        )

    def constriction_v(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Not available on a measurement: the constriction velocity is read from a fitted model.

        Use ``PhaseModel(measurement).metrics.constriction_v()``, see ``PhaseModelMetrics``.

        Raises:
            AttributeError: Always, with a pointer to the model.
        """
        raise self._fit_metric_error("constriction_v")

    def peak_constriction(self, peak_constriction_window: float = 1.0) -> MetricResult:
        """Peak constriction amplitude: the smallest pupil size after light onset.

        Adhikari, Zele & Feigl (2015), Table 2: "Minimum pupil size during light presentation (% of
        baseline)". After a short stimulus the pupil keeps constricting past light offset, so the
        search runs until light offset + ``peak_constriction_window``.
        The value is in the units of the data. ``time_to_peak`` uses the same peak.

        Args:
            peak_constriction_window (float, optional): End of the search window, in seconds after
                light offset. Defaults to 1.0 (Adhikari et al., 2015; Amiot et al., 2024).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The smallest pupil size in [light onset, light offset + peak_constriction_window].
            NaN, with a warning, if the window has no valid samples. Warns when the peak is the last
            valid sample of the window.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Adhikari, P., Pearson, C. A., Anderson, A. M., Zele, A. J., & Feigl, B. (2015). Effect of Age and Refractive Error on the Melanopsin Mediated Post-Illumination Pupil Response (PIPR). Scientific Reports, 5(1), 17610. https://doi.org/10.1038/srep17610
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        _, peak_time, peak_size = self._peak("peak_constriction", peak_constriction_window)
        return MetricResult(
            peak_size,
            self._m.get_baseline_correction(),
            "PeakConstriction",
            (peak_time,),
            (peak_size,),
        )

    def time_to_peak(self, peak_constriction_window: float = 1.0) -> MetricResult:
        """Time to peak: the time from light onset to the peak constriction.

        Adhikari, Zele & Feigl (2015), Table 2: "Time to peak pupil constriction". The peak is the one
        of ``peak_constriction``: the smallest pupil size in [light onset, light offset +
        ``peak_constriction_window``].

        Args:
            peak_constriction_window (float, optional): End of the search window, in seconds after
                light offset. Defaults to 1.0 (Adhikari et al., 2015; Amiot et al., 2024).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The time to peak, in seconds. NaN, with a warning, if the window has no valid
            samples. Warns when the peak is the last valid sample of the window.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Adhikari, P., Pearson, C. A., Anderson, A. M., Zele, A. J., & Feigl, B. (2015). Effect of Age and Refractive Error on the Melanopsin Mediated Post-Illumination Pupil Response (PIPR). Scientific Reports, 5(1), 17610. https://doi.org/10.1038/srep17610
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        onset, peak_time, peak_size = self._peak("time_to_peak", peak_constriction_window)
        return MetricResult(
            peak_time - onset,
            self._m.get_baseline_correction(),
            "TimeToPeak",
            (onset, peak_time),
            (peak_size, peak_size),
        )

    def pupil_escape(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Not available on a measurement: the pupil escape is read from a fitted model.

        Use ``PhaseModel(measurement).metrics.pupil_escape()``, see ``PhaseModelMetrics``.

        Raises:
            AttributeError: Always, with a pointer to the model.
        """
        raise self._fit_metric_error("pupil_escape")

    def redilation_v(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Not available on a measurement: the redilation velocity is read from a fitted model.

        Use ``PhaseModel(measurement).metrics.redilation_v()``, see ``PhaseModelMetrics``.

        Raises:
            AttributeError: Always, with a pointer to the model.
        """
        raise self._fit_metric_error("redilation_v")

    def pipr_xs(self, start: float, end: float) -> MetricResult:
        """The x second PIPR amplitude between start and end time after light offset.

        Args:
            start (float): Start of the window, in seconds after light offset.
            end (float): End of the window, in seconds after light offset.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The average pupil size in the window.
        """
        return self._pipr("pipr_xs", start, end, f"PIPR_{start:g}-{end:g}")

    def pipr_6s(self) -> MetricResult:
        """6 s PIPR: the average pupil size from 5.95 to 6.05 s after light offset.

        Adhikari, Zele & Feigl (2015), Table 2: "Pupil size at 6 s after light offset (% of baseline)".
        The value is ``pipr_xs(5.95, 6.05)``, in the units of the data, e.g. a fraction of the baseline
        on ratio-corrected data. The 0.1 s window follows Amiot et al. (2024). Kelbsch et al. (2019)
        use a 1 s window around 6 s.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The average pupil size in the window.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
            - Kelbsch, C., Strasser, T., Chen, Y., Feigl, B., Gamlin, P. D., Kardon, R., Peters, T., Roecklein, K. A., Steinhauer, S. R., Szabadi, E., Zele, A. J., Wilhelm, H., & Wilhelm, B. J. (2019). Standards in pupillography. Frontiers in Neurology, 10, 129. https://doi.org/10.3389/fneur.2019.00129
        """
        return self._pipr("pipr_6s", 5.95, 6.05, "6s-PIPR_5.95-6.05")

    def plateau_pipr(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Not available on a measurement: the plateau PIPR is read from a fitted model.

        Use ``PhaseModel(measurement).metrics.plateau_pipr()``, see ``PhaseModelMetrics``.

        Raises:
            AttributeError: Always, with a pointer to the model.
        """
        raise self._fit_metric_error("plateau_pipr")

    def auc(self, start: float, end: float, baseline_duration: float = 10) -> MetricResult:
        """Area between the baseline and the pupil size, from start to end seconds after light offset.

        Adhikari et al. (2015), Table 2: sum(BPD - APD), with BPD the baseline pupil diameter and APD
        the actual pupil diameter.

        - The sum is taken as an integral over time with the trapezoid rule, so the value is in size·s
          and does not depend on the sampling rate.
        - BPD is ``baseline(baseline_duration)`` of the data themselves, so any baseline correction,
          or none, works.
        - NaN samples are left out, so a gap is bridged by a straight line.

        Args:
            start (float): Start of the window, in seconds after light offset.
            end (float): End of the window, in seconds after light offset.
            baseline_duration (float, optional): Length of the baseline window before light onset, in
                seconds. Defaults to 10 (Adhikari et al., 2015, Table 2).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The area, in size·s; positive when the pupil is smaller than its baseline.
            NaN, with a warning, if the window has fewer than 2 valid samples. Warns when the
            window runs past the data, and when the pupil is larger than the baseline in part of
            the window, as that part counts negative.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
        """
        _, offset = self._stimulus_timing("auc")
        bpd = float(self.baseline(baseline_duration))
        start_time, end_time = offset + start, offset + end
        time_data, size = self._m.get_time(), self._m.get_size()
        if time_data[0] > start_time:
            warnings.warn("Start time is before the first data point.")
        if time_data[-1] < end_time:
            warnings.warn("End time is after the last data point.")
        valid = (time_data >= start_time) & (time_data <= end_time) & ~np.isnan(size)
        if np.sum(valid) < 2:
            warnings.warn(
                "Fewer than 2 valid samples in the auc window, so auc is NaN."
            )
            value = np.nan
        else:
            difference = bpd - size[valid]
            value = float(np.trapezoid(difference, time_data[valid]))
            if np.any(difference < 0):
                negative = float(np.trapezoid(np.minimum(difference, 0), time_data[valid]))
                warnings.warn(
                    f"The pupil is larger than the baseline ({bpd:.4g}) in part of the auc window "
                    f"({start:g} to {end:g} s after light offset), and that part adds {negative:.4g} to auc."
                )
        return MetricResult(
            value,
            self._m.get_baseline_correction(),
            f"AUC_{start:g}-{end:g}",
            time_data[valid],
            size[valid],
            area_level=bpd,
        )

    def auc_early(self, baseline_duration: float = 10) -> MetricResult:
        """AUC early: the area between the baseline and the pupil size from 0 to 10 s after light offset.

        Adhikari et al. (2015), Table 2: "sum(BPD - APD) over 0-10 s after light offset". See ``auc``.

        Args:
            baseline_duration (float, optional): Length of the baseline window before light onset, in
                seconds. Defaults to 10 (Adhikari et al., 2015, Table 2).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: ``auc(0, 10)``, in size·s.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
        """
        return self.auc(0, 10, baseline_duration)

    def auc_late(self, baseline_duration: float = 10) -> MetricResult:
        """AUC late: the area between the baseline and the pupil size from 10 to 30 s after light offset.

        Adhikari et al. (2015), Table 2: "sum(BPD - APD) over 10-30 s after light offset". See ``auc``.

        Args:
            baseline_duration (float, optional): Length of the baseline window before light onset, in
                seconds. Defaults to 10 (Adhikari et al., 2015, Table 2).

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: ``auc(10, 30)``, in size·s.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
        """
        return self.auc(10, 30, baseline_duration)

    def pipr_duration(self, baseline_percent: float = 99.0) -> MetricResult:
        """PIPR duration: the time from light offset until the pupil has returned to a percentage of its baseline.

        Adhikari, Zele & Feigl (2015), Table 2: "Time to return to baseline after light offset". The
        return is taken at the first sample at or after light offset whose size is at least
        baseline_percent % of the baseline. The baseline is the one stored with the baseline correction,
        or ``baseline()`` when the data are not corrected. On percent-corrected data the level is in
        percentage points, and on subtractive-corrected data in the units of the data. These three cases
        warn.

        In noisy data, a noise peak can reach the level before the pupil has returned. A smaller
        ``baseline_percent`` makes this less likely.

        Args:
            baseline_percent (float, optional): The size, in percent of the baseline, that counts as
                returned. Defaults to 99.0. Table 2 says "return to baseline" (100), which a
                redilation that only approaches the baseline never reaches.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            MetricResult: The duration, in seconds. NaN, with a warning, if the pupil does not return
            to that level before the end of the data.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
        """
        _, offset = self._stimulus_timing("pipr_duration")
        level = self._percent_of_baseline(baseline_percent, "pipr_duration")
        time_data, size = self._m.get_time(), self._m.get_size()
        reached = np.flatnonzero((time_data >= offset) & (size >= level))
        returned = float(time_data[reached[0]]) if reached.size else np.nan
        if np.isnan(returned):
            warnings.warn(
                f"The pupil does not return to {baseline_percent:g}% of baseline before the end of the data, "
                "so pipr_duration is NaN."
            )
        return MetricResult(
            returned - offset,
            self._m.get_baseline_correction(),
            f"{baseline_percent:g}%-PIPRDuration",
            (offset, returned),
            (level, level),
        )

    def get_average_size(self, start_time: float, end_time: float) -> float:
        """Calculate average pupil size within a time range.

        Args:
            start_time (float): Start time for averaging window
            end_time (float): End time for averaging window

        Raises:
            ValueError: If no data points are found in the specified time range.

        Returns:
            float: Average pupil size in the specified time range
        """
        time_data = self._m.get_time()
        size_data = self._m.get_size()
        if min(time_data) > start_time:
            warnings.warn("Start time is before the first data point.")
        if max(time_data) < end_time:
            warnings.warn("End time is after the last data point.")

        mask = (time_data >= start_time) & (time_data <= end_time)
        if not np.any(mask):
            warnings.warn("No data points in the specified time range.")
            return np.nan
        return np.nanmean(size_data[mask])  # type: ignore

    def _stimulus_timing(self, metric: str) -> tuple[float, float]:
        stimulus = self._m.get_light_stimulus()
        if stimulus is None:
            raise ValueError(f"Light stimulus not set. Cannot calculate {metric}.")
        return stimulus.get_timing()

    def _pipr(self, metric: str, start: float, end: float, name: str) -> MetricResult:
        _, stimulus_end = self._stimulus_timing(metric)
        start_time, end_time = stimulus_end + start, stimulus_end + end
        value = float(self.get_average_size(start_time, end_time))
        return MetricResult(
            value,
            self._m.get_baseline_correction(),
            name,
            (start_time, end_time),
            (value, value),
        )

    def _percent_of_baseline(self, percent: float, metric: str) -> float:
        correction = self._m.get_baseline_correction()
        if correction is None:
            baseline = float(self.baseline())
            warnings.warn(
                f"{metric}: the data are not baseline corrected, so the level is {percent:g}% of "
                f"metrics.baseline() ({baseline:.4g})."
            )
            return percent / 100 * baseline
        method = correction.get_method()
        if method == "ratio":
            return percent / 100
        if method == "percent":
            level = percent - 100
            warnings.warn(
                f"{metric}: the data are percent corrected, so the level is {level:g} percentage point "
                f"({percent:g}% of the stored baseline)."
            )
            return level
        baseline = correction.get_baseline()
        level = (percent / 100 - 1) * baseline
        warnings.warn(
            f"{metric}: the data are subtractive corrected, so the level is {percent:g}% of the stored "
            f"baseline ({baseline:.4g}): {level:.4g}."
        )
        return level

    def _peak(self, metric: str, peak_constriction_window: float) -> tuple[float, float, float]:
        onset, offset = self._stimulus_timing(metric)
        end = offset + peak_constriction_window
        time_data, size = self._m.get_time(), self._m.get_size()
        valid = (time_data >= onset) & (time_data <= end) & ~np.isnan(size)
        if not np.any(valid):
            warnings.warn(
                "No valid samples between light onset and end, so the peak is NaN."
            )
            return onset, np.nan, np.nan
        time_data, size = time_data[valid], size[valid]
        peak = int(np.argmin(size))
        if peak == size.size - 1:
            warnings.warn(
                f"The peak is the last valid sample before end (t = {time_data[peak]:.4g} s), "
                "so the pupil may still be constricting. Pass a larger peak_constriction_window."
            )
        return onset, float(time_data[peak]), float(size[peak])

    @staticmethod
    def _fit_metric_error(name: str) -> AttributeError:
        return AttributeError(
            f"{name} comes from a fit. Use the model:\n"
            f"    model = piprkit.PhaseModel(pm)\n"
            f"    model.metrics.{name}()"
        )


class PhaseModelMetrics:
    """Metrics read from a fitted PhaseModel.

    Use the ``metrics`` property of a fitted PhaseModel rather than creating this class::

        model = PhaseModel(measurement)
        model.metrics.constriction_v()
        model.metrics.plateau_pipr()

    Each metric is a parameter of one phase of the model. Metrics computed from data
    (``pipr_6s``, ``baseline``, ...) are not available here: use the measurement's ``.metrics``, or
    ``model.to_measurement().metrics`` for the fitted curve.

    References:
        - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
    """

    def __init__(self, model: PhaseModel) -> None:
        """Use the ``metrics`` property of a fitted PhaseModel instead of calling this.

        Args:
            model (PhaseModel): The fitted model to read the metrics from.
        """
        self._model = model

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("_") and hasattr(PupilMetrics, name):
            raise AttributeError(
                f"{name} is computed from data. Use pm.metrics.{name}(), "
                f"or model.to_measurement().metrics.{name}() for the fitted curve."
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def constriction_v(self) -> MetricResult:
        """Constriction velocity: the slope m of the constriction phase.

        Adhikari et al. (2015), Table 2: "Stimulus gradient of linear model at light onset". The
        value is in the units of the data per second (e.g. mm/s, or %/s on percent-corrected data),
        and is negative for a constriction.

        Raises:
            ValueError: If the model was fitted without the constriction phase.

        Returns:
            MetricResult: The slope m. NaN, with a warning, when the phase has no fit;
            ``result.get_flags()`` holds the flags of the phase.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return self._phase_metric(
            "constriction", "m", "constriction_v", "ConstrictionVelocity", self._fitted_line
        )

    def pupil_escape(self) -> MetricResult:
        """Pupil escape: the slope m of the maintained constriction phase.

        Adhikari et al. (2015), Table 2: "Stimulus gradient of linear model during light
        stimulation". The value is in the units of the data per second, and is positive when the
        pupil redilates while the light is still on.

        Raises:
            ValueError: If the model was fitted without the maintained_constriction phase.

        Returns:
            MetricResult: The slope m. NaN, with a warning, when the phase has no fit;
            ``result.get_flags()`` holds the flags of the phase.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return self._phase_metric(
            "maintained_constriction", "m", "pupil_escape", "PupilEscape", self._fitted_line
        )

    def redilation_v(self) -> MetricResult:
        """Redilation velocity: the rate constant k of the redilation phase.

        Adhikari et al. (2015), Table 2: "Global rate constant (k) of exponential model". k is in
        1/s. The fit bounds it to k <= 0 (``redilation_k_bounds``).

        Raises:
            ValueError: If the model was fitted without the redilation phase.

        Returns:
            MetricResult: The rate constant k. NaN, with a warning, when the phase has no fit;
            ``result.get_flags()`` holds the flags of the phase.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return self._phase_metric(
            "redilation", "k", "redilation_v", "RedilationVelocity", self._fitted_curve
        )

    def plateau_pipr(self) -> MetricResult:
        """Plateau PIPR: the plateau P of the redilation phase.

        Adhikari et al. (2015), Table 2: "Plateau of exponential model". The value is in the units of
        the data, e.g. a ratio of the baseline on ratio-corrected data.

        Raises:
            ValueError: If the model was fitted without the redilation phase.

        Returns:
            MetricResult: The plateau P. NaN, with a warning, when the phase has no fit;
            ``result.get_flags()`` holds the flags of the phase.

        References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return self._phase_metric(
            "redilation", "P", "plateau_pipr", "PlateauPIPR", self._level
        )

    def _phase_metric(
        self,
        phase: str,
        param: str,
        metric: str,
        name: str,
        plot_data: Callable[[Phase, float], tuple[np.ndarray, np.ndarray]],
    ) -> MetricResult:
        phase_models = self._model.get_phases()
        if phase not in phase_models:
            raise ValueError(
                f"{metric} needs the {phase} phase, which the model does not have. Phases: {list(phase_models)}"
            )
        phase_model = phase_models[phase]
        value = float(dict(zip(phase_model.get_param_names(), phase_model.get_params()))[param])
        flags = phase_model.get_flags()
        if np.isnan(value):
            set_flags = [
                flag
                for flag in ("not_defined", "not_converged", "too_few_samples")
                if flags[flag]
            ]
            reason = f" ({flags['reason']})" if flags["reason"] else ""
            warnings.warn(
                f"{metric} is NaN: {phase} {', '.join(set_flags) or 'has no fit'}{reason}"
            )
        elif flags["at_bounds"]:
            warnings.warn(
                f"{metric}: the {phase} fit has {', '.join(flags['at_bounds'])} at its bounds."
            )
        time_data, size_data = plot_data(phase_model, value)
        return MetricResult(
            value,
            self._model.get_baseline_correction(),
            name,
            time_data,
            size_data,
            flags,
        )

    @staticmethod
    def _fitted_line(phase_model: Phase, value: float) -> tuple[np.ndarray, np.ndarray]:
        if np.isnan(value):
            return np.array([]), np.array([])
        time_data = np.array([phase_model.get_start_time(), phase_model.get_end_time()])
        return time_data, phase_model.get_predicted(time_data, extrapolation="allow")

    @staticmethod
    def _fitted_curve(phase_model: Phase, value: float) -> tuple[np.ndarray, np.ndarray]:
        return phase_model.get_time(), phase_model.get_predicted()

    @staticmethod
    def _level(phase_model: Phase, value: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([phase_model.get_start_time(), phase_model.get_end_time()]), np.array(
            [value, value]
        )
