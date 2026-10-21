from collections.abc import Sequence
from typing import Any, override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text
from numpy.typing import NDArray

from ...core.pupil_base import PupilBase
from ...core.pupil_measurement import PupilMeasurement
from ...utils.baseline import BaselineCorrection
from ...utils.light_stim import LightStimulus
from ..pupil_metrics import PhaseModelMetrics
from .base_model import BaseModel, Phase
from .breakpoints import find_constriction_start_and_end, find_redilation_start_and_end
from .phases import (
    BaselinePhase,
    ConstrictionPhase,
    MaintainedConstrictionPhase,
    RedilationPhase,
)

PHASES = ("baseline", "constriction", "maintained_constriction", "redilation")


class _TimeLabel(Text):
    # A label at the line of a start or end. It is left out where it would overlap the nearest shown
    # label before it, so the labels of nearby times don't run into each other at any zoom
    def __init__(self, *args: Any, previous: "_TimeLabel | None" = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._previous = previous

    def _shown(self, renderer: Any) -> bool:
        previous = self._previous
        while previous is not None and not previous._shown(renderer):
            previous = previous._previous
        return previous is None or not self.get_window_extent(renderer).overlaps(previous.get_window_extent(renderer))

    def draw(self, renderer: Any) -> None:
        if self._shown(renderer):
            super().draw(renderer)


class PhaseModel(BaseModel):
    """Piecewise model of a pupil light response, with a fit for each phase.

    The response is split into up to four phases. Each phase is fitted on its own, with its own
    time window and parameters, with x = 0 at the start of the phase:

    - ``"baseline"`` (BaselinePhase): a constant c.
    - ``"constriction"`` (ConstrictionPhase): a line m·x + c.
    - ``"maintained_constriction"`` (MaintainedConstrictionPhase): a line m·x + c.
    - ``"redilation"`` (RedilationPhase): an exponential S·e^(k·x) + P.

    The model is fitted when it is created. For example::

        model = piprkit.PhaseModel(measurement)
        model.plot_phases()
        model.get_all_params()
        model.metrics.redilation_v()

    All times are in seconds.

    The constriction start, the constriction end and the redilation start are found by a segmented
    fit in two stages. Each stage keeps the breakpoints with the smallest squared error:

    1. A flat line, a falling line and another line, joined at the constriction start and the
       constriction end, from light onset - ``baseline_duration`` to light offset +
       ``peak_constriction_window``. The end is moved later if needed, so that ``min_samples``
       valid samples follow the peak constriction.
    2. A flat line and an exponential, joined at the redilation start, from the peak constriction
       to the redilation end. The redilation start is not before the constriction end.

    The peak constriction is the smallest pupil size from light onset to light offset +
    ``peak_constriction_window``. Between two phases, the model is interpolated with
    :meth:`PupilMeasurement.interpolate`. Before the baseline it continues the baseline, and after
    the redilation end the redilation. A phase without a fit and the stretches next to it are NaN.

    References:
        - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        - Markwell, E. L., Feigl, B., & Zele, A. J. (2010). Intrinsically photosensitive melanopsin retinal ganglion cell contributions to the pupillary light reflex and circadian rhythm. Clinical and Experimental Optometry, 93(3), 137–149. https://doi.org/10.1111/j.1444-0938.2010.00479.x
        - Zele, A. J., Feigl, B., Smith, S. S., & Markwell, E. L. (2011). The Circadian Response of Intrinsically Photosensitive Retinal Ganglion Cells. PLOS ONE, 6(3), e17860. https://doi.org/10.1371/journal.pone.0017860
        - Feigl, B., Zele, A. J., Fader, S. M., Howes, A. N., Hughes, C. E., Jones, K. A., & Jones, R. (2012). The post‐illumination pupil response of melanopsin‐expressing intrinsically photosensitive retinal ganglion cells in diabetes. Acta Ophthalmologica, 90(3). https://doi.org/10.1111/j.1755-3768.2011.02226.x
        - Adhikari, P., Pearson, C. A., Anderson, A. M., Zele, A. J., & Feigl, B. (2015). Effect of Age and Refractive Error on the Melanopsin Mediated Post-Illumination Pupil Response (PIPR). Scientific Reports, 5(1), 17610. https://doi.org/10.1038/srep17610
        - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
    """
    def __init__(
        self,
        pupil_measurement: PupilBase,
        phases: Sequence[str] = PHASES,
        baseline_duration: float = 10.0,
        constriction_latency: bool = False,
        redilation_delay: float = 0.0,
        redilation_end: float | None = None,
        peak_constriction_window: float = 1.0,
        min_samples: int = 3,
        breakpoint_resolution: float = 0.01,
        breakpoint_min_duration: float = 0.1,
        redilation_k_min: float = -10.0,
        redilation_k_max: float = -0.01,
        redilation_k_count: int = 40,
        redilation_k_estimate: float = -0.4,
        redilation_k_bounds: tuple[float, float] = (-np.inf, 0.0),
    ) -> None:
        """Fit the model to a pupil measurement.

        Args:
            pupil_measurement (PupilBase): The measurement to fit. It needs a light stimulus. The model
                keeps a copy.
            phases (Sequence[str], optional): The phases to fit: any of "baseline", "constriction",
                "maintained_constriction" and "redilation". They are always fitted in that order.
                Defaults to all four.
            baseline_duration (float, optional): Length of the baseline window before light onset, in
                seconds. Defaults to 10.0 (Feigl et al., 2011).
            constriction_latency (bool, optional): Whether the constriction may start only after the PLR
                latency (``metrics.plr_latency()``). If False, it may
                start at light onset. Defaults to False.
            redilation_delay (float, optional): Earliest start of the redilation phase,
                in seconds after light offset. Defaults to 0.0.
            redilation_end (float | None, optional): End of the redilation phase, in seconds after
                light offset. Samples after it are left out of the fit and of the search for the
                redilation start. None means the last sample. Defaults to None.
            peak_constriction_window (float, optional): How far after light offset, in seconds, the
                constriction start and end and the peak constriction are searched. Defaults to 1.0
                (Adhikari et al., 2015; Amiot et al., 2024).
            min_samples (int, optional): Fewest valid samples a phase needs to be fitted. With fewer, its
                too_few_samples flag is set. Also the fewest samples per segment in the breakpoint search.
                Defaults to 3.
            breakpoint_resolution (float, optional): Spacing of the candidate breakpoints, in seconds.
                Defaults to 0.01.
            breakpoint_min_duration (float, optional): Shortest constriction the search accepts, from
                constriction start to constriction end, and shortest maintained constriction, in
                seconds. Defaults to 0.1.
            redilation_k_min (float, optional): Most negative rate constant k tried in the search
                for the redilation start, in 1/s. Defaults to -10.0.
            redilation_k_max (float, optional): Rate constant k closest to 0 tried in the search for
                the redilation start, in 1/s. If the best k is this one, the redilation phase is
                not defined. Defaults to -0.01.
            redilation_k_count (int, optional): Number of k values tried, log-spaced from k_max to
                k_min. k is then refined between the neighbours of the best one. Defaults to 40.
            redilation_k_estimate (float, optional): Start value of k for the exponential fit when
                the search finds no redilation start. Defaults to -0.4 (Feigl et al., 2011).
            redilation_k_bounds (tuple[float, float], optional): Lower and upper bound of k in the
                exponential fit. Defaults to (-inf, 0.0).

        Raises:
            ValueError: If a phase is unknown, or if the measurement has no light stimulus.

        References:
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
        """
        unknown = [phase for phase in phases if phase not in PHASES]
        if unknown:
            raise ValueError(f"Unknown phases: {unknown}. Options: {list(PHASES)}")
        light_stimulus = pupil_measurement.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set. Cannot setup fits.")
        super().__init__(pupil_measurement.get_time(), pupil_measurement.get_size())
        self._measurement = pupil_measurement.copy()
        self._set_light_stimulus(light_stimulus)
        self.baseline_correction = pupil_measurement.get_baseline_correction()
        self.phases = tuple(phase for phase in PHASES if phase in phases)
        self.baseline_duration = baseline_duration
        self.constriction_latency = constriction_latency
        self.redilation_delay = redilation_delay
        self.redilation_end = redilation_end
        self.peak_constriction_window = peak_constriction_window
        self.min_samples = min_samples
        self.breakpoint_resolution = breakpoint_resolution
        self.breakpoint_min_duration = breakpoint_min_duration
        self.redilation_k_min = redilation_k_min
        self.redilation_k_max = redilation_k_max
        self.redilation_k_count = redilation_k_count
        self.redilation_k_estimate = redilation_k_estimate
        self.redilation_k_bounds = redilation_k_bounds
        self._setup_phases()
        self.fit()

    def get_baseline_correction(self) -> BaselineCorrection | None:
        """Get the baseline correction of the measurement the model was fitted to.

        Returns:
            BaselineCorrection | None: The baseline correction, or None if none was applied.
        """
        return self.baseline_correction

    def _set_light_stimulus(self, stimulus: LightStimulus) -> None:
        self.stimulus = stimulus

    def get_light_stimulus(self) -> LightStimulus:
        """Return the light stimulus of the measurement the model was fitted to.

        Returns:
            LightStimulus: The light stimulus. Its onset and offset set the search windows of the phases.
        """
        return self.stimulus

    def plot_light_stimulus(self, ax: Axes | None = None, show: bool = False, **kwargs: Any) -> Axes:
        """Plot the light stimulus. See ``LightStimulus.plot``.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``LightStimulus.plot``.

        Returns:
            Axes: The axes with the plotted light stimulus.
        """
        return self.get_light_stimulus().plot(ax, show, **kwargs)

    def get_sections(self) -> dict[str, dict[str, float]]:
        """Get the pre-stimulus, during-stimulus and post-stimulus sections of the fitted data.

        The phases don't have to line up with the sections. See :meth:`LightStimulus.get_sections`.

        Returns:
            dict[str, dict[str, float]]: For each section, its ``start`` and ``end`` in seconds.
        """
        time_data = self.get_time()
        return self.get_light_stimulus().get_sections(float(time_data[0]), float(time_data[-1]))

    def plot_sections(self, ax: Axes | None = None, show: bool = False, **kwargs: Any) -> Axes:
        """Plot the sections as a labelled band along the top of the axes. See :meth:`LightStimulus.plot_sections`.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for each section, see :meth:`LightStimulus.plot_sections`.

        Returns:
            Axes: The axes with the plotted sections.
        """
        time_data = self.get_time()
        return self.get_light_stimulus().plot_sections(float(time_data[0]), float(time_data[-1]), ax, show, **kwargs)

    def _setup_phases(self) -> None:
        time_data, size_data = self.get_time(), self.get_size()
        onset, offset = self.get_light_stimulus().get_timing()
        self._phase_models: dict[str, Phase] = {}
        self._undefined: dict[str, str] = {}
        self._redilation_start_estimate: tuple[float, float, float] | None = None
        peak_constriction = onset + float(self._measurement.metrics.time_to_peak(self.peak_constriction_window))
        search_end = self._constriction_search_end(peak_constriction)
        breakpoints = find_constriction_start_and_end(
            time_data,
            size_data,
            onset - self.baseline_duration,
            search_end,
            onset + self._constriction_delay(),
            min_duration=self.breakpoint_min_duration,
            resolution=self.breakpoint_resolution,
            min_samples=self.min_samples,
        )
        if "maintained_constriction" in self.phases or "redilation" in self.phases:
            redilation_start = self._find_redilation_start(breakpoints, peak_constriction)
        for phase in self.phases:
            if phase == "baseline":
                self._phase_models[phase] = BaselinePhase(
                    time_data, size_data, onset - self.baseline_duration, onset, min_samples=self.min_samples
                )
            elif phase == "constriction":
                if breakpoints is None:
                    self._phase_models[phase] = ConstrictionPhase(
                        time_data, size_data, onset, search_end, min_samples=self.min_samples
                    )
                    self._undefined[phase] = "No constriction found in the search range."
                else:
                    self._phase_models[phase] = ConstrictionPhase(
                        time_data, size_data, *breakpoints, min_samples=self.min_samples
                    )
            elif phase == "maintained_constriction":
                self._phase_models[phase] = MaintainedConstrictionPhase(
                    time_data, size_data, peak_constriction, redilation_start, min_samples=self.min_samples
                )
            else:
                self._phase_models[phase] = RedilationPhase(
                    time_data,
                    size_data,
                    redilation_start,
                    end_time=self._redilation_end_time(),
                    min_samples=self.min_samples,
                )

    def _constriction_search_end(self, peak_constriction: float) -> float:
        # The constriction can end at the peak constriction only with min_samples valid samples after it, so the search
        # runs on past the window when the peak constriction lies closer to its end
        time_data, size_data = self.get_time(), self.get_size()
        search_end = self.get_light_stimulus().get_timing()[1] + self.peak_constriction_window
        after = time_data[(time_data > peak_constriction) & np.isfinite(size_data)]
        if after.size < self.min_samples:
            return search_end
        return max(search_end, float(after[self.min_samples - 1]))

    def _find_redilation_start(self, breakpoints: tuple[float, float] | None, peak_constriction: float) -> float:
        time_data, size_data = self.get_time(), self.get_size()
        offset = self.get_light_stimulus().get_timing()[1]
        # The flat line starts at the peak constriction; the redilation starts at or after light offset
        # + redilation_delay, the peak constriction and the constriction end
        search_start = peak_constriction if np.isfinite(peak_constriction) else offset
        earliest = max(offset + self.redilation_delay, search_start)
        if breakpoints is not None:
            earliest = max(earliest, breakpoints[1])
        end = self._redilation_end_time()
        found = find_redilation_start_and_end(
            time_data,
            size_data,
            search_start,
            earliest,
            end=np.inf if end is None else end,
            k_min=self.redilation_k_min,
            k_max=self.redilation_k_max,
            k_count=self.redilation_k_count,
            resolution=self.breakpoint_resolution,
            min_samples=self.min_samples,
        )
        if found is None:
            return earliest
        redilation_start, _, estimate = found
        # At the k closest to 0 the exponential cannot be told apart from a flat line
        if estimate[1] >= self.redilation_k_max and "redilation" in self.phases:
            self._undefined["redilation"] = "No redilation found in the search range."
        self._redilation_start_estimate = estimate
        return redilation_start

    def _redilation_end_time(self) -> float | None:
        if self.redilation_end is None:
            return None
        return self.get_light_stimulus().get_timing()[1] + self.redilation_end

    def _constriction_delay(self) -> float:
        """References:
            - Adhikari, P., Zele, A. J., & Feigl, B. (2015). The post-illumination pupil response (PIPR). Investigative Ophthalmology and Visual Science, 56(6), 3838–3849. https://doi.org/10.1167/iovs.14-16233
            - Feigl, B., Zele, A. J., Fader, S. M., Howes, A. N., Hughes, C. E., Jones, K. A., & Jones, R. (2012). The post‐illumination pupil response of melanopsin‐expressing intrinsically photosensitive retinal ganglion cells in diabetes. Acta Ophthalmologica, 90(3). https://doi.org/10.1111/j.1755-3768.2011.02226.x
        """
        if not self.constriction_latency:
            return 0.0
        return float(self._measurement.metrics.plr_latency())

    @override
    def fit(self) -> None:
        """Fit each phase to the pupil size data in its window.

        The model calls this when it is created. Each phase is fitted on its own:

        - baseline: c is ``metrics.baseline(baseline_duration)``, the mean of the valid samples in its
          window. Not defined if the window has no valid samples.
        - constriction and maintained constriction: a line fitted with ``scipy.optimize.curve_fit``,
          started from the line through the first and last valid sample. The maintained constriction is
          not defined if it is shorter than ``breakpoint_min_duration``.
        - redilation: an exponential fitted with ``curve_fit``, started from the estimate of the
          search for the redilation start, with k between ``redilation_k_bounds``.

        A phase whose window was not found is not fitted. Its parameters are NaN and its flags say
        why, see :meth:`get_flags`.
        """
        phase_models = self._phase_models
        for phase, reason in self._undefined.items():
            phase_models[phase]._set_not_defined(reason)
        if "baseline" in phase_models and "baseline" not in self._undefined:
            phase_model = phase_models["baseline"]
            c = float(self._measurement.metrics.baseline(self.baseline_duration))
            if np.isnan(c):
                phase_model._set_not_defined("No valid samples in the baseline window.")
            else:
                phase_model.fit(c)
        if "constriction" in phase_models and "constriction" not in self._undefined:
            phase_model = phase_models["constriction"]
            phase_model.fit(self._constriction_estimate(phase_model))
        if "maintained_constriction" in phase_models and "maintained_constriction" not in self._undefined:
            phase_model = phase_models["maintained_constriction"]
            duration = phase_model.get_end_time() - phase_model.get_start_time()
            if not duration >= self.breakpoint_min_duration:
                phase_model._set_not_defined(
                    "The maintained constriction is shorter than breakpoint_min_duration "
                    f"({self.breakpoint_min_duration:g} s)."
                )
            else:
                phase_model.fit(self._maintained_constriction_estimate(phase_model))
        if "redilation" in phase_models and "redilation" not in self._undefined:
            phase_model = phase_models["redilation"]
            k_lower, k_upper = self.redilation_k_bounds
            phase_model.fit(
                p0=self._redilation_estimate(phase_model),
                bounds=((-np.inf, k_lower, -np.inf), (np.inf, k_upper, np.inf)),
            )

    @staticmethod
    def _valid_data(phase_model: Phase) -> tuple[np.ndarray, np.ndarray]:
        time_data, size = phase_model.get_time(), phase_model.get_size()
        mask = ~np.isnan(size)
        return time_data[mask], size[mask]

    def _constriction_estimate(self, phase_model: Phase) -> tuple[float, float] | None:
        time_data, size = self._valid_data(phase_model)
        if size.size < 2:
            return None
        slope = (size[-1] - size[0]) / (time_data[-1] - time_data[0])
        return float(slope), float(size[0])

    def _maintained_constriction_estimate(self, phase_model: Phase) -> tuple[float, float] | None:
        return self._constriction_estimate(phase_model)

    def _redilation_estimate(self, phase_model: RedilationPhase) -> tuple[float, float, float] | None:
        if self._redilation_start_estimate is not None:
            return self._redilation_start_estimate
        return phase_model._data_estimate(self.redilation_k_estimate)

    def get_phases(self) -> dict[str, Phase]:
        """Get the fitted phases.

        Returns:
            dict[str, Phase]: The phases, by name, in the order baseline, constriction,
            maintained_constriction, redilation. Each phase holds its own window, pupil size
            data, parameters and flags, and can be plotted on its own. The dict is a copy; the phases
            are shared with the model.
        """
        return dict(self._phase_models)

    def plot_phases(self, ax: Axes | None = None, show: bool = False, **kwargs: Any) -> Axes:
        """Plot the phases: a labelled band per phase along the bottom of the axes, the fitted curve
        of each phase, the stretches between the phases, and a dotted line at each start and end of
        a phase with a fit.

        Each phase has its own colour. A phase without a fit is drawn hatched, and the legend shows
        any flags that are set (not_defined, not_converged, too_few_samples, at_bounds). A stretch
        that is interpolated between two phases is a grey dashed line, and the stretch before the
        baseline, where the model continues the baseline, is a dashed line in the colour of the
        baseline, and the stretch after the redilation end, where the model continues the
        redilation, a dashed line in the colour of the redilation. Each phase has its name in the band where the name fits, also after zooming in or
        out. The y margin of the axes is set to at least 0.1, so that the data on the axes stay above
        the band.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.plot`` for the fitted curves.

        Returns:
            Axes: The axes with the plotted phases.
        """
        import matplotlib.pyplot as plt
        from matplotlib.transforms import blended_transform_factory

        from ...utils.light_stim import _BandLabel

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
            ax.set_xlabel("Time (s)")  # type: ignore
            ax.set_ylabel(self._y_label())  # type: ignore
        ax.margins(y=max(ax.margins()[1], 0.1))
        # x in data coordinates, y in axes coordinates, so the band stays at the bottom
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = {name: cycle[index % len(cycle)] for index, name in enumerate(PHASES)}
        times: dict[float, list[str]] = {}
        for name, phase_model in self._phase_models.items():
            flags = phase_model.get_flags()
            set_flags = [flag for flag in ("not_defined", "not_converged", "too_few_samples") if flags[flag]]
            if flags["at_bounds"]:
                set_flags.append(f"at_bounds: {', '.join(flags['at_bounds'])}")
            label = f"{name} ({', '.join(set_flags)})" if set_flags else name
            band = {"ymin": 0.0, "ymax": 0.06, "color": colors[name], "label": label}
            start, end = phase_model.get_start_time(), phase_model.get_end_time()
            readable = name.replace("_", " ")
            if self._has_fit(phase_model):
                phase_model.plot_window(ax, alpha=0.5, **band)
                phase_model.plot_predicted(ax, color=colors[name], label="_nolegend_", **kwargs)
                times.setdefault(round(start, 9), []).append(f"{readable} start")
                times.setdefault(round(end, 9), []).append(f"{readable} end")
            else:
                phase_model.plot_window(ax, alpha=0.6, fill=False, hatch="//", **band)
            if end > start:
                band_label = _BandLabel(start, end, 0.03, readable, transform=transform, fontsize=7)
                band_label.set_in_layout(False)
                ax.add_artist(band_label)
        fitted = {name: phase_model for name, phase_model in self._phase_models.items() if self._has_fit(phase_model)}
        first_time, last_time = float(self.get_time()[0]), float(self.get_time()[-1])
        if "baseline" in fitted and first_time < fitted["baseline"].get_start_time():
            stretch = np.linspace(first_time, fitted["baseline"].get_start_time(), 50)
            ax.plot(stretch, self._predict(stretch), color=colors["baseline"], linestyle="--", label="_nolegend_")  # type: ignore
        if "redilation" in fitted and fitted["redilation"].get_end_time() < last_time:
            stretch = np.linspace(fitted["redilation"].get_end_time(), last_time, 50)
            ax.plot(stretch, self._predict(stretch), color=colors["redilation"], linestyle="--", label="_nolegend_")  # type: ignore
        for first, second in zip(PHASES, PHASES[1:]):
            if first in fitted and second in fitted:
                end, start = fitted[first].get_end_time(), fitted[second].get_start_time()
                if end < start:
                    stretch = np.linspace(end, start, 50)
                    ax.plot(stretch, self._predict(stretch), color="grey", linestyle="--", label="_nolegend_")  # type: ignore
        previous = None
        for time, names in sorted(times.items()):
            ax.axvline(time, color="grey", linestyle=":", linewidth=0.8, label="_nolegend_")  # type: ignore
            previous = _TimeLabel(
                time,
                0.07,
                ", ".join(names),
                previous=previous,
                transform=transform,
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=6,
                clip_on=True,
            )
            previous.set_in_layout(False)
            ax.add_artist(previous)
        if show:
            plt.show()  # type: ignore
        return ax

    @override
    def get_window(self) -> dict[str, Any]:
        """Get the time window of each phase.

        Returns:
            dict[str, Any]: For each phase name, a dict with the ``start`` and ``end`` of the window in
            seconds, the ``delay`` (time after the start that is left out of the fit; 0 in a
            PhaseModel), ``n_samples``, the number of valid samples in the window, and ``sections``,
            the names of the sections the window overlaps (see :meth:`get_sections`).
        """
        sections = self.get_sections()
        windows = {}
        for name, phase_model in self._phase_models.items():
            window = phase_model.get_window()
            window["sections"] = tuple(
                section
                for section, bounds in sections.items()
                if window["start"] < bounds["end"] and window["end"] > bounds["start"]
            )
            windows[name] = window
        return windows

    @override
    def get_flags(self) -> dict[str, Any]:
        """Get the flags of each phase.

        Returns:
            dict[str, Any]: For each phase name, the flags of that phase:

            - ``not_defined`` (bool): the window was not found, so the phase was not fitted;
              ``reason`` (str | None) says why.
            - ``not_converged`` (bool): ``curve_fit`` did not converge.
            - ``too_few_samples`` (bool): the window has fewer than ``min_samples`` valid samples.
            - ``at_bounds`` (tuple[str, ...]): the parameters that ended at a bound.

            The parameters are NaN when not_defined, not_converged or too_few_samples is set.
        """
        return {name: phase_model.get_flags() for name, phase_model in self._phase_models.items()}

    @override
    def get_options(self) -> dict[str, Any]:
        """Get the options used for the fit.

        Returns:
            dict[str, Any]: The arguments the model was created with, by name, and under
            ``"phase_options"`` the fit options of each phase: the start estimate ``p0``, the
            ``bounds``, the ``delay`` and ``min_samples``. These are empty for the baseline phase,
            which is not fitted with curve_fit.
        """
        return {
            "phases": self.phases,
            "baseline_duration": self.baseline_duration,
            "constriction_latency": self.constriction_latency,
            "redilation_delay": self.redilation_delay,
            "redilation_end": self.redilation_end,
            "peak_constriction_window": self.peak_constriction_window,
            "min_samples": self.min_samples,
            "breakpoint_resolution": self.breakpoint_resolution,
            "breakpoint_min_duration": self.breakpoint_min_duration,
            "redilation_k_min": self.redilation_k_min,
            "redilation_k_max": self.redilation_k_max,
            "redilation_k_count": self.redilation_k_count,
            "redilation_k_estimate": self.redilation_k_estimate,
            "redilation_k_bounds": self.redilation_k_bounds,
            "phase_options": {name: phase_model.get_options() for name, phase_model in self._phase_models.items()},
        }

    @override
    def _predict(self, time_data: np.ndarray) -> np.ndarray:
        """Predict pupil size for given time points using the fitted models.

        Args:
            time_data (np.ndarray): Time array to predict for

        Returns:
            np.ndarray: Predicted pupil sizes
        """
        size = np.full_like(time_data, np.nan, dtype=np.float64)
        fitted = {name: phase_model for name, phase_model in self._phase_models.items() if self._has_fit(phase_model)}
        for name, phase_model in fitted.items():
            # Before the baseline start and after the redilation end, the model extrapolates
            start = -np.inf if name == "baseline" else phase_model.get_start_time()
            end = np.inf if name == "redilation" else phase_model.get_end_time()
            in_phase = (time_data >= start) & (time_data < end)
            if np.any(in_phase):
                size[in_phase] = phase_model._predict(time_data[in_phase])
        # A stretch between two neighbouring phases with a fit is filled with interpolate(). Next to a
        # phase without a fit it stays NaN.
        for first, second in zip(PHASES, PHASES[1:]):
            if first not in fitted or second not in fitted:
                continue
            end, start = fitted[first].get_end_time(), fitted[second].get_start_time()
            in_stretch = (time_data >= end) & (time_data < start)
            if not np.any(in_stretch):
                continue
            ends = PupilMeasurement(
                np.array([end, start]),
                np.concatenate((fitted[first]._predict(np.array([end])), fitted[second]._predict(np.array([start])))),
            )
            size[in_stretch] = ends.interpolate(time_data[in_stretch], inplace=False).get_size()
        return size

    @staticmethod
    def _has_fit(phase_model: Phase) -> bool:
        return bool(np.all(np.isfinite(phase_model.get_params())))

    def get_param(self, phase_name: str, param_name: str) -> float:
        """Get one fitted parameter of one phase.

        Args:
            phase_name (str): The phase: "baseline", "constriction", "maintained_constriction" or
                "redilation".
            param_name (str): The parameter: "c" for baseline, "m" or "c" for the two constrictions,
                "S", "k" or "P" for redilation.

        Raises:
            ValueError: If the model has no such phase, or the phase has no such parameter.

        Returns:
            float: The parameter value. NaN if the phase was not fitted, see :meth:`get_flags`.
        """
        params = self.get_all_params()
        if phase_name not in params:
            raise ValueError(
                f"Phase '{phase_name}' not found. Available phases: {list(params.keys())}"
            )
        phase_params = params[phase_name]
        if param_name not in phase_params:
            raise ValueError(
                f"Parameter '{param_name}' not found in phase '{phase_name}'"
            )
        return params[phase_name][param_name]

    def get_all_params(self) -> dict[str, dict[str, float]]:
        """Get all fitted parameters organized by phase.

        Returns:
            dict[str, dict[str, float]]: For each phase name, its parameters by name, e.g.
            ``{"baseline": {"c": 6.1}, "constriction": {"m": -3.0, "c": 6.0}, ...}``.
        """
        return {
            name: dict(zip(phase_model.get_param_names(), phase_model.get_params()))
            for name, phase_model in self._phase_models.items()
        }

    def get_simulation_args(self) -> dict[str, Any]:
        """Get the arguments of :func:`~piprkit.simulate_pupil_measurement` that simulate this fit.

        For example::

            simulation = piprkit.simulate_pupil_measurement(**model.get_simulation_args())

        The simulation has the time points and light stimulus of the fitted data. It takes these values
        from the fit:

        - ``constriction_delay``: the constriction start, from light onset.
        - ``time_to_peak``: the peak constriction (the maintained constriction start), from light onset.
        - ``redilation_delay``: the redilation start, from light offset.
        - ``baseline_c``, ``maintained_constriction_c``, ``maintained_constriction_m``, ``redilation_k``
          and ``redilation_P``: the fitted parameters.

        The other parameters (the constriction m and c, and the redilation S) are not used: the
        simulation derives them so that the curve has no jumps. The simulated constriction runs from the
        constriction start straight to the peak constriction, without the interpolated stretch between
        the constriction end and the peak constriction.

        Raises:
            ValueError: If the model does not have all four phases, or if a phase has no fit (see
                :meth:`get_flags`).

        Returns:
            dict[str, Any]: The arguments, by name. Add ``noise_std``, ``trend_std`` and ``rng`` to
            simulate noise.
        """
        missing = [phase for phase in PHASES if phase not in self._phase_models]
        if missing:
            raise ValueError(f"A simulation needs all four phases. Missing: {missing}")
        for name, phase_model in self._phase_models.items():
            if not self._has_fit(phase_model):
                raise ValueError(f"Cannot simulate: the {name} phase has no fit. See get_flags().")
        onset, offset = self.get_light_stimulus().get_timing()
        window, params = self.get_window(), self.get_all_params()
        return {
            "time_data": self.get_time().copy(),
            "t_light_onset": float(onset),
            "t_light_offset": float(offset),
            "constriction_delay": float(window["constriction"]["start"] - onset),
            "time_to_peak": float(window["maintained_constriction"]["start"] - onset),
            "redilation_delay": float(window["redilation"]["start"] - offset),
            "baseline_c": float(params["baseline"]["c"]),
            "maintained_constriction_c": float(params["maintained_constriction"]["c"]),
            "maintained_constriction_m": float(params["maintained_constriction"]["m"]),
            "redilation_k": float(params["redilation"]["k"]),
            "redilation_P": float(params["redilation"]["P"]),
        }

    @override
    def get_formula_string(self) -> str:
        """Get formula strings for all fitted models.

        Returns:
            str: Formatted string
        """
        labels = {
            "baseline": "Baseline",
            "constriction": "Constriction",
            "maintained_constriction": "Maintained constriction",
            "redilation": "Redilation",
        }
        string = "Formula:\n"
        for name, phase_model in self._phase_models.items():
            string += f"\t{labels[name]}: {phase_model.get_formula_string()}\n"
        return string

    @property
    @override
    def metrics(self) -> PhaseModelMetrics:
        """The metrics read from the fitted phases, such as the constriction velocity.

        Each metric is a method, for example ``model.metrics.constriction_v()``. See
        :class:`~piprkit.analysis.pupil_metrics.PhaseModelMetrics` for all of them. Metrics computed from
        data, such as ``pipr_6s``, are on the ``metrics`` of the measurement instead.
        """
        return PhaseModelMetrics(self)

    @override
    def to_measurement(
        self, time_data: NDArray[np.number] | None = None, extrapolation: str = "error"
    ) -> PupilMeasurement:
        """Convert the fitted model to a PupilMeasurement, with the light stimulus and baseline correction of the fitted data.

        Args:
            time_data (NDArray[np.number] | None, optional): Time points to predict at. If None, the
                time points of the fitted data are used. Defaults to None.
            extrapolation (str, optional): What to do when time_data runs outside the fitted data:
                "allow", "warn" or "error". Defaults to "error".

        Raises:
            ValueError: If extrapolation is unknown, or if it is "error" and time_data runs outside
                the fitted data.

        Returns:
            PupilMeasurement: The predicted pupil size as a measurement.
        """
        measurement = super().to_measurement(time_data, extrapolation)
        measurement.set_light_stimulus(*self.get_light_stimulus().get_timing())
        correction = self.get_baseline_correction()
        if correction is not None:
            measurement._set_baseline_correction(correction)
        return measurement
