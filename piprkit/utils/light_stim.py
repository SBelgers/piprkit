from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

SECTIONS = ("pre_stimulus", "during_stimulus", "post_stimulus")
_SECTION_LABELS = {"pre_stimulus": "pre-stimulus", "during_stimulus": "during stimulus", "post_stimulus": "post-stimulus"}


class _BandLabel(Text):
    # A label in a band of sections or phases: drawn in the middle of the visible part of its window,
    # and only when it fits there, so it follows zooming
    def __init__(self, start: float, end: float, y: float, text: str, **kwargs: Any) -> None:
        super().__init__((start + end) / 2, y, text, ha="center", va="center", **kwargs)
        self._window = (start, end)

    def draw(self, renderer: Any) -> None:
        left, right = sorted(self.axes.get_xlim())  # type: ignore
        low, high = max(self._window[0], left), min(self._window[1], right)
        if high <= low:
            return
        self.set_x((low + high) / 2)
        pixels = self.axes.transData.transform([(low, 0), (high, 0)])[:, 0]  # type: ignore
        if self.get_window_extent(renderer).width <= abs(pixels[1] - pixels[0]):
            super().draw(renderer)


class LightStimulus:
    """A light stimulus: the period in which the light that evokes the pupil response is on.

    It runs from ``start_time`` (light onset) to ``end_time`` (light offset), in seconds on the
    same time axis as the pupil data. A measurement uses it as the time reference for its metrics
    and baseline correction. The colour is only used to draw the stimulus in plots.
    """

    def __init__(
        self,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        end_time: Optional[float] = None,
        color: Optional[str] = None,
    ):
        """Create a light stimulus from its onset and either its duration or its offset.

        Args:
            start_time (float, optional): Light onset, in seconds. Defaults to 0.0.
            duration (float | None, optional): How long the light is on, in seconds. Give either
                duration or end_time. Defaults to None.
            end_time (float | None, optional): Light offset, in seconds. Give either duration or
                end_time. Defaults to None.
            color (str | None, optional): Colour used to draw the stimulus in plots, any matplotlib
                colour such as "blue" or "#1f77b4". It does not set or record the wavelength of the
                light. None gives "yellow". Defaults to None.

        Raises:
            ValueError: If both or neither of duration and end_time are given, or if color is not a
                matplotlib colour.
        """
        self._set_time(start_time=start_time, duration=duration, end_time=end_time)
        self._set_color(color=color)

    def __repr__(self):
        return f"LightStimulus(start_time={self.start_time}, end_time={self.end_time}, color={self.color})"

    def _set_time(
        self,
        start_time: float,
        duration: float | None = None,
        end_time: float | None  = None,
    ) -> None:
        if end_time is None and duration is None:
            raise ValueError("Either duration or end_time must be specified.")
        if end_time is not None and duration is not None:
            raise ValueError("Only one of duration or end_time can be specified.")
        self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
            return
        elif duration is not None:
            self.end_time = start_time + duration
            return
        raise ValueError("Invalid parameters for setting time.")

    def _set_color(self, color: Optional[str]) -> None:
        if color is not None:
            from matplotlib.colors import is_color_like

            if not is_color_like(color):
                raise ValueError(f"Invalid color: {color}")
            self.color = color
        else:
            self.color = "yellow"

    def get_timing(self) -> tuple[float, float]:
        """Get the onset and offset time of the stimulus.

        Returns:
            tuple[float, float]: The start and end time of the stimulus, in seconds.
        """
        return self.start_time, self.end_time

    def get_color(self) -> str:
        """Return the colour used to draw the stimulus in plots.

        Returns:
            str: A matplotlib colour; "yellow" unless another colour was given.
        """
        return self.color

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for the stimulus. This shifts the start and end times by the specified offset.

        Args:
            offset (float): The offset to apply to the start and end times.
        """
        start_time, end_time = self.get_timing()
        self._set_time(start_time=start_time + offset, end_time=end_time + offset)

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the light stimulus as a shaded area with dashed lines at onset and offset.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for the shaded area. Defaults: label "Light Stimulus",
                color the stimulus colour, alpha 0.3.

        Returns:
            Axes: The axes with the plotted light stimulus.
        """
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        start_time, end_time = self.get_timing()
        if "label" not in kwargs:
            kwargs["label"] = "Light Stimulus"
        if "color" not in kwargs:
            kwargs["color"] = self.get_color()
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.3
        ax.axvspan(start_time, end_time, **kwargs)  # type: ignore
        ax.axvline(  # type: ignore
            start_time, color="black", linestyle="--", alpha=0.7, label="_Light Onset"
        )
        ax.axvline(  # type: ignore
            end_time, color="black", linestyle="--", alpha=0.7, label="_Light Offset"
        )
        if show:
            plt.show()  # type: ignore
        return ax

    def get_sections(self, start_time: float, end_time: float) -> dict[str, dict[str, float]]:
        """Divide a time range into the pre-stimulus, during-stimulus and post-stimulus sections.

        The pre-stimulus section runs from ``start_time`` to light onset, the during-stimulus section
        from light onset to light offset, and the post-stimulus section from light offset to
        ``end_time``. When the time range starts after light onset, the pre-stimulus section is empty:
        it starts and ends at light onset. The same holds for the post-stimulus section when the time
        range ends before light offset.

        Args:
            start_time (float): Start of the time range, in seconds, e.g. the first sample.
            end_time (float): End of the time range, in seconds, e.g. the last sample.

        Returns:
            dict[str, dict[str, float]]: For ``"pre_stimulus"``, ``"during_stimulus"`` and
            ``"post_stimulus"``, the ``start`` and ``end`` of the section, in seconds.
        """
        onset, offset = (float(time) for time in self.get_timing())
        return {
            "pre_stimulus": {"start": min(start_time, onset), "end": onset},
            "during_stimulus": {"start": onset, "end": offset},
            "post_stimulus": {"start": offset, "end": max(end_time, offset)},
        }

    def plot_sections(
        self,
        start_time: float,
        end_time: float,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the sections of a time range as a labelled band along the top of the axes.

        The pre-stimulus and post-stimulus sections are light grey, and the during-stimulus section has
        the colour of the stimulus. Each section has its name in the band where the name fits, also
        after zooming in or out. Empty sections are left out. The y margin of the axes is set to at
        least 0.1, so that the data on the axes stay below the band. See :meth:`get_sections`.

        Args:
            start_time (float): Start of the time range, in seconds, e.g. the first sample.
            end_time (float): End of the time range, in seconds, e.g. the last sample.
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for each section. Defaults: ymin 0.94 and ymax 1 (the
                top 6 % of the axes), alpha 0.5.

        Returns:
            Axes: The axes with the plotted sections.
        """
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        kwargs.setdefault("ymin", 0.94)
        kwargs.setdefault("ymax", 1.0)
        kwargs.setdefault("alpha", 0.5)
        ax.margins(y=max(ax.margins()[1], 0.1))
        # x in data coordinates, y in axes coordinates, so the band stays at the top
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for name, window in self.get_sections(start_time, end_time).items():
            start, end = window["start"], window["end"]
            if end <= start:
                continue
            color = self.get_color() if name == "during_stimulus" else "lightgrey"
            ax.axvspan(start, end, **{"color": color, "label": "_nolegend_", **kwargs})  # type: ignore
            label = _BandLabel(
                start, end, (kwargs["ymin"] + kwargs["ymax"]) / 2, _SECTION_LABELS[name], transform=transform, fontsize=7
            )
            label.set_in_layout(False)
            ax.add_artist(label)
        if show:
            plt.show()  # type: ignore
        return ax


class LightStimuliSeries:
    """The light stimuli of a pupil series, in the order they were added.

    Each stimulus is a :class:`LightStimulus`. A ``PupilSeries`` keeps one of these, and
    :meth:`PupilSeries.split` cuts the series into one measurement per stimulus.
    """

    def __init__(self):
        """Create an empty series. Add stimuli with :meth:`add_stimulus`."""
        self.stimuli: list[LightStimulus] = []

    def _add_stimulus_from_class(self, stimulus: LightStimulus) -> None:
        self.stimuli.append(stimulus)

    def add_stimulus(
        self,
        start_time: float,
        duration: Optional[float] = None,
        end_time: Optional[float] = None,
        color: Optional[str] = None,
    ) -> None:
        """Add a light stimulus to the end of the series.

        Args:
            start_time (float): Light onset, in seconds.
            duration (float | None, optional): How long the light is on, in seconds. Give either
                duration or end_time. Defaults to None.
            end_time (float | None, optional): Light offset, in seconds. Give either duration or
                end_time. Defaults to None.
            color (str | None, optional): Colour used to draw the stimulus in plots, any matplotlib
                colour. None gives "yellow". Defaults to None.

        Raises:
            ValueError: If both or neither of duration and end_time are given, or if color is not a
                matplotlib colour.
        """
        stimulus = LightStimulus(
            start_time=start_time, duration=duration, end_time=end_time, color=color
        )
        self._add_stimulus_from_class(stimulus)

    def get_stimuli(self) -> list[LightStimulus]:
        """Get the LightStimulus objects.

        Returns:
            list[LightStimulus]: The stimuli, in the order they were added. This is the stored list,
            not a copy.
        """
        return self.stimuli

    def get_timings(self) -> list[tuple[float, float]]:
        """ Get the start and end times of all light stimuli in the series.

        Returns:
            list[tuple[float, float]]: A list of tuples, where each tuple contains the start and end time of a light stimulus.
        """
        return [stimulus.get_timing() for stimulus in self.stimuli]

    def get_colors(self) -> list[str]:
        """Get the colors of all light stimuli in the series.

        Returns:
            list[str]: The plot colour of each stimulus, see :meth:`LightStimulus.get_color`.
        """
        return [stimulus.get_color() for stimulus in self.stimuli]

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for all light stimuli in the series. This shifts the start and end times of each stimulus by the specified offset.

        Args:
            offset (float): Time in seconds to add to the start and end of each stimulus. A negative value
                shifts to earlier times.
        """
        for stimulus in self.stimuli:
            stimulus.set_time_offset(offset)

    def plot(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all light stimuli in the series, each as in :meth:`LightStimulus.plot`.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created.
                Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to :meth:`LightStimulus.plot` for each stimulus. Only the first
                stimulus gets a legend label.

        Returns:
            Axes: The axes with the plotted light stimuli.
        """
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        for i, stimulus in enumerate(self.stimuli):
            if i > 0:
                kwargs["label"] = "_nolegend_"
            stimulus.plot(ax=ax, show=False, **kwargs)
        if show:
            plt.show()  # type: ignore
        return ax
