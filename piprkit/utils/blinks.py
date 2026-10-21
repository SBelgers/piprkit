from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class PupilBlinks:
    """Class for blink operations."""

    def __init__(self) -> None:
        self.blink_list: list[tuple[float, float]] = []

    def find_blinks(self):  # type: ignore
        """Find blinks in the pupil size data.
        Raises:
            NotImplementedError: Blink detection is not implemented yet.
        """
        warnings.warn(
            "Finding blink based on pupil size traces is not recommended. Use a more reliable method, such as video analysis."
        )
        raise NotImplementedError("Blink detection is not implemented yet.")

    def set_blinks(self, blink_list: list[tuple[float, float]] | None) -> None:
        """Set the blink periods so they can be used for filtering or plotting.

        Args:
            blink_list (list[tuple[float, float]] | None): A list of blink periods, where each period is defined by a start and end time. If None, the blink list is cleared.
        """
        self.blink_list = list(blink_list) if blink_list is not None else []

    def get_blinks(self) -> list[tuple[float, float]]:
        """Get the blink periods.

        Returns:
            list[tuple[float, float]]: The list of blink periods, or an empty list if there are none.
        """
        return self.blink_list

    def set_time_offset(self, offset: float) -> None:
        """Set a time offset for all blinks. This shifts the start and end times of each blink by the specified offset.

        Args:
            offset (float): Time in seconds to add to the start and end of each blink. A negative value
                shifts to earlier times.
        """
        self.blink_list = [
            (start + offset, end + offset) for start, end in self.blink_list
        ]

    def drop_blinks_out_of_time_range(self, start_time: float, end_time: float) -> None:
        """Drop blinks that are outside the specified time range.

        Args:
            start_time (float): The start of the time range.
            end_time (float): The end of the time range.
        """
        self.blink_list = [
            (start, end)
            for start, end in self.blink_list
            if not (end < start_time or start > end_time)
        ]

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all blinks.

        Args:
            ax (Axes | None, optional): Axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            **kwargs: Passed to ax.axvspan. Defaults to color="red" and alpha=0.3.

        Returns:
            Axes: The axes with the plotted blinks.
        """
        blinks = self.get_blinks()
        if "color" not in kwargs:
            kwargs["color"] = "red"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.3
        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        for blink_start, blink_end in blinks:
            ax.axvspan(blink_start, blink_end, **kwargs)  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax
