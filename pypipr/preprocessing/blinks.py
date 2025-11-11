from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..core.pupil_base import PupilBase


class BlinkMixin(PupilBase):
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

    def remove_blinks(self) -> None:
        """Remove blinks from the pupil size data by setting them to NaN."""
        blink_list = self.get_blinks()
        size = self.get_size()
        time_data = self.get_time()
        for blink_start, blink_end in blink_list:
            mask = (time_data >= blink_start) & (time_data <= blink_end)
            size[mask] = np.nan
        self.set_time_and_size(time_data, size)

    def plot_blinks(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot all blinks.

        Args:
            ax (Optional[Axes], optional): Axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.

        Returns:
            Axes: The axes with the plotted light stimuli.
        """
        blinks = self.get_blinks()
        if "color" not in kwargs:
            kwargs["color"] = "red"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.3
        if ax is None:
            fig, ax = plt.subplots()
        for blink_start, blink_end in blinks:
            ax.axvspan(blink_start, blink_end, **kwargs)
        if show:
            plt.show()
        return ax
