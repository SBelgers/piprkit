from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..core.pupil_measurement import PupilMeasurement


class PupilBlinks:
    """Organised namespace for blink operations on a PupilMeasurement.

    Access via ``measurement.blinks.<method>()``, e.g.::

        pupMeas.blinks.set_blinks([(0.5, 1.2)])
        pupMeas.blinks.remove_blinks()
    """

    def __init__(self, measurement: "PupilMeasurement") -> None:
        self._m = measurement

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
        self._m.blink_list = blink_list

    def get_blinks(self) -> tuple[tuple[float, float]]:
        """Get the blink periods.

        Raises:
            ValueError: Blink list not set.

        Returns:
            tuple[tuple[float, float]]: The list of blink periods.
        """
        if not hasattr(self._m, "blink_list"):
            raise ValueError("Blink list not set.")
        return self._m.blink_list

    def remove_blinks(self, buffer_seconds: float = 0.0) -> None:
        """Remove blinks from the pupil size data by setting them to NaN."""
        blink_list = self.get_blinks()
        size = self._m.get_size()
        time_data = self._m.get_time()
        for blink_start, blink_end in blink_list:
            mask = (time_data >= blink_start - buffer_seconds) & (time_data <= blink_end + buffer_seconds)
            size[mask] = np.nan
        self._m.set_time_and_size(time_data, size)

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
