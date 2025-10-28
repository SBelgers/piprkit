from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from .pupil_base import PupilBase
from ..preprocessing.filtering import FilterMixin
from ..analysis.pupil_metrics import PupilMetricsMixin

class PupilMeasurement(PupilMetricsMixin, FilterMixin, PupilBase):
    """A class representing a single pupil measurement with time series data."""

    # ============================================================================
    # INITIALIZATION AND CORE DATA METHODS
    # ============================================================================

    def __init__(
        self, time_data: NDArray[np.number], size_data: NDArray[np.number]
    ) -> None:
        """Initialize PupilMeasurement with time and size data.

        Args:
            time_data (NDArray[np.number]): Array of time data.
            size (NDArray[np.number]): Array of size data.

        Raises:
            ValueError: If the time and size arrays do not have the same length.
        """
        if len(size_data) != len(time_data):
            raise ValueError("Size and time_data must have the same length")
        self.set_time_and_size(time_data, size_data)






    # ============================================================================
    # BLINK MANAGEMENT
    # ============================================================================

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
