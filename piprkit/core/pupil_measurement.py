from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..analysis.pupil_metrics import PupilMetricsMixin
from ..preprocessing.filtering import FilterMixin
from ..preprocessing.blinks import BlinkMixin
from .pupil_base import PupilBase


class PupilMeasurement(PupilMetricsMixin, FilterMixin, BlinkMixin, PupilBase):
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
        