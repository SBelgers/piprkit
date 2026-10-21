from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..analysis.pupil_metrics import PupilMetrics
from .pupil_base import PupilBase


class PupilMeasurement(PupilBase):
    """A class representing a single pupil measurement with time series data."""

    def __init__(
        self, time_data: NDArray[np.number], size_data: NDArray[np.number]
    ) -> None:
        """Initialize PupilMeasurement with time and pupil size data.

        Args:
            time_data (NDArray[np.number]): 1-D array with the time of each sample in seconds,
                strictly increasing and without NaN.
            size_data (NDArray[np.number]): 1-D array with the pupil size of each sample (usually
                the diameter in mm), same length as ``time_data``. NaN marks missing samples.

        Raises:
            ValueError: If the arrays differ in length, or if time_data is not strictly
                increasing or contains NaN.
        """
        if len(size_data) != len(time_data):
            raise ValueError("Size and time_data must have the same length")
        self.set_time_and_size(time_data, size_data)

    @property
    def metrics(self) -> PupilMetrics:
        """The pupil metrics of this measurement.

        Each metric is a method, for example ``measurement.metrics.pipr_6s()``.
        See :class:`~piprkit.analysis.pupil_metrics.PupilMetrics` for all metrics.
        """
        return PupilMetrics(self)
