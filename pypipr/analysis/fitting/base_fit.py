from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Optional, Sequence, Union, override

import numpy as np
from scipy.optimize import curve_fit  # type: ignore

from ...core.pupil_base import PupilBase

"""
As described by Feigl et al (20...), the pupillary light response can be modelled using linear and exponential models.




Sections Overview:
==================
| Shorthand   | Function Name           | Description                                       | Function                      |
|============ |======================== |===================================================|===============================|
| BASELINE    | Baseline Estimation     | Fit a flat linear model (y = c) to the 10s        | y = c                         |
|             |                         | prestimulus period to estimate the mean baseline  |                               |
|             |                         | pupil diameter.                                   |                               |
|------------ |------------------------ |---------------------------------------------------|-------------------------------|
| CONSTRICT   | Constriction Phase      | Fit a linear model from light onset to            | y = m * (x + t) + c           |
|             |                         | maximum constriction to capture constriction      |                               |
|             |                         | velocity.                                         |                               |
|------------ |------------------------ |---------------------------------------------------|-------------------------------|
| SUSTAINED   | Sustained Constriction  | Fit a linear model from maximum constriction      | y = m * x + c                 |
|             |                         | to light offset to describe maintained            |                               |
|             |                         | constriction.                                     |                               |
|------------ |------------------------ |---------------------------------------------------|-------------------------------|
| REDILATION  | Redilation (PIPR)       | Fit an exponential model to describe redilation   | y = S * exp(k * x) + P        |
|             |                         | kinetics after light offset. Parameters:          |                               |
|             |                         | - S: scaling constant                             |                               |
|             |                         | - k: redilation velocity (GR)                     |                               |
|             |                         | - x: time in seconds                              |                               |
|             |                         | - P: sustained plateau pupil diameter             |                               |
|------------ |------------------------ |---------------------------------------------------|-------------------------------|

"""


class BaseFit(PupilBase):
    """
    Abstract base class for fitting pupil response phases.

    This class provides common functionality for all phase-specific fits,
    including parameter fitting, prediction, and scoring.
    """

    # ============================================================================
    # INITIALIZATION
    # ============================================================================

    def __init__(
        self,
        time_data: np.ndarray,
        size: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> None:
        """Initialize BaseFit with time series data and phase boundaries.

        Args:
            time_data (np.ndarray): Time data for the phase.
            size (np.ndarray): Pupil size data for the phase.
            start_time (float): Start time of the phase.
            end_time (float): End time of the phase.

        Raises:
            ValueError: If the input arrays have incompatible shapes.
        """
        self.start_time = start_time
        self.end_time = end_time
        if len(time_data) != len(size):
            raise ValueError("time and size must have the same length")
        time_data = np.asarray(time_data, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        mask = (time_data >= start_time) & (time_data <= end_time)
        self.time_data = time_data[mask]
        self.size = size[mask]

    # ============================================================================
    # CORE DATA ACCESS METHODS
    # ============================================================================

    @override
    def get_time(self) -> np.ndarray:
        return self.time_data

    @override
    def get_size(self) -> np.ndarray:
        return self.predict(self.get_time())
    
    def get_start_time(self) -> float:
        """Get the start time of the phase."""
        return self.start_time
    def get_end_time(self) -> float:
        """Get the end time of the phase."""
        return self.end_time

    # ============================================================================
    # PARAMETER MANAGEMENT
    # ============================================================================

    def get_params(self) -> tuple[float, ...]:
        return self._params

    def _set_params(self, *params: float) -> None:
        self._params = params

    @abstractmethod
    def get_param_names(self) -> Sequence[str]:
        pass

    # ============================================================================
    # TIME OFFSET MANAGEMENT
    # ============================================================================

    def set_time_offset(self, offset: float) -> None:
        """Set time offset for model alignment."""
        self._time_offset = offset

    def get_time_offset(self) -> float:
        """Get the current time offset for model alignment."""
        return self._time_offset if hasattr(self, "_time_offset") else 0.0

    # ============================================================================
    # FITTING METHODS
    # ============================================================================
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _fit_with_p0(
        self,
        p0: Union[None, tuple[float, ...]],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if p0 is not None:
            if kwargs is None:
                kwargs = {"p0": p0}
            else:
                kwargs["p0"] = p0
        if kwargs is None:
            kwargs = {}
        if "nan_policy" not in kwargs:
            kwargs["nan_policy"] = "omit"

        try:
            time_offset = self.get_time() + self.get_time_offset()
            popt, _ = curve_fit(self._model_function, time_offset, self.size, **kwargs)  # type: ignore
            self._set_params(*popt)  # type: ignore
        except RuntimeError as e:
            warnings.warn(f"Fit failed: {e}")
            self._set_params(*([np.nan] * len(self.get_param_names())))  # type: ignore

    # ============================================================================
    # PREDICTION AND EVALUATION
    # ============================================================================

    def predict(self, time_data: np.ndarray) -> np.ndarray:
        """Predict pupil size for given time points using the fitted model.

        Args:
            time_data (np.ndarray): Time array to predict for

        Returns:
            np.ndarray: Predicted pupil sizes
        """
        time_offset = time_data + self.get_time_offset()
        return self._model_function(time_offset, *self.get_params())

    # ============================================================================
    # ABSTRACT METHODS (TO BE IMPLEMENTED BY SUBCLASSES)
    # ============================================================================

    @staticmethod
    @abstractmethod
    def _model_function(time_data: np.ndarray, *params: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_formula_string(self) -> str:
        """Get a string representation of the fitted model formula."""
        pass
