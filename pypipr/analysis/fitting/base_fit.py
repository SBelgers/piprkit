from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Optional, Sequence, Union, override

import numpy as np
from scipy.optimize import (  # type: ignore
    curve_fit,  # type: ignore
    minimize,  # type: ignore
)

from ...core.pupil_base import PupilBase

# TODO: Add automatic phase detection and fitting. Check if t_constrict, t_sustain, and t_redilation is correctly implemented.

"""
Pupillary Light Response Model

Models pupil diameter changes in response to light stimuli using linear and
exponential functions, based on Feigl et al. (2011).


Phases:
-------

1. BASELINE (y = y0)
    - Description: Flat linear model fit to 10s prestimulus period.
    - Defined from:
        - Start Time: -infinity
        - End Time: light stimuli onset 
    - Variables:
        - y0: Baseline pupil diameter (mean before stimulus)
2. PLR LATENCY PERIOD (y = undefined)
    - Description: Undefined period after light onset and before constriction.
    - Defined from:
        - Start Time: light onset
        - End Time: light onset + PLR latency
3. CONSTRICT (y = cv * (x + t_constrict) + y0)
    - Description: Linear model from light onset to max constriction.
    - Defined from:
        - Start: light onset + PLR latency
        - End: time of maximum constriction
    - Variables:
        - x: Time since light onset
        - t_constrict: Time shift to light onset + PLR latency
        - cv: Constriction velocity
        - y0: Baseline pupil diameter
4. SUSTAINED (y = m * (x + t_sustained) + c)
    - Description: Linear model from max constriction to light offset.
    - Defined from:
        - Start: time of maximum constriction
        - End: light offset
    - Variables:
        - x: Time since max constriction
        - t_sustained: Time shift to max constriction
        - m: Slope of sustained constriction
        - c: Diameter at max constriction
5. REDILATION (y = S * exp(k * (x + t_redilation) + P)
    - Description: Exponential model for redilation after light offset.
    - Defined from:
        - Start: light offset
        - End: + infinity
    - Variables:
        - x: Time since light offset
        - t_redilation: Time shift to light offset
        - S: Scaling constant
        - k: Redilation velocity
        - P: Plateau pupil diameter

Notes:
------
- All time values (x) are in seconds.
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
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Initialize BaseFit with time series data and phase boundaries.
        1. BASELINE (y = y_0)
            - Description: Flat linear model fit to 10s prestimulus period.
            - Defined from:
                - Start Time: -infinity
                - End Time: light stimuli onset
            - Variables:
                - y_0: Baseline pupil diameter (mean before stimulus)


        Args:
            time_data (np.ndarray): Time data for the phase.
            size (np.ndarray): Pupil size data for the phase.
            estimated_start_time (Optional[float]): Estimated start time of the phase.
            estimated_end_time (Optional[float]): Estimated end time of the phase.

        Raises:
            ValueError: If the input arrays have incompatible shapes.
        """
        if start_time is None or end_time is None:
            raise NotImplementedError("Automatic phase detection not implemented yet.")
        self.start_time = start_time
        self.end_time = end_time
        if len(time_data) != len(size):
            raise ValueError("time and size must have the same length")
        time_data = np.asarray(time_data, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        mask = (time_data >= start_time) & (time_data <= end_time)
        self.time_data = time_data[mask]
        self.size = size[mask]
        self.set_time_offset(self.start_time)

    # ============================================================================
    # CORE DATA ACCESS METHODS
    # ============================================================================

    @override
    def get_time(self) -> np.ndarray:
        return self.time_data

    @override
    def get_size(self) -> np.ndarray:
        return self.predict(self.get_time())

    def set_start_time(self, start_time: float) -> None:
        """Set the start time of the phase."""
        self.start_time = start_time

    def get_start_time(self) -> float:
        """Get the start time of the phase."""
        return self.start_time

    def set_end_time(self, end_time: float) -> None:
        """Set the end time of the phase."""
        self.end_time = end_time

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
        if self.get_size().shape[0] < 3:
            warnings.warn(
                "Not enough data points to perform fit. Need at least 3 data points."
            )
            self._set_params(*([np.nan] * len(self.get_param_names())))  # type: ignore
            return    
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
            popt, _ = curve_fit(self._model_function, time_offset, self.get_size(), **kwargs)  # type: ignore
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

    def goodness_of_fit(self, method: str = "MAE") -> float:
        options = ["MAE"]
        if method not in options:
            raise ValueError(f"Unknown method: {method}")
        if method == "MAE":
            return np.nanmean(np.abs(self.size - self.predict(self.get_time()))).astype(
                float
            )  # type: ignore
        return np.nan

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

    # -----------------------------------------------------------------------------
    # Utility: Optimize phase boundaries for best fit
    # -----------------------------------------------------------------------------

    def optimize_phase_boundary_and_params(
        self,
        time_data: np.ndarray,
        size: np.ndarray,
        estimated_time: tuple[float, float],
        time_bounds: list[tuple[float, float]],
        estimated_fit_params: list[float],
        fit_param_bounds: list[tuple[float, float]],
    ):
        """
        Optimize start/end times and fit parameters to minimize fit error.

        Args:
            time_data (np.ndarray): Full time data.
            size (np.ndarray): Full size data.
            estimated_time (tuple[float, float]): Initial guess for (start, end) times.
            time_bounds (list[tuple[float, float]]): Bounds for start/end times.
            estimated_fit_params (list[float]): Initial guess for fit parameters.
            fit_param_bounds (list[tuple[float, float]]): Bounds for fit parameters.

        Returns:
            result: scipy.optimize.OptimizeResult
        """

        def optimize_time_func(start_time: float, end_time: float):
            # create a new copy of self
            new_fit = type(self)(
                time_data=time_data,
                size=size,
                start_time=start_time,
                end_time=end_time,
            )
            new_fit.fit()
            return new_fit.goodness_of_fit()

        x0 = estimated_time
        bounds = time_bounds + fit_param_bounds
        result = minimize(optimize_time_func, x0, bounds=bounds, method="L-BFGS-B")
        return result
