from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Union, override

import numpy as np

from .base_fit import BaseFit


class FitBaseline(BaseFit):
    """
    Fit a constant baseline model to pre-stimulus data.

    | Shorthand   | Function Name           | Description                                       | Function                      |
    |============ |======================== |===================================================|===============================|
    | BASELINE    | Baseline Estimation     | Fit a flat linear model (y = c) to the 10s        | y = c                         |
    |             |                         | prestimulus period to estimate the mean baseline  |                               |
    |             |                         | pupil diameter.                                   |                               |
    |------------ |------------------------ |---------------------------------------------------|-------------------------------|
    """

    def __init__(
        self,
        time_data: np.ndarray,
        size: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> None:
        if end_time - start_time != 10:
            warnings.warn(
                "Baseline measurement is often 10 seconds long. If this is not the case, results may be inaccurate."
            )
        super().__init__(time_data, size, start_time, end_time)

    @staticmethod
    @override
    def _model_function(time_data: np.ndarray, c: float) -> np.ndarray:  # type: ignore
        return np.full_like(time_data, c)

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["c"]

    @override
    def get_formula_string(self) -> str:
        c = self.get_params()[0]
        return f"y = {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        baseline_estimate: Union[None, float] = 1,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the baseline model.

        Args:
            baseline_estimate (Union[None, float], optional): Initial estimate for the baseline. Defaults to 1.
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        if baseline_estimate is None:
            p0 = None
        else:
            p0 = (baseline_estimate,)
        self._fit_with_p0(p0, kwargs)


class FitConstrict(BaseFit):
    """
    Fit a linear model for the constriction phase of the pupillary light response.

    | Shorthand   | Function Name           | Description                                       | Function                      |
    |============ |======================== |===================================================|===============================|
    | CONSTRICT   | Constriction Phase      | Fit a linear model from light onset to            | y = m * (x + t) + c           |
    |             |                         | maximum constriction to capture constriction      |                               |
    |             |                         | velocity.                                         |                               |
    |------------ |------------------------ |---------------------------------------------------|-------------------------------|
    """

    @staticmethod
    @override
    def _model_function( # type: ignore
        time_data: np.ndarray, m: float, t: float, c: float
    ) -> np.ndarray:  # type: ignore
        return m * (time_data + t) + c

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["m", "t", "c"]

    @override
    def get_formula_string(self) -> str:
        m, t, c = self.get_params()
        return f"y = {m:.4f} * (x + {t:.4f}) + {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        initial_estimate: Union[None, tuple[float, float, float]] = (0.5, 0.2, 1.0),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the constriction model.

        Args:
            initial_estimate (Union[None, tuple[float, float, float]], optional): Initial estimate for the parameters (m, t, c). Defaults to (0.5, 0.2, 1.0).
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        p0 = initial_estimate
        self._fit_with_p0(p0, kwargs)
        # TODO: There is an error here. For now, set the fit to np.nan.
        warnings.warn(
            "There is an issue with the implementation of this fit. Defaulting to NaN."
        )
        self._set_params(*([np.nan] * len(self.get_param_names())))


class FitSustain(BaseFit):
    """
    Fit a linear model for sustained constriction phase.

    | Shorthand   | Function Name           | Description                                       | Function                      |
    |============ |======================== |===================================================|===============================|
    | SUSTAINED   | Sustained Constriction  | Fit a linear model from maximum constriction      | y = m * x + c                 |
    |             |                         | to light offset to describe maintained            |                               |
    |             |                         | constriction.                                     |                               |
    |------------ |------------------------ |---------------------------------------------------|-------------------------------|
    """

    @staticmethod
    @override
    def _model_function(time_data: np.ndarray, m: float, c: float) -> np.ndarray:  # type: ignore
        return m * time_data + c

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["m", "c"]

    @override
    def get_formula_string(self) -> str:
        m, c = self.get_params()
        time_offset = self.get_time_offset()
        return f"y = {m:.4f} * (x + {time_offset:.4f}) + {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        estimate: Union[None, tuple[float, float]] = (0.005, 0.7),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the sustained constriction model.

        Args:
            estimate (Union[None, tuple[float, float]], optional): Initial estimate for the parameters (m, c). Defaults to (0.005, 0.7).
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        p0 = estimate
        self._fit_with_p0(p0, kwargs)


class FitRedilation(BaseFit):
    """
        Fit an exponential redilation model for post-illumination pupil response (PIPR).

    | Shorthand   | Function Name           | Description                                       | Function                      |
    |============ |======================== |===================================================|===============================|
    | REDILATION  | Redilation (PIPR)       | Fit an exponential model to describe redilation   | y = S * exp(k * x) + P        |
    |             |                         | kinetics after light offset. Parameters:          |                               |
    |             |                         | - S: scaling constant                             |                               |
    |             |                         | - k: redilation velocity (GR)                     |                               |
    |             |                         | - x: time in seconds                              |                               |
    |             |                         | - P: sustained plateau pupil diameter             |                               |
    |------------ |------------------------ |---------------------------------------------------|-------------------------------|
    """

    @staticmethod
    @override
    def _model_function( # type: ignore
        time_data: np.ndarray, s: float, k: float, p: float
    ) -> np.ndarray:  # type: ignore
        return -s * np.exp(-k * time_data) + p

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["s", "k", "p"]

    @override
    def get_formula_string(self) -> str:
        s, k, p = self.get_params()
        time_offset = self.get_time_offset()
        return f"y = -{s:.4f} * exp(-{k:.4f} * (x + {time_offset:.4f})) + {p:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        estimate: Union[None, tuple[float, float, float]] = (0.7, 0.6, 1),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the redilation model.

        Args:
            estimate (Union[None, tuple[float, float, float]], optional): Initial estimate for the parameters (s, k, p). Defaults to (0.7, 0.6, 1).
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        p0 = estimate
        self._fit_with_p0(p0, kwargs)
