from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Union, override

import numpy as np

from .base_fit import BaseFit


class FitBaseline(BaseFit):
    """
    Class used to fit the Baseline phase of the pupil response.

    1. BASELINE (y = y_0)
        - Description: Flat linear model fit to 10s prestimulus period.
        - Defined from:
            - Start Time: -infinity
            - End Time: light stimuli onset
        - Variables:
            - y_0: Baseline pupil diameter (mean before stimulus)
    """

    def __init__(
        self,
        time_data: np.ndarray,
        size: np.ndarray,
        start: float,
        end: float,
    ) -> None:
        if end - start != 10:
            warnings.warn(
                "Baseline measurement is often 10 seconds long. If this is not the case, results may be inaccurate."
            )
        super().__init__(time_data, size, start, end)

    @staticmethod
    @override
    def _model_function(time_data_start_at_0: np.ndarray, c: float) -> np.ndarray:  # type: ignore
        return np.full_like(time_data_start_at_0, c)

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["c"]

    @override
    def get_formula_string(self) -> str:
        c = self.get_params()[0]
        return (
            f"y = {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"
        )

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


class FitLatency(BaseFit):
    """
    Class used to fit the pupillary latency phase of the pupil response.

    2. PLR LATENCY PERIOD (y = undefined)
    - Description: Undefined period after light onset and before constriction.
    - Defined from:
        - Start Time: light onset
        - End Time: light onset + PLR latency
    """

    @staticmethod
    @override
    def _model_function(time_data_start_at_0: np.ndarray) -> np.ndarray:  # type: ignore
        """Model function for latency fitting."""
        return np.full_like(time_data_start_at_0, np.nan)

    @override
    def get_param_names(self) -> Sequence[str]:
        """Get parameter names for latency fitting."""
        return []

    @override
    def get_formula_string(self) -> str:
        """Get formula string for latency fitting."""
        return "The pupil response from light onset to the latency is not defined in this model."

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit the latency model."""
        warnings.warn(
            "Latency fitting is not implemented yet. Returning NaN parameters."
        )


class FitConstrict(BaseFit):
    """
    Class used to fit the constriction phase of the pupil response.

    3. CONSTRICT (y = c_v * (x + t_constrict) + y_0)
        - Description: Linear model from light onset to max constriction.
        - Defined from:
            - Start: light onset + PLR latency
            - End: time of maximum constriction
        - Variables:
            - x: Time since light onset
            - t_constrict: Time shift to light onset + PLR latency
            - c_v: Constriction velocity
            - y_0: Diameter at light onset
    """

    @staticmethod
    @override
    def _model_function(  # type: ignore
        time_data_start_at_0: np.ndarray, c_v: float, y_0: float
    ) -> np.ndarray:  # type: ignore
        return c_v * (time_data_start_at_0) + y_0

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["c_v", "y_0"]

    @override
    def get_formula_string(self) -> str:
        c_v, y_0 = self.get_params()
        return f"y = {c_v:.4f} * (x) + {y_0:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    # TODO Check this initial estimate.
    def fit(
        self,
        initial_estimate: Union[None, tuple[float, float]] = (-3.0, 1.0),
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
            "There is an issue with the implementation of the constriction phase. Defaulting to NaN."
        )
        self._set_params(*([np.nan] * len(self.get_param_names())))


class FitSustain(BaseFit):
    """
    Class used to fit the sustained response phase of the pupil response.

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
    """

    @staticmethod
    @override
    def _model_function(time_data_start_at_0: np.ndarray, m: float, c: float) -> np.ndarray:  # type: ignore
        return m * time_data_start_at_0 + c

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
        estimate: Union[None, tuple[float, float]] = (0.025, 55),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the sustained constriction model.

        Args:
            estimate (Union[None, tuple[float, float]], optional): Initial estimate for the parameters (m, c). Defaults to (0.005, 0.7).
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        p0 = estimate
        self._fit_with_p0(p0, kwargs)
        self._set_params(*([np.nan] * len(self.get_param_names())))


class FitRedilation(BaseFit):
    """
    Class used to fit the redilation phase of the pupil response.

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
    """

    @staticmethod
    @override
    def _model_function(  # type: ignore
        time_data_start_at_0: np.ndarray, s: float, k: float, p: float
    ) -> np.ndarray:  # type: ignore
        return -s * np.exp(k * time_data_start_at_0) + p

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
        estimate: Union[None, tuple[float, float, float]] = (0.45, -0.4, 1),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fit the redilation model.

        Args:
            estimate (Union[None, tuple[float, float, float]], optional): Initial estimate for the parameters (s, k, p). Defaults to (0.7, 0.6, 1).
            kwargs (Optional[dict[str, Any]], optional): Additional arguments for curve fitting. Defaults to None.
        """
        p0 = estimate
        self._fit_with_p0(p0, kwargs)
