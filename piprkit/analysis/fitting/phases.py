from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import numpy as np

from .base_model import Phase


class BaselinePhase(Phase):
    """Baseline phase: a constant pupil size.

    .. math:: y = c

    c is the baseline pupil size. A PhaseModel sets it to ``metrics.baseline(baseline_duration)``,
    the mean of the valid samples from light onset - baseline_duration to light onset; it does not
    use curve_fit.

    References:
        - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
    """

    @staticmethod
    @override
    def _model_function(time_data_start_at_0: np.ndarray, c: float) -> np.ndarray:  # type: ignore
        """References:
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
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

    @override
    def fit(self, p0: float) -> None:
        """Set the baseline c; there is nothing to fit.

        Args:
            p0 (float): The baseline pupil size c, e.g. the mean of the samples in the window.
        """
        self._set_params(p0)


class ConstrictionPhase(Phase):
    """Constriction phase: a straight line.

    .. math:: y = m x + c

    x is the time since the start of the window, in seconds. m is the slope in size units per second
    (negative for a constriction), and c the size at the start of the window. A PhaseModel uses the
    window from the constriction start to the constriction end, found by its breakpoint search.
    """

    @staticmethod
    @override
    def _model_function(  # type: ignore
        time_data_start_at_0: np.ndarray, m: float, c: float
    ) -> np.ndarray:  # type: ignore
        """References:
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return m * time_data_start_at_0 + c

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["m", "c"]

    @override
    def get_formula_string(self) -> str:
        m, c = self.get_params()
        time_offset = self.get_time_offset()
        return f"y = {m:.4f} * (x - {time_offset:.4f}) + {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        p0: None | tuple[float, float] = (-0.5, 1.0),
        kwargs: dict[str, Any] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        """Fit m and c to the samples in the window with scipy.optimize.curve_fit.

        NaN samples are skipped. The result sets the parameters, the flags (see get_flags) and the
        options (see get_options).

        Args:
            p0 (None | tuple[float, float], optional): Start estimate of the parameters (m, c). If it lies
                outside the bounds it is moved to the nearest bound, with a warning. Defaults to (-0.5, 1.0).
            kwargs (dict[str, Any] | None, optional): Extra keyword arguments for curve_fit. Defaults to None.
            bounds (tuple[Sequence[float], Sequence[float]] | None, optional): Lower and upper bounds of the
                parameters, as (lower, upper). None means unbounded. Defaults to None.
        """
        self._fit_with_p0(p0, kwargs, bounds)


class MaintainedConstrictionPhase(Phase):
    """Maintained constriction phase: a straight line.

    .. math:: y = m x + c

    x is the time since the start of the window, in seconds. m is the slope in size units per second,
    and c the size at the start of the window. A PhaseModel uses the window from the peak
    constriction to the redilation start.
    """

    @staticmethod
    @override
    def _model_function(time_data_start_at_0: np.ndarray, m: float, c: float) -> np.ndarray:  # type: ignore
        """References:
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return m * time_data_start_at_0 + c

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["m", "c"]

    @override
    def get_formula_string(self) -> str:
        m, c = self.get_params()
        time_offset = self.get_time_offset()
        return f"y = {m:.4f} * (x - {time_offset:.4f}) + {c:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        p0: None | tuple[float, float] = (0.025, 0.55),
        kwargs: dict[str, Any] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        """Fit m and c to the samples in the window with scipy.optimize.curve_fit.

        NaN samples are skipped. The result sets the parameters, the flags (see get_flags) and the
        options (see get_options).

        Args:
            p0 (None | tuple[float, float], optional): Start estimate of the parameters (m, c). If it lies
                outside the bounds it is moved to the nearest bound, with a warning. Defaults to (0.025, 0.55).
            kwargs (dict[str, Any] | None, optional): Extra keyword arguments for curve_fit. Defaults to None.
            bounds (tuple[Sequence[float], Sequence[float]] | None, optional): Lower and upper bounds of the
                parameters, as (lower, upper). None means unbounded. Defaults to None.
        """
        self._fit_with_p0(p0, kwargs, bounds)


class RedilationPhase(Phase):
    """Redilation phase: an exponential.

    .. math:: y = S e^{k x} + P

    This is the exponential model of the post-illumination pupil response (Feigl et al., 2011). x is
    the time since the start of the window, in seconds. P is the level the curve approaches, k
    the rate constant in 1/s (negative for a curve that levels off), and S the difference between
    the size at the start of the window and P. A PhaseModel uses the window from the redilation
    start to the redilation end, by default the last sample. A fit with S > 0, a curve that falls instead of redilating, is
    marked not defined.

    The phase can also be fitted on its own, to any window of pupil size data, without a
    PhaseModel. For example, from light offset (here 0 s) to the end of the data; ``get_params()``
    then returns S, k and P::

        phase = piprkit.RedilationPhase(pm.get_time(), pm.get_size(), start_time=0.0)
        phase.fit()
        phase.get_params()

    References:
        - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
    """


    @staticmethod
    @override
    def _model_function(  # type: ignore
        time_data_start_at_0: np.ndarray, S: float, k: float, P: float
    ) -> np.ndarray:  # type: ignore
        """References:
            - Feigl, B., Mattes, D., Thomas, R., & Zele, A. J. (2011). Intrinsically Photosensitive (Melanopsin) Retinal Ganglion Cell Function in Glaucoma. Investigative Ophthalmology & Visual Science, 52(7), 4362–4367. https://doi.org/10.1167/iovs.10-7069
        """
        return S * np.exp(k * time_data_start_at_0) + P

    @override
    def get_param_names(self) -> Sequence[str]:
        return ["S", "k", "P"]

    @override
    def get_formula_string(self) -> str:
        S, k, P = self.get_params()
        time_offset = self.get_time_offset()
        return f"y = {S:.4f} * exp({k:.4f} * (x - {time_offset:.4f})) + {P:.4f} from {self.get_start_time():.4f} to {self.get_end_time():.4f}"

    def fit(
        self,
        p0: None | tuple[float, float, float] = (-0.45, -0.4, 1.0),
        kwargs: dict[str, Any] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = (
            (-np.inf, -np.inf, -np.inf),
            (np.inf, 0.0, np.inf),
        ),
    ) -> None:
        """Fit S, k and P to the samples in the window with scipy.optimize.curve_fit.

        NaN samples are skipped. The result sets the parameters, the flags (see get_flags) and the
        options (see get_options).

        Args:
            p0 (None | tuple[float, float, float], optional): Start estimate of the parameters (S, k, P). If
                it lies outside the bounds it is moved to the nearest bound, with a warning. Defaults to
                (-0.45, -0.4, 1.0), with k from Feigl et al. (2011).
            kwargs (dict[str, Any] | None, optional): Extra keyword arguments for curve_fit. Defaults to None.
            bounds (tuple[Sequence[float], Sequence[float]] | None, optional): Lower and upper bounds of the
                parameters, as (lower, upper). None means unbounded. Defaults to k <= 0, the rest unbounded.
        """
        self._fit_with_p0(p0, kwargs, bounds)
        S = self.get_params()[0]
        if np.isfinite(S) and S > 0:
            self._set_not_defined("S > 0: the fitted curve falls instead of redilating.")

    # k: Feigl et al. (2011)
    def _data_estimate(self, k: float = -0.4) -> tuple[float, float, float] | None:
        size = self.get_size()
        size = size[~np.isnan(size)]
        if size.size < 2:
            return None
        plateau = float(np.mean(size[-max(1, size.size // 5):]))
        return float(size[0]) - plateau, k, plateau
