from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, override

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore

from ...core.pupil_base import TimeSeriesBase
from ...core.pupil_measurement import PupilMeasurement
from ...utils.utils import check_time_series


class BaseModel(TimeSeriesBase):
    def __init__(self, time_data: np.ndarray, size_data: np.ndarray) -> None:
        """Base class for fitting pupil response models to time series data.

        Args:
            time_data (np.ndarray): Time data for the pupil response.
            size_data (np.ndarray): Pupil size data corresponding to the time data.

        Raises:
            ValueError: If the input arrays have incompatible shapes.
        """
        if len(time_data) != len(size_data):
            raise ValueError("time and size must have the same length")
        self.time_data = np.asarray(time_data, dtype=np.float64)
        self.size_data = np.asarray(size_data, dtype=np.float64)

    @override
    def get_time(self) -> np.ndarray:
        """Return the time of each sample the model is fitted to.

        Returns:
            np.ndarray: 1-D float array of times in seconds. For a phase, only the samples inside its
            window. This is the stored array, not a copy.
        """
        return self.time_data

    @override
    def get_size(self) -> np.ndarray:
        """Return the pupil size of each sample the model is fitted to.

        Returns:
            np.ndarray: 1-D float array with the same length as :meth:`get_time`, in the units of the
            fitted data. NaN marks missing samples. This is the stored array, not a copy.
        """
        return self.size_data

    # Prediction

    @abstractmethod
    def _predict(self, time_data: np.ndarray) -> np.ndarray:
        pass

    def get_predicted(
        self, time_data: NDArray[np.number] | None = None, extrapolation: str = "error"
    ) -> np.ndarray:
        """Get the pupil size predicted by the fitted model.

        Args:
            time_data (NDArray[np.number] | None, optional): Time points to predict at, in seconds,
                strictly increasing. If None, the time points of the fitted data are used. Defaults to None.
            extrapolation (str, optional): What to do when time_data runs outside the time of the fitted
                data: "allow", "warn" or "error". Defaults to "error".

        Raises:
            ValueError: If time_data contains NaNs, is not sorted or contains duplicates, if extrapolation
                is unknown, or if it is "error" and time_data runs outside the fitted data.

        Returns:
            np.ndarray: Predicted pupil size at each time point. NaN in the delay of a phase, and for a
            PhaseModel in a phase without a fit and in the stretches next to it.
        """
        if time_data is None:
            return self._predict(self.get_time())
        check_time_series(time_data)
        time_data = np.asarray(time_data, dtype=np.float64)
        self._check_extrapolation(time_data, extrapolation)
        return self._predict(time_data)

    def to_measurement(
        self, time_data: NDArray[np.number] | None = None, extrapolation: str = "error"
    ) -> PupilMeasurement:
        """Convert the fitted model to a PupilMeasurement instance.

        Args:
            time_data (NDArray[np.number] | None, optional): The time points at which to predict the pupil size. Defaults to None.
            extrapolation (str, optional): The method to use for handling time points outside the range of the fitted data. Options are "allow", "warn", and "error". Defaults to "error".

        Returns:
            PupilMeasurement: A PupilMeasurement instance containing the time and size data of the fitted model.
        """        
        
        time_data = self.get_time() if time_data is None else np.asarray(time_data, dtype=np.float64)
        return PupilMeasurement(time_data, self.get_predicted(time_data, extrapolation))

    def _check_extrapolation(self, time_data: np.ndarray, extrapolation: str) -> None:
        options = ["allow", "warn", "error"]
        if extrapolation not in options:
            raise ValueError(f"Unknown extrapolation: {extrapolation}. Options: {options}")
        if extrapolation == "allow":
            return
        original_time = self.get_time()
        if original_time.size == 0:
            outside = time_data.size > 0
        else:
            outside = bool(np.any((time_data < np.min(original_time)) | (time_data > np.max(original_time))))
        if not outside:
            return
        message = "time_data runs outside the time of the fitted data, so the result is extrapolated."
        if extrapolation == "error":
            raise ValueError(message)
        warnings.warn(message)

    @property
    def metrics(self):
        """Not available on a single phase.

        The metrics of a fit are on the ``metrics`` of a PhaseModel. For other metrics, convert the fit
        with :meth:`to_measurement` and use the ``metrics`` of the result.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(
            "A fit has no metrics. Use to_measurement() first and compute metrics on the result."
        )

    def plot_predicted(
        self,
        ax: Axes | None = None,
        show: bool = False,
        time_data: NDArray[np.number] | None = None,
        extrapolation: str = "error",
        **kwargs: Any,
    ) -> Axes:
        """Plot the predicted pupil size of the fitted model.

        Args:
            ax (Axes | None, optional): The axes to plot on. If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
            time_data (NDArray[np.number] | None, optional): Time points to predict at. If None, the
                time points of the fitted data are used. Defaults to None.
            extrapolation (str, optional): What to do when time_data runs outside the time of the fitted
                data: "allow", "warn" or "error". Defaults to "error".
            **kwargs: Passed to ``ax.plot``.

        Returns:
            Axes: The axes with the plotted prediction.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
            ax.set_xlabel("Time (s)")  # type: ignore
            ax.set_ylabel(self._y_label())  # type: ignore
        time_data = self.get_time() if time_data is None else np.asarray(time_data, dtype=np.float64)
        ax.plot(time_data, self.get_predicted(time_data, extrapolation), **kwargs)  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax

    # Fitting

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_formula_string(self) -> str:
        """Get a string representation of the fitted model formula."""

    def get_goodness_of_fit(self, method: str = "MAE") -> float:
        """Calculate the goodness of fit for the model.

        Args:
            method (str, optional): "MAE" (mean absolute error), "RMSE" (root mean square error) or
                "R2" (coefficient of determination). Defaults to "MAE".

        Raises:
            ValueError: Raised if an unknown method is provided.

        Returns:
            float: The goodness of fit value
        """
        return self._calculate_goodness_of_fit(
            self.get_size(), self.get_predicted(), method
        )

    @staticmethod
    def _calculate_goodness_of_fit(
        measured: np.ndarray, predicted: np.ndarray, method: str = "MAE"
    ) -> float:
        """Calculate the goodness of fit between measured and predicted data.

        Args:
            measured (np.ndarray): Measured data points
            predicted (np.ndarray): Predicted data points from the model
            method (str, optional): The method to use for calculating goodness of fit. Defaults to "MAE".
                Options:
                    - "MAE": Mean Absolute Error
                    - "RMSE": Root Mean Square Error
                    - "R2": Coefficient of determination

        Raises:
            ValueError: Raised if an unknown method is provided.

        Returns:
            float: The goodness of fit value, which can be MAE, RMSE, or R2 depending on the selected method.
        """
        options = ["MAE", "RMSE", "R2"]
        if method not in options:
            raise ValueError(f"Unknown method: {method}. Options: {options}")
        valid = np.isfinite(measured) & np.isfinite(predicted)
        if not np.any(valid):
            return np.nan
        residuals = measured[valid] - predicted[valid]
        if method == "MAE":
            return float(np.mean(np.abs(residuals)))
        if method == "RMSE":
            return float(np.sqrt(np.mean(residuals**2)))
        total = np.sum((measured[valid] - np.mean(measured[valid])) ** 2)
        return float(1 - np.sum(residuals**2) / total) if total > 0 else np.nan

    # Fit records

    @abstractmethod
    def get_window(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_flags(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_options(self) -> dict[str, Any]:
        pass


class Phase(BaseModel):
    """
    Abstract base class for fitting pupil response phases.

    This class provides common functionality for all phase-specific fits,
    including parameter fitting, prediction, and scoring.

    """

    def __init__(
        self,
        time_data: np.ndarray,
        size_data: np.ndarray,
        start_time: float,
        end_time: float | None = None,
        delay: float = 0.0,
        min_samples: int = 3,
    ) -> None:
        """Select the samples of one phase and prepare it for fitting.

        The phase keeps the samples from start_time + delay to end_time. x = 0 in its formula is at
        start_time.

        Args:
            time_data (np.ndarray): Time of each sample of the whole measurement, in seconds.
            size_data (np.ndarray): Pupil size of each sample, same length as time_data.
            start_time (float): Start of the window, in seconds. Also x = 0 of the formula.
            end_time (float | None, optional): End of the window, in seconds. None means the last sample.
                Defaults to None.
            delay (float, optional): Time after start_time, in seconds, whose samples are left out of the
                fit. The prediction is NaN there, and x = 0 stays at start_time. Defaults to 0.0.
            min_samples (int, optional): Fewest valid samples needed for a fit. With fewer, the
                too_few_samples flag is set and the parameters are NaN. Defaults to 3.

        Raises:
            ValueError: If time_data and size_data differ in length.
        """
        if len(time_data) != len(size_data):
            raise ValueError("time and size must have the same length")
        time_data = np.asarray(time_data, dtype=np.float64)
        size_data = np.asarray(size_data, dtype=np.float64)
        if end_time is None:
            end_time = float(np.max(time_data)) if time_data.size else np.inf
        mask = (time_data >= start_time + delay) & (time_data <= end_time)
        super().__init__(time_data[mask], size_data[mask])
        self.start_time = start_time
        self.end_time = end_time
        self.delay = delay
        self.min_samples = min_samples
        self._set_time_offset(self.start_time)
        self._flags = self._no_flags()
        self._options: dict[str, Any] = {}
        self._set_params(*([np.nan] * len(self.get_param_names())))

    def _set_start_time(self, start_time: float) -> None:
        self.start_time = start_time

    def get_start_time(self) -> float:
        """Return the start of the window. It is also x = 0 of the formula.

        Returns:
            float: The start of the window, in seconds.
        """
        return self.start_time

    def _set_end_time(self, end_time: float) -> None:
        self.end_time = end_time

    def get_end_time(self) -> float:
        """Return the end of the window.

        Returns:
            float: The end of the window, in seconds.
        """
        return self.end_time

    def get_delay(self) -> float:
        """Return the delay: the time after the start of the window whose samples are left out of the fit.

        The prediction is NaN in [start, start + delay). The delay is 0 unless it was given when the
        phase was created; a PhaseModel uses 0.

        Returns:
            float: The delay, in seconds.
        """
        return self.delay

    def plot_window(
        self, ax: Axes | None = None, show: bool = False, **kwargs: Any
    ) -> Axes:
        """Plot the time window of the phase on the given axes.

        Args:
            ax (Axes | None, optional): The axes on which to plot the data. If None, a new figure and axes will be created. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to False.
            **kwargs: Passed to ``ax.axvspan`` for the window. Defaults: alpha 0.15.

        Returns:
            Axes: The axes on which the data was plotted.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
        kwargs.setdefault("alpha", 0.15)
        start, end, delay = self.get_start_time(), self.get_end_time(), self.get_delay()
        ax.axvspan(start + delay, end, **kwargs)  # type: ignore
        if delay > 0:
            ax.axvspan(  # type: ignore
                start,
                start + delay,
                fill=False,
                hatch="//",
                color=kwargs.get("color"),
                label="_nolegend_",
            )
        if show:
            plt.show()  # type: ignore
        return ax

    # Parameter stuff

    def get_params(self) -> tuple[float, ...]:
        """Return the fitted parameters, in the order of :meth:`get_param_names`.

        Returns:
            tuple[float, ...]: The parameters. NaN before the fit, and when the fit failed (see
            :meth:`get_flags`).
        """
        return self._params

    def _set_params(self, *params: float) -> None:
        self._params = params

    @abstractmethod
    def get_param_names(self) -> Sequence[str]:
        """Return the names of the parameters, in the order of :meth:`get_params`.

        Returns:
            Sequence[str]: The parameter names, e.g. ["m", "c"].
        """

    # Fit records

    @staticmethod
    def _no_flags() -> dict[str, Any]:
        return {
            "not_defined": False,
            "reason": None,
            "not_converged": False,
            "at_bounds": (),
            "too_few_samples": False,
        }

    def _set_not_defined(self, reason: str) -> None:
        self._flags = self._no_flags()
        self._flags["not_defined"] = True
        self._flags["reason"] = reason
        self._set_params(*([np.nan] * len(self.get_param_names())))

    @override
    def get_window(self) -> dict[str, Any]:
        """Get the time window of the phase.

        Returns:
            dict[str, Any]: A dict with the ``start`` and ``end`` of the window in seconds, the
            ``delay``, and ``n_samples``, the number of valid samples in the window after the delay.
        """
        n_samples = int(np.sum(np.isfinite(self.get_size())))
        return {
            "start": self.get_start_time(),
            "end": self.get_end_time(),
            "delay": self.get_delay(),
            "n_samples": n_samples,
        }

    @override
    def get_flags(self) -> dict[str, Any]:
        """Get the flags that say whether the fit worked.

        - ``not_defined`` (bool): the phase is marked as not defined, e.g. because its window was not
          found; ``reason`` (str | None) says why.
        - ``not_converged`` (bool): ``curve_fit`` did not converge.
        - ``too_few_samples`` (bool): the window has fewer than ``min_samples`` valid samples.
        - ``at_bounds`` (tuple[str, ...]): the parameters that ended at a bound.

        The parameters are NaN when not_defined, not_converged or too_few_samples is set.

        Returns:
            dict[str, Any]: A copy of the flags.
        """
        return dict(self._flags)

    @override
    def get_options(self) -> dict[str, Any]:
        """Get the options of the last fit.

        - ``p0``: the start estimate, after it was moved inside the bounds.
        - ``bounds``: the lower and upper bounds of the parameters, or None.
        - ``delay`` and ``min_samples``: see the arguments of the phase.

        The options are empty if the phase was not fitted with ``curve_fit``: before fit(), with too
        few samples, and for a BaselinePhase.

        Returns:
            dict[str, Any]: A copy of the options.
        """
        return dict(self._options)

    # Time offset

    def _set_time_offset(self, offset: float) -> None:
        self._time_offset = offset

    def get_time_offset(self) -> float:
        """Return the time that counts as x = 0 in the formula.

        This is the start of the window: the model is evaluated at x = t - offset.

        Returns:
            float: The time offset, in seconds.
        """
        return self._time_offset if hasattr(self, "_time_offset") else 0.0

    def _fit_with_p0(
        self,
        p0: None | tuple[float, ...],
        kwargs: dict[str, Any] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        self._flags = self._no_flags()
        if np.sum(np.isfinite(self.get_size())) < self.min_samples:
            warnings.warn(
                f"Not enough data points to perform fit. Need at least {self.min_samples} data points."
            )
            self._flags["too_few_samples"] = True
            self._set_params(*([np.nan] * len(self.get_param_names())))  # type: ignore
            return
        if kwargs is None:
            kwargs = {}
        if p0 is not None:
            if bounds is not None:
                p0 = self._move_estimate_inside_bounds(p0, bounds)
            kwargs["p0"] = p0
        if bounds is not None:
            kwargs["bounds"] = bounds
        if "nan_policy" not in kwargs:
            kwargs["nan_policy"] = "omit"

        self._options = {"p0": p0, "bounds": bounds, "delay": self.get_delay(), "min_samples": self.min_samples}
        try:
            time_with_offset = self.get_time() - self.get_time_offset()
            popt, _ = curve_fit(  # type: ignore
                self.get_model_function(),
                time_with_offset,
                self.get_size(),
                **kwargs,
            )  # type: ignore
            self._set_params(*popt)  # type: ignore
            if bounds is not None:
                self._flags["at_bounds"] = self._params_at_bounds(popt, bounds)  # type: ignore
        except RuntimeError as e:
            warnings.warn(f"Fit failed: {e}")
            self._flags["not_converged"] = True
            self._set_params(*([np.nan] * len(self.get_param_names())))  # type: ignore

    def _move_estimate_inside_bounds(
        self,
        p0: tuple[float, ...],
        bounds: tuple[Sequence[float], Sequence[float]],
    ) -> tuple[float, ...]:
        estimate = np.asarray(p0, dtype=np.float64)
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), estimate.shape)
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), estimate.shape)
        moved_estimate = np.clip(estimate, lower, upper)
        moved = (moved_estimate != estimate) & ~np.isnan(estimate)
        if np.any(moved):
            names = [name for name, m in zip(self.get_param_names(), moved) if m]
            warnings.warn(
                f"Initial estimate for {', '.join(names)} is outside the bounds "
                "and was moved to the nearest bound."
            )
        return tuple(float(value) for value in moved_estimate)

    def _params_at_bounds(
        self, params: Sequence[float], bounds: tuple[Sequence[float], Sequence[float]]
    ) -> tuple[str, ...]:
        values = np.asarray(params, dtype=np.float64)
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), values.shape)
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), values.shape)
        at_bounds = np.isclose(values, lower) | np.isclose(values, upper)
        return tuple(name for name, at in zip(self.get_param_names(), at_bounds) if at)

    # Prediction

    @override
    def _predict(self, time_data: np.ndarray) -> np.ndarray:
        """Predict pupil size for given time points using the fitted model.

        Args:
            time_data (np.ndarray): Time array to predict for

        Returns:
            np.ndarray: Predicted pupil sizes
        """
        time_with_offset = time_data - self.get_time_offset()
        size = self.get_model_function()(time_with_offset, *self.get_params())
        in_delay = (time_data >= self.start_time) & (
            time_data < self.start_time + self.delay
        )
        return np.where(in_delay, np.nan, size)

    @classmethod
    def get_model_function(cls) -> Callable[..., np.ndarray]:
        """Get the model function used for fitting and prediction.

        Returns:
            Callable[..., np.ndarray]: The model function. Takes time data and parameters as input and returns predicted sizes.

        """
        return cls._model_function

    # Abstract methods

    @staticmethod
    @abstractmethod
    def _model_function(time_data: np.ndarray, *params: float) -> np.ndarray:
        pass
