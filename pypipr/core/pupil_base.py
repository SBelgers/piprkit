import copy
import warnings
from abc import ABC
from typing import Any, Optional, Self

import numpy as np
from matplotlib.axes import Axes

from ..utils.light_stimuli import LightStimulus


class PupilBase(ABC):
    # ============================================================================
    # CORE DATA METHODS
    # ============================================================================

    def set_time_and_size(self, time_data: np.ndarray, size: np.ndarray) -> None:
        """Set time and size arrays with validation.

        Args:
            time_data (np.ndarray): Array of time data.
            size (np.ndarray): Array of size data.

        Raises:
            ValueError: If the time and size arrays do not have the same length.
        """
        if len(time_data) != len(size):
            raise ValueError("Time and size must have the same length")
        self.time_data = time_data.copy()
        self.size = size.copy()

    def get_time(self) -> np.ndarray:
        return self.time_data

    def get_size(self) -> np.ndarray:
        return self.size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"time_data.shape={getattr(self, 'time_data', np.array([])).shape}, "
            f"size.shape={getattr(self, 'size', np.array([])).shape}, "
            f"light_stimulus={getattr(self, 'light_stimulus', None)})"
        )

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def copy(self) -> Self:
        """Create a deep copy of this object.

        Returns:
            Self: A deep copy of the object.
        """
        return copy.deepcopy(self)

    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================

    def plot(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        scatter: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot pupil size over time.

        Args:
            ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
            ax.set_xlabel("Time (s)")  # type: ignore
            ax.set_ylabel("Pupil Diameter")  # type: ignore
        kwargs.pop("scatter", None)
        kwargs.pop("show", None)
        kwargs.pop('ax', None)
        if scatter:
            ax.scatter(self.get_time(), self.get_size(), **kwargs) # type: ignore
        else:
            ax.plot(self.get_time(), self.get_size(), **kwargs)  # type: ignore
        ax.set_xlabel("Time (s)")  # type: ignore
        ax.set_ylabel("Pupil Diameter")  # type: ignore
        if show:
            plt.show()  # type: ignore
        return ax

    # ============================================================================
    # LIGHT STIMULUS MANAGEMENT
    # ============================================================================

    def set_light_stimulus(self, start: float, end: float) -> None:
        """Set the light stimulus period.

        Args:
            start (float): The start time of the light stimulus.
            end (float): The end time of the light stimulus.
        """
        self.light_stimulus = LightStimulus(start_time=start, end_time=end)

    def get_light_stimulus(self) -> Optional[LightStimulus]:
        """Get the light stimulus object.

        Returns:
            Optional[LightStimulus]: The light stimulus object or None if not set.
        """
        if not hasattr(self, "light_stimulus"):
            warnings.warn("Light stimulus not set.")
            return None
        return self.light_stimulus

    def plot_light_stimulus(
        self,
        ax: Optional[Axes] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Axes:
        """Plot the light stimulus period.

        Args:
            ax (Optional[Axes], optional): The axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.

        Raises:
            ValueError: If the light stimulus is not set.

        Returns:
            Axes: The axes with the light stimulus plotted.
        """
        light_stimulus = self.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set.")
        return light_stimulus.plot(ax, show, **kwargs)
