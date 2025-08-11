import copy
from abc import ABC
from typing import Any, Optional, Self

import numpy as np

from matplotlib.axes import Axes


class PupilBase(ABC):

    
    # ============================================================================
    # CORE DATA METHODS
    # ============================================================================
    
    def set_time_and_size(
        self, time_data: np.ndarray, size: np.ndarray
    ) -> None:
        """Set time and size arrays with validation.

        Args:
            time_data (np.ndarray): Array of time data.
            size (np.ndarray): Array of size data.

        Raises:
            ValueError: If the time and size arrays do not have the same length.
        """
        if len(time_data) != len(size):
            raise ValueError("Time and size must have the same length")
        self.time_data = time_data
        self.size = size
        
    def get_time(self) -> np.ndarray: 
        return self.time_data

    def get_size(self) -> np.ndarray:
        return self.size

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
        **kwargs: Any,
    ) -> None:
        """Plot pupil size over time.

        Args:
            ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()  # type: ignore
            ax.set_xlabel("Time (s)") # type: ignore
            ax.set_ylabel("Pupil Diameter (mm)") # type: ignore
        ax.plot(self.get_time(), self.get_size(), **kwargs) # type: ignore
        if show:
            plt.show()  # type: ignore
