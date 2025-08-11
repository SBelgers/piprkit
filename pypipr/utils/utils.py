import numpy as np
from numpy.typing import NDArray
def check_time_series(time_data: NDArray[np.number]) -> None:
    """
    Check if the time series is sorted and does not contain duplicates.
    
    Raises:
        ValueError: If the time series contains NaNs,is not sorted or contains duplicates.
    """
    if np.any(np.isnan(time_data)):
        raise ValueError("Time series contains NaN values.")
    if np.any(np.diff(time_data) < 0):
        raise ValueError("Time array is not sorted in ascending order.")
    if np.any(np.diff(time_data) == 0):
        raise ValueError("Time array contains duplicate values.")

