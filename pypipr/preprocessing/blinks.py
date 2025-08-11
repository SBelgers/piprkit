# TODO: fix this file.

# from typing import Any, Optional

# from matplotlib import pyplot as plt
# from ..core.pupil_measurement import PupilMeasurement

# from scipy.signal import find_peaks # type: ignore
# import numpy as np

# def find_potential_peaks(
#     pupil_measurement: PupilMeasurement,
#     detection_threshold: float = 0.1,
#     min_duration: float = 0.1,

# ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    
#     pupil_measurement  = pupil_measurement.copy()
#     pupil_measurement.drop_nan()
#     pupil_measurement.rolling_mean(4)
#     dt = np.nanmean(np.diff(pupil_measurement.get_time()))
#     min_distance = max(1, int(min_duration / dt))  # Ensure peaks are at least 100 ms apart

#     signal = -pupil_measurement.get_size()
#     threshold = np.percentile(signal, (1 - detection_threshold) * 100)
#     print(threshold)

#     potential_peaks, properties = find_peaks( # type: ignore
#         x=signal, height=threshold
#     )
#     print(np.diff(potential_peaks)) # type: ignore
#     print(potential_peaks < min_distance) # type: ignore
#     return potential_peaks, properties # type: ignore

# def manual_detection_plot(
#     pupil_measurement: PupilMeasurement,
#     detection_sensitivity: float = 0.1,
#     ax: Optional[Any] = None,
#     show: bool = False,
#     **kwargs: Any,
# ) -> None:
    

#     if ax is None:
#         fig, ax = plt.subplots(1,1) # type: ignore
#     pupil_measurement.plot(ax=ax, **kwargs)
    
#     dt = np.nanmean(np.diff(pupil_measurement.get_time()))
#     min_distance = max(1, int(0.1 / dt)) # Ensure peaks are at least 100 ms apart
    
#     threshold = np.percentile(-pupil_measurement.get_size(), detection_sensitivity * 100)
#     print(threshold)

#     potential_peaks, _ = find_potential_peaks(
#         pupil_measurement, detection_threshold=detection_sensitivity, min_duration=0.1
#     )
#     print(f"Potential peaks found at indices: {potential_peaks}")
#     ax.vlines(pupil_measurement.get_time()[potential_peaks], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label='Potential Peaks')
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Pupil Diameter (mm)")
