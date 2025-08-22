# Import main classes and functions for flat API
from .core.pupil_measurement import PupilMeasurement
from .core.pupil_series import PupilSeries
from .analysis.fitting.pupil_fit import PupilFit
from .analysis import pupil_metrics
from .analysis.fitting.phase_fits import (
    FitBaseline,
    FitLatency,
    FitSustain,
    FitRedilation,
    FitConstrict,
)
from .data.loaders import (
    load_simulated_pupil,
    load_real_series,
    simulate_pupil_measurement,
)
from .preprocessing import filtering
from .utils.utils import check_time_series
from .utils.light_stimuli import LightStimulus, LightStimuliSeries
from ._version import __version__



__all__ = [
    # Main classes
    "PupilMeasurement",
    "PupilSeries",
    "PupilFit",
    # Data loading functions
    "load_simulated_pupil",
    "load_real_series",
    "simulate_pupil_measurement",
    # Analysis/metrics functions
    "pupil_metrics",
    "FitBaseline",
    "FitLatency",
    "FitSustain",
    "FitRedilation",
    "FitConstrict",
    # Preprocessing functions
    "filtering",
    # Utility functions and classes
    "check_time_series",
    "LightStimulus",
    "LightStimuliSeries",
    # Version
    "__version__",
]