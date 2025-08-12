# Import main classes and functions for flat API
from .core import PupilMeasurement, PupilSeries
from .analysis import (
    PupilFit,
    baseline,
    transient_plr,
    plr_latency,
    constriction_v,
    peak_constriction,
    time_to_peak,
    pupil_escape,
    redilation_v,
    pipr_xs,
    pipr_6s,
    plateau,
    auc_early,
    auc_late,
    pipr_duration,
    net_pipr,
    get_average_size,
    calculate_baseline,
    apply_baseline_correction,
)
from .data import load_simulated_pupil, load_real_series
from .preprocessing import (
    rolling_filter,
    rolling_mean,
    rolling_median,
    get_rate_of_change,
    limit_rate_of_change,
)
from .utils import check_time_series, LightStimulus, LightStimuliSeries
from ._version import __version__

# Keep submodules available for advanced users
from . import core
from . import analysis
from . import preprocessing
from . import data
from . import utils

__all__ = [
    # Main classes
    "PupilMeasurement",
    "PupilSeries", 
    "PupilFit",
    # Data loading functions
    "load_simulated_pupil",
    "load_real_series",
    # Analysis/metrics functions
    "baseline",
    "transient_plr",
    "plr_latency",
    "constriction_v",
    "peak_constriction",
    "time_to_peak",
    "pupil_escape",
    "redilation_v",
    "pipr_xs",
    "pipr_6s",
    "plateau",
    "auc_early",
    "auc_late",
    "pipr_duration",
    "net_pipr",
    "get_average_size",
    "calculate_baseline",
    "apply_baseline_correction",
    # Preprocessing functions
    "rolling_filter",
    "rolling_mean",
    "rolling_median",
    "get_rate_of_change",
    "limit_rate_of_change",
    # Utility functions and classes
    "check_time_series",
    "LightStimulus",
    "LightStimuliSeries",
    # Version
    "__version__",
    # Submodules for advanced use
    "core",
    "analysis", 
    "preprocessing",
    "data",
    "utils",
]