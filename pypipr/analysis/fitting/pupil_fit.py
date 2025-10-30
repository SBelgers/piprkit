import numpy as np

from ...core.pupil_measurement import PupilBase, PupilMeasurement
from ...utils.light_stimuli import LightStimulus
from .phase_fits import FitBaseline, FitConstrict, FitRedilation, FitSustain
import warnings
# TODO Update to new formulas.

class PupilFit(PupilMeasurement):
    """Fit a model to the pupil measurement data.

    Phases
    ------

    1. BASELINE (y = y0)
        - Description: Flat linear model fit to 10s prestimulus period.
        - Defined from:
            - Start Time: -infinity
            - End Time: light stimuli onset
        - Variables:
            - y0: Baseline pupil diameter (mean before stimulus)
    2. PLR LATENCY PERIOD (y = undefined)
        - Description: Undefined period after light onset and before constriction.
        - Defined from:
            - Start Time: light onset
            - End Time: light onset + PLR latency
    3. CONSTRICT (y = cv * (x + t_constrict) + y0)
        - Description: Linear model from light onset to max constriction.
        - Defined from:
            - Start: light onset + PLR latency
            - End: time of maximum constriction
        - Variables:
            - x: Time since light onset
            - t_constrict: Time shift to light onset + PLR latency
            - cv: Constriction velocity
            - y0: Baseline pupil diameter
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

    Notes
    -----
    - All time values (x) are in seconds.
    """
    def __init__(self, pupil_measurement: PupilBase) -> None:
        """Initialize the PupilFit class.

        Args:
            pupil_measurement (PupilMeasurement): The pupil measurement data to fit.
        """
        light_stimulus = pupil_measurement.get_light_stimulus()
        if light_stimulus is None:
            raise ValueError("Light stimulus not set. Cannot setup fits.")
        self._set_light_stimulus(light_stimulus)
        self.pupil_measurement = pupil_measurement.copy()
        self._set_time(pupil_measurement.get_time())
        self._setup_fits()
        self._fit_all()

    def _set_time(self, time_data: np.ndarray) -> None:
        self.time_data = np.asarray(time_data, dtype=np.float64)

    def get_time(self) -> np.ndarray:
        return self.time_data
    
    def get_size(self) -> np.ndarray:
        return self.predict(self.get_time())

    def _set_light_stimulus(self, stimulus: LightStimulus) -> None:
        self.stimulus = stimulus

    def get_light_stimulus(self) -> LightStimulus:
        return self.stimulus

    def _setup_fits(self) -> None:
        pupil_size = self.pupil_measurement.get_size()
        stimulus_start, stimulus_end = self.get_light_stimulus().get_time()
        baseline_start = -np.inf
        baseline_end = stimulus_start
        self.baseline_fit = FitBaseline(
            self.get_time(), pupil_size, baseline_start, baseline_end
        )
        constrict_start = stimulus_start
        constrict_end = stimulus_end
        self.constrict_fit = FitConstrict(
            self.time_data, pupil_size, constrict_start, constrict_end
        )
        sustain_start = stimulus_start
        sustain_end = stimulus_end
        self.sustain_fit = FitSustain(
            self.time_data, pupil_size, sustain_start, sustain_end
        )
        redilation_start = stimulus_end
        redilation_end = np.inf
        self.redilation_fit = FitRedilation(
            self.time_data, pupil_size, redilation_start, redilation_end
        )

    def _fit_all(self) -> None:
        # TODO: Improve the fit p0 guesses.
        self.baseline_fit.fit()
        self.constrict_fit.fit()
        self.sustain_fit.fit()
        self.sustain_fit._set_params(*([np.nan] * len(self.sustain_fit.get_param_names())))
        warnings.warn("Automatic sustain fit is not yet implemented. Defaulting to NaN.")
        self.redilation_fit.fit()

    def get_fits(self) -> tuple[FitBaseline, FitConstrict, FitSustain, FitRedilation]:
        return (
            self.baseline_fit,
            self.constrict_fit,
            self.sustain_fit,
            self.redilation_fit,
        )

    def predict(self, time_data: np.ndarray) -> np.ndarray:
        """Predict pupil size for given time points using the fitted models.

        Args:
            time_data (np.ndarray): Time array to predict for

        Returns:
            np.ndarray: Predicted pupil sizes
        """
        baseline_fit, constrict_fit, sustain_fit, redilation_fit = self.get_fits()

        size = np.full_like(time_data, np.nan, dtype=np.float64)

        # Baseline phase
        baseline_start, baseline_end = (
            baseline_fit.get_start_time(),
            baseline_fit.get_end_time(),
        )
        baseline_time = (baseline_end > time_data) & (time_data >= baseline_start)

        if np.any(baseline_time):
            size[baseline_time] = baseline_fit.predict(time_data[baseline_time])
        # constriction phase
        constrict_start, constrict_end = (
            constrict_fit.get_start_time(),
            constrict_fit.get_end_time(),
        )
        constrict_time = (constrict_end > time_data) & (time_data >= constrict_start)

        if np.any(constrict_time):
            size[constrict_time] = constrict_fit.predict(time_data[constrict_time])
        # sustained phase
        sustain_start, sustain_end = (
            sustain_fit.get_start_time(),
            sustain_fit.get_end_time(),
        )
        sustain_time = (sustain_end > time_data) & (time_data >= sustain_start)
        if np.any(sustain_time):
            size[sustain_time] = sustain_fit.predict(time_data[sustain_time])
        # redilation phase
        redilation_start, redilation_end = (
            redilation_fit.get_start_time(),
            redilation_fit.get_end_time(),
        )
        redilation_time = (redilation_end > time_data) & (time_data >= redilation_start)
        if np.any(redilation_time):
            size[redilation_time] = redilation_fit.predict(time_data[redilation_time])
        return size

    def get_param(self, phase_name: str, param_name: str) -> float:
        """
        Get a specific parameter from a specific phase.

        Args:
            phase_name: One of 'baseline', 'constriction', 'sustained', 'redilation'
            param_name: Name of the parameter to retrieve

        Returns:
            Parameter value
        """
        params = self.get_all_params()
        if phase_name not in params:
            raise ValueError(
                f"Phase '{phase_name}' not found. Available phases: {list(params.keys())}"
            )
        phase_params = params[phase_name]
        if param_name not in phase_params:
            raise ValueError(
                f"Parameter '{param_name}' not found in phase '{phase_name}'"
            )
        return params[phase_name][param_name]

    def get_all_params(self) -> dict[str, dict[str, float]]:
        """Get all fitted parameters organized by phase.

        Returns:
            dict[str, dict[str, float]]: Dictionary with phase names as keys and parameter dictionaries as values
        """
        baseline_fit, constrict_fit, sustain_fit, redilation_fit = self.get_fits()
        return {
            "baseline": dict(
                zip(baseline_fit.get_param_names(), baseline_fit.get_params())
            ),
            "constriction": dict(
                zip(
                    constrict_fit.get_param_names(),
                    constrict_fit.get_params(),
                )
            ),
            "sustained": dict(
                zip(sustain_fit.get_param_names(), sustain_fit.get_params())
            ),
            "redilation": dict(
                zip(
                    redilation_fit.get_param_names(),
                    redilation_fit.get_params(),
                )
            ),
        }

    def get_formula_strings(self) -> str:
        """Get formula strings for all fitted models.

        Returns:
            str: Formatted string
        """
        baseline_fit, constrict_fit, sustain_fit, redilation_fit = self.get_fits()
        string = "Formula:\n"
        string += f"\tBaseline: {baseline_fit.get_formula_string()}\n"
        string += f"\tConstriction: {constrict_fit.get_formula_string()}\n"
        string += f"\tSustained: {sustain_fit.get_formula_string()}\n"
        string += f"\tRedilation: {redilation_fit.get_formula_string()}\n"
        return string
