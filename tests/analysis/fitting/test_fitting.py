
# ===================== General PupilFit Tests =====================
def test_pupil_fit_from_measurement():
    import numpy as np
    from pypipr.core.pupil_measurement import PupilMeasurement
    from pypipr.utils.light_stimuli import LightStimulus
    from pypipr.analysis.fitting.pupil_fit import PupilFit

    # Create dummy time and size data
    time_data = np.linspace(0, 20, 100)
    size_data = np.sin(time_data / 3) + 5  # simple synthetic pupil size

    # Create a measurement
    measurement = PupilMeasurement(time_data, size_data)

    # Attach a light stimulus
    stimulus = LightStimulus(start_time=5, duration=10, colour="yellow")
    measurement._set_light_stimulus = lambda s: setattr(measurement, "_light_stimulus", s)
    measurement.get_light_stimulus = lambda: getattr(measurement, "_light_stimulus", None)
    measurement._set_light_stimulus(stimulus)

    # Create PupilFit from measurement
    fit = PupilFit(measurement)

    # Check that fits are created
    assert hasattr(fit, "baseline_fit")
    assert hasattr(fit, "constrict_fit")
    assert hasattr(fit, "sustain_fit")
    assert hasattr(fit, "redilation_fit")

    # Check that time and size are correct
    np.testing.assert_array_equal(fit.get_time(), time_data)

def test_pupil_fit_from_measurement_with_custom_phases():
    # TODO: Implement test for custom phases
    pass

def test_pupil_fit_from_time_size():
    # TODO: Implement test for creating PupilFit from time and size arrays
    pass

def test_pupil_fit_fit_all():
    # TODO: Implement test for fitting all phases
    pass

def test_pupil_fit_fit_all_with_failed_fits():
    # TODO: Implement test for handling failed fits
    pass

def test_pupil_fit_predict():
    # TODO: Implement test for prediction
    pass

def test_pupil_fit_predict_phases():
    # TODO: Implement test for phase-specific prediction
    pass

def test_pupil_fit_get_all_params():
    # TODO: Implement test for retrieving all parameters
    pass

def test_pupil_fit_get_all_params_unfitted():
    # TODO: Implement test for unfitted parameters
    pass

def test_pupil_fit_plot():
    # TODO: Implement test for plotting
    pass

# ===================== Baseline Phase Tests =====================
def test_fit_baseline():
    # TODO: Implement test for baseline fitting
    pass

def test_fit_baseline_model_function():
    # TODO: Implement test for baseline model function
    pass

def test_fit_baseline_formula_string():
    # TODO: Implement test for baseline formula string
    pass

# ===================== Constriction Phase Tests =====================
def test_fit_constrict():
    # TODO: Implement test for constriction fitting
    pass

def test_fit_constrict_model_function():
    # TODO: Implement test for constriction model function
    pass

def test_fit_constrict_formula_string():
    # TODO: Implement test for constriction formula string
    pass

# ===================== Sustained Phase Tests =====================
def test_fit_sustain():
    # TODO: Implement test for sustained fitting
    pass

def test_fit_sustain_model_function():
    # TODO: Implement test for sustained model function
    pass

def test_fit_sustain_formula_string():
    # TODO: Implement test for sustained formula string
    pass

# ===================== Redilation Phase Tests =====================
def test_fit_redilation():
    # TODO: Implement test for redilation fitting
    pass

def test_fit_redilation_model_function():
    # TODO: Implement test for redilation model function
    pass

def test_fit_redilation_formula_string():
    # TODO: Implement test for redilation formula string
    pass

# ===================== Latency Phase Tests (Placeholder) =====================
def test_fit_latency():
    # TODO: Implement test for latency phase if needed
    pass

# ===================== Phase Fit Utility/Edge Tests =====================
def test_phase_fits_inheritance():
    # TODO: Implement test for inheritance structure
    pass

def test_phase_fits_with_noisy_data():
    # TODO: Implement test for noisy data handling
    pass

def test_phase_fits_with_insufficient_data():
    # TODO: Implement test for insufficient data handling
    pass

def test_phase_fits_parameter_bounds():
    # TODO: Implement test for parameter bounds
    pass
