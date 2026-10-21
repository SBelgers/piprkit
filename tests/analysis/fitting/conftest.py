import numpy as np
import pytest

import piprkit


@pytest.fixture
def measurement():
    # Noise-free simulation with dt = 0.01 s, so peak constriction (-0.75 s) lies on a sample
    return piprkit.simulate_pupil_measurement(t_start=-12, t_end=10, num_samples=2201)


@pytest.fixture
def plot_phase(save_plot):
    # Draws only with --plots: all data, the phase's data, its window and the fitted curve
    def plot(phase, time_data=None, size_data=None):
        if not save_plot:
            return
        if time_data is None:
            ax = phase.plot(scatter=True, s=12, color="grey", label="phase data")
        else:
            measurement = piprkit.PupilMeasurement(time_data, size_data)
            ax = measurement.plot(scatter=True, s=12, color="lightgrey", label="all data")
            phase.plot(ax, scatter=True, s=12, color="grey", label="phase data")
        phase.plot_window(ax, label="window")
        curve_time = np.linspace(phase.get_start_time(), phase.get_end_time(), 200)
        phase.plot_predicted(ax, time_data=curve_time, extrapolation="allow", label="fit")
        save_plot(ax, phase.get_formula_string())

    return plot
