import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import piprkit
from piprkit.analysis.fitting.phase_model import PHASES


def test_plot_draws_the_fitted_data():
    size = np.linspace(1, 2, 10)
    phase = piprkit.RedilationPhase(np.arange(10.0), size, 2)
    ax = phase.plot(scatter=True)
    # Only the samples from t = 2 on belong to the phase
    offsets = np.asarray(ax.collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], np.arange(2.0, 10.0))
    np.testing.assert_allclose(offsets[:, 1], size[2:])


def test_plot_window_without_delay():
    phase = piprkit.RedilationPhase(np.arange(10.0), np.ones(10), 2, 8)
    ax = phase.plot_window()
    assert [(patch.get_x(), patch.get_width()) for patch in ax.patches] == [(2, 6)]


def test_plot_window_hatches_the_delay():
    phase = piprkit.RedilationPhase(np.arange(10.0), np.ones(10), 2, 8, delay=1)
    ax = phase.plot_window()
    spans = [(patch.get_x(), patch.get_width(), patch.get_hatch()) for patch in ax.patches]
    assert spans == [(3, 5, None), (2, 1, "//")]


def test_plot_light_stimulus(measurement):
    ax = piprkit.PhaseModel(measurement).plot_light_stimulus()
    assert [(patch.get_x(), patch.get_width()) for patch in ax.patches] == [(-1, 1)]


def test_plot_phases_one_band_and_curve_per_phase(measurement):
    ax = piprkit.PhaseModel(measurement).plot_phases()
    assert [patch.get_label() for patch in ax.patches] == list(PHASES)
    curves = [line for line in ax.lines if line.get_linestyle() == "-"]
    assert [to_rgb(line.get_color()) for line in curves] == [to_rgb(patch.get_facecolor()) for patch in ax.patches]


def test_plot_phases_shows_flags():
    # Peak 0.99 s after onset leaves a maintained constriction shorter than breakpoint_min_duration
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=0.99
    )
    ax = piprkit.PhaseModel(measurement).plot_phases()
    assert ax.patches[2].get_label() == "maintained_constriction (not_defined)"


def test_plot_phases_keeps_one_colour_per_phase(measurement):
    full = piprkit.PhaseModel(measurement).plot_phases()
    subset = piprkit.PhaseModel(measurement, phases=("baseline", "redilation")).plot_phases()
    # redilation has the same colour with or without the other phases
    assert to_rgb(subset.patches[1].get_facecolor()) == to_rgb(full.patches[3].get_facecolor())


def test_plot_phases_draws_on_given_axes(measurement):
    _, ax = plt.subplots()
    assert piprkit.PhaseModel(measurement).plot_phases(ax) is ax
    assert len(plt.get_fignums()) == 1


def test_plot_phases_band_along_the_bottom(measurement):
    ax = piprkit.PhaseModel(measurement).plot_phases()
    for patch in ax.patches:
        _, bottom, _, height = patch.get_bbox().bounds
        assert (bottom, height) == (0, 0.06)
    assert [text.get_text() for text in ax.texts][:4] == ["baseline", "constriction", "maintained constriction", "redilation"]


def test_plot_phases_hatches_a_phase_without_a_fit():
    measurement = piprkit.simulate_pupil_measurement(
        t_start=-12, t_end=10, num_samples=2201, time_to_peak=0.99
    )
    patch = piprkit.PhaseModel(measurement).plot_phases().patches[2]
    assert patch.get_hatch() == "//"
    assert not patch.get_fill()


def test_plot_phases_draws_the_stretches_dashed(measurement):
    model = piprkit.PhaseModel(measurement)
    ax = model.plot_phases()
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    # Before the baseline (from -12 s), in the colour of the baseline, and the latency, in grey
    assert [to_rgb(line.get_color()) for line in dashed] == [to_rgb(ax.patches[0].get_facecolor()), to_rgb("grey")]
    assert dashed[0].get_xdata()[0] == -12
    assert (dashed[1].get_xdata()[0], dashed[1].get_xdata()[-1]) == (-1, pytest.approx(-0.9))


def test_plot_phases_names_each_start_and_end(measurement):
    ax = piprkit.PhaseModel(measurement).plot_phases()
    names = [text.get_text() for text in ax.texts if text.get_rotation() == 90]
    assert "maintained constriction end, redilation start" in names
    # The constriction end lands one sample after the maintained constriction start, so each has its own label
    assert "constriction end" in names
    assert "maintained constriction start" in names
