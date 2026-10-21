import numpy as np
import pytest

import piprkit

# The default response of simulate_pupil_measurement, sampled every 0.01 s from -12 to 30 s
DEFAULT = dict(
    time_data=np.linspace(-12, 30, 4201),
    t_light_onset=-1.0,
    t_light_offset=0.0,
    constriction_delay=0.1,
    time_to_peak=0.25,
    redilation_delay=0.0,
    baseline_c=1.0,
    maintained_constriction_c=0.55,
    maintained_constriction_m=0.025,
    redilation_k=-0.4,
    redilation_P=1.0,
)
# The peak comes 0.5 s after light offset, and the plateau lasts until 5 s after light offset
PLATEAU_AFTER_LIGHT_OFFSET = dict(
    DEFAULT,
    constriction_delay=0.2,
    time_to_peak=1.5,
    redilation_delay=5.0,
    baseline_c=6.0,
    maintained_constriction_c=3.0,
    maintained_constriction_m=0.0,
    redilation_k=-0.3,
    redilation_P=5.0,
)


def without_time_data(args):
    # pytest.approx cannot compare an array inside a dict
    return {name: value for name, value in args.items() if name != "time_data"}


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(DEFAULT, id="default"),
        pytest.param(dict(DEFAULT, constriction_delay=0.0), id="constriction_from_light_onset"),
        pytest.param(PLATEAU_AFTER_LIGHT_OFFSET, id="plateau_after_light_offset"),
    ],
)
def test_fit_recovers_simulation_args(args, plot_phase_model):
    # Sampled every 0.01 s, so the constriction start, the peak constriction and the redilation start lie on a sample
    # and the fit finds them exactly
    measurement = piprkit.simulate_pupil_measurement(**args)
    model = piprkit.PhaseModel(measurement)
    plot_phase_model(model, measurement)
    simulation_args = model.get_simulation_args()
    np.testing.assert_array_equal(simulation_args["time_data"], args["time_data"])
    assert without_time_data(simulation_args) == pytest.approx(without_time_data(args), rel=1e-6, abs=1e-9)


def test_fit_simulate_fit_gives_the_same_args(plot_phase_model):
    # The first fit differs from the simulation because of the noise; simulating it and fitting again must give the same values
    first = piprkit.PhaseModel(piprkit.simulate_pupil_measurement(**DEFAULT, noise_std=0.005, rng=0))
    simulation = piprkit.simulate_pupil_measurement(**first.get_simulation_args())
    second = piprkit.PhaseModel(simulation)
    plot_phase_model(second, simulation)
    first_args, second_args = first.get_simulation_args(), second.get_simulation_args()
    np.testing.assert_array_equal(second_args["time_data"], first_args["time_data"])
    assert without_time_data(second_args) == pytest.approx(without_time_data(first_args), rel=1e-6, abs=1e-9)


def test_simulation_of_a_real_trial_gives_its_constriction_slope(plot_phase_model):
    # The real samples are unevenly spaced. On evenly spaced samples, the constriction start of trial 1 fell between
    # two samples, and the search took a piece of the baseline as the constriction
    trial = piprkit.load_real_series().split(10, 60)[0]
    args = piprkit.PhaseModel(trial).get_simulation_args()
    simulation = piprkit.simulate_pupil_measurement(**args)
    model = piprkit.PhaseModel(simulation)
    plot_phase_model(model, simulation)
    slope = (args["maintained_constriction_c"] - args["baseline_c"]) / (args["time_to_peak"] - args["constriction_delay"])
    assert model.get_param("constriction", "m") == pytest.approx(slope, rel=1e-6)


def test_simulation_args_need_all_phases():
    model = piprkit.PhaseModel(piprkit.simulate_pupil_measurement(**DEFAULT), phases=("baseline", "redilation"))
    with pytest.raises(ValueError, match="A simulation needs all four phases"):
        model.get_simulation_args()


def test_simulation_args_need_a_fit_in_each_phase():
    # The peak constriction at light offset leaves a maintained constriction shorter than breakpoint_min_duration
    model = piprkit.PhaseModel(piprkit.simulate_pupil_measurement(**dict(DEFAULT, time_to_peak=1.0)))
    with pytest.raises(ValueError, match="the maintained_constriction phase has no fit"):
        model.get_simulation_args()
