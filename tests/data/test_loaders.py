import json

import numpy as np
import pytest

from piprkit import PupilMeasurement, PupilSeries
from piprkit.data.loaders import (
    load_csv,
    load_real_series,
    load_simulated_pupil,
    save_csv,
    simulate_pupil_measurement,
)

def test_load_sample():
    # Test loading blue trace
    blue_trace = load_simulated_pupil("blue")
    assert isinstance(blue_trace, PupilMeasurement)

    # Test loading red trace
    red_trace = load_simulated_pupil("red")
    assert isinstance(red_trace, PupilMeasurement)
    assert np.allclose(blue_trace.get_time(), red_trace.get_time(), atol=1e-8)
    assert len(blue_trace.get_size()) == 15999
    # Row "8.952,7.1509962250612364" in simulated_trace_blue.csv
    index = np.argmin(np.abs(blue_trace.get_time() - 8.952))
    assert blue_trace.get_time()[index] == pytest.approx(8.952)
    assert blue_trace.get_size()[index] == pytest.approx(7.1509962250612364)



    # Test invalid input
    with pytest.raises(ValueError):
        load_simulated_pupil("green")


def test_load_real_series():
    ps = load_real_series()
    assert isinstance(ps, PupilSeries)
    assert len(ps.get_light_stimuli_series().get_stimuli()) == 6
    assert len(ps.get_time()) == len(ps.get_size())
    assert np.all(np.diff(ps.get_time()) > 0)


def test_simulate_pupil_measurement_defaults():
    pm = simulate_pupil_measurement()
    assert isinstance(pm, PupilMeasurement)
    assert len(pm.get_time()) == 1000
    assert pm.get_time()[0] == pytest.approx(-10)
    assert pm.get_time()[-1] == pytest.approx(10)
    assert pm.get_light_stimulus().get_timing() == (-1.0, 0.0)


def test_simulate_pupil_measurement_time_data():
    # On the times of the default simulation, it gives the default simulation
    time_data = np.linspace(-10, 10, 1000)
    pm = simulate_pupil_measurement(time_data=time_data)
    np.testing.assert_array_equal(pm.get_time(), time_data)
    np.testing.assert_array_equal(pm.get_size(), simulate_pupil_measurement().get_size())


def test_simulate_pupil_measurement_baseline_without_noise():
    pm = simulate_pupil_measurement(baseline_c=6.5)
    np.testing.assert_allclose(pm.get_size()[pm.get_time() < -1], 6.5)


def test_simulate_pupil_measurement_short_constriction_delay():
    # A constriction delay shorter than the sample interval used to crash (min() of an empty array).
    pm = simulate_pupil_measurement(constriction_delay=0.001)
    assert len(pm.get_size()) == 1000


def test_simulate_pupil_measurement_bad_timing_raises():
    # The peak constriction, 2 s after light onset, lies after the redilation start at light offset.
    with pytest.raises(ValueError, match="Expected t_start"):
        simulate_pupil_measurement(time_to_peak=2.0)


@pytest.mark.parametrize(
    "response, steepest",
    [
        ({}, 3.0),
        (
            dict(
                constriction_delay=0.2,
                time_to_peak=1.5,
                redilation_delay=5.0,
                baseline_c=6.0,
                maintained_constriction_c=3.0,
                maintained_constriction_m=0.0,
                redilation_k=-0.3,
                redilation_P=5.0,
            ),
            3 / 1.3,
        ),
    ],
    ids=["default", "plateau_after_light_offset"],
)
def test_simulate_pupil_measurement_is_continuous(response, steepest):
    # The phases join up: no step between samples is larger than the steepest slope of the response allows
    pm = simulate_pupil_measurement(t_start=-12, t_end=40, num_samples=5201, **response)
    assert np.max(np.abs(np.diff(pm.get_size()))) <= steepest * 0.01 + 1e-9


def test_simulate_pupil_measurement_rng():
    kwargs = dict(noise_std=0.1, trend_std=0.01)
    a = simulate_pupil_measurement(**kwargs, rng=1)
    np.testing.assert_array_equal(simulate_pupil_measurement(**kwargs, rng=1).get_size(), a.get_size())
    np.testing.assert_array_equal(
        simulate_pupil_measurement(**kwargs, rng=np.random.default_rng(1)).get_size(), a.get_size()
    )
    assert not np.allclose(simulate_pupil_measurement(**kwargs, rng=2).get_size(), a.get_size(), equal_nan=True)


def _write_csv(path, rows=((0, 5.0), (1, 5.5), (2, 6.0), (3, 5.0)), header="time,size"):
    path.write_text("\n".join([header] + [f"{t},{s}" for t, s in rows]) + "\n")
    return path


@pytest.mark.parametrize("time_col, size_col", [("time", "size"), (0, 1)])
def test_load_csv_columns(tmp_path, time_col, size_col):
    pm = load_csv(_write_csv(tmp_path / "pupil.csv"), time_col, size_col)
    assert isinstance(pm, PupilMeasurement)
    np.testing.assert_allclose(pm.get_time(), [0, 1, 2, 3])
    np.testing.assert_allclose(pm.get_size(), [5.0, 5.5, 6.0, 5.0])


def test_load_csv_header_with_spaces(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv", header="time (s),pupil size")
    pm = load_csv(path, "time (s)", "pupil size")
    np.testing.assert_allclose(pm.get_size(), [5.0, 5.5, 6.0, 5.0])


def test_load_csv_one_stimulus(tmp_path):
    pm = load_csv(_write_csv(tmp_path / "pupil.csv"), "time", "size", stimuli=[(1, 2)])
    assert isinstance(pm, PupilMeasurement)
    assert pm.get_light_stimulus().get_timing() == (1, 2)


def test_load_csv_several_stimuli(tmp_path):
    ps = load_csv(_write_csv(tmp_path / "pupil.csv"), "time", "size", stimuli=[(0, 1), (2, 3)])
    assert isinstance(ps, PupilSeries)
    assert ps.get_light_stimuli_series().get_timings() == [(0, 1), (2, 3)]


def test_load_csv_blinks(tmp_path):
    pm = load_csv(_write_csv(tmp_path / "pupil.csv"), "time", "size", blinks=[(1, 1.5)])
    assert pm.get_blinks() == [(1, 1.5)]


def test_load_csv_unknown_column_raises(tmp_path):
    with pytest.raises(ValueError, match="Column 'diameter' not found"):
        load_csv(_write_csv(tmp_path / "pupil.csv"), "time", "diameter")


def test_load_csv_unsorted_time_raises(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv", rows=((0, 5.0), (2, 5.0), (1, 5.0)))
    with pytest.raises(ValueError, match="not sorted"):
        load_csv(path, "time", "size")


def _write_sidecar(csv_path, **metadata):
    csv_path.with_suffix(".json").write_text(json.dumps(metadata))


def test_load_csv_sidecar_one_stimulus(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, light_stimuli=[{"onset": 1, "duration": 1, "intensity": 0.9}])
    pm = load_csv(path, "time", "size")
    assert isinstance(pm, PupilMeasurement)
    assert pm.get_light_stimulus().get_timing() == (1, 2)


def test_load_csv_sidecar_several_stimuli(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, light_stimuli=[{"onset": 0, "duration": 1}, {"onset": 2, "duration": 1}])
    ps = load_csv(path, "time", "size")
    assert isinstance(ps, PupilSeries)
    assert ps.get_light_stimuli_series().get_timings() == [(0, 1), (2, 3)]


def test_load_csv_sidecar_blinks(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, blinks=[{"onset": 1, "duration": 0.5}])
    assert load_csv(path, "time", "size").get_blinks() == [(1, 1.5)]


def test_load_csv_sidecar_ignores_events(tmp_path):
    # "events" is part of the format but not supported yet
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, events=[{"onset": 1, "duration": 0, "label": "button"}])
    pm = load_csv(path, "time", "size")
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        assert pm.get_light_stimulus() is None
    assert pm.get_blinks() == []


def test_load_csv_arguments_override_sidecar(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, light_stimuli=[{"onset": 0, "duration": 1}], blinks=[{"onset": 1, "duration": 0.5}])
    pm = load_csv(path, "time", "size", stimuli=[(2, 3)], blinks=[])
    assert pm.get_light_stimulus().get_timing() == (2, 3)
    assert pm.get_blinks() == []


def test_load_csv_sidecar_disabled(tmp_path):
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, light_stimuli=[{"onset": 0, "duration": 1}])
    pm = load_csv(path, "time", "size", sidecar=False)
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        assert pm.get_light_stimulus() is None


def test_save_csv_round_trip_measurement(tmp_path):
    pm = simulate_pupil_measurement(noise_std=0.1, rng=0)
    pm.set_blinks([(-5.0, -4.7)])
    save_csv(pm, tmp_path / "pupil.csv")
    loaded = load_csv(tmp_path / "pupil.csv")
    assert isinstance(loaded, PupilMeasurement)
    np.testing.assert_array_equal(loaded.get_time(), pm.get_time())
    np.testing.assert_array_equal(loaded.get_size(), pm.get_size())
    np.testing.assert_allclose(loaded.get_light_stimulus().get_timing(), pm.get_light_stimulus().get_timing())
    np.testing.assert_allclose(loaded.get_blinks(), pm.get_blinks())


def test_save_csv_round_trip_series(tmp_path):
    pm = simulate_pupil_measurement(noise_std=0.1, rng=0)
    ps = PupilSeries(pm.get_time(), pm.get_size(), [(-6.0, -5.0), (-1.0, 0.0)])
    save_csv(ps, tmp_path / "pupil.csv")
    loaded = load_csv(tmp_path / "pupil.csv")
    assert isinstance(loaded, PupilSeries)
    np.testing.assert_allclose(loaded.get_light_stimuli_series().get_timings(), ps.get_light_stimuli_series().get_timings())


def test_save_csv_replaces_stale_sidecar(tmp_path):
    # Saving without stimuli must not keep the stimuli of an earlier sidecar
    path = _write_csv(tmp_path / "pupil.csv")
    _write_sidecar(path, light_stimuli=[{"onset": 0, "duration": 1}])
    save_csv(PupilMeasurement(np.arange(4.0), np.full(4, 5.0)), path)
    pm = load_csv(path)
    with pytest.warns(UserWarning, match="Light stimulus not set"):
        assert pm.get_light_stimulus() is None

