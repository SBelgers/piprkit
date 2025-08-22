import numpy as np
import pytest

import pypipr


@pytest.mark.xfail(reason="Not implemented")
def test_transient_plr():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.transient_plr(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_plr_latency():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.plr_latency(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_constriction_v():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.constriction_v(pupil) == 0.0


def test_peak_constriction():
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25])
    size = np.array([7,7,6,5,6,7,7])
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.peak_constriction(pupil) == 5


def test_time_to_peak():
    time_data = np.array([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25])
    size = np.array([7, 7, 6, 5, 6, 7, 7])
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.time_to_peak(pupil) == 0.5


@pytest.mark.xfail(reason="Not implemented")
def test_pupil_escape():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.pupil_escape(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_redilation_v():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.redilation_v(pupil) == 0.0


def test_pipr_xs():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    # print(np.vstack((time_data, size)).T)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.pipr_xs(pupil, 5.5, 6.5) == 8.4
    assert pypipr.pupil_metrics.pipr_xs(pupil, 1, 5) == 7.2


def test_pipr_6s():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.pipr_6s(pupil) == 8.4


@pytest.mark.xfail(reason="Not implemented")
def test_plateau():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.plateau(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_auc_early():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.auc_early(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_auc_late():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.auc_late(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_pipr_duration():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.pipr_duration(pupil) == 0.0


@pytest.mark.xfail(reason="Not implemented")
def test_net_pipr():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    pupil.set_light_stimulus(-1, 0)
    assert pypipr.pupil_metrics.net_pipr(pupil) == 0.0


def test_get_average_size():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    print(np.vstack((time_data, size)).T)
    pupil = pypipr.PupilMeasurement(time_data, size)
    # Wanr because the baseline is determined in an invalid range
    with pytest.warns():
        pypipr.pupil_metrics.get_average_size(pupil, 5, 20)
    assert pypipr.pupil_metrics.get_average_size(pupil, -6, -1) == pytest.approx(4.6)
    assert pypipr.pupil_metrics.get_average_size(pupil, 0, 5) == pytest.approx(7)
    assert pypipr.pupil_metrics.get_average_size(pupil, -10, 3) == pytest.approx(4.6)


def test_calculate_baseline():
    time_data = np.linspace(-10, 10, 21)
    size = np.linspace(2, 10, 21)
    pupil = pypipr.PupilMeasurement(time_data, size)
    # Fail because the light stimulus is not set.
    with pytest.raises(ValueError):
        pypipr.pupil_metrics.calculate_baseline(pupil)
    pupil.set_light_stimulus(-1, 0)
    # Warn because the baseline is determined in an invalid range
    with pytest.warns():
        pypipr.pupil_metrics.calculate_baseline(pupil, 20)
    assert pypipr.pupil_metrics.calculate_baseline(pupil, 5) == pytest.approx(4.6)
