import numpy as np
import pytest

from piprkit.analysis.fitting.breakpoints import find_constriction_start_and_end, find_redilation_start_and_end

TIME = np.round(np.arange(-5, 3.001, 0.01), 10)
# Flat at 1 until 0.2, falling at -2 per s until 0.8, rising at 0.1 per s afterwards
SIZE = np.where(TIME < 0.2, 1.0, np.where(TIME <= 0.8, 1 - 2 * (TIME - 0.2), -0.2 + 0.1 * (TIME - 0.8)))


def test_find_constriction_start_and_end_on_exact_data():
    assert find_constriction_start_and_end(TIME, SIZE, -5, 3, 0) == pytest.approx((0.2, 0.8))


def test_find_constriction_start_and_end_with_noise():
    noise = np.random.default_rng(0).normal(0, 0.01, TIME.size)
    assert find_constriction_start_and_end(TIME, SIZE + noise, -5, 3, 0) == pytest.approx((0.2, 0.8), abs=0.02)


def test_find_constriction_start_and_end_respects_earliest_constriction_start():
    start, _ = find_constriction_start_and_end(TIME, SIZE, -5, 3, 0.4)
    assert start >= 0.4


def test_find_constriction_start_and_end_without_constriction():
    # The size rises, so the best fit has no falling segment
    assert find_constriction_start_and_end(TIME, 1 + 0.1 * TIME, -5, 3, 0) is None


def test_find_constriction_start_and_end_with_too_few_samples():
    assert find_constriction_start_and_end(TIME[:4], SIZE[:4], -5, 3, -5) is None


def test_find_constriction_start_and_end_on_the_simulation(measurement):
    # The size leaves the baseline at the constriction start (-0.9 s) and reaches the peak constriction at -0.75 s. The
    # constriction end lands one sample later, because the line after it also has to follow the start of the redilation
    start, end = find_constriction_start_and_end(measurement.get_time(), measurement.get_size(), -11, 1, -1)
    assert start == pytest.approx(-0.9)
    assert end == pytest.approx(-0.74)


def test_find_redilation_start_and_end_on_the_simulation(measurement):
    # Light offset at 0 s; S = -0.43125, k = -0.4, P = 1
    start, end, (S, k, P) = find_redilation_start_and_end(measurement.get_time(), measurement.get_size(), -0.75, 0)
    assert start == pytest.approx(0, abs=0.05)
    assert end == measurement.get_time()[-1]
    assert (S, k, P) == pytest.approx((-0.43125, -0.4, 1), rel=0.05)


def test_find_redilation_start_and_end_stops_at_the_end(measurement):
    start, end, _ = find_redilation_start_and_end(measurement.get_time(), measurement.get_size(), -0.75, 0, end=6)
    assert start == pytest.approx(0, abs=0.05)
    assert end == pytest.approx(6)


def test_find_redilation_start_and_end_respects_the_earliest_start(measurement):
    start, _, _ = find_redilation_start_and_end(measurement.get_time(), measurement.get_size(), -0.75, 0.5)
    assert start >= 0.5


def test_find_redilation_start_and_end_with_empty_range(measurement):
    # The earliest start lies after the last sample
    assert find_redilation_start_and_end(measurement.get_time(), measurement.get_size(), -0.75, 20) is None


def test_find_redilation_start_and_end_with_too_few_samples():
    assert find_redilation_start_and_end(np.arange(4.0), np.ones(4), 0, 0) is None


def test_find_constriction_start_and_end_use_min_samples():
    # 801 samples, fewer than 3 * 300
    assert find_constriction_start_and_end(TIME, SIZE, -5, 3, 0, min_samples=300) is None


@pytest.mark.parametrize(
    "k_min, k_max, k_count",
    [(-1, -2, 40), (-10, 0, 40), (-10, -0.01, 0)],
)
def test_find_redilation_start_and_end_rejects_invalid_k_grid(measurement, k_min, k_max, k_count):
    with pytest.raises(ValueError, match="k_min < k_max < 0"):
        find_redilation_start_and_end(
            measurement.get_time(), measurement.get_size(), -0.75, 0, k_min=k_min, k_max=k_max, k_count=k_count
        )


def test_find_redilation_start_and_end_uses_k_range(measurement):
    _, _, (_, k, _) = find_redilation_start_and_end(
        measurement.get_time(), measurement.get_size(), -0.75, 0, k_min=-0.02, k_max=-0.01, k_count=2
    )
    assert k == pytest.approx(-0.02) or k == pytest.approx(-0.01)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_find_redilation_start_and_end_after_a_plateau():
    # Flat at 0.5 until 6 s, then an exponential back towards 1 with k = -0.3, between two grid values. The data run
    # 40 s, past the 35 s at which the search overflowed with k = -10
    time_data = np.round(np.arange(0, 40.001, 0.01), 10)
    size = np.where(time_data < 6, 0.5, 1 - 0.5 * np.exp(-0.3 * (time_data - 6)))
    start, end, (S, k, P) = find_redilation_start_and_end(time_data, size, 0, 0)
    assert start == pytest.approx(6)
    assert end == 40
    assert (S, k, P) == pytest.approx((-0.5, -0.3, 1), rel=1e-5)
