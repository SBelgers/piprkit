import numpy as np
import pytest

from piprkit.analysis.pupil_metrics import MetricResult
from piprkit.utils.baseline import BaselineCorrection
from piprkit.utils.utils import _add_buffer_seconds, check_time_series, net_pipr


def test_check_time_series_accepts_sorted_unique_time():
    check_time_series(np.array([0.0, 0.5, 2.0, 3.0]))


def test_check_time_series_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        check_time_series(np.array([0.0, np.nan, 2.0]))


def test_check_time_series_rejects_unsorted():
    with pytest.raises(ValueError, match="not sorted"):
        check_time_series(np.array([0.0, 2.0, 1.0]))


def test_check_time_series_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        check_time_series(np.array([0.0, 1.0, 1.0, 2.0]))


@pytest.mark.parametrize("buffer_seconds", [0.0, -1.0])
def test_add_buffer_seconds_without_buffer_returns_mask(buffer_seconds):
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    mask[5] = False
    np.testing.assert_array_equal(
        _add_buffer_seconds(time_data, mask, buffer_seconds), mask
    )


def test_add_buffer_seconds_nothing_removed():
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 2.0), mask)


def test_add_buffer_seconds_everything_removed():
    time_data = np.arange(10.0)
    mask = np.zeros(10, dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 2.0), mask)


def test_add_buffer_seconds_edge_is_inclusive():
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    mask[5] = False
    # t = 3 and t = 7 are exactly 2 s from t = 5, so they are removed too.
    expected = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 2.0), expected)


def test_add_buffer_seconds_between_samples():
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    mask[5] = False
    # Window [3.5, 6.5] around t = 5.
    expected = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 1.5), expected)


def test_add_buffer_seconds_several_removed_samples():
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    mask[[1, 8]] = False
    # Windows [0, 2] and [7, 9].
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 1.0), expected)


def test_add_buffer_seconds_irregular_time():
    time_data = np.array([0.0, 0.1, 0.5, 2.0, 2.05, 3.0])
    mask = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    # Window [1.5, 2.5] around t = 2.0 holds only t = 2.0 and t = 2.05.
    expected = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    np.testing.assert_array_equal(_add_buffer_seconds(time_data, mask, 0.5), expected)


def test_add_buffer_seconds_does_not_change_input_mask():
    time_data = np.arange(10.0)
    mask = np.ones(10, dtype=bool)
    mask[5] = False
    original = mask.copy()
    _add_buffer_seconds(time_data, mask, 2.0)
    np.testing.assert_array_equal(mask, original)


RATIO = BaselineCorrection("ratio", 5.0, 10)


def test_net_pipr():
    # The baselines themselves may differ; only the correction method has to match
    blue = MetricResult(0.6, RATIO, "pipr_6s")
    red = MetricResult(0.9, BaselineCorrection("ratio", 4.0, 10), "pipr_6s")
    result = net_pipr(blue, red)
    assert isinstance(result, MetricResult)
    assert result == pytest.approx(-0.3)
    assert result.get_name() == "net_pipr(pipr_6s, pipr_6s)^R"
    assert result.get_baseline_correction() == RATIO


def test_net_pipr_uncorrected():
    assert net_pipr(MetricResult(4.0, None, "plateau"), MetricResult(5.5, None, "plateau")) == pytest.approx(-1.5)


@pytest.mark.parametrize("correction", [BaselineCorrection("percent", 5.0, 10), None])
def test_net_pipr_needs_the_same_correction_method(correction):
    with pytest.raises(ValueError, match="different baseline corrections"):
        net_pipr(MetricResult(0.6, RATIO, "pipr_6s"), MetricResult(0.9, correction, "pipr_6s"))


def test_net_pipr_needs_metric_results():
    with pytest.raises(TypeError, match="two MetricResults"):
        net_pipr(MetricResult(0.6, RATIO, "pipr_6s"), 0.9)
