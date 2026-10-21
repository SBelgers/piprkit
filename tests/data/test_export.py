import csv
import math

import numpy as np
import pytest

import piprkit
from piprkit.analysis.pupil_metrics import MetricResult

# The flags of a fit whose S and k ended at their bounds
AT_BOUNDS = {"not_defined": False, "reason": None, "not_converged": False, "at_bounds": ("S", "k"), "too_few_samples": False}


def two_trials():
    # Light from 12 to 13 s and from 52 to 53 s. The size is 5 mm up to each light onset, then
    # 4 mm (until 40 s) after the first and 3 mm after the second
    time_data = np.arange(0, 80, 0.5)
    size = np.select([time_data <= 12, time_data <= 40, time_data <= 52], [5.0, 4.0, 5.0], 3.0)
    return piprkit.PupilSeries(time_data, size, [(12.0, 13.0), (52.0, 53.0)])


def test_metric_rows_of_a_split_series():
    rows = []
    for trial, (pm, condition) in enumerate(zip(two_trials().split(10, 20), ["blue", "red"]), start=1):
        pm.filters.apply_baseline_correction()
        results = [pm.metrics.pipr_6s(), pm.metrics.auc_early()]
        rows += piprkit.metric_rows(results, participant="P01", condition=condition, trial=trial)
    # Ratio 0.8 after the first light and 0.6 after the second, so the areas over 10 s are 2 and 4
    assert rows == [
        {"participant": "P01", "condition": "blue", "trial": 1, "metric": "6s-PIPR_5.95-6.05^R", "value": pytest.approx(0.8)},
        {"participant": "P01", "condition": "blue", "trial": 1, "metric": "AUC_0-10^R", "value": pytest.approx(2)},
        {"participant": "P01", "condition": "red", "trial": 2, "metric": "6s-PIPR_5.95-6.05^R", "value": pytest.approx(0.6)},
        {"participant": "P01", "condition": "red", "trial": 2, "metric": "AUC_0-10^R", "value": pytest.approx(4)},
    ]
    # The labels come first, in the order they were passed
    assert list(rows[0]) == ["participant", "condition", "trial", "metric", "value"]


def test_metric_rows_add_the_flags_of_a_fit():
    (row,) = piprkit.metric_rows([MetricResult(1.0, None, "PlateauPIPR", flags=AT_BOUNDS)], trial=1)
    assert row == {"trial": 1, "metric": "PlateauPIPR^mm", "value": 1.0, **AT_BOUNDS}


def test_save_metric_rows(tmp_path):
    results = [MetricResult(np.nan, None, "TransientPLR_0.18-0.5"), MetricResult(1.0, None, "PlateauPIPR", flags=AT_BOUNDS)]
    rows = piprkit.metric_rows(results, participant="P01", trial=1)
    assert math.isnan(rows[0]["value"])
    piprkit.save_metric_rows(rows, tmp_path / "metrics.csv")
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as file:
        table = list(csv.DictReader(file))
    assert list(table[0]) == ["participant", "trial", "metric", "value", *AT_BOUNDS]
    # The NaN value is an empty cell, and so are the flags of the metric computed from data
    assert table[0] == {
        "participant": "P01", "trial": "1", "metric": "TransientPLR_0.18-0.5^mm", "value": "",
        "not_defined": "", "reason": "", "not_converged": "", "at_bounds": "", "too_few_samples": "",
    }
    assert table[1] == {
        "participant": "P01", "trial": "1", "metric": "PlateauPIPR^mm", "value": "1.0",
        "not_defined": "False", "reason": "", "not_converged": "False", "at_bounds": "S k", "too_few_samples": "False",
    }


@pytest.mark.parametrize("label", ["metric", "value", "at_bounds"])
def test_metric_rows_label_with_the_name_of_a_column_raises(label):
    with pytest.raises(ValueError, match=label):
        piprkit.metric_rows([MetricResult(1.0, None, "PlateauPIPR", flags=AT_BOUNDS)], **{label: "x"})


def test_metric_rows_needs_a_list_of_metric_results():
    with pytest.raises(TypeError, match="list"):
        piprkit.metric_rows(MetricResult(0.8, None, "6s-PIPR"), trial=1)
    with pytest.raises(TypeError, match="MetricResult"):
        piprkit.metric_rows([0.8], trial=1)

