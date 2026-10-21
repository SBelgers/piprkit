from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..analysis.pupil_metrics import MetricResult


def metric_rows(results: Iterable[MetricResult], /, **labels: Any) -> list[dict[str, Any]]:
    """Turn the metric results of one trial into rows of a long table, one row per result.

    Each row holds the labels of the trial, the name of the metric (see
    :meth:`~piprkit.analysis.pupil_metrics.MetricResult.get_name`), its value, and the flags of a
    metric read from a fitted model (see :meth:`~piprkit.analysis.pupil_metrics.MetricResult.get_flags`).
    Together, the rows of all trials form one long table. Write them with :func:`save_metric_rows`,
    or pass them to ``pandas.DataFrame``::

        rows = []
        for trial, pm in enumerate(series.split(10, 60), start=1):
            pm.filters.apply_baseline_correction()
            results = [pm.metrics.pipr_6s(), pm.metrics.auc(10, 30)]
            rows += piprkit.metric_rows(results, participant="P01", trial=trial)
        piprkit.save_metric_rows(rows, "metrics.csv")

    Args:
        results (Iterable[MetricResult]): The metric results of one trial, from the ``metrics`` of
            a measurement or of a fitted PhaseModel.
        **labels: The labels of the trial, such as ``participant``, ``condition`` and ``trial``.
            Each label becomes a column, in the order given.

    Raises:
        TypeError: If results is a single MetricResult instead of a list, or holds anything other
            than MetricResults.
        ValueError: If a label has the name of another column: "metric", "value" or a flag.

    Returns:
        list[dict[str, Any]]: One dict per result, with the labels, "metric", "value" (NaN when the
        metric could not be computed) and, for a metric read from a fitted model, its flags.
    """
    if isinstance(results, MetricResult):
        raise TypeError(
            "metric_rows needs a list of MetricResults, e.g. metric_rows([pm.metrics.pipr_6s()], trial=1)."
        )
    rows = []
    for result in results:
        if not isinstance(result, MetricResult):
            raise TypeError(f"metric_rows needs MetricResults, not {type(result).__name__}.")
        row = {"metric": result.get_name(), "value": float(result), **(result.get_flags() or {})}
        clash = [label for label in labels if label in row]
        if clash:
            raise ValueError(f"The labels {clash} have the names of other columns. Use other names.")
        rows.append({**labels, **row})
    return rows


def save_metric_rows(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write rows of :func:`metric_rows` to a CSV file, with a header row.

    The columns are those of the rows, in the order they first appear. A cell is empty when its
    row has no such column (e.g. the flags of a metric computed from data), and when the value is
    NaN or None, so R and pandas read an empty number as missing. The parameters in ``at_bounds``
    are separated by spaces. pandas reads the flags as True/False with ``dtype="boolean"``. An
    existing file is overwritten.

    Args:
        rows (Sequence[dict[str, Any]]): The rows of all trials, from :func:`metric_rows`.
        path (str | Path): Path to the CSV file.
    """
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(value) for column, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value)
    return value
