import matplotlib

matplotlib.use("Agg")

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--plots", action="store_true", help="Save a figure of each fitting test in tests/plots/")
    parser.addoption(
        "--data", action="append", default=[], help="Recording CSV (with JSON sidecar) for tests/test_private_data.py; repeatable"
    )
    parser.addoption("--data-column", default=None, help="Size column of the --data recordings. Defaults to the second column")
    parser.addoption(
        "--data-size-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Size range kept in the --data recordings (trim_size). Without it the recordings are not trimmed",
    )


def _file_name(name):
    return re.sub(r"[^\w.\-\[\]]", "_", name)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def save_plot(request):
    # None unless pytest runs with --plots, so tests can skip drawing
    if not request.config.getoption("--plots"):
        return None
    folder = Path(__file__).parent / "plots" / request.node.path.stem

    def save(ax, title="", name=None):
        if name is None:
            path = folder / f"{_file_name(request.node.name)}.png"
        else:
            # Stage figures are grouped per recording: plots/<test file>/<parameter id>/<name>.png
            callspec = getattr(request.node, "callspec", None)
            group = callspec.id if callspec else request.node.originalname
            path = folder / _file_name(group) / f"{name}.png"
        ax.set_title(title.expandtabs(2), fontsize=7, loc="left")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(path, dpi=120, bbox_inches="tight")

    return save


@pytest.fixture
def plot_phase_model(save_plot):
    # Draws only with --plots: the data, the expected curve, the sections, the phases and the light stimulus
    def plot(model, expected=None, name=None):
        if not save_plot:
            return
        ax = model.plot(scatter=True, s=4, color="grey", label="data")
        if expected is not None:
            expected.plot(ax, linestyle="--", color="black", label="expected")
        model.plot_sections(ax)
        model.plot_phases(ax)
        model.plot_light_stimulus(ax)
        save_plot(ax, model.get_formula_string(), name)

    return plot


@pytest.fixture
def plot_stage(save_plot):
    # Draws only with --plots: one pipeline stage, over the stage before it when given
    def plot(current, previous=None, *, name):
        if not save_plot:
            return
        ax = None if previous is None else previous.plot(color="lightgrey", label="before")
        ax = current.plot(ax, label="data" if previous is None else "after")
        current.plot_light_stimulus(ax)
        if current.get_blinks():
            current.plot_blinks(ax)
        valid = np.isfinite(current.get_size()).sum()
        save_plot(ax, f"{name}: {valid} of {current.get_size().size} samples valid", name)

    return plot


@pytest.fixture
def plot_metrics(save_plot):
    # Draws only with --plots: the measurement and each metric where it was measured
    def plot(measurement, results, *, name):
        if not save_plot:
            return
        ax = measurement.plot(color="grey", label="data")
        measurement.plot_light_stimulus(ax)
        for result in results:
            result.plot(ax)
        save_plot(ax, "\n".join(f"{result.get_label()}: {result:.4g}" for result in results), name)

    return plot
