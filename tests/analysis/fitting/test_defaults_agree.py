import inspect

import pytest

import piprkit
from piprkit.analysis.fitting.base_model import Phase
from piprkit.analysis.fitting.breakpoints import find_constriction_start_and_end, find_redilation_start_and_end
from piprkit.analysis.pupil_metrics import PupilMetrics
from piprkit.preprocessing.filtering import PupilFilters


def default(function, name):
    return inspect.signature(function).parameters[name].default


def phase_model(name):
    return default(piprkit.PhaseModel, name)


# PhaseModel passes its own values down, so these only guard the defaults of direct calls against drifting apart
@pytest.mark.parametrize(
    "model_name, function, name",
    [
        ("peak_constriction_window", PupilMetrics.peak_constriction, "peak_constriction_window"),
        ("peak_constriction_window", PupilMetrics.time_to_peak, "peak_constriction_window"),
        ("peak_constriction_window", PupilMetrics.transient_plr, "peak_constriction_window"),
        ("baseline_duration", PupilMetrics.baseline, "duration"),
        ("baseline_duration", PupilMetrics.auc, "baseline_duration"),
        ("baseline_duration", PupilMetrics.auc_early, "baseline_duration"),
        ("baseline_duration", PupilMetrics.auc_late, "baseline_duration"),
        ("baseline_duration", PupilFilters.apply_baseline_correction, "duration"),
        ("min_samples", Phase, "min_samples"),
        ("min_samples", find_constriction_start_and_end, "min_samples"),
        ("min_samples", find_redilation_start_and_end, "min_samples"),
        ("breakpoint_resolution", find_constriction_start_and_end, "resolution"),
        ("breakpoint_resolution", find_redilation_start_and_end, "resolution"),
        ("breakpoint_min_duration", find_constriction_start_and_end, "min_duration"),
        ("redilation_k_min", find_redilation_start_and_end, "k_min"),
        ("redilation_k_max", find_redilation_start_and_end, "k_max"),
        ("redilation_k_count", find_redilation_start_and_end, "k_count"),
    ],
)
def test_defaults_agree(model_name, function, name):
    assert default(function, name) == phase_model(model_name)
