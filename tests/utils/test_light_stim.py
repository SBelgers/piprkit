import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb, to_rgba

from piprkit import LightStimuliSeries, LightStimulus
from piprkit.utils.light_stim import SECTIONS


def _spans(ax):
    return [(patch.get_x(), patch.get_x() + patch.get_width()) for patch in ax.patches]


def test_light_stimulus_from_duration():
    stimulus = LightStimulus(start_time=1, duration=2)
    assert stimulus.get_timing() == (1, 3)


def test_light_stimulus_from_end_time():
    stimulus = LightStimulus(start_time=1, end_time=3)
    assert stimulus.get_timing() == (1, 3)


def test_light_stimulus_needs_duration_or_end_time():
    with pytest.raises(ValueError, match="Either duration or end_time"):
        LightStimulus(start_time=1)


def test_light_stimulus_rejects_duration_and_end_time():
    with pytest.raises(ValueError, match="Only one of duration or end_time"):
        LightStimulus(start_time=1, duration=2, end_time=3)


def test_light_stimulus_default_color():
    assert LightStimulus(start_time=0, duration=1).get_color() == "yellow"


def test_light_stimulus_custom_color():
    # The parameter used to be spelled colour, unlike the color attribute and get_color().
    assert LightStimulus(start_time=0, duration=1, color="blue").get_color() == "blue"


def test_light_stimulus_rejects_invalid_color():
    with pytest.raises(ValueError, match="Invalid color"):
        LightStimulus(start_time=0, duration=1, color="not-a-color")


def test_light_stimulus_colour_keyword_removed():
    # The old spelling colour is removed, not kept as an alias.
    with pytest.raises(TypeError):
        LightStimulus(start_time=0, duration=1, colour="blue")


def test_light_stimulus_set_time_offset():
    stimulus = LightStimulus(start_time=1, end_time=3)
    stimulus.set_time_offset(-1.5)
    np.testing.assert_allclose(stimulus.get_timing(), (-0.5, 1.5))


def test_light_stimulus_plot():
    ax = LightStimulus(start_time=1, end_time=3).plot()
    assert isinstance(ax, Axes)
    np.testing.assert_allclose(_spans(ax), [(1, 3)])
    # Dashed lines at onset and offset.
    np.testing.assert_allclose([line.get_xdata()[0] for line in ax.lines], [1, 3])
    assert ax.get_legend_handles_labels()[1] == ["Light Stimulus"]


def test_light_stimulus_plot_custom_label():
    ax = LightStimulus(start_time=1, end_time=3).plot(label="Blue light")
    assert ax.get_legend_handles_labels()[1] == ["Blue light"]


def test_light_stimulus_plot_default_style():
    ax = LightStimulus(start_time=1, end_time=3, color="blue").plot()
    np.testing.assert_allclose(ax.patches[0].get_facecolor(), to_rgba("blue", 0.3))


def test_light_stimulus_plot_style_kwargs():
    ax = LightStimulus(start_time=1, end_time=3).plot(color="red", alpha=0.5, hatch="//")
    patch = ax.patches[0]
    np.testing.assert_allclose(patch.get_facecolor(), to_rgba("red", 0.5))
    assert patch.get_hatch() == "//"


def test_light_stimuli_series_plot_style_kwargs():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, end_time=3)
    series.add_stimulus(start_time=5, end_time=6)
    ax = series.plot(alpha=0.5)
    for patch in ax.patches:
        np.testing.assert_allclose(patch.get_facecolor(), to_rgba("yellow", 0.5))


def test_light_stimulus_plot_on_given_axes():
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    assert LightStimulus(start_time=1, end_time=3).plot(ax=ax) is ax


def test_light_stimuli_series_add_stimulus():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, duration=2)
    series.add_stimulus(start_time=5, end_time=6, color="blue")
    assert all(isinstance(s, LightStimulus) for s in series.get_stimuli())
    assert series.get_timings() == [(1, 3), (5, 6)]
    assert series.get_colors() == ["yellow", "blue"]


def test_light_stimuli_series_set_time_offset():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, end_time=3)
    series.add_stimulus(start_time=5, end_time=6)
    series.set_time_offset(-1)
    np.testing.assert_allclose(series.get_timings(), [(0, 2), (4, 5)])


def test_light_stimuli_series_plot():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, end_time=3)
    series.add_stimulus(start_time=5, end_time=6)
    ax = series.plot()
    np.testing.assert_allclose(_spans(ax), [(1, 3), (5, 6)])


def test_light_stimuli_series_plot_has_one_legend_entry():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, end_time=3)
    series.add_stimulus(start_time=5, end_time=6)
    ax = series.plot()
    assert ax.get_legend_handles_labels()[1] == ["Light Stimulus"]


def test_light_stimuli_series_plot_custom_label_once():
    series = LightStimuliSeries()
    series.add_stimulus(start_time=1, end_time=3)
    series.add_stimulus(start_time=5, end_time=6)
    ax = series.plot(label="Blue light")
    assert ax.get_legend_handles_labels()[1] == ["Blue light"]


def test_get_sections():
    sections = LightStimulus(start_time=-1, end_time=0).get_sections(-5, 10)
    assert list(sections) == list(SECTIONS)
    assert sections == {
        "pre_stimulus": {"start": -5, "end": -1},
        "during_stimulus": {"start": -1, "end": 0},
        "post_stimulus": {"start": 0, "end": 10},
    }


def test_get_sections_outside_the_time_range_are_empty():
    sections = LightStimulus(start_time=-1, end_time=0).get_sections(-0.5, -0.2)
    assert sections["pre_stimulus"] == {"start": -1, "end": -1}
    assert sections["post_stimulus"] == {"start": 0, "end": 0}


def test_plot_sections():
    ax = LightStimulus(start_time=-1, end_time=0, color="blue").plot_sections(-5, 10)
    assert _spans(ax) == [(-5, -1), (-1, 0), (0, 10)]
    colors = [to_rgb(patch.get_facecolor()) for patch in ax.patches]
    assert colors == [to_rgb("lightgrey"), to_rgb("blue"), to_rgb("lightgrey")]
    assert [text.get_text() for text in ax.texts] == ["pre-stimulus", "during stimulus", "post-stimulus"]


def test_plot_sections_leaves_out_empty_sections():
    ax = LightStimulus(start_time=-1, end_time=0).plot_sections(-0.5, -0.2)
    assert [text.get_text() for text in ax.texts] == ["during stimulus"]
