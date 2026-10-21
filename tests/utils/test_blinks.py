import numpy as np
import pytest
from matplotlib.colors import to_rgba

from piprkit.utils.blinks import PupilBlinks


def test_new_blinks_are_empty():
    assert PupilBlinks().get_blinks() == []


def test_set_blinks():
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2), (3, 4)])
    assert blinks.get_blinks() == [(1, 2), (3, 4)]


def test_set_blinks_none_clears():
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2)])
    blinks.set_blinks(None)
    assert blinks.get_blinks() == []


def test_set_blinks_copies_the_list():
    blink_list = [(1, 2)]
    blinks = PupilBlinks()
    blinks.set_blinks(blink_list)
    blink_list.append((3, 4))
    assert blinks.get_blinks() == [(1, 2)]


def test_set_time_offset():
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2), (3, 4)])
    blinks.set_time_offset(2)
    assert blinks.get_blinks() == [(3, 4), (5, 6)]


def test_drop_blinks_out_of_time_range():
    blinks = PupilBlinks()
    blinks.set_blinks(
        [
            (0, 1),  # before the range: dropped
            (1, 2),  # ends exactly at the range start: kept
            (1.5, 2.5),  # overlaps the range start: kept
            (3, 4),  # inside: kept
            (4.5, 6),  # overlaps the range end: kept
            (5, 6),  # starts exactly at the range end: kept
            (6, 7),  # after the range: dropped
        ]
    )
    blinks.drop_blinks_out_of_time_range(2, 5)
    # Kept blinks are not clipped to the range.
    assert blinks.get_blinks() == [(1, 2), (1.5, 2.5), (3, 4), (4.5, 6), (5, 6)]


def test_find_blinks_not_implemented():
    with pytest.warns(UserWarning, match="not recommended"):
        with pytest.raises(NotImplementedError):
            PupilBlinks().find_blinks()


def test_plot():
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2), (3, 4)])
    ax = blinks.plot()
    spans = [(patch.get_x(), patch.get_x() + patch.get_width()) for patch in ax.patches]
    np.testing.assert_allclose(spans, [(1, 2), (3, 4)])
    for patch in ax.patches:
        np.testing.assert_allclose(patch.get_facecolor(), to_rgba("red", 0.3))


def test_plot_custom_colour():
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2)])
    ax = blinks.plot(color="blue", alpha=0.5)
    np.testing.assert_allclose(ax.patches[0].get_facecolor(), to_rgba("blue", 0.5))


def test_plot_on_given_axes():
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    blinks = PupilBlinks()
    blinks.set_blinks([(1, 2)])
    assert blinks.plot(ax=ax) is ax
