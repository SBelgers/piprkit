import numpy as np

from pypipr.core.pupil_base import PupilBase
import pytest


class DummyPupil(PupilBase):
    def __init__(self):
        pass
    
def test_get_time():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    dummy = DummyPupil()
    dummy.set_time_and_size(time_data, size)
    np.testing.assert_array_equal(dummy.get_time(), time_data)


def test_get_size():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    dummy = DummyPupil()
    dummy.set_time_and_size(time_data, size)
    np.testing.assert_array_equal(dummy.get_size(), size)


def test_set_size_and_time():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    dummy = DummyPupil()
    dummy.set_time_and_size(time_data, size)
    np.testing.assert_array_equal(dummy.get_time(), time_data)
    new_time = np.array([10, 11, 12])
    new_size = np.array([16.0, 16.1, 16.2])
    dummy.set_time_and_size(new_time, new_size)
    np.testing.assert_array_equal(dummy.get_time(), new_time)
    np.testing.assert_array_equal(dummy.get_size(), new_size)

def test_copy():
    time_data = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    dummy = DummyPupil()
    dummy.set_time_and_size(time_data, size)
    dummy_copy = dummy.copy()
    assert vars(dummy).keys() == vars(dummy_copy).keys()
    np.testing.assert_array_equal(dummy.get_time(), dummy_copy.get_time())
    np.testing.assert_array_equal(dummy.get_size(), dummy_copy.get_size())
    dummy.set_light_stimulus(1,2)
    dummy_copy = dummy.copy()
    assert vars(dummy).keys() == vars(dummy_copy).keys()
    assert vars(dummy.get_light_stimulus()) == vars(dummy_copy.get_light_stimulus())
