import pytest
import numpy as np

from src.core.idealization import Idealizer


# (trace, events)
test_traces = [
    (
        np.array([1, 1, 2, 1, 1, 1], dtype=float),
        np.array([[1, 2, 0, 1],
                  [2, 1, 2, 2],
                  [1, 3, 3, 5]], dtype=float),
    ),
    (
        np.array([1, 1, 1, 2, 2, 3], dtype=float),
        np.array([[1, 3, 0, 2],
                  [2, 2, 3, 4],
                  [3, 1, 5, 5]], dtype=float),
    ),
    (
        np.array([2, 1, 1, 2, 2, 3], dtype=float),
        np.array([[2, 1, 0, 0],
                  [1, 2, 1, 2],
                  [2, 2, 3, 4],
                  [3, 1, 5, 5]], dtype=float),
    ),
    (
        np.array([2, 1, 1, 2, 2, 3, 3], dtype=float),
        np.array([[2, 1, 0, 0],
                  [1, 2, 1, 2],
                  [2, 2, 3, 4],
                  [3, 2, 5, 6]], dtype=float),
    ),
]

resolution_test_event_series = [
    (
        2,
        np.array([1, 1, 1, 2, 2, 3]),
    ),
    (
        2,
        np.array([2, 1, 1, 2, 2, 3]),
    ),
    (
        4,
        np.array([2,2,2,2,1,2,3,3,3,4,6,6,6,6,6,6,1,1,1,1 , 1, 2, 2, 3, 2,2,2,2,5,5,5,5,5,5,3]),
    ),
    (
        4,
        np.array([2, 1, 1, 2, 2, 3, 3,3,3,3,3,3,3, 5,5,5,5,5, 2,2,2,2,2,2,2, 1,1,1,1,1,1,1]),
    ),
    (
        4,
        np.array([2, 1, 1, 2, 2, 3, 3]),
    ),
]


@pytest.mark.parametrize("resolution, trace", resolution_test_event_series)
def test_extract_events_with_resolution(resolution, trace):
    idealization = Idealizer.apply_resolution(trace, np.arange(len(trace)), resolution)
    out = Idealizer.extract_events(idealization, np.arange(len(idealization)))
    assert np.all(out[:, 1] >= resolution)

@pytest.mark.parametrize("trace, events", test_traces)
def test_extract_events(trace, events):
    out = Idealizer.extract_events(trace, np.arange(len(trace)))
    print(out)
    print(events)
    assert np.all(out == events)

def test_single_amplitude():
    # Test case with only one amplitude
    signal = np.random.rand(100)
    amplitudes = np.array([1])
    idealization = Idealizer.threshold_crossing(signal, amplitudes)
    assert np.allclose(idealization, np.ones(100))

def test_two_amplitudes():
    # Test case with default thresholds
    signal = np.random.rand(100)
    amplitudes = np.array([0, 1])
    idealization = Idealizer.threshold_crossing(signal, amplitudes)
    assert np.all([np.isclose(i, 0) or np.isclose(i, 1) for i in idealization])

def test_custom_thresholds():
    # Test case with custom thresholds
    signal = np.random.rand(100)
    amplitudes = np.array([1, 2, 3])
    thresholds = np.array([0.5, 1.8])
    idealization = Idealizer.threshold_crossing(signal, amplitudes, thresholds)
    assert np.all([np.isclose(i, 1) or np.isclose(i, 3) or np.isclose(i, 2) for i in idealization])

def test_wrong_thresholds():
    # Test case with wrong number of thresholds
    signal = np.random.rand(100)
    amplitudes = np.array([1, 2, 3])
    thresholds = np.array([0.5, 0.8, 0.9])
    with pytest.raises(ValueError):
        Idealizer.threshold_crossing(signal, amplitudes, thresholds)
