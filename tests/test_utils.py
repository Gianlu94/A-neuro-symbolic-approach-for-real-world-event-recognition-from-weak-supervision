import pytest

from utils import get_data

@pytest.fixture()
def video1():
    return {
        "name": "video1",
        "duration": 90,
        "event": "high_jump",
        "intervals": [(10, 20), (30, 40), (50, 60), (70, 80)],
        "num_features": 100,
        "output": [
            ["video1", 90, "high_jump", 0, 30], ["video1", 90, "high_jump", 20, 50],
            ["video1", 90, "high_jump", 40, 70], ["video1", 90, "high_jump", 60, 90]
        ]
    }

@pytest.fixture()
def video2():
    return {
        "name": "video2",
        "duration": 70,
        "event": "high_jump",
        "intervals": [(10, 20), (30, 40), (50, 60)],
        "num_features": 100,
        "output": [
            ["video2", 70, "high_jump", 0, 30], ["video2", 70, "high_jump", 20, 50],
            ["video2", 70, "high_jump", 40, 70]
        ]
    }
    
def test_get_data_even_events(video1):
    video = video1["name"]
    duration = video1["duration"]
    event_name = video1["event"]
    event_intervals = video1["intervals"]
    num_features = video1["num_features"]
    
    true_output = video1["output"]
    obtained_output = get_data(video, duration, event_name, event_intervals, num_features)

    for true_interval, obtained_interval in zip(true_output, obtained_output):
        for i in range(len(true_interval)):
            assert true_interval[i] == obtained_interval[i]


def test_get_data_odd_events(video2):
    video = video2["name"]
    duration = video2["duration"]
    event_name = video2["event"]
    event_intervals = video2["intervals"]
    num_features = video2["num_features"]
    
    true_output = video2["output"]
    obtained_output = get_data(video, duration, event_name, event_intervals, num_features)
    
    for true_interval, obtained_interval in zip(true_output, obtained_output):
        for i in range(len(true_interval)):
            assert true_interval[i] == obtained_interval[i]
    
    
    

