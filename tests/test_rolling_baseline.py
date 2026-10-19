import pandas as pd
from src.detectors import RollingZScoreDetector

def test_spike_after_constant_history_is_detected():
    df = pd.DataFrame({'latency_ms': [10] * 5 + [100], 'error_rate': [0] * 6})
    assert RollingZScoreDetector(window=5).predict(df).tolist() == [False] * 5 + [True]

def test_normal_constant_history_stays_normal():
    df = pd.DataFrame({'latency_ms': [10] * 12, 'error_rate': [0] * 12})
    assert not RollingZScoreDetector(window=5).predict(df).any()

def test_current_sample_does_not_dilute_its_score():
    df = pd.DataFrame({'latency_ms': [9, 10, 11, 10, 10, 30], 'error_rate': [0] * 6})
    assert RollingZScoreDetector(window=5).predict(df)[-1]
