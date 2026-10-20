import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["latency_ms", "error_rate"]


class RollingZScoreDetector:
    """Flags a minute as anomalous if either feature's rolling z-score exceeds the threshold."""

    def __init__(self, window: int = 60, threshold: float = 3.5):
        if type(window) is not int or window < 2:
            raise ValueError("window must be an integer of at least 2")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a finite positive number")
        self.window = window
        self.threshold = threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        flagged = np.zeros(len(df), dtype=bool)
        for col in FEATURES:
            series = df[col]
            # The current sample must not dilute its own anomaly score.
            history = series.shift(1).rolling(self.window, min_periods=self.window)
            rolling_mean = history.mean()
            rolling_std = history.std()
            deviation = (series - rolling_mean).abs()
            z = deviation / rolling_std.replace(0, np.nan)
            constant_baseline_spike = rolling_std.eq(0) & deviation.gt(0)
            flagged |= ((z > self.threshold) | constant_baseline_spike).fillna(False).to_numpy()
        return flagged


class IsolationForestDetector:
    """Flags a minute as anomalous using IsolationForest over the raw feature vector."""

    def __init__(self, contamination: float = 0.03, seed: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].to_numpy()
        labels = self.model.fit_predict(X)  # -1 = anomaly, 1 = normal
        return labels == -1


