import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["latency_ms", "error_rate"]


class RollingZScoreDetector:
    """Flags a minute as anomalous if either feature's rolling z-score exceeds the threshold."""

    def __init__(self, window: int = 60, threshold: float = 3.5):
        self.window = window
        self.threshold = threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        flagged = np.zeros(len(df), dtype=bool)
        for col in FEATURES:
            series = df[col]
            rolling_mean = series.rolling(self.window, min_periods=self.window).mean()
            rolling_std = series.rolling(self.window, min_periods=self.window).std()
            z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            flagged |= (z.abs() > self.threshold).fillna(False).to_numpy()
        return flagged


class IsolationForestDetector:
    """Flags a minute as anomalous using IsolationForest over the raw feature vector."""

    def __init__(self, contamination: float = 0.03, seed: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].to_numpy()
        labels = self.model.fit_predict(X)  # -1 = anomaly, 1 = normal
        return labels == -1
