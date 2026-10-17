import numpy as np
import pandas as pd


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels must be one-dimensional arrays of equal length.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels must contain only boolean or binary values.")
    tp = int(((y_pred == True) & (y_true == True)).sum())
    fp = int(((y_pred == True) & (y_true == False)).sum())
    fn = int(((y_pred == False) & (y_true == True)).sum())
    tn = int(((y_pred == False) & (y_true == False)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(false_positive_rate, 4),
    }


def group_into_incidents(df: pd.DataFrame, flagged: pd.Series, gap_minutes: int = 5) -> list[dict]:
    """Collapses consecutive/nearby flagged minutes into single incidents instead of one alert per minute."""
    if gap_minutes < 0:
        raise ValueError('gap_minutes must be nonnegative')
    ordered = df.copy()
    ordered['timestamp'] = pd.to_datetime(ordered['timestamp'], utc=True)
    if ordered['timestamp'].isna().any():
        raise ValueError('timestamps must not be missing')
    flagged_times = ordered.loc[flagged, 'timestamp'].sort_values()
    if flagged_times.empty:
        return []
    ordered = ordered.sort_values('timestamp')
    incidents = []
    start = previous = flagged_times.iloc[0]
    for timestamp in flagged_times.iloc[1:]:
        if timestamp - previous <= pd.Timedelta(minutes=gap_minutes):
            previous = timestamp
            continue
        incidents.append(_build_incident(ordered, start, previous))
        start = previous = timestamp
    incidents.append(_build_incident(ordered, start, previous))
    return incidents


def _build_incident(df: pd.DataFrame, start, end) -> dict:
    window = df.loc[df['timestamp'].between(start, end)]
    return {
        'start': start,
        'end': end,
        'duration_minutes': (end - start).total_seconds() / 60 + 1,
        'peak_latency_ms': round(window['latency_ms'].max(), 1),
        'peak_error_rate': round(window['error_rate'].max(), 4),
    }

