import pandas as pd


def score(y_true, y_pred) -> dict:
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
    flagged_idx = df.index[flagged]
    if len(flagged_idx) == 0:
        return []

    incidents = []
    start = prev = flagged_idx[0]
    for idx in flagged_idx[1:]:
        if idx - prev <= gap_minutes:
            prev = idx
            continue
        incidents.append(_build_incident(df, start, prev))
        start = prev = idx
    incidents.append(_build_incident(df, start, prev))
    return incidents


def _build_incident(df: pd.DataFrame, start_idx: int, end_idx: int) -> dict:
    window = df.loc[start_idx:end_idx]
    return {
        "start": window["timestamp"].iloc[0],
        "end": window["timestamp"].iloc[-1],
        "duration_minutes": end_idx - start_idx + 1,
        "peak_latency_ms": round(window["latency_ms"].max(), 1),
        "peak_error_rate": round(window["error_rate"].max(), 4),
    }
