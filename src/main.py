import os
import pandas as pd

from generate_data import generate
from detectors import RollingZScoreDetector, IsolationForestDetector
from evaluate import score, group_into_incidents

DATA_PATH = "data/synthetic_metrics.csv"
ALERTS_PATH = "output/alerts.log"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        df = generate()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def write_alerts(path: str, detector_name: str, incidents: list[dict], mode: str = "w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        f.write(f"\n=== Alerts from {detector_name} ({len(incidents)} incidents) ===\n")
        for inc in incidents:
            f.write(
                f"[{inc['start']} -> {inc['end']}] duration={inc['duration_minutes']}m "
                f"peak_latency={inc['peak_latency_ms']}ms peak_error_rate={inc['peak_error_rate']}\n"
            )


def main():
    df = load_data()
    y_true = df["is_anomaly"]

    detectors = {
        "RollingZScore": RollingZScoreDetector(window=60, threshold=3.5),
        "IsolationForest": IsolationForestDetector(contamination=0.03),
    }

    os.makedirs("output", exist_ok=True)
    open(ALERTS_PATH, "w").close()  # reset alerts file for this run

    print(f"Loaded {len(df)} minutes of data ({y_true.sum()} true anomalous minutes)\n")

    for name, detector in detectors.items():
        y_pred = pd.Series(detector.predict(df), index=df.index)
        metrics = score(y_true, y_pred)
        incidents = group_into_incidents(df, y_pred)

        print(f"--- {name} ---")
        print(f"  precision={metrics['precision']}  recall={metrics['recall']}  "
              f"f1={metrics['f1']}  false_positive_rate={metrics['false_positive_rate']}")
        print(f"  {len(incidents)} incidents raised (vs {len(incidents)} would-be-alerts "
              f"instead of {int(y_pred.sum())} raw flagged minutes)\n")

        write_alerts(ALERTS_PATH, name, incidents, mode="a")

    print(f"Alert details written to {ALERTS_PATH}")


if __name__ == "__main__":
    main()
