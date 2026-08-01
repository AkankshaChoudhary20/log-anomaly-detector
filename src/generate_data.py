import numpy as np
import pandas as pd

RNG_SEED = 42
MINUTES = 60 * 24 * 3  # 3 days at 1-minute resolution
ANOMALY_WINDOWS = [
    (700, 715, "latency_spike"),
    (1800, 1850, "sustained_error_rate"),
    (3200, 3210, "latency_spike"),
    (4000, 4060, "sustained_error_rate"),
]


def _daily_seasonality(n_minutes: int) -> np.ndarray:
    minute_of_day = np.arange(n_minutes) % (60 * 24)
    return 0.5 + 0.5 * np.sin(2 * np.pi * (minute_of_day - 360) / (60 * 24))


def generate(n_minutes: int = MINUTES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-07-01", periods=n_minutes, freq="min")

    traffic_factor = _daily_seasonality(n_minutes)
    base_latency = 80 + 40 * traffic_factor
    latency = base_latency + rng.normal(0, 5, n_minutes)

    base_error_rate = 0.01 + 0.01 * traffic_factor
    error_rate = np.clip(base_error_rate + rng.normal(0, 0.003, n_minutes), 0, 1)

    is_anomaly = np.zeros(n_minutes, dtype=bool)
    anomaly_type = np.full(n_minutes, "", dtype=object)

    for start, end, kind in ANOMALY_WINDOWS:
        idx = slice(start, min(end, n_minutes))
        is_anomaly[idx] = True
        anomaly_type[idx] = kind
        if kind == "latency_spike":
            latency[idx] += rng.uniform(250, 400, latency[idx].shape)
        elif kind == "sustained_error_rate":
            error_rate[idx] = np.clip(
                error_rate[idx] + rng.uniform(0.15, 0.35, error_rate[idx].shape), 0, 1
            )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "latency_ms": latency,
            "error_rate": error_rate,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
    )


if __name__ == "__main__":
    df = generate()
    out_path = "data/synthetic_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Injected anomalies: {df['is_anomaly'].sum()} minutes across {len(ANOMALY_WINDOWS)} windows")
