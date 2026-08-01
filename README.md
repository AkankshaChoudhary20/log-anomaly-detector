# Log/Metric Anomaly Detector

A small AIOps project: detect anomalies in service metrics (latency, error rate)
and turn raw flagged minutes into a handful of grouped incidents instead of
alert-spamming per minute.

## Why this project

Built as a first, self-contained AIOps exercise — no live cluster required,
synthetic data with known ground-truth anomalies so detector quality can be
measured honestly rather than eyeballed.

## Design

- `src/generate_data.py` — generates 3 days of 1-minute latency/error-rate data
  with daily seasonality (traffic pattern) plus 4 injected anomaly windows
  (2 latency spikes, 2 sustained error-rate incidents). Ground truth
  (`is_anomaly`) is preserved for evaluation.
- `src/detectors.py` — two interchangeable detectors:
  - `RollingZScoreDetector`: per-feature rolling mean/std, flags |z| > threshold.
  - `IsolationForestDetector`: multivariate outlier detection over
    `[latency_ms, error_rate]` jointly.
- `src/evaluate.py` — precision/recall/F1/false-positive-rate against ground
  truth, plus `group_into_incidents()` which collapses nearby flagged minutes
  (within a 5-minute gap) into single incidents — this is the alert-correlation
  piece, not just detection.
- `src/main.py` — orchestrates: load/generate data, run both detectors, print
  metrics, write grouped incidents to `output/alerts.log`.

## Assumptions

- Synthetic data, not a live cluster (no Prometheus/kubectl integration yet).
- Anomalies are injected as either a latency spike or a sustained elevated
  error rate — these are meant to be representative, not exhaustive.
- Isolation Forest's `contamination` parameter is set to roughly match the
  known injected anomaly rate (~3%), which is a simplification — in a real
  deployment you would not know this in advance.

## Results (this run)

| Detector | Precision | Recall | F1 | False Positive Rate | Incidents Raised |
|---|---|---|---|---|---|
| Rolling Z-Score | 0.842 | 0.119 | 0.208 | 0.0007 | 7 |
| Isolation Forest | 1.000 | 0.963 | 0.981 | 0.0000 | 4 |

**Isolation Forest won clearly.** The rolling z-score detector's window
adapts too quickly to a sustained-but-moderate error-rate shift, so it only
catches the first minute or two of each sustained incident and misses the
rest — visible in `output/alerts.log`, where its incidents for the sustained
windows are fragmented into 1-7 minute blips instead of the true 50-60 minute
duration that Isolation Forest correctly captured. This is a real, honest
result of the comparison, not a tuning failure to paper over — it demonstrates
why naive single-metric thresholding is a common but weak choice for AIOps
anomaly detection.

Incident grouping meaningfully reduced alert volume: raw flagged minutes
(19-130) collapsed into 4-7 actual incidents per detector.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```

Re-running regenerates nothing if `data/synthetic_metrics.csv` already
exists; delete it to regenerate with a different seed/window.

## Next steps (not built yet)

- Wire into a real cluster: pull actual `kubectl top` / Prometheus metrics
  instead of synthetic data.
- Containerize and deploy as a scheduled Job or sidecar using the
  `kubernetes-specialist` skill (resource limits, probes, RBAC, NetworkPolicy).
- Push grouped incidents to Slack/webhook instead of a local log file.
- Try an unsupervised recall improvement for z-score (e.g., separate
  short-term and long-term rolling windows) as a fairer comparison.
