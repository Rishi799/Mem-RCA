# Phase 2 recorded replay and alert timing

## What is implemented

- `scripts/serve_live_replay.py` serves a local browser dashboard and streams one recorded incident over server-sent events. It advances time, updates service scores, shows the observed call graph, applies a validation-normal threshold with a two-window persistence rule, and reports the observed time gap between connected services' persistent alerts.
- `scripts/analyze_propagation.py` measures alert coverage and connected-service alert delays over held-out cases. Thresholds come from normal validation scores; graph topology comes from training cases for the corresponding fold. The script excludes pre-injection score windows from alert-delay events.
- Outputs are CSV files under `experiments/propagation/`: validation thresholds, case-level root detection, per-edge alert-pair events, and per-edge delay summaries.

## Current exploratory result

With selected anomaly scores and the 99th-percentile validation-normal threshold, a persistent root-service alert was found in **231 of 270 held-out cases (85.6%)**. Coverage varied by system: Online Boutique **88/90 (97.8%)**, Sock Shop **65/90 (72.2%)**, and Train Ticket **78/90 (86.7%)**. Sock Shop has no trace graph in this dataset.

For graph-covered systems, there were **259 ordered edge alert pairs** across **50 service edges** (79 pairs in Online Boutique and 180 in Train Ticket). In **50.6%** of pairs, both services first alerted in the same sampled window. The remaining delays are broadly variable across incidents. This is evidence that the current scores can show recorded alert ordering; it is not a stable per-edge propagation-time estimate.

## Run it

From the repository root, with the RE2 data and selected Phase 2 scores available:

```powershell
python scripts/serve_live_replay.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/lstm_selected/anomaly/fold1/ob/service_window_scores.csv
```

Open the printed local URL, usually `http://127.0.0.1:8765`, and choose **Start stream**. Stop the Python process with Ctrl+C.

Recreate the held-out alert timing analysis:

```powershell
python scripts/analyze_propagation.py --scores experiments/lstm_selected/anomaly --out experiments/propagation
```

## Limits and next research step

The local stream replays precomputed benchmark telemetry and scores. It is not connected to a live service, message broker, or telemetry collector. The dashboard reports alert timing after the downstream service crosses the detector threshold; it does not forecast a future impact time. Trace edges indicate observed parent-to-child calls and cannot establish that a caller's anomaly caused the callee's anomaly. Injection time is only an approximate symptom onset, and two-window persistence adds detection latency. The dataset also lacks traces for Sock Shop.

To claim predictive propagation timing, collect synchronized onset labels on independent incidents, train a time-to-event model using only information available before downstream onset, evaluate censoring and missed events, and compare against simple fixed-delay and graph-distance baselines. The current delay tables are exploratory and should not be presented as causal or predictive evidence.
