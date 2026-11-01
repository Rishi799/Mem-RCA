# Phase 2 LSTM workflow

Mem-RCA trains a per-service LSTM forecaster using only pre-injection training telemetry. For each 30-second window it observes the first 29 samples, predicts the final sample, and uses prediction error as an anomaly score. Graph ranking remains a separate trace-based component; this is not a learned graph neural network.

## Preprocessing options

The trainer supports two comparable pipelines:

- `--preprocessing legacy --score-calibration robust_z` reproduces the original median/std metric scaling and robust z-score calibration.
- `--preprocessing robust --score-calibration empirical_tail` filters features below 80% observed coverage using training data, applies signed `log1p`, scales with training-only median/IQR statistics, clips to ±10, adds the target-window missing fraction to the error, and maps service errors to empirical upper-tail rarity using normal validation scores.

All imputation and scaling parameters are fitted on normal training rows. The empirical calibration reference contains only normal validation scores. The evaluator selects a system alarm threshold from the validation normal quantile and uses strict exceedance to handle tied discrete tail scores. A two-of-three-window persistence rule can reduce isolated alarms.

## Reproduce training and model selection

Run from the repository root after downloading RE2 and generating the split manifest. PyTorch must be installed in the active environment with CUDA support for the GPU commands.

```powershell
# Legacy z-score model family
python scripts/train_lstm_grid.py --device cuda --preprocessing legacy --score-calibration robust_z --out experiments/lstm

# Calibration-only ablation derived from the same legacy raw errors
python scripts/recalibrate_lstm_grid.py --source experiments/lstm --out experiments/lstm_rankcal

# Robust preprocessing model family
python scripts/train_lstm_grid.py --device cuda --preprocessing robust --score-calibration empirical_tail --out experiments/lstm_robust

# Choose anomaly and RCA variants independently using validation data only
python scripts/select_lstm_variants.py --legacy-z experiments/lstm --legacy-rankcal experiments/lstm_rankcal --robust experiments/lstm_robust

# Evaluate the selected anomaly and RCA score sets
python scripts/evaluate_lstm_grid.py --root experiments/lstm_selected/anomaly --persistence-hits 2
python scripts/evaluate_lstm_grid.py --root experiments/lstm_selected/rca --persistence-hits 2
```

The grid runner skips a fold/system when its score file already exists. Use `--force` to retrain the selected jobs. Checkpoints, histories, scores, and aggregate metrics are regenerable and ignored by Git under `experiments/`.

## Demo

Build the recorded Online Boutique replay from the validation-selected RCA scores:

```powershell
python scripts/build_replay_demo.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/lstm_selected/rca/fold1/ob/service_window_scores.csv --model-name "Validation-selected LSTM RCA" --out experiments/demo/re2ob_checkoutservice_cpu_3_fold1_selected_rca.html
```

The HTML replay includes recorded telemetry, time-local service scores, the observed trace graph, and a root-cause ranking. It is a recorded incident replay, not a live production connection. The dataset's ground-truth root is shown separately from the model's ranking.

For an interactive local event stream, start the dashboard in a second terminal:

```powershell
python scripts/serve_live_replay.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/lstm_selected/anomaly/fold1/ob/service_window_scores.csv
```

Open `http://127.0.0.1:8765` in a browser. Play streams the incident in time steps; service alerts use the 99th percentile of normal validation window scores and require two consecutive windows. This demonstrates a replay stream, not an ingestion adapter for production telemetry.

Run the held-out connected-service alert timing analysis with:

```powershell
python scripts/analyze_propagation.py --scores experiments/lstm_selected/anomaly --out experiments/propagation
```

The script writes validation thresholds, per-case detection coverage, alert-pair events, and per-edge delay summaries. It uses training-fold trace topology and only post-injection scores. Interpret delays as observed alert ordering; the graph does not prove causation or predict a downstream alert before it happens. See [the propagation report](PHASE2_PROPAGATION.md).

`src/memrca/models/inference.py` loads checkpoints and applies their stored preprocessing and validation calibration when scoring new telemetry. It can be used to connect the model to a future streaming interface.

## Interpretation

- The robust preprocessing variant improves thresholded detection for Sock Shop, but its results are worse for Train Ticket. The model selector can retain different variants for anomaly detection and RCA.
- Graph edges represent observed call structure. Alert delays summarize when connected services crossed a threshold in the recorded incidents; they are not causal proof or a forecast.
- Fault injection time is an imperfect symptom-onset label. No unknown-fault rejection, uncertainty estimate, future propagation-delay prediction, or production data ingestion is implemented yet.
- The benchmark has three repetitions per configuration. Test folds have been inspected during development; the metrics in [the results report](PHASE2_RESULTS.md) are exploratory and need confirmation on new seeds or an external dataset.
