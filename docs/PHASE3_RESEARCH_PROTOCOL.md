# Mem-RCA research protocol (candidate)

## Research question

Can a temporal graph model diagnose service anomalies and rank root causes reliably when some service telemetry is missing and test faults were not represented during training, while reporting when its diagnosis is uncertain?

This is a candidate research direction, not a novelty or publication guarantee. A focused literature review and comparison table must be completed before claiming a research gap.

## Proposed contribution to investigate

The candidate method is a missingness-aware, selective temporal-graph diagnosis model:

1. A service-level LSTM encodes each service's metric history and its observation mask. Per-service forecast residuals provide anomaly scores; the existing normal-only LSTM remains the primary baseline.
2. Directed graph attention passes contextual evidence across observed caller-to-callee dependencies. The graph is built from training cases only for benchmark evaluation.
3. During training, service/metric telemetry is deliberately masked. A consistency objective encourages stable forecasts and root-cause rankings across clean and masked views of the same incident.
4. The model reports a calibrated confidence or abstains when the root-cause ranking is not sufficiently supported.

The proposed contribution to test is the combination of dropout-consistent training and calibrated selective root-cause diagnosis under unseen faults and missing telemetry. LSTM-plus-GNN fusion alone is not a novelty claim: prior methods already combine those families, including [TraceGra](https://doi.org/10.1016/j.comcom.2023.03.028), and newer trace graph autoencoders include [TraceDAE](https://doi.org/10.1109/TNSM.2025.3583213) and [TraceHG](https://doi.org/10.1109/TSC.2026.3667576). The review must search for work on missing telemetry, open-set/selective RCA, robustness training, and calibration before finalizing this contribution.

## Hypotheses

- **H1: Diagnosis under missing telemetry.** The proposed model retains better root-cause ranking and anomaly detection than the current LSTM and graph reranker as telemetry is removed.
- **H2: Selective reliability.** At the same retained-case coverage, calibrated abstention reduces root-cause errors compared with always returning a root cause.
- **H3: Unseen faults.** The model maintains useful service-level diagnosis on fault classes excluded from training, without tuning on the final test cases.

If any hypothesis is not supported, report the negative result and do not change the held-out protocol after seeing it.

## Data and leakage controls

- Existing RE2 experiments are exploratory: their test cases have already been inspected. Preserve them as pilot results only.
- Use case-level splits. No windows from a case may cross train, validation, and test.
- Fit feature selection, imputation, scaling, graph topology, thresholds, and calibration on training/validation data only, as appropriate. Do not use test labels for tuning.
- Report the exact suite, cases, fault holdouts, random seeds, and graph construction for every result.
- Prefer a new independent evaluation on RCAEval RE1/RE3 or a separately collected controlled-fault dataset. The RCAEval benchmark contains 735 cases across RE1, RE2, and RE3; the current project primarily evaluated the 270 RE2 cases. See the [RCAEval paper](https://arxiv.org/abs/2412.17015).
- Because repeated configurations are limited, bootstrap confidence intervals by incident/case, not by overlapping windows. Use multiple training seeds and describe the limited number of independent incidents.

## Comparisons and ablations

Required comparisons:

- Existing Isolation Forest baseline.
- Existing normal-only LSTM.
- Existing metric-only RCA ranking.
- Existing validation-tuned trace graph reranker.
- Proposed temporal graph model.

Required ablations:

- Remove graph message passing.
- Remove the missingness mask.
- Remove the consistency objective.
- Remove confidence calibration/abstention.
- Compare random metric masking with whole-service masking.

For every model, report parameter count and inference latency so accuracy changes can be interpreted against cost.

## Evaluation measures

| Task | Primary measures |
|---|---|
| Anomaly detection | PR-AUC, ROC-AUC, precision, recall, F1, false alarms per incident/time, detection delay |
| Root-cause ranking | Hit@1, Hit@3, MRR; report known and unseen faults separately |
| Selective diagnosis | Coverage-risk curve, risk at fixed coverage, Brier score and calibration error |
| Missing telemetry | Repeat primary metrics under 0%, 10%, 30%, and 50% masking; test both random metric loss and whole-service loss |
| Efficiency | Training time, inference latency per window, parameter count, peak memory |

Tune thresholds and calibration on validation data only. Aggregate over cases and seeds; report confidence intervals and per-system results, not only a pooled mean.

## Scope boundary

The first research version predicts anomaly scores and root-cause rankings with uncertainty. It does not claim causal propagation. Predicting the future time at which a downstream service will fail remains a separate task requiring independently labeled, synchronized propagation events and a survival/time-to-event evaluation.

## Frontend target

Use a small Streamlit application around a stable inference API. The final demo should have:

1. **Incident replay/live view:** telemetry, anomaly score timeline, pause/speed controls, and missing-data indicators.
2. **Service graph:** directed observed dependencies, current alert state, and ranked root-cause candidates.
3. **Evidence and confidence:** metric contributors, trace support, calibrated confidence, and an explicit abstention state.
4. **Evaluation:** baseline/ablation comparisons, fault-known status, missingness setting, and detection/RCA metrics.

Keep the recorded RCAEval replay available as the reliable offline demo. Add a true live telemetry adapter only after the model API and offline results are stable. A local OpenTelemetry Demo is a possible data source; it is a separately instrumented sample application, not part of the current RCAEval data.

The inference wrapper is `src/memrca/models/temporal_graph_inference.py`. It loads the saved checkpoint and its training-fitted scalers, accepts synchronized raw metric DataFrames per service, and returns anomaly scores, relative per-window root rankings, and graph edges. `predict_incident` aggregates the calibrated incident decision and abstention state; calibration must not be applied to a single window. The replay server can call this same inference API on raw telemetry windows with `--checkpoint`, while using the matching score CSV for validation-only alert thresholds and the saved case-level file for the incident-calibrated decision. It remains a recorded replay front end until a real telemetry adapter is connected.

Example with the single-system pilot artifacts:

```powershell
python scripts/serve_live_replay.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/temporal_graph_pilot/fold1/ob/service_window_scores.csv
```

To run the inference API directly on replay windows, pass the checkpoint paired with the same score export:

```powershell
python scripts/serve_live_replay.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/temporal_graph_pilot/fold1/ob/service_window_scores.csv --checkpoint experiments/temporal_graph_pilot/fold1/ob/model.pt --device cuda
```

## Two-person work interface

- **Person A (GPU):** model implementation/training, baselines, controlled ablations, calibration, and locked test evaluation.
- **Person B:** literature comparison, dataset manifests and missingness experiments, inference API integration, Streamlit dashboard, and demo packaging.
- Agree on a shared prediction record before parallel implementation: `timestamp`, `service`, `anomaly_score`, `root_cause_rank`, `confidence`, `abstain`, `missing_fraction`, and `evidence`.

## Completion criteria for this research version

- Literature comparison includes at least the closest graph-temporal and selective/open-set RCA methods, with a precise distinction from the proposed consistency objective.
- The split manifest and training configuration are versioned; test data are untouched until method and thresholds are frozen.
- Baselines, ablations, uncertainty calibration, missingness stress tests, multiple seeds, and case-level confidence intervals are all reported.
- The same saved preprocessing/model artifact powers both evaluation and the dashboard.
- The paper/demo clearly separate measured alert ordering from causal or future impact prediction.

## Candidate trainer status

The initial implementation scaffold is in `src/memrca/models/temporal_graph.py`; aligned windows and training-only topology preparation are in `src/memrca/data/temporal_graph.py`. The current trainer is a pilot path, not a finished research evaluation. It trains forecasting loss on normal training windows, root ranking on training fault windows, and a consistency term across artificially masked input views.

## Independent suite evaluation

For a stronger calibration/evaluation split, use RE2 train only, RE1 as validation/calibration, and RE3 as the final external test. `scripts/build_external_manifest.py` builds this manifest for a selected fold/system while preserving the RE2 training rows and graph topology. The model feature schema and transforms are fit from RE2 training cases only; RE1 is used for early stopping and calibration; RE3 must remain untouched until all choices are frozen.

Download the two additional suites on a machine with Hugging Face access, then build a fold manifest:

```powershell
python scripts/download.py --suite re1
python scripts/download.py --suite re3
python scripts/build_external_manifest.py --fold fold1 --validation-suite RE1 --suite RE3 --out experiments/splits/fold1_re2_train_re1_val_re3_test.csv
```

Run the complete seed/ablation matrix for that fold:

```powershell
python scripts/run_temporal_graph_grid.py --folds fold1 --systems ob ss tt --seeds 42 43 44 --ablations full no_graph no_mask no_consistency --manifest experiments/splits/fold1_re2_train_re1_val_re3_test.csv --out experiments/temporal_graph_external --device cuda
```

Repeat manifest creation and the grid command for `fold2` and `fold3`, changing the file names and `--folds` value. The grid writes per-run scores and aggregate anomaly, RCA, and selective-confidence CSVs. The full matrix is compute-heavy; it can be resumed without `--force` because completed checkpoints and score files are skipped.

In this workspace, RE1/RE3 case metadata are present in `cases.parquet`, but their case folders are not downloaded. The RE3 download attempt failed because DNS/network access to Hugging Face was unavailable. Until those case folders are downloaded, independent-suite metrics cannot be produced here.

To run one pilot job on the GPU after the existing RE2 split and training-edge manifests are available:

```powershell
python scripts/train_temporal_graph.py --fold fold1 --system ob --device cuda --out experiments/temporal_graph
python scripts/evaluate_lstm_scores.py --scores experiments/temporal_graph/fold1/ob/service_window_scores.csv --persistence-hits 2 --out experiments/temporal_graph/fold1/ob/anomaly_evaluation
python scripts/evaluate_temporal_graph.py --scores experiments/temporal_graph/fold1/ob/service_window_scores.csv
```

These commands use the already-inspected RE2 test partition and are for integration/pilot analysis only. The pilot trainer fits a temperature and an abstention threshold using validation incidents only, then writes `root_cause_case_predictions.csv`, `root_calibration.json`, and a risk/coverage report. The threshold is a configurable 20% maximum validation risk by default; few validation incidents can make this estimate unstable. Seed/repetition aggregation, complete ablations, and independent data evaluation are still required before research claims.
