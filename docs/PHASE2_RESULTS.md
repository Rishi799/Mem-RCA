# Phase 2 model and evaluation results

## Pipeline

The current experiment trains one normal-only LSTM forecaster per service. Each 30-second window uses the first 29 metric samples to predict the last sample; mean squared forecast error is the raw anomaly score. Train/validation/test assignments are case-level (repetitions 1/2/3), and unseen fault classes are excluded from training and validation.

Three score variants were compared:

- **Legacy z-score:** the original median/std feature scaling and robust z-score calibration.
- **Legacy empirical-tail:** the legacy model's raw forecast errors mapped to per-service empirical tail rarity using normal validation scores.
- **Robust empirical-tail:** features with under 80% training coverage are dropped; signed `log1p`, training-only median/IQR scaling and clipping to ±10 are applied. The target missing fraction contributes to the error, then scores use the same validation-only empirical tail mapping.

For anomaly selection, each fold/system chooses the candidate with the best known-fault validation F1 after a 99th-percentile normal threshold and a two-of-three-window persistence rule. For RCA selection, it chooses the candidate with the best validation MRR. The graph reranker weight is also tuned on validation cases, with topology from training cases only. The decisions and copied score files are in `experiments/lstm_selected/` (ignored by Git); `model_selection.csv` records the chosen variant and validation metrics.

The threshold uses strict exceedance over the validation normal quantile because empirical-tail scores are discrete and can have ties. This produced very low observed validation false-positive rates, but test false-positive rates still differ across repetitions, so the validation target is not a guarantee under drift.

## Selected test results

Means across the three held-out folds, using the validation-selected anomaly variant and the two-of-three persistence rule:

| System | ROC-AUC | Average precision | F1 | Recall | Test FPR | Incidents detected within 120 s |
|---|---:|---:|---:|---:|---:|---:|
| Online Boutique | 0.952 | 0.933 | 0.924 | 0.915 | 0.067 | 1.000 |
| Sock Shop | 0.862 | 0.870 | 0.633 | 0.470 | 0.011 | 0.611 |
| Train Ticket | 0.844 | 0.835 | 0.742 | 0.615 | 0.042 | 0.733 |

Means across held-out folds using the separately validation-selected RCA variant:

| System | Metric Hit@1 | Metric Hit@3 | Metric MRR | Graph Hit@1 | Graph Hit@3 | Graph MRR |
|---|---:|---:|---:|---:|---:|---:|
| Online Boutique | 0.889 | 0.967 | 0.931 | 0.889 | 0.978 | 0.931 |
| Sock Shop | 0.856 | 0.944 | 0.906 | 0.856 | 0.944 | 0.906 |
| Train Ticket | 0.633 | 0.811 | 0.734 | 0.633 | 0.800 | 0.733 |

Graph evidence did not consistently improve RCA, and the graph is observed caller/callee structure, not causal proof. Sock Shop has no trace graph, so it uses metric-only RCA.

## Interpretation and limits

- Robust preprocessing materially improved Sock Shop thresholded recall and F1, while its ranking metrics were lower than the original model. It performed poorly on Train Ticket, so the validation selector retains legacy variants there.
- Per-fold selection uses only validation data, but it is noisy: each validation fold has a small set of known-fault incidents. A single selected model family per system, chosen through nested cross-validation, would be a stronger final research design.
- The test folds have been inspected during development and should be treated as exploratory results, not an untouched final benchmark. For a paper, freeze the method and rerun on new seeds or an external benchmark.
- Fault injection time is an imperfect anomaly-onset label; symptoms may begin later. The current system does not reject unknown faults, estimate uncertainty, predict propagation delays, or consume live production telemetry.

Detailed fold-level and known/unseen results are in `experiments/lstm_selected/anomaly/summary_persist2/` and `experiments/lstm_selected/rca/summary_persist2/`.
