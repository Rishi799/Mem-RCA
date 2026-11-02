# Phase 3 temporal-graph pilot (exploratory)

## Run configuration

- One system/fold: Online Boutique, `fold1`.
- Training: CUDA, early stopping on validation root MRR, best validation checkpoint selected from 10 epochs.
- Model: masked per-service LSTM encoders, directed training-case graph attention, normal-only next-step forecast loss, training-incident root ranking loss, and artificial metric masking consistency.
- Calibration: temperature and abstention threshold fit on the 20 validation incidents only; target validation risk 20%.
- Test: 30 RE2 incidents (20 known faults and 10 held-out fault types). These cases were already examined in earlier development, so this is not confirmatory evidence.

## Observed pilot metrics

| Test measure | Full model | Existing LSTM, same fold/system |
|---|---:|---:|
| Anomaly ROC-AUC | 0.923 | 0.943 |
| Anomaly F1 | 0.910 | 0.890 |
| Test false-positive rate | 0.133 | 0.066 |
| Root Hit@1, all cases | 1.000 (root head) | 0.867 (forecast-score ranking) |
| Root MRR, all cases | 1.000 (root head) | 0.919 (forecast-score ranking) |

The pilot root head ranked the label first in 20/20 known and 10/10 unseen cases. This unusually high score is based on only 30 correlated incidents from one system/fold and cannot be treated as a generalization claim. The anomaly detector traded higher recall for more false alarms than the existing model. It did not improve every measure.

## One-fold ablation pilot

All variants used seed 42 on the same single Online Boutique fold. Results are included to check the experiment path, not to support claims:

| Variant | ROC-AUC | F1 | Test FPR | Root-head Hit@1 / MRR |
|---|---:|---:|---:|---:|
| Full | 0.923 | 0.910 | 0.133 | 1.000 / 1.000 |
| No graph | 0.924 | 0.916 | 0.123 | 1.000 / 1.000 |
| No observation-mask input | 0.924 | 0.915 | 0.124 | 1.000 / 1.000 |
| No consistency loss | 0.924 | 0.910 | 0.133 | 1.000 / 1.000 |

These ablations do **not** show a measurable contribution from graph attention, the mask input, or the consistency loss on this case set. The identical perfect root-head ranking across variants suggests the current test split is too limited/repetitive to establish the proposed research contribution. The next evaluation must use multiple seeds and the independent RE1-validation/RE3-test protocol before deciding which components to retain.

The fitted validation temperature reached its lower bound (0.05), with a zero validation abstention threshold and 100% validation coverage. Test ECE was 0.0 because every one of the 30 cases was ranked correctly and the probabilities saturated. This risk/coverage result is not informative about real-world confidence. Calibration and abstention need a larger independent validation set or cross-fitted calibration before they can support a reliability claim.

## Artifacts

Ignored experiment outputs are under `experiments/temporal_graph_pilot/fold1/ob/`:

- `model.pt`, `training_history.json`
- `service_window_scores.csv`
- `root_cause_case_predictions.csv`, `root_calibration.json`
- `anomaly_evaluation/metrics.csv`
- `root_head_evaluation/root_head_metrics_summary.csv`
- `root_head_evaluation/selective_risk_coverage.csv`

The existing evaluator was adjusted to omit undefined service scores when a service has no observed target metrics in a window. This prevents missing values from breaking root-rank calculations; such windows remain excluded from that service's score ranking.

## Decision

Continue with multi-seed and all-system validation on the independent RE1-validation/RE3-test protocol. Keep the test results marked exploratory. Do not add the pilot values to headline project claims. The current ablations do not validate the proposed contribution; the next research gate is independent evaluation and calibration, not frontend polish.
