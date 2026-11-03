# Phase 3 RE2 full-model grid results (exploratory)

## Configuration

- Full temporal-graph variant, CUDA, five epochs, seeds 42/43/44.
- Three RE2 folds and systems: Online Boutique (OB), Sock Shop (SS), Train Ticket (TT).
- 27 trained/evaluated runs total. Means and standard deviations below aggregate the nine fold/seed runs per system.
- RE2 test cases were inspected earlier in development. Treat these as exploratory results, not confirmatory evidence. Folds may reuse incident IDs, so the nine runs are not nine independent test sets.

## Anomaly detection

| System | ROC-AUC | Average precision | F1 | False-positive rate | Recall |
|---|---:|---:|---:|---:|---:|
| OB | 0.953 ± 0.019 | 0.929 ± 0.030 | 0.924 ± 0.018 | 0.085 ± 0.043 | 0.931 ± 0.020 |
| SS | 0.882 ± 0.019 | 0.890 ± 0.020 | 0.650 ± 0.119 | 0.013 ± 0.017 | 0.499 ± 0.141 |
| TT | 0.807 ± 0.018 | 0.807 ± 0.022 | 0.573 ± 0.093 | 0.012 ± 0.005 | 0.411 ± 0.091 |

The low false-positive rates on SS and TT come with low recall. Detection quality varies substantially by system; pooled metrics would hide that weakness.

## Root-cause ranking

These are the learned root-head scores, evaluated separately for known and unseen fault labels.

| System | Fault status | Hit@1 | MRR |
|---|---|---:|---:|
| OB | Known | 1.000 ± 0.000 | 1.000 ± 0.000 |
| OB | Unseen | 1.000 ± 0.000 | 1.000 ± 0.000 |
| SS | Known | 0.939 ± 0.055 | 0.967 ± 0.030 |
| SS | Unseen | 0.889 ± 0.105 | 0.936 ± 0.064 |
| TT | Known | 0.983 ± 0.025 | 0.988 ± 0.018 |
| TT | Unseen | 0.956 ± 0.073 | 0.970 ± 0.050 |

These high ranking scores, especially the perfect OB scores, require an external test before they support a generalization claim. The benchmark has repeated service/fault configurations, and its test partition has already been viewed during development.

## Calibration

Validation temperature fitting used 20 incidents per run. Across the nine runs per system, fit-set mean Brier/ECE were OB 0.000005/0.000252, SS 0.043008/0.023672, and TT 0.036186/0.024269. These measurements are on the validation cases used to fit temperature, so they are optimistic fit diagnostics rather than independent calibration estimates. Test selective metrics and per-run values are in `experiments/temporal_graph_re2_grid/selective_metrics_by_run.csv`.

On the previously inspected test cases, mean incident-level test coverage was 100% for all systems because the validation-selected abstention threshold was zero. Mean selective risk / Brier / ECE were OB 0.007 / 0.010 / 0.006, SS 0.085 / 0.150 / 0.074, and TT 0.022 / 0.037 / 0.026. Since no cases were rejected, the abstention rule did not demonstrate a coverage-risk tradeoff; these test values are exploratory and do not establish calibration generalization.

## Paired RE2 ablations

Each ablation was trained with seed 42 on all three folds and systems, and paired against the matching seed-42 full model. Values below are mean F1 deltas (ablation minus full) across three folds:

| Ablation | OB ΔF1 | SS ΔF1 | TT ΔF1 |
|---|---:|---:|---:|
| No graph | +0.000 | +0.000 | -0.092 |
| No observation mask | +0.002 | -0.000 | -0.052 |
| No consistency loss | -0.001 | -0.001 | -0.101 |

The graph and consistency terms show little F1 change on OB/SS; TT falls when those components are removed, but the fold-to-fold variation is substantial. Removing the mask has mixed, small effects. Root-head ranking is mostly unchanged in these single-seed comparisons. This is evidence to investigate, not a robust component-contribution claim: the ablations use one seed, the RE2 test cases were already inspected, and the results need replication on untouched data.

## Artifacts

The run-level and aggregated full-model CSVs are under the ignored `experiments/temporal_graph_re2_grid/` directory. Ablation outputs and paired deltas are under `experiments/temporal_graph_re2_ablation_grid/`. The calibration report is [PHASE3_CALIBRATION_STATUS.md](PHASE3_CALIBRATION_STATUS.md). RE1 and RE3 were later downloaded and used for the open-world external validation ([EXTERNAL_VALIDATION_PROTOCOL.md](EXTERNAL_VALIDATION_PROTOCOL.md)); the Phase 3 grid itself was not re-run on them.
