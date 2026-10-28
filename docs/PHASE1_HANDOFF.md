# Phase 1 completion and handoff

Phase 1 establishes the reproducible data and baseline workstream for Mem-RCA. It is complete as a project baseline; it does not include the proposed graph-temporal deep-learning model or live replay application.

## Deliverables

- Dataset audit: `experiments/audit/re2_summary.json`
- Case-level fold manifest and counts: `experiments/splits/re2_folds.csv` and `experiments/splits/re2_folds_summary.json`
- Trace quality report and reusable edge list: `experiments/graph/re2_graph_all_summary.csv`, `experiments/graph/re2_graph_all_diagnostics.csv`, and `experiments/graph/re2_graph_all_edges.csv`
- Baseline anomaly metrics, combined test-window alarms, per-service test-window scores, and root rankings: `experiments/baseline/`
- Validation-tuned graph ablation and held-out test comparison: `experiments/graph_rca/`
- Window-size comparison: `experiments/sensitivity/window_comparison.csv` and its 10/60-second run directories

All experiment outputs are generated locally and ignored by Git. The RE2 download itself is also ignored.

## Data and split verification

The RE2 index contains 270 cases, balanced as 90 cases per system, 45 per fault, and 90 per repetition. All 270 metric and injection-time files are present. Logs are present for 269 cases and traces for 180; these counts match the index. There are no incomplete system/fault/repetition groups or metadata-to-file mismatches.

The fold builder produces 630 fold-case rows: 210 cases in each of three folds. Per fold, repetition 1 supplies 60 train cases, repetition 2 supplies 60 validation cases, and repetition 3 supplies 90 test cases. Unseen fault types are excluded from train and validation and appear only in test. A case remains intact across its split; overlapping metric windows do not cross case-level splits. The design has only three repetitions per configuration, so statistical claims need cautious wording and stronger follow-up evaluation.

## Graph audit

The full trace audit covers 90 Online Boutique and 90 Train Ticket cases. Sock Shop has no trace files. Online Boutique has 9 distinct observed inter-service edges per case; Train Ticket has 20 to 55 (median 55). The audit found 294 repeated span rows in Online Boutique and none in Train Ticket; it found no duplicate span IDs with conflicting services or parents. It counted 17,801 unmatched parent references in Online Boutique and 1,033,279 in Train Ticket. Duplicate spans are collapsed before extracting edges, and missing graph evidence safely yields metric-only RCA. These edges describe observed calls and are not proof of causal fault propagation.

## Baseline and graph ablation

The Isolation Forest is trained on pre-injection training windows only. Per-service score calibration and the system alarm threshold use pre-injection validation windows only. Held-out test cases are used once for final metrics. The 30-second-window baseline's mean metrics across the three folds were:

| System | ROC-AUC | Average precision | F1 | RCA Hit@1 |
|---|---:|---:|---:|---:|
| Online Boutique | 0.799 | 0.734 | 0.016 | 0.233 |
| Sock Shop | 0.935 | 0.928 | 0.206 | 0.800 |
| Train Ticket | 0.908 | 0.916 | 0.352 | 0.733 |

The graph reranker selects its weight on validation MRR using training-case edges only. On held-out tests, equal-weight averages over fold/system were metric-only versus graph-supported Hit@1 0.589 versus 0.593, Hit@3 0.826 versus 0.837, and MRR 0.728 versus 0.732. The aggregate gain is small and comes from Online Boutique; the selected graph weight is zero for Sock Shop (no traces) and Train Ticket. This is an exploratory reranking result, not evidence of learned causality.

## Score schema for integration

`experiments/baseline/test_service_window_scores.csv` contains one row per test case, fold, window, and service:

| Column | Meaning |
|---|---|
| `fold`, `system`, `case` | Evaluation partition identifiers |
| `fault`, `fault_status` | Fault label and known/unseen test status |
| `window_start`, `seconds_from_injection` | Window timestamp and offset |
| `is_anomaly` | Evaluation label derived from fault injection time |
| `service`, `anomaly_score` | Service-level score used to support RCA |

`test_root_rankings.csv` aggregates service scores over the configured post-injection RCA horizon and records the true root, service score, and rank. `test_window_scores.csv` contains the combined system-level anomaly score and alarm. The graph edge CSV joins on `case` and `system` and has `source`, `target`, and `span_count` columns. The graph evaluator reads the incident-level rankings, builds topology from training cases only, tunes on validation, and reports metric-only and graph-supported ranks on test.

## Reproduction

Run from the repository root after installing project dependencies and downloading RE2:

```powershell
python scripts/audit_dataset.py --suite re2
python scripts/build_splits.py
python scripts/audit_graphs.py --all
python scripts/run_baseline.py
python scripts/evaluate_graph_ranker.py
python scripts/run_baseline.py --window 10 --stride 10 --out experiments/sensitivity/window_10s
python scripts/run_baseline.py --window 60 --stride 10 --out experiments/sensitivity/window_60s
python scripts/compare_window_runs.py
python -m unittest discover -s tests -v
```

The full graph audit and baseline run process large trace/metric data and can take several minutes. To evaluate graph handling on a small representative subset before the full audit, omit `--all`.

## Remaining work

Phase 2 is to build and compare the graph-temporal deep model, implement explicit unknown-fault and uncertainty handling, evaluate missing-telemetry ablations, and create the recorded-incident replay interface. Reliable downstream impact-time labels are not provided by the present Phase 1 pipeline, so propagation-time prediction remains future work.
