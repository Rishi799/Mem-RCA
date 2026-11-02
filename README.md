# Mem-RCA

**Deep-learning anomaly detection and root-cause ranking for microservices**

**Course:** AI253IA — Artificial Neural Networks & Deep Learning (PBL), RV College of Engineering<br>
**Team:** Priyanshi Gupta (1RV24CI088) · Rishi Agarwal (1RV24CI099) · Sambhav Badaya (1RV24CI106)

Mem-RCA is a research prototype for detecting abnormal microservice behavior and ranking likely root-cause services. It includes a normal-only LSTM baseline and a Phase 3 temporal-graph candidate with incident-level temperature calibration and abstention. RE2 results are exploratory; external validation and reliable selective diagnosis have not been established.

## Research direction

The main research question is whether graph-informed diagnosis can remain reliable when fault types are held out from training and telemetry is incomplete. The intended system will:

- detect abnormal behavior from service telemetry;
- rank likely root-cause services using metric evidence and service dependencies;
- distinguish known fault classes from unfamiliar or uncertain incidents;
- evaluate how diagnosis changes when telemetry sources are missing.

The demonstration replays recorded RCAEval incidents through a local event stream and visualizes anomaly alerts across the observed service graph. A held-out analysis estimates the time gap between persistent alerts on connected services. These are recorded-data results, not a live production monitor or a causal forecast. The temporal-graph inference API can score raw windows during replay with `--checkpoint`; incident-level calibration is only reported after aggregation.

## Current implementation

The repository contains data preparation, baseline training, and evaluation tools:

- `src/memrca/data/loader.py` reads the RCAEval case index, metrics, logs, traces, and fault injection time.
- `src/memrca/data/windows.py` converts one-second metrics into fixed-width labeled windows and omits windows that cross the injection boundary.
- `src/memrca/data/graph.py` extracts observed parent-service to child-service edges from trace span IDs and collapses duplicate span IDs.
- `src/memrca/baselines/isolation_forest.py` implements a normal-only per-service Isolation Forest baseline.
- `scripts/download.py` downloads dataset suites from Hugging Face.
- `scripts/audit_dataset.py` checks downloaded-file coverage and label/repetition balance from the case index.
- `scripts/build_splits.py` validates the repetition/fault folds and writes a case-level manifest.
- `scripts/audit_graphs.py` reports dataset-wide trace coverage and audits representative graphs by default. Use `--all` for per-case diagnostics across every trace file.
- `scripts/run_baseline.py` trains and evaluates anomaly scores and service root-cause rankings; it saves both combined-window alarms and per-service window scores for evidence traceability.
- `scripts/evaluate_graph_ranker.py` tunes a graph-supported reranker on validation cases and compares it with metric-only RCA on test cases.
- `scripts/serve_live_replay.py` streams one recorded incident to a local browser dashboard, with thresholded service alerts and observed alert-delay updates.
- `scripts/analyze_propagation.py` measures persistent alert coverage and connected-service alert delays on test cases using validation thresholds and training-fold trace topology.
- `scripts/build_external_manifest.py` combines RE2 training cases with independent RE1 validation and RE3 test suites for cross-suite evaluation.
- `scripts/run_temporal_graph_grid.py` runs seeds and graph/mask/consistency ablations with aggregate reports.
- `scripts/compare_temporal_graph_ablations.py` computes paired deltas against the matching full-model runs.
- `scripts/summarize_temporal_graph_calibration.py` aggregates validation-only calibration diagnostics.
- `src/memrca/models/temporal_graph_inference.py` loads a candidate temporal-graph checkpoint and applies its saved preprocessing and calibration.
- `scripts/explore_case.py` summarizes one case, ranks services by a simple metric-shift score, and saves plots.
- `configs/data.yaml` records the RE2 systems, fault classes, known/unseen folds, and initial 30-second windows scored every 10 seconds.

**Phase 1 status:** the RE2 data pipeline, split validation, Isolation Forest baseline, trace graph audit, and graph-supported root-cause evaluation are implemented. The Isolation Forest and graph reranker are baselines, not learned graph models.

**Phase 2 status:** legacy and robust LSTM pipelines have been trained and compared across all nine RE2 fold/system combinations. Validation-only selection produces separate score sets for anomaly detection and RCA because the best preprocessing differs by system and objective. See the [Phase 2 guide](docs/PHASE2_LSTM_START.md) and [results and limitations](docs/PHASE2_RESULTS.md). The LSTM is metric-only; graph evidence is a separately tuned trace reranker, not a learned graph neural network.

**Phase 3 status:** the LSTM plus directed graph-attention candidate, validation calibration, inference API, and recorded replay integration are implemented. The full model completed 27 RE2 runs (three folds, three systems, three seeds); graph, observation-mask, and consistency-loss ablations completed 27 more runs (three folds, three systems, one seed per ablation). RE2 results remain exploratory because test cases were viewed during development and folds reuse incident configurations. Independent RE1 validation and untouched RE3 testing could not run: Hugging Face DNS resolution fails in this environment. See the [research protocol](docs/PHASE3_RESEARCH_PROTOCOL.md), [pilot report](docs/PHASE3_PILOT.md), [RE2 grid results](docs/PHASE3_RE2_GRID_RESULTS.md), and [calibration status](docs/PHASE3_CALIBRATION_STATUS.md). Future-impact-time prediction and a production telemetry adapter remain separate work.

## Dataset

The project uses [RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval), a public benchmark containing 735 cases from Online Boutique (`ob`), Sock Shop (`ss`), and Train Ticket (`tt`). The case index includes metadata for all suites; download RE2 for the initial multi-source experiments.

| Suite | Cases | Telemetry | Intended use |
|---|---:|---|---|
| RE1 | 375 | Metrics | Optional metric-only comparison |
| RE2 | 270 | Metrics, logs, traces | Main initial dataset |
| RE3 | 90 | Metrics, logs, traces | Optional code-level fault evaluation |

An RE2 case folder is named like `re2ob_checkoutservice_cpu_1` and contains `metrics.parquet`, `logs.parquet`, `traces.parquet`, and `inject_time.txt`. The timestamp separates the recorded pre-injection and post-injection periods. Some cases or suites may not include every telemetry file.

The full RE2 download is approximately 3 GB. Raw data and generated experiment outputs are excluded from Git; download the dataset separately in each working copy.

The downloaded RE2 index reports logs for 269 of 270 cases and traces for 180 of 270 cases: Sock Shop has no traces, and one Train Ticket case has no logs. The index lists about 43.6 million log rows and 101.8 million trace spans. A full graph audit processed all 180 trace cases: Online Boutique has 9 observed inter-service edges per case; Train Ticket has 20–55 (median 55); Sock Shop has no trace graph. The audit found 294 repeated span rows in Online Boutique and none in Train Ticket, with no duplicate span IDs mapped to conflicting services or parents. Unmatched parent references were 17,801 / 32.3 million in Online Boutique and 1.03 million / 66.8 million in Train Ticket. Graphs are built after de-duplicating span IDs, and their edges represent observed call structure, not proof that one service caused another to fail. The large trace and log tables should be processed per case or aggregated before model training. Metric files have 930–1,441 one-second samples and 69–376 metric columns per case.

## Setup

Windows with Python 3.11 or later:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is optional for Phase 1 and required for Phase 2 LSTM training. Install a Windows/Python/CUDA-compatible build from the [official PyTorch installer](https://pytorch.org/get-started/locally/) on the training laptop. The correct command depends on the GPU, driver, and Python environment.

## Download and explore RE2

From the repository root:

```powershell
python scripts/download.py --suite re2
python scripts/audit_dataset.py --suite re2
python scripts/explore_case.py
python scripts/build_splits.py
python scripts/audit_graphs.py
python scripts/audit_graphs.py --all
python scripts/run_baseline.py
python scripts/evaluate_graph_ranker.py
python scripts/build_replay_demo.py --case re2ob_checkoutservice_cpu_3 --fold fold1
python -m unittest discover -s tests -v
```

To train and evaluate the Phase 2 LSTM (requires the configured PyTorch environment):

```powershell
# Train legacy scores and calibrate them for the calibration-only ablation
python scripts/train_lstm_grid.py --device cuda --preprocessing legacy --score-calibration robust_z --out experiments/lstm
python scripts/recalibrate_lstm_grid.py --source experiments/lstm --out experiments/lstm_rankcal

# Train robust scores, then select the best variant on validation data
python scripts/train_lstm_grid.py --device cuda --preprocessing robust --score-calibration empirical_tail --out experiments/lstm_robust
python scripts/select_lstm_variants.py --legacy-z experiments/lstm --legacy-rankcal experiments/lstm_rankcal --robust experiments/lstm_robust
python scripts/evaluate_lstm_grid.py --root experiments/lstm_selected/anomaly --persistence-hits 2
python scripts/evaluate_lstm_grid.py --root experiments/lstm_selected/rca --persistence-hits 2
python scripts/build_replay_demo.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/lstm_selected/rca/fold1/ob/service_window_scores.csv --model-name "Validation-selected LSTM RCA" --out experiments/demo/re2ob_checkoutservice_cpu_3_fold1_selected_rca.html

# Start the local time-stepped replay dashboard (open the printed localhost URL)
python scripts/serve_live_replay.py --case re2ob_checkoutservice_cpu_3 --fold fold1 --scores experiments/lstm_selected/anomaly/fold1/ob/service_window_scores.csv

# Estimate alert delays for connected services across held-out cases
python scripts/analyze_propagation.py --scores experiments/lstm_selected/anomaly --out experiments/propagation
```

To download one system only:

```powershell
python scripts/download.py --suite re2 --system ob
```

To inspect a specific downloaded case:

```powershell
python scripts/explore_case.py --case re2ob_checkoutservice_cpu_1
```

The explorer prints sampling interval, missing-value rate, metrics per service, and the top services by metric shift. It writes figures to `experiments/explore/` by default.

The baseline uses 30-second metric-mean windows with a 10-second stride. For each fold and service, an Isolation Forest is fitted only on pre-injection windows from training cases. Missing values are imputed using training windows. Per-service scores are standardized using pre-injection validation windows; a system alarm threshold is selected from the validation normal-score quantile configured by `--normal-fpr` (default 1%). Test cases are repetition 3, including both known and held-out fault types. RCA ranks use the mean of the strongest quarter of each service's standardized scores in the first 120 seconds after injection.

The run writes `experiments/splits/` and `experiments/baseline/` outputs, which are ignored by Git. Baseline results should be treated as an initial reference only. In the current recorded run, mean-over-fold ROC-AUC / average precision / F1 / RCA Hit@1 were: Online Boutique 0.799 / 0.734 / 0.016 / 0.233; Sock Shop 0.935 / 0.928 / 0.206 / 0.800; Train Ticket 0.908 / 0.916 / 0.352 / 0.733. The anomaly label treats every window after the injection timestamp as anomalous, although symptoms may start later. The threshold prioritizes a low validation false-alarm rate, which explains the low recall for some systems. Metric-only RCA is the reference ranker. The graph-supported reranker is tuned on validation labels using training-case edges only; validation and test case traces are excluded from the topology. Trace edges remain observed-call evidence, not proof of causality.

Window sensitivity was evaluated at 10, 30, and 60 seconds while keeping the scoring stride at 10 seconds. Equal-weight averages across the three systems and three folds were:

| Window | ROC-AUC | Avg. precision | F1 | Incident detection rate | RCA Hit@1 | RCA Hit@3 | MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 s | 0.859 | 0.840 | 0.209 | 0.326 | 0.589 | 0.804 | 0.723 |
| 30 s | 0.881 | 0.859 | 0.191 | 0.241 | 0.589 | 0.826 | 0.728 |
| 60 s | 0.893 | 0.875 | 0.190 | 0.222 | 0.615 | 0.833 | 0.744 |

Longer windows improved ranking and score discrimination on average; shorter windows flagged more incidents. Per-system results differ, and the dataset has only three repetitions per configuration, so 30 seconds remains a practical initial setting rather than a proven optimum. Reproduce the comparison with:

```powershell
python scripts/run_baseline.py --window 10 --stride 10 --out experiments/sensitivity/window_10s
python scripts/run_baseline.py --window 60 --stride 10 --out experiments/sensitivity/window_60s
python scripts/compare_window_runs.py
```

For graph-supported RCA, `scripts/evaluate_graph_ranker.py` chooses a weight per fold/system from validation MRR. On the held-out test cases, mean-over-fold/system metric-only versus graph-supported Hit@1 / Hit@3 / MRR were 0.589 / 0.826 / 0.728 versus 0.593 / 0.837 / 0.732. The small aggregate change came from Online Boutique; Sock Shop had no trace graph, and validation selected zero graph weight for Train Ticket. Treat this as an exploratory reranking result, not evidence of learned causality.

## Project layout

```text
configs/data.yaml          Dataset settings and provisional open-world folds
scripts/download.py        RCAEval download command
scripts/audit_dataset.py   Dataset coverage and split-balance report
scripts/build_splits.py    Case-level train/validation/test manifest
scripts/audit_graphs.py    Trace coverage and representative graph audit
scripts/explore_case.py    Case summary and exploratory figures
scripts/run_baseline.py    Isolation Forest and root-cause evaluation
scripts/evaluate_graph_ranker.py  Validation-tuned graph RCA comparison
scripts/compare_window_runs.py  Window-size sensitivity comparison
scripts/build_replay_demo.py  Self-contained recorded incident replay
scripts/serve_live_replay.py  Local streamed replay dashboard
scripts/analyze_propagation.py Held-out connected-service alert-delay analysis
scripts/build_external_manifest.py RE2 train / RE1 validation / RE3 test manifest
scripts/train_temporal_graph.py  Candidate missingness-aware graph-temporal trainer
scripts/run_temporal_graph_grid.py Seed and ablation matrix runner
scripts/evaluate_temporal_graph.py  Root-head and selective-risk evaluation
src/memrca/data/           RCAEval loading, windowing, and trace-derived edges
src/memrca/data/temporal_graph.py  Train-normal scaling and aligned service graph windows
src/memrca/baselines/      Normal-only anomaly baseline
src/memrca/models/         LSTM, inference helper, and Phase 3 temporal graph candidate
scripts/train_temporal_graph.py  Candidate graph-temporal model trainer
scripts/evaluate_temporal_graph.py  Root-head versus forecast-score RCA evaluation
scripts/train_lstm_forecaster.py  Per-system/fold LSTM training and score export
scripts/train_lstm_grid.py   Sequential fold/system training runner
scripts/evaluate_lstm_scores.py  Validation-calibrated anomaly and RCA metrics
scripts/evaluate_lstm_grid.py  Fold/system evaluation and aggregate results
scripts/recalibrate_lstm_grid.py  Empirical-tail calibration ablation
scripts/select_lstm_variants.py  Validation-only model selection by objective
tests/                     Unit tests for windows, graph edges, and baseline scores
data/raw/                  Downloaded dataset; ignored by Git
experiments/               Generated figures and results; ignored by Git
docs/PHASE2_LSTM_START.md  Phase 2 training, handoff, and limitations
docs/PHASE2_RESULTS.md    Phase 2 results, selected variants, and research limits
docs/PHASE2_PROPAGATION.md  Alert-delay method, findings, and limitations
docs/PHASE3_RESEARCH_PROTOCOL.md  Candidate research question and evaluation protocol
docs/PHASE3_PILOT.md   One-system exploratory temporal-graph pilot results
```

## Phase 1 deliverables and current limits

Phase 1 covers the reproducible RE2 data and baseline work, not the final deep-learning system:

- Dataset download and audit tools check case coverage, telemetry files, label balance, and metadata/file mismatches.
- `build_splits.py` creates a case-level manifest for each fold, checks repetition assignments, keeps unseen faults out of train and validation, and reports per-split counts. Repetition 1/2/3 map to train/validation/test; cases are never split across partitions.
- `audit_graphs.py` reports trace coverage and exports service edges with case/system identifiers. Missing traces are represented by absent edge rows and graph coverage metadata; Sock Shop therefore falls back to metric-only ranking.
- `run_baseline.py` saves test-window anomaly scores and per-service rankings, plus validation rankings used by the graph ablation.
- `evaluate_graph_ranker.py` compares metric-only and graph-supported Hit@1, Hit@3, and MRR; it selects the graph weight on validation and evaluates on test.
- `compare_window_runs.py` summarizes the 10/30/60-second sensitivity runs.
- `tests/` covers the core window, graph, baseline, and graph-ranker behaviors.

See [the Phase 1 handoff](docs/PHASE1_HANDOFF.md) for deliverables, integration schema, reproducible commands, and the current validation results.

The dataset has only three repetitions per configuration, limiting estimates of generalization. Fault injection time is used to construct evaluation labels, and symptoms may begin later. Graph edges show observed calls, not causal propagation. Phase 2 now includes a normal-only LSTM, validation-only variant selection, recorded-incident replay, a local replay stream, and exploratory alert-delay analysis. The selected anomaly pipeline averages F1 0.924 / 0.633 / 0.742 for Online Boutique / Sock Shop / Train Ticket; test false-positive rates are 0.067 / 0.011 / 0.042. The separately selected RCA pipeline averages graph Hit@1 0.889 / 0.856 / 0.633 and MRR 0.931 / 0.906 / 0.733. Full metrics are in [the Phase 2 results report](docs/PHASE2_RESULTS.md) and [alert-delay report](docs/PHASE2_PROPAGATION.md). These test folds were inspected during development, so treat the values as exploratory and confirm on new seeds or an external dataset before making research claims. Unknown-fault rejection, uncertainty estimation, future impact-time prediction, production telemetry ingestion, and a learned graph component remain future work.

## Evaluation considerations

The configured folds are case-level: repetition 1 is train, repetition 2 is validation, and repetition 3 is test; held-out faults appear only in test. Overlapping windows from one incident never cross splits, but the same fault/service configuration recurs across repetitions. Report anomaly detection and root-cause ranking separately. The model selector uses validation known-fault F1 for anomaly detection and validation MRR for RCA, and writes separate selected score sets. The benchmark has only three repetitions per fault/service configuration, and the test folds have already been explored during development. Use new seeds or a separate benchmark for confirmatory research.
