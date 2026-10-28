# Mem-RCA

**Uncertainty-aware root-cause diagnosis for microservices**

**Course:** AI253IA — Artificial Neural Networks & Deep Learning (PBL), RV College of Engineering<br>
**Team:** Priyanshi Gupta (1RV24CI088) · Rishi Agarwal (1RV24CI099) · Sambhav Badaya (1RV24CI106)

Mem-RCA is a deep learning project for detecting abnormal microservice behavior and identifying the service most likely to have initiated an incident. It uses service dependencies as context for diagnosis and is designed to report when a fault is unfamiliar or the available evidence is insufficient, rather than always forcing a known-fault prediction.

## Research direction

The main research question is whether graph-informed diagnosis can remain reliable when fault types are held out from training and telemetry is incomplete. The intended system will:

- detect abnormal behavior from service telemetry;
- rank likely root-cause services using metric evidence and service dependencies;
- distinguish known fault classes from unfamiliar or uncertain incidents;
- evaluate how diagnosis changes when telemetry sources are missing.

The demonstration will replay recorded RCAEval incidents and visualize their telemetry and service relationships. This is a repeatable replay, not a live production monitoring system. Predicting downstream impact times is a possible extension if reliable onset labels are available.

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
- `scripts/explore_case.py` summarizes one case, ranks services by a simple metric-shift score, and saves plots.
- `configs/data.yaml` records the RE2 systems, fault classes, known/unseen folds, and initial 30-second windows scored every 10 seconds.

**Phase 1 status:** the data pipeline, split validation, anomaly baseline, trace graph audit, and graph-supported root-cause evaluation are implemented. The Isolation Forest and graph reranker are transparent baselines; they are not the proposed graph-temporal deep model. Unknown-fault rejection, calibrated uncertainty, downstream impact-time prediction, and the replay interface remain future work.

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

PyTorch is not currently required by the implemented baseline tools. Install the appropriate PyTorch build when the graph-temporal deep model is implemented.

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
python -m unittest discover -s tests -v
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
src/memrca/data/           RCAEval loading, windowing, and trace-derived edges
src/memrca/baselines/      Normal-only anomaly baseline
tests/                     Unit tests for windows, graph edges, and baseline scores
data/raw/                  Downloaded dataset; ignored by Git
experiments/               Generated figures and results; ignored by Git
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

The dataset has only three repetitions per configuration, limiting estimates of generalization. Fault injection time is used to construct evaluation labels, and symptoms may begin later. The graph method is a reranking baseline over observed call structure; it does not establish causal propagation. Phase 2 should implement and evaluate the graph-temporal deep model, unknown-fault handling, uncertainty, telemetry ablations, and a replay demonstration.

## Evaluation considerations

The configured folds are case-level: repetition 1 is train, repetition 2 is validation, and repetition 3 is test; held-out faults appear only in test. Overlapping windows from one incident never cross splits, but the same fault/service configuration recurs across repetitions. Report anomaly detection and root-cause ranking separately. Unknown-fault rejection, uncertainty estimation, and the graph-temporal deep model are Phase 2 work. RE2 has only three repetitions per fault/service configuration, so results need confirmation with additional seeds, stronger baselines, and further experiments before making research claims.
