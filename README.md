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

The repository currently contains the project scaffold and data exploration tools:

- `src/memrca/data/loader.py` reads the RCAEval case index, metrics, logs, traces, and fault injection time.
- `src/memrca/data/graph.py` extracts observed parent-service to child-service edges from trace span IDs.
- `scripts/download.py` downloads dataset suites from Hugging Face.
- `scripts/audit_dataset.py` checks downloaded-file coverage and label/repetition balance from the case index.
- `scripts/explore_case.py` summarizes one case, ranks services by a simple metric-shift score, and saves plots.
- `configs/data.yaml` records the RE2 systems, fault classes, and provisional known/unseen folds.

Model training, graph-based diagnosis, unknown-fault handling, and the replay interface have not yet been implemented. The current metric-shift ranking in the explorer is a data-inspection aid, not a trained model or a project result.

## Dataset

The project uses [RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval), a public benchmark containing 735 cases from Online Boutique (`ob`), Sock Shop (`ss`), and Train Ticket (`tt`). The case index includes metadata for all suites; download RE2 for the initial multi-source experiments.

| Suite | Cases | Telemetry | Intended use |
|---|---:|---|---|
| RE1 | 375 | Metrics | Optional metric-only comparison |
| RE2 | 270 | Metrics, logs, traces | Main initial dataset |
| RE3 | 90 | Metrics, logs, traces | Optional code-level fault evaluation |

An RE2 case folder is named like `re2ob_checkoutservice_cpu_1` and contains `metrics.parquet`, `logs.parquet`, `traces.parquet`, and `inject_time.txt`. The timestamp separates the recorded pre-injection and post-injection periods. Some cases or suites may not include every telemetry file.

The full RE2 download is approximately 3 GB. Raw data and generated experiment outputs are excluded from Git; download the dataset separately in each working copy.

The downloaded RE2 index reports logs for 269 of 270 cases and traces for 180 of 270 cases: Sock Shop has no traces, and one Train Ticket case has no logs. The index lists about 43.6 million log rows and 101.8 million trace spans, so process these modalities per case or aggregate them before model training. Metric files have 930–1,441 one-second samples and 69–376 metric columns per case. The trace-derived graph represents observed call structure; an edge alone is not evidence that one service caused another to fail.

## Setup

Windows with Python 3.11 or later:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is not currently required by the implemented tools. Install the appropriate PyTorch build separately when model training is added.

## Download and explore RE2

From the repository root:

```powershell
python scripts/download.py --suite re2
python scripts/audit_dataset.py --suite re2
python scripts/explore_case.py
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

## Project layout

```text
configs/data.yaml          Dataset settings and provisional open-world folds
scripts/download.py        RCAEval download command
scripts/audit_dataset.py   Dataset coverage and split-balance report
scripts/explore_case.py    Case summary and exploratory figures
src/memrca/data/           RCAEval loading and trace-derived service edges
data/raw/                  Downloaded dataset; ignored by Git
experiments/               Generated figures and results; ignored by Git
```

## Evaluation considerations

The configured folds are starting points and must be checked before model evaluation. Keep incidents, rather than overlapping windows from one incident, separated across splits. Tune any unknown-fault threshold using validation data only. Report anomaly detection, root-cause ranking, unknown-fault detection, and robustness to missing telemetry separately. RE2 has three repetitions per fault/service configuration, so results should be interpreted with that limited repetition count in mind.
