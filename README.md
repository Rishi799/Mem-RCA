# Mem-RCA

**Open-world root cause analysis for microservices via propagation-aware incident memory**

**Course:** AI253IA — Artificial Neural Networks & Deep Learning (PBL), RV College of Engineering<br>
**Team:** Priyanshi Gupta (1RV24CI088) · Rishi Agarwal (1RV24CI099) · Sambhav Badaya (1RV24CI106)

Root-cause analysis (RCA) models are usually trained and tested on the same fixed set of failure types. Real systems keep producing failures nobody has labelled yet. Mem-RCA handles this open-world setting:

1. A deep localiser (per-service LSTM encoders + directed graph attention) ranks the services most likely to be the root cause.
2. The incident is summarised by a **root-anchored fingerprint**: the root service's own metric shifts, plus the symptoms it caused elsewhere.
3. An **incident memory** compares the fingerprint with stored incidents of every known failure type. If nothing is familiar enough (below a threshold τ set on validation data), the incident is flagged **NOVEL**.
4. NOVEL incidents are grouped. An engineer names a group, and it is **enrolled as a new failure type without retraining**.
5. **PropagationNet**, a graph neural network with a hazard head, forecasts **which other services the fault will reach and when**. It updates the forecast as new telemetry arrives.
6. Evaluation is **leakage-aware**: fault types are held out, there are no-telemetry and candidate-only baselines, roots are held out, and two independent suites (RE1, RE3) were checked under a protocol fixed in advance.

## Results at a glance

Novelty AUROC: how well unseen-type incidents are separated from known ones (0.5 = chance).

| Experiment | Online Boutique | Sock Shop | Train Ticket | Status |
|---|---:|---:|---:|---|
| RE2, 2 fault types unseen (development) — Mem-RCA | 0.78 | 0.60 | 0.79 | optimistic: used to design the method |
| RE2 — average over services (global mean) | 0.66 | 0.56 | 0.55 | baseline |
| RE1, 1 fault type unseen (independent suite) — Mem-RCA | 0.82 | 0.76 | 0.65 | confirmatory |
| RE1 — best fixed fingerprint / global mean | 0.84 / 0.82 | 0.71 / 0.54 | 0.62 / 0.60 | baselines |
| RE3 code-level faults vs RE2 memory — Mem-RCA | 0.84 | 0.83 | 0.96 | confirmatory, p < 0.001 each |
| RE3 — average over services (global mean) | 0.50 | 0.59 | 0.35 | baseline |

Pre-registered hypotheses ([protocol](docs/EXTERNAL_VALIDATION_PROTOCOL.md)):

- **H1 is not supported.** On RE1 Train Ticket, the root-anchored fingerprint did not beat global pooling (0.59 vs 0.60 / 0.62).
- **H2 is supported.** Averaged over the RE1 systems, Mem-RCA scores 0.74, above every fixed fingerprint (0.72 / 0.67 / 0.66).
- **H3 is supported.** RE3 code faults look unfamiliar in every system.

Findings:

- **No single fingerprint is safe everywhere; choosing it per system is.**
  - Averaging over services (the usual summary) is near or below chance on Train Ticket in RE2 and RE3, and weak on Sock Shop.
  - Root anchoring helped on RE2 and RE3 Train Ticket but did not replicate on RE1.
  - The per-system choice made on known incidents only (Mem-RCA) is the most reliable option across suites.
- **Discovery and enrolment.** On RE2 Train Ticket, 47 % of unseen fault types are discovered from the incident stream, with clusters 98 % pure. After enrolment, 90 % of later incidents of the new type are named correctly. With 3 examples, a new type is recognised 87 % of the time. On RE1, k-shot enrolment beats the averaged fingerprint in every system.
- **The root-cause benchmark has a shortcut.** RE2 has only five root services per system. A rule with no learning that only looks at those five nearly matches the learned localiser (Train Ticket Hit@1 0.95 vs 0.97). When a root service is held out of training, the learned localiser drops to Hit@1 0.20 / 0.13 / 0.09 (OB / SS / TT), while the learning-free largest-shift rule keeps 0.93 / 0.84 / 0.79. The localiser has largely learned *which services are usually roots*.

- **Time-to-impact forecasting (C5).** Faults travel mainly up the call chain: on Online Boutique, 67 % of the root's callers are hit, with a median of 11 s, against 36 % of its callees at 70 s.
  - **RE2:** PropagationNet beats simple forecasts on unseen root services (AUROC 0.70 vs 0.66 / 0.63).
  - **RE3, frozen protocol:**
    - H5 supported: it orders who is hit first better, C-index 0.64 vs 0.55.
    - H6 supported: its forecast improves as evidence arrives (0.68 → 0.73).
    - H4 not supported: a simple per-hop table had the higher pooled AUROC, 0.74 vs 0.70.

Full results, protocols and caveats:
- [docs/OPEN_WORLD_RESULTS.md](docs/OPEN_WORLD_RESULTS.md)
- [docs/EXTERNAL_VALIDATION_PROTOCOL.md](docs/EXTERNAL_VALIDATION_PROTOCOL.md)
- [docs/PROPAGATION_PROTOCOL.md](docs/PROPAGATION_PROTOCOL.md)

## Quick start: the Mem-RCA console

```powershell
python scripts/serve_console.py
```

Open <http://127.0.0.1:8765>, pick an environment and press **Start monitoring**. The console has two modes:

- **Live monitoring** (default) streams each recorded incident's telemetry at 5–60× speed, like a monitoring dashboard. A **Service health** card shows every service as a tile, the anomaly level of the most anomalous service against the alert threshold, and an activity feed. The detector uses the normal period only, so nothing alerts before the fault; the incident opens by itself when the first service's anomaly is confirmed (onset + 5 s). Root cause and pattern appear once the 2-minute evidence window the models use has passed, and the blast radius then runs on the replay clock. After the 5-minute forecast window the incident closes and the next recording starts. **Skip ahead** jumps to the alert, the analysis or the end of the incident. Nothing on screen uses data after the replay clock.
- **Step-through** opens one recorded incident at a time with **Next incident** (or **▶ Auto**).

Every incident gets an ID (`INC-0001`, …) and a header with the three answers an on-call engineer needs: root cause, diagnosis (known pattern or new pattern) and blast radius. Below it:

- **Blast radius forecast**: a dependency graph of every service, with the faulty service in the centre and one ring per call hop. Solid links are calls seen in the traces. Dashed links are documented dependencies that the traces miss ([architecture.py](src/memrca/propagation/architecture.py)): the Online Boutique and Sock Shop architectures (RCAEval has no Sock Shop traces) and each Train Ticket database's owner service. These links are for drawing only; PropagationNet uses the traced graph. Services with no link sit in a band below the graph; if the origin itself has no link, the graph is centred on the best-connected service. Each service is coloured by its state (affected, expected with time and chance, predicted but not hit, possible, unlikely). The spread path is drawn along the links, labelled with the delay after the previous service. **Play** replays the five minutes after the fault (one 10-s step per 0.9 s) and pauses the live incident stream. By default the forecast stays fixed from the moment it was made, so the only changes on screen are services that really get affected; **Update forecast every 10 s** switches to the live forecast. A grouped list, a table of every service and a forecast check are below it;
- **Root cause analysis**: the deep localiser's ranking of likely origins, next to the largest-metric-deviation ranking, and the origin's telemetry;
- **Failure pattern match**: the incident signature, its similarity to each known pattern and the match threshold τ; an incident below τ is held as a new pattern instead of being forced into a known one;
- **Pattern library**: a map of past incidents, the known patterns, and groups of unclassified incidents that can be named and turned into a new pattern with one click (no retraining);
- **Incident history** for the replay.

**Evaluation mode** reveals the recorded ground truth and scores every decision (origin, pattern, forecast). The **Model insights** tab shows the benchmark results and the architecture.

The console reads the outputs of the experiment scripts below, so run those first. Everything is recorded-data replay, not live telemetry.

## Reproducing the open-world experiments

After installing (see [Setup](#setup)) and downloading the data (`python scripts/download.py --suite all`):

```powershell
# development set: RE2, using the Phase 3 temporal-graph checkpoints
python scripts/extract_incident_features.py        # localiser scores + embeddings per incident (GPU, ~12 min)
python scripts/extract_shift_profiles.py           # learning-free metric-shift profiles for RE1/RE2/RE3 (~5 min)
python scripts/evaluate_open_world.py              # E2: root-anchored vs global fingerprints; C4 shortcut check
python scripts/select_fingerprint.py               # validation-only selection of view and memory (Mem-RCA)
python scripts/simulate_discovery.py --variants shift:global_mean+prototype shift:global_max+split1 shift:global_max+split3 shift:anchor_pred+split1
python scripts/run_service_holdout.py              # C4: retrain without each root service (GPU)
# confirmatory set: RE1 and RE3, protocol in docs/EXTERNAL_VALIDATION_PROTOCOL.md
python scripts/train_external_localisers.py        # localiser per RE1 fold (GPU)
python scripts/evaluate_external.py                # H1-H3 checks, discovery and few-shot on RE1
# time-to-impact forecasting (C5), protocol in docs/PROPAGATION_PROTOCOL.md
python scripts/propagation_feasibility.py          # 1-second onset labels and propagation structure on RE2
python scripts/train_propagation.py                # PropagationNet vs baselines on RE2 (GPU, ~35 min)
python scripts/train_propagation.py --confirm      # H4-H6 on RE3, run once
python scripts/make_figures.py                     # figures in docs/figures/
python -m unittest discover -s tests
```

Open-world code lives in `src/memrca/openworld/`:

| Module | Contents |
|---|---|
| [fingerprint.py](src/memrca/openworld/fingerprint.py) | metric-shift profiles, pooling (global mean / global max / root-anchored / hybrid) |
| [memory.py](src/memrca/openworld/memory.py) | the incident memory: prototypes or k-NN, NOVEL threshold, enrolment |
| [metrics.py](src/memrca/openworld/metrics.py) | novelty AUROC and OSCR |
| [incidents.py](src/memrca/openworld/incidents.py) | incident sets for RE1 folds and the RE2 → RE3 cross-suite test |
| [console.py](src/memrca/openworld/console.py) | the interactive session behind the console |
| [propagation/](src/memrca/propagation/) | 1-second onset detection, time-to-impact samples, PropagationNet, forecasts and live-replay telemetry for the console |

## Development history (Phases 1–3)

The sections below describe the earlier data pipeline, LSTM and temporal-graph work, which the open-world study builds on.

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

**Phase 3 status:** the LSTM plus directed graph-attention candidate, validation calibration, inference API, and recorded replay integration are implemented. The full model completed 27 RE2 runs (three folds, three systems, three seeds); graph, observation-mask, and consistency-loss ablations completed 27 more runs (three folds, three systems, one seed per ablation). RE2 results remain exploratory because test cases were viewed during development and folds reuse incident configurations. RE1 and RE3 were later downloaded and evaluated under a frozen protocol for the open-world study; see [EXTERNAL_VALIDATION_PROTOCOL.md](docs/EXTERNAL_VALIDATION_PROTOCOL.md). See the [research protocol](docs/PHASE3_RESEARCH_PROTOCOL.md), [pilot report](docs/PHASE3_PILOT.md), [RE2 grid results](docs/PHASE3_RE2_GRID_RESULTS.md), and [calibration status](docs/PHASE3_CALIBRATION_STATUS.md). Future-impact-time prediction and a production telemetry adapter remain separate work.

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
app/                       Mem-RCA console front end (HTML/CSS/JS, no external libraries)
scripts/serve_console.py   Local server for the console (python scripts/serve_console.py)
src/memrca/openworld/      Fingerprints, incident memory, metrics, incident sets, console session
scripts/extract_incident_features.py  RE2 localiser scores and embeddings per incident
scripts/extract_shift_profiles.py     Learning-free metric-shift profiles for RE1/RE2/RE3
scripts/evaluate_open_world.py        E2 fingerprint comparison and C4 shortcut check
scripts/select_fingerprint.py         Validation-only choice of fingerprint view and memory
scripts/simulate_discovery.py         Discover-and-enrol stream (C3) and k-shot enrolment (E4)
scripts/run_service_holdout.py        Root-service hold-out retraining (C4)
scripts/train_external_localisers.py  Localiser per RE1 open-world fold
scripts/evaluate_external.py          Confirmatory RE1/RE3 evaluation (H1-H3)
scripts/propagation_feasibility.py    Onset labels and propagation structure (C5 feasibility)
scripts/train_propagation.py          PropagationNet training, baselines, RE2 dev and RE3 confirmatory tests
src/memrca/propagation/               Onset detection, time-to-impact dataset, PropagationNet, console forecasts
docs/PROPAGATION_PROTOCOL.md          C5 protocol frozen before the RE3 run, with its outcome
scripts/make_figures.py               Result figures in docs/figures/
docs/OPEN_WORLD_RESULTS.md            Open-world results, lessons and caveats
docs/EXTERNAL_VALIDATION_PROTOCOL.md  Protocol frozen before the RE1/RE3 evaluation
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
