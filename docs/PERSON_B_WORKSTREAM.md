# Person B Workstream: Graph Evidence, Root-Cause Evaluation & Data Integrity

## Overview
This document summarizes the completion of **Person B's Phase 1 deliverables** for Mem-RCA (AI253IA - Artificial Neural Networks & Deep Learning PBL).

Person B delivers the graph evidence pipeline, split integrity audit, and root-cause evaluation baseline that plugs directly into Person A's anomaly detector outputs.

---

## 1. Task 1: Experimental Split & Label Audit
- **Script**: [`scripts/verify_split.py`](../scripts/verify_split.py)
- **Manifest**: `experiments/split_manifest.csv`
- **Audit Report**: `experiments/split_audit_report.json`

### Key Findings & Verification:
- **Repetitions strictly partitioned**:
  - Repetition 1 $\rightarrow$ `train` (60 cases per fold)
  - Repetition 2 $\rightarrow$ `validation` (60 cases per fold)
  - Repetition 3 $\rightarrow$ `test` known faults (60 cases per fold)
- **Open-world Unseen Faults**:
  - Unseen fault types (`disk, socket` in Fold 1; `cpu, loss` in Fold 2; `mem, delay` in Fold 3) appear **only** in the `test` split (90 cases per fold across repetitions 1, 2, and 3).
  - Total test cases per fold: **150** ($60\text{ known} + 90\text{ unseen}$).
- **No data leakage detected**: No unseen fault cases or windows cross into training or validation splits.
- **Three-Repetition Design Note**: RCAEval RE2 provides 3 repetitions per fault/service configuration ($3 \times 15 = 45$ cases per fault type across 3 systems). Because of this limited repetition count, repetitions are kept strictly partitioned to avoid optimistic leakage between overlapping incident time windows.

---

## 2. Task 2: Service Graph Extraction & Diagnostics
- **Module**: [`src/memrca/data/graph.py`](../src/memrca/data/graph.py)
- **Validation Script**: [`scripts/validate_graphs.py`](../scripts/validate_graphs.py)
- **Quality Report**: `experiments/graph_quality_report.json`

### Improvements Made:
- **Parent-Span Matching**: Validates parent-child spans matching strictly within each `traceID`.
- **Missing Parents & Root Spans**: Accurately classifies root spans (`parentSpanID` empty or 0) and flags orphan spans whose parent span ID is missing.
- **Service Name Normalization**: Maps trace service identifiers (e.g., `frontendservice`) to metric telemetry identifiers (`frontend`).
- **Duplicate & Self-Edge Aggregation**: Aggregates duplicate trace call occurrences and isolates internal same-service calls.
- **Missing Trace Instrumentation**:
  - Explicitly records and handles that **Sock Shop (`ss`) has 0 trace spans** in RE2.
  - Documents that **trace edges reflect invocation dependency evidence, not proof of causality**.

---

## 3. Task 3: Root-Cause Ranking & Evaluation
- **Module**: [`src/memrca/evaluation/ranker.py`](../src/memrca/evaluation/ranker.py)
- **Execution Script**: [`scripts/evaluate_root_cause.py`](../scripts/evaluate_root_cause.py)
- **Metrics Computed**: **Hit@1**, **Hit@3**, and **Mean Reciprocal Rank (MRR)**.

### Score-Only vs Graph-Informed Ablation:
- Benchmarked baseline pure anomaly score ranking vs. graph-informed propagation ranking ($\alpha=0.3$).
- Gracefully handles missing trace graphs (such as in Sock Shop) without crashing by falling back to pure score ranking.
- Normalization parameters are fit strictly on training splits.

---

## 4. Handoff to Person A: Anomaly Score Schema

To plug Person A's anomaly detector into the Person B root-cause evaluation pipeline, save detector predictions as a CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| `case` | string | Case identifier (e.g. `re2ob_checkoutservice_cpu_1`) |
| `fold` | string | Fold identifier (`fold1`, `fold2`, or `fold3`) |
| `split` | string | `train`, `validation`, or `test` |
| `window_start` | integer / float | Start timestamp of the detection window |
| `service` | string | Microservice name |
| `anomaly_score` | float | Anomaly score produced by detector |

### Commands to Run the Pipeline:
```bash
# 1. Verify split and generate manifest
python scripts/verify_split.py

# 2. Validate trace graphs and diagnostics
python scripts/validate_graphs.py

# 3. Evaluate root-cause ranking (using Person A's score CSV)
python scripts/evaluate_root_cause.py --scores path/to/person_a_scores.csv --edges experiments/graphs/re2_case_edges.csv
```
