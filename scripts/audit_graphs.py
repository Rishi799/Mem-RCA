"""Report full trace coverage and audit representative service graphs.

By default, one median-size trace case per system is selected for edge-quality
checks; dataset-wide coverage comes from cases.parquet. Use --all to inspect all
trace files (the suite contains over 100 million spans).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from memrca.data.graph import service_edges_from_traces
from memrca.data.loader import load_case, metric_columns_by_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--suite", default="re2")
    parser.add_argument("--out", default="experiments/graph")
    parser.add_argument("--all", action="store_true", help="Audit every trace case")
    args = parser.parse_args()

    raw, suite = Path(args.raw), args.suite.lower()
    index = pd.read_parquet(raw / "cases.parquet")
    index = index[index["suite"].astype(str).str.lower().eq(suite)].copy()
    coverage = index.groupby("system").agg(
        indexed_cases=("case", "count"), cases_with_traces=("has_traces", "sum"),
        indexed_trace_spans=("n_traces", "sum"),
    ).reset_index()

    traced = index[index["has_traces"].fillna(False)].copy()
    if args.all:
        selected = traced
    else:
        # Select an actual median trace-count row per system.
        selected_rows = []
        for _, group in traced.groupby("system"):
            group = group.sort_values(["n_traces", "case"])
            selected_rows.append(group.iloc[len(group) // 2])
        selected = pd.DataFrame(selected_rows)

    edge_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    selected_rows = selected.to_dict("records")
    total = len(selected_rows)
    for number, row in enumerate(selected_rows, start=1):
        name = str(row["case"])
        if number == 1 or number % 10 == 0 or number == total:
            print(f"Auditing trace case {number}/{total}: {name}", flush=True)
        trace_path = raw / name / "traces.parquet"
        traces = pd.read_parquet(trace_path, columns=["traceID", "spanID", "parentSpanID", "serviceName"])
        metric_services = set(metric_columns_by_service(load_case(raw, name).metrics.columns))
        spans = traces.dropna(subset=["traceID", "spanID", "serviceName"])[
            ["traceID", "spanID", "parentSpanID", "serviceName"]
        ].copy()
        spans["traceID"] = spans["traceID"].astype(str)
        spans["spanID"] = spans["spanID"].astype(str)
        spans["parentSpanID"] = spans["parentSpanID"].fillna("").astype(str)
        spans["serviceName"] = spans["serviceName"].astype(str)
        duplicate_rows = int(spans.duplicated(["traceID", "spanID"], keep=False).sum())
        conflicts_service = int((spans.groupby(["traceID", "spanID"])["serviceName"].nunique() > 1).sum())
        conflicts_parent = int((spans.groupby(["traceID", "spanID"])["parentSpanID"].nunique() > 1).sum())
        canonical = spans.drop_duplicates(["traceID", "spanID"], keep="first")
        parent_refs = canonical[canonical["parentSpanID"].ne("")]
        parents = canonical[["traceID", "spanID", "serviceName"]].rename(
            columns={"spanID": "parentSpanID", "serviceName": "parent_service"}
        )
        matched = parent_refs.merge(parents, on=["traceID", "parentSpanID"], how="inner", validate="many_to_one")
        edges = service_edges_from_traces(traces, metric_services)
        if not edges.empty:
            edges.insert(0, "case", name)
            edges.insert(1, "system", str(row["system"]))
            edge_frames.append(edges)
        diagnostics.append({
            "case": name, "system": row["system"], "indexed_spans": int(row["n_traces"]),
            "loaded_span_rows": int(len(spans)), "duplicate_span_rows": duplicate_rows,
            "conflicting_service_span_ids": conflicts_service,
            "conflicting_parent_span_ids": conflicts_parent,
            "canonical_spans": int(len(canonical)), "parent_references": int(len(parent_refs)),
            "matched_parent_references": int(len(matched)),
            "orphan_parent_references": int(len(parent_refs) - len(matched)),
            "inter_service_edges": int(len(edges)),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "all" if args.all else "sample"
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(
        columns=["case", "system", "source", "target", "span_count"]
    )
    diag = pd.DataFrame(diagnostics)
    coverage.to_csv(out / f"{suite}_graph_coverage.csv", index=False)
    diag.to_csv(out / f"{suite}_graph_{mode}_diagnostics.csv", index=False)
    edges.to_csv(out / f"{suite}_graph_{mode}_edges.csv", index=False)
    if not diag.empty:
        graph_stats = diag.groupby("system").agg(
            audited_trace_cases=("case", "count"), loaded_span_rows=("loaded_span_rows", "sum"),
            duplicate_span_rows=("duplicate_span_rows", "sum"),
            conflicting_service_span_ids=("conflicting_service_span_ids", "sum"),
            conflicting_parent_span_ids=("conflicting_parent_span_ids", "sum"),
            parent_references=("parent_references", "sum"),
            orphan_parent_references=("orphan_parent_references", "sum"),
            min_inter_service_edges=("inter_service_edges", "min"),
            median_inter_service_edges=("inter_service_edges", "median"),
            max_inter_service_edges=("inter_service_edges", "max"),
        ).reset_index()
        summary = coverage.merge(graph_stats, on="system", how="left")
        summary.to_csv(out / f"{suite}_graph_{mode}_summary.csv", index=False)
    print("Dataset-wide coverage:")
    print(coverage.to_string(index=False))
    print(f"\nGraph quality ({mode}):")
    print(diag.to_string(index=False))
    print(f"Wrote coverage plus {mode} diagnostics, edges, and summary under {out}")


if __name__ == "__main__":
    main()
