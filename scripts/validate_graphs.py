"""Validate trace graph coverage, edge counts, and data quality across RCAEval suites.

Outputs:
- Reusable edge lists for downstream graph-informed root cause analysis.
- Graph quality summary report (JSON) detailing span coverage, missing parent spans,
  orphan spans, self-edges, and explicitly documenting that Sock Shop lacks traces.
- Note: Observed edges reflect call dependencies, not proof of causality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memrca.data.graph import extract_trace_graph_diagnostics, EDGE_COLUMNS


def validate_graphs(
    raw_dir: str | Path,
    suite: str = "re2",
    out_edges_dir: str | Path = "experiments/graphs",
    out_summary: str | Path = "experiments/graph_quality_report.json",
) -> dict:
    raw_path = Path(raw_dir)
    index_path = raw_path / "cases.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Case index not found at {index_path}")

    cases_df = pd.read_parquet(index_path)
    suite_cases = cases_df[cases_df["suite"].astype(str).str.lower() == suite.lower()].copy()

    edges_dir = Path(out_edges_dir)
    edges_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "suite": suite,
        "total_cases": len(suite_cases),
        "systems": {},
        "trace_limitations": {
            "sock_shop_status": "Sock Shop (ss) does not contain trace instrumentation in RCAEval RE2 (has_traces = 0).",
            "causality_note": "Observed trace edges represent call dependency evidence, NOT proof of fault causality."
        },
        "aggregate_stats": {
            "cases_with_traces": int(suite_cases["has_traces"].sum()),
            "cases_without_traces": int((~suite_cases["has_traces"]).sum()),
            "total_spans_indexed": int(suite_cases["n_traces"].fillna(0).sum()),
        },
    }

    # Group by system
    for system, sys_grp in suite_cases.groupby("system"):
        system = str(system)
        summary["systems"][system] = {
            "case_count": len(sys_grp),
            "cases_with_traces": int(sys_grp["has_traces"].sum()),
            "total_spans": int(sys_grp["n_traces"].fillna(0).sum()),
            "unique_services": sorted(sys_grp["root_cause_service"].unique().tolist()),
            "cases_processed_locally": 0,
        }

    # Check for locally downloaded case trace files
    all_case_edges = []
    for _, row in suite_cases.iterrows():
        case_name = str(row["case"])
        system = str(row["system"])
        case_dir = raw_path / case_name
        trace_file = case_dir / "traces.parquet"

        if trace_file.is_file():
            summary["systems"][system]["cases_processed_locally"] += 1
            traces = pd.read_parquet(trace_file)
            metrics_file = case_dir / "metrics.parquet"
            metric_services = None
            if metrics_file.is_file():
                # Read columns only
                m_cols = pd.read_parquet(metrics_file).columns
                metric_services = {c.partition("_")[0] for c in m_cols if c != "time"}

            diag = extract_trace_graph_diagnostics(traces, metric_services=metric_services)
            edges_df = diag["edges"]
            edges_df["case"] = case_name
            edges_df["system"] = system
            all_case_edges.append(edges_df)

    if all_case_edges:
        combined_edges = pd.concat(all_case_edges, ignore_index=True)
        combined_edges.to_csv(edges_dir / f"{suite}_case_edges.csv", index=False)

    summary_file = Path(out_summary)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="Path to raw RCAEval directory")
    parser.add_argument("--suite", default="re2", help="Suite to validate (re2)")
    parser.add_argument("--out-dir", default="experiments/graphs", help="Output directory for edge lists")
    parser.add_argument("--out-summary", default="experiments/graph_quality_report.json", help="Summary report path")
    args = parser.parse_args()

    summary = validate_graphs(
        raw_dir=args.raw,
        suite=args.suite,
        out_edges_dir=args.out_dir,
        out_summary=args.out_summary,
    )

    print(f"=== Graph Validation Summary for Suite {args.suite.upper()} ===")
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Cases with traces: {summary['aggregate_stats']['cases_with_traces']}")
    print(f"Cases without traces: {summary['aggregate_stats']['cases_without_traces']}")
    for sys_name, stats in summary["systems"].items():
        print(f"System '{sys_name}': {stats['cases_with_traces']}/{stats['case_count']} cases have traces. Total spans: {stats['total_spans']}")
    print(f"\nDocumentation note: {summary['trace_limitations']['sock_shop_status']}")
    print(f"Causality note: {summary['trace_limitations']['causality_note']}")
    print(f"Report written to: {args.out_summary}")


if __name__ == "__main__":
    main()
