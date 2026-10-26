"""Audit RCAEval dataset coverage without loading large telemetry tables.

Run from the repository root:
    python scripts/audit_dataset.py --suite re2

The script compares cases.parquet metadata with downloaded case folders and files,
checks label/repetition balance, prints a summary, and writes JSON under experiments/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def summarize(values: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(values.min()),
        "median": float(values.median()),
        "max": int(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="RCAEval raw-data directory")
    parser.add_argument("--suite", default="re2", choices=["re1", "re2", "re3"])
    parser.add_argument(
        "--out",
        default=None,
        help="JSON report path (default: experiments/audit/<suite>_summary.json)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    index_path = raw_dir / "cases.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Case index not found: {index_path}. Download the suite first.")

    index = pd.read_parquet(index_path)
    suite = args.suite.upper()
    cases = index[index["suite"].astype(str).str.upper() == suite].copy()
    if cases.empty:
        raise ValueError(f"No {suite} entries found in {index_path}")

    file_counts: Counter[str] = Counter()
    missing: dict[str, list[str]] = {key: [] for key in ("folder", "metrics", "inject_time", "logs", "traces")}
    modality_mismatches: list[dict[str, object]] = []

    for row in cases.to_dict("records"):
        name = str(row["case"])
        folder = raw_dir / name
        if not folder.is_dir():
            missing["folder"].append(name)
            continue

        present = {
            "metrics": (folder / "metrics.parquet").is_file(),
            "inject_time": (folder / "inject_time.txt").is_file(),
            "logs": (folder / "logs.parquet").is_file(),
            "traces": (folder / "traces.parquet").is_file(),
        }
        for key, exists in present.items():
            if exists:
                file_counts[key] += 1
            else:
                missing[key].append(name)

        for modality in ("logs", "traces"):
            indexed = bool(row.get(f"has_{modality}", False))
            if indexed != present[modality]:
                modality_mismatches.append(
                    {"case": name, "modality": modality, "index_says_present": indexed,
                     "file_present": present[modality]}
                )

    group_sizes = cases.groupby(["system", "fault", "repetition"], dropna=False).size()
    incomplete_groups = [
        {"system": str(key[0]), "fault": str(key[1]), "repetition": int(key[2]), "cases": int(size)}
        for key, size in group_sizes.items()
        if size != 5
    ]

    report: dict[str, object] = {
        "suite": suite,
        "raw_dir": str(raw_dir),
        "index_rows": int(len(cases)),
        "systems": {str(k): int(v) for k, v in cases["system"].value_counts().sort_index().items()},
        "faults": {str(k): int(v) for k, v in cases["fault"].value_counts().sort_index().items()},
        "repetitions": {str(k): int(v) for k, v in cases["repetition"].value_counts().sort_index().items()},
        "cases_per_system_fault_repetition": {
            "group_count": int(len(group_sizes)),
            "size_counts": {str(k): int(v) for k, v in group_sizes.value_counts().sort_index().items()},
            "expected_cases_per_group": 5,
            "incomplete_groups": incomplete_groups,
        },
        "telemetry_metadata": {
            "files_found": dict(file_counts),
            "missing_files_or_folders": missing,
            "index_file_mismatches": modality_mismatches,
            "indexed_cases_with_logs": int(cases["has_logs"].fillna(False).sum()),
            "indexed_cases_with_traces": int(cases["has_traces"].fillna(False).sum()),
            "total_log_rows": int(pd.to_numeric(cases["n_logs"], errors="coerce").fillna(0).sum()),
            "total_trace_spans": int(pd.to_numeric(cases["n_traces"], errors="coerce").fillna(0).sum()),
        },
        "case_size_metadata": {
            "metric_columns": summarize(cases["n_metrics"]),
            "timesteps": summarize(cases["n_timesteps"]),
            "duration_minutes": {
                "min": float(cases["duration_minutes"].min()),
                "median": float(cases["duration_minutes"].median()),
                "max": float(cases["duration_minutes"].max()),
            },
        },
    }

    out_path = Path(args.out) if args.out else Path("experiments/audit") / f"{suite.lower()}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{suite}: {len(cases)} indexed cases")
    print("Cases by system:", report["systems"])
    print("Cases by fault:", report["faults"])
    print("Cases by repetition:", report["repetitions"])
    print("Downloaded telemetry files:", dict(file_counts))
    print("Indexed logs/traces:", report["telemetry_metadata"]["indexed_cases_with_logs"],
          "/", report["telemetry_metadata"]["indexed_cases_with_traces"])
    print("Total log rows / trace spans:", report["telemetry_metadata"]["total_log_rows"],
          "/", report["telemetry_metadata"]["total_trace_spans"])
    print("Case-size metadata:", report["case_size_metadata"])
    print("Incomplete label groups:", len(incomplete_groups))
    print("Index/file modality mismatches:", len(modality_mismatches))
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
