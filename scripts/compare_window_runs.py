"""Compare held-out baseline results from multiple window-size runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", default=[
            "10s=experiments/sensitivity/window_10s",
            "30s=experiments/baseline",
            "60s=experiments/sensitivity/window_60s",
        ], help="Named run folders as LABEL=PATH",
    )
    parser.add_argument("--out", default="experiments/sensitivity/window_comparison.csv")
    args = parser.parse_args()
    pieces = []
    columns = ["roc_auc", "average_precision", "f1", "false_positive_rate",
               "incident_detection_rate", "mean_detection_delay_seconds",
               "hit_at_1", "hit_at_3", "mean_reciprocal_rank"]
    for spec in args.runs:
        label, _, folder = spec.partition("=")
        if not folder:
            raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
        path = Path(folder) / "metrics.csv"
        metrics = pd.read_csv(path)
        metrics = metrics[metrics["fault_status"].eq("all")]
        grouped = metrics.groupby("system")[columns].mean(numeric_only=True).reset_index()
        grouped.insert(0, "window_run", label)
        pieces.append(grouped)
    comparison = pd.concat(pieces, ignore_index=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
