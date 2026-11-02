"""Summarize validation-only calibration diagnostics across grid runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "temperature", "validation_abstention_threshold", "validation_risk_at_threshold",
    "validation_coverage_at_threshold", "validation_brier_score",
    "validation_expected_calibration_error", "validation_incidents",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/temporal_graph_re2_grid")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in root.glob("*/seed*/fold*/**/root_calibration.json"):
        try:
            ablation, seed, fold, system = path.relative_to(root).parts[:4]
            calibration = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
        rows.append({"ablation": ablation, "seed": seed, "fold": fold,
                     "system": system, **{metric: calibration.get(metric) for metric in METRICS}})
    if not rows:
        raise SystemExit(f"No root_calibration.json files found under {root}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "calibration_metrics_by_run.csv", index=False)
    summaries = frame.groupby(["ablation", "system"])[list(METRICS)].agg(["mean", "std"]).reset_index()
    summaries.columns = ["_".join(str(x) for x in col if x).rstrip("_") if isinstance(col, tuple) else col
                         for col in summaries.columns]
    summaries.to_csv(root / "calibration_metrics_summary.csv", index=False)
    print(frame.groupby(["ablation", "system"])[["validation_incidents", "validation_brier_score",
                                                   "validation_expected_calibration_error"]].agg(["count", "mean", "std"]).to_string())
    print(f"Wrote calibration reports under {root.resolve()}")


if __name__ == "__main__":
    main()
