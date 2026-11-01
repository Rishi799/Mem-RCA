"""Evaluate and summarize every trained Phase 2 LSTM fold/system run."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/lstm")
    parser.add_argument("--systems", nargs="+", choices=["ob", "ss", "tt"], default=["ob", "ss", "tt"])
    parser.add_argument("--folds", nargs="+", choices=["fold1", "fold2", "fold3"], default=["fold1", "fold2", "fold3"])
    parser.add_argument("--persistence-hits", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    evaluator = Path(__file__).with_name("evaluate_lstm_scores.py")
    metric_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    graph_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for fold in args.folds:
        for system in args.systems:
            run = root / fold / system
            scores = run / "service_window_scores.csv"
            if not scores.exists():
                missing.append(f"{fold}/{system}")
                continue
            out = run / ("evaluation" if args.persistence_hits == 1 else f"evaluation_persist{args.persistence_hits}")
            command = [sys.executable, str(evaluator), "--scores", str(scores), "--out", str(out),
                       "--persistence-hits", str(args.persistence_hits)]
            subprocess.run(command, cwd=project, check=True)
            metric_frames.append(pd.read_csv(out / "metrics.csv"))
            ranking_frames.append(pd.read_csv(out / "root_cause_rankings.csv"))
            graph_frames.append(pd.read_csv(out / "graph" / "graph_metrics.csv"))

    if missing:
        raise SystemExit("Missing trained score files: " + ", ".join(missing))
    summary_dir = root / ("summary" if args.persistence_hits == 1 else f"summary_persist{args.persistence_hits}")
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    graph = pd.concat(graph_frames, ignore_index=True)
    metrics.to_csv(summary_dir / "lstm_metrics_by_fold.csv", index=False)
    rankings.to_csv(summary_dir / "lstm_root_rankings.csv", index=False)
    graph.to_csv(summary_dir / "lstm_graph_metrics_by_fold.csv", index=False)
    numeric = [column for column in metrics.select_dtypes(include="number").columns
               if column not in {"fold", "test_windows", "positive_test_windows", "rca_cases"}]
    grouped = metrics.groupby(["system", "fault_status"], as_index=False)[numeric].mean()
    grouped.to_csv(summary_dir / "lstm_metrics_summary.csv", index=False)
    graph_numeric = [column for column in graph.select_dtypes(include="number").columns
                     if column not in {"fold", "cases"}]
    graph.groupby(["system", "fault_status"], as_index=False)[graph_numeric].mean().to_csv(
        summary_dir / "lstm_graph_metrics_summary.csv", index=False
    )
    print("\nLSTM anomaly metrics (mean across folds):")
    print(grouped.to_string(index=False))
    print("\nGraph RCA metrics (mean across folds):")
    print(graph.groupby(["system", "fault_status"], as_index=False)[graph_numeric].mean().to_string(index=False))
    print(f"\nWrote aggregate files under {summary_dir}")


if __name__ == "__main__":
    main()
