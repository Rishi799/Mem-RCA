"""Run reproducible temporal-graph seeds and ablations over folds/systems."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import json

import pandas as pd


ABLATIONS = ("full", "no_graph", "no_mask", "no_consistency")


def aggregate_mean_std(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def run_evaluator(command: list[str], project: Path) -> None:
    """Retry transient Windows subprocess crashes while evaluating large CSVs."""
    for attempt in range(3):
        try:
            subprocess.run(command, cwd=project, check=True, stdout=subprocess.DEVNULL)
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            print(f"Evaluator process failed; retrying ({attempt + 2}/3)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", choices=["fold1", "fold2", "fold3"],
                        default=["fold1", "fold2", "fold3"])
    parser.add_argument("--systems", nargs="+", choices=["ob", "ss", "tt"],
                        default=["ob", "ss", "tt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=["full"])
    parser.add_argument("--out", default="experiments/temporal_graph_grid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    trainer = Path(__file__).with_name("train_temporal_graph.py")
    anomaly_eval = Path(__file__).with_name("evaluate_lstm_scores.py")
    root_eval = Path(__file__).with_name("evaluate_temporal_graph.py")
    output_root = Path(args.out)
    grid_config = {
        "folds": args.folds, "systems": args.systems, "seeds": args.seeds,
        "ablations": args.ablations, "device": args.device, "epochs": args.epochs,
        "batch_size": args.batch_size, "manifest": args.manifest, "edges": args.edges,
        "raw": args.raw,
    }
    config_path = output_root / "grid_config.json"
    if config_path.exists() and not args.force:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != grid_config:
            raise ValueError(f"Grid configuration differs from {config_path}; use --force for a clean rerun or a new --out directory")
    output_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(grid_config, indent=2), encoding="utf-8")
    anomaly_rows: list[pd.DataFrame] = []
    root_rows: list[pd.DataFrame] = []
    selective_rows: list[dict] = []

    for ablation in args.ablations:
        for seed in args.seeds:
            run_root = output_root / ablation / f"seed{seed}"
            for fold in args.folds:
                for system in args.systems:
                    run = run_root / fold / system
                    score_path = run / "service_window_scores.csv"
                    command = [
                        sys.executable, str(trainer), "--raw", args.raw,
                        "--manifest", args.manifest, "--edges", args.edges,
                        "--fold", fold, "--system", system, "--device", args.device,
                        "--seed", str(seed), "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size), "--out", str(run_root),
                    ]
                    if args.force:
                        command.append("--force")
                    if ablation == "no_graph":
                        command.append("--disable-graph")
                    elif ablation == "no_mask":
                        command.append("--disable-mask-input")
                    elif ablation == "no_consistency":
                        command.extend(["--consistency-weight", "0"])
                    print(f"RUN {ablation} seed={seed} {fold}/{system}", flush=True)
                    run.mkdir(parents=True, exist_ok=True)
                    anomaly_out = run / "anomaly_evaluation"
                    root_out = run / "root_head_evaluation"
                    complete = (score_path.exists()
                                and (anomaly_out / "metrics.csv").exists()
                                and (root_out / "root_head_metrics_summary.csv").exists()
                                and not args.force)
                    if complete:
                        print(f"REUSE completed outputs for {ablation} seed={seed} {fold}/{system}", flush=True)
                    else:
                        with (run / "training.log").open("w", encoding="utf-8") as log:
                            try:
                                subprocess.run(command, cwd=project, check=True, stdout=log, stderr=subprocess.STDOUT)
                            except subprocess.CalledProcessError:
                                tail = (run / "training.log").read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                                raise RuntimeError("Training failed; final log lines:\n" + "\n".join(tail))
                        run_evaluator([
                            sys.executable, str(anomaly_eval), "--scores", str(score_path),
                            "--persistence-hits", "2", "--out", str(anomaly_out),
                        ], project)
                        run_evaluator([
                            sys.executable, str(root_eval), "--scores", str(score_path), "--out", str(root_out),
                        ], project)
                    metrics = pd.read_csv(anomaly_out / "metrics.csv")
                    metrics = metrics[metrics["fault_status"].eq("all")].copy()
                    metrics.insert(0, "ablation", ablation)
                    metrics.insert(1, "seed", seed)
                    anomaly_rows.append(metrics)
                    root_metrics = pd.read_csv(root_out / "root_head_metrics_summary.csv")
                    root_metrics.insert(0, "ablation", ablation)
                    root_metrics.insert(1, "seed", seed)
                    root_metrics.insert(2, "fold", fold)
                    root_rows.append(root_metrics)
                    selective_path = root_out / "selective_evaluation.json"
                    if selective_path.exists():
                        record = json.loads(selective_path.read_text(encoding="utf-8"))
                        calibration_path = run / "root_calibration.json"
                        if calibration_path.exists():
                            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                            record.update({f"calibration_{key.removeprefix('validation_')}": calibration.get(key) for key in (
                                "temperature", "validation_abstention_threshold", "validation_risk_at_threshold",
                                "validation_coverage_at_threshold", "validation_brier_score",
                                "validation_expected_calibration_error", "validation_incidents",
                            )})
                        selective_rows.append({"ablation": ablation, "seed": seed,
                                               "fold": fold, "system": system, **record})

    output_root.mkdir(parents=True, exist_ok=True)
    if anomaly_rows:
        anomaly = pd.concat(anomaly_rows, ignore_index=True)
        anomaly.to_csv(output_root / "anomaly_metrics_by_run.csv", index=False)
        numeric = [column for column in anomaly.select_dtypes(include="number").columns
                   if column not in {"seed", "fold", "test_windows", "positive_test_windows", "rca_cases"}]
        aggregate_mean_std(anomaly, ["ablation", "system", "fault_status"], numeric).to_csv(
            output_root / "anomaly_metrics_summary.csv", index=False)
    if root_rows:
        root = pd.concat(root_rows, ignore_index=True)
        root.to_csv(output_root / "root_metrics_by_run.csv", index=False)
        aggregate_mean_std(root, ["ablation", "system", "fault_status", "score_source"],
                            ["hit_at_1", "hit_at_3", "mrr"]).to_csv(
            output_root / "root_metrics_summary.csv", index=False)
    if selective_rows:
        selective = pd.DataFrame(selective_rows)
        selective.to_csv(output_root / "selective_metrics_by_run.csv", index=False)
        calibration_metrics = [column for column in selective.select_dtypes(include="number").columns
                               if column.startswith("validation_")]
        if calibration_metrics:
            aggregate_mean_std(selective, ["ablation", "system"], calibration_metrics).to_csv(
                output_root / "calibration_metrics_summary.csv", index=False)
    print(f"Grid complete; aggregate reports are under {output_root.resolve()}")


if __name__ == "__main__":
    main()
