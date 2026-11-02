"""Create paired seed-42 comparisons between the full model and ablations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def paired_deltas(base: pd.DataFrame, variants: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    base = base[keys + metrics].copy()
    variants = variants[keys + ["ablation"] + metrics].copy()
    merged = variants.merge(base, on=keys, how="inner", suffixes=("", "_full"), validate="many_to_one")
    for metric in metrics:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_full"]
    return merged


def write_summary(grouped: pd.DataFrame, path: Path) -> None:
    grouped.columns = ["_".join(str(part) for part in column if part)
                       if isinstance(column, tuple) else str(column)
                       for column in grouped.columns]
    grouped.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", default="experiments/temporal_graph_re2_grid")
    parser.add_argument("--ablation-root", default="experiments/temporal_graph_re2_ablation_grid")
    args = parser.parse_args()
    full_root, ablation_root = Path(args.full_root), Path(args.ablation_root)

    anomaly_full = pd.read_csv(full_root / "anomaly_metrics_by_run.csv")
    anomaly_variants = pd.read_csv(ablation_root / "anomaly_metrics_by_run.csv")
    anomaly_full = anomaly_full[anomaly_full.seed.eq(42) & anomaly_full.fault_status.eq("all")]
    anomaly_variants = anomaly_variants[anomaly_variants.fault_status.eq("all")]
    anomaly_metrics = [name for name in ("roc_auc", "average_precision", "precision", "recall", "f1",
                                         "false_positive_rate", "incident_detection_rate_120s")
                       if name in anomaly_full and name in anomaly_variants]
    anomaly = paired_deltas(anomaly_full, anomaly_variants, ["fold", "system", "fault_status"], anomaly_metrics)
    anomaly.to_csv(ablation_root / "anomaly_paired_deltas.csv", index=False)
    delta_cols = [f"{name}_delta" for name in anomaly_metrics]
    write_summary(anomaly.groupby(["ablation", "system"])[delta_cols].agg(["mean", "std"]).reset_index(),
                  ablation_root / "anomaly_paired_delta_summary.csv")

    root_full = pd.read_csv(full_root / "root_metrics_by_run.csv")
    root_variants = pd.read_csv(ablation_root / "root_metrics_by_run.csv")
    root_full = root_full[root_full.seed.eq(42)]
    root_metrics = ["hit_at_1", "hit_at_3", "mrr"]
    root = paired_deltas(root_full, root_variants,
                         ["fold", "system", "fault_status", "score_source"], root_metrics)
    root.to_csv(ablation_root / "root_paired_deltas.csv", index=False)
    write_summary(root.groupby(["ablation", "system", "fault_status", "score_source"])[
        [f"{name}_delta" for name in root_metrics]
    ].agg(["mean", "std"]).reset_index(), ablation_root / "root_paired_delta_summary.csv")
    print("Wrote paired seed-42 anomaly and root-cause deltas under", ablation_root.resolve())


if __name__ == "__main__":
    main()
