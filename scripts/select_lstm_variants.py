"""Select legacy vs robust LSTM score variants using validation data only.

Creates separate score sets for anomaly detection and root-cause ranking because
the best model for one objective need not be best for the other.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from evaluate_lstm_scores import persistent_alarms, root_rankings, window_scores


def validation_metrics(path: Path, normal_fpr: float, persistence_hits: int) -> tuple[float, float]:
    scores = pd.read_csv(path)
    windows = window_scores(scores)
    validation = windows[windows["split"].eq("validation")].copy()
    normal = validation[~validation["is_anomaly"].astype(bool)]
    if normal.empty:
        raise ValueError(f"No normal validation windows in {path}")
    threshold = float(normal["anomaly_score"].quantile(1.0 - normal_fpr))
    y = validation["is_anomaly"].astype(bool).to_numpy()
    alarms = persistent_alarms(validation, validation["anomaly_score"].gt(threshold).to_numpy(), persistence_hits)
    anomaly_f1 = float(f1_score(y, alarms, zero_division=0))
    roots = root_rankings(scores, 120, split="validation")
    rca_mrr = float(roots["mrr"].mean()) if not roots.empty else float("nan")
    return anomaly_f1, rca_mrr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-z", default="experiments/lstm")
    parser.add_argument("--legacy-rankcal", default="experiments/lstm_rankcal")
    parser.add_argument("--robust", default="experiments/lstm_robust")
    parser.add_argument("--out", default="experiments/lstm_selected")
    parser.add_argument("--normal-fpr", type=float, default=0.01)
    parser.add_argument("--persistence-hits", type=int, default=2)
    args = parser.parse_args()
    sources = {
        "legacy_z": Path(args.legacy_z),
        "legacy_rankcal": Path(args.legacy_rankcal),
        "robust": Path(args.robust),
    }
    out = Path(args.out)
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for fold in ("fold1", "fold2", "fold3"):
        for system in ("ob", "ss", "tt"):
            results: dict[str, tuple[float, float]] = {}
            paths: dict[str, Path] = {}
            for name, root in sources.items():
                path = root / fold / system / "service_window_scores.csv"
                if not path.exists():
                    missing.append(f"{name}:{fold}/{system}")
                    continue
                paths[name] = path
                results[name] = validation_metrics(path, args.normal_fpr, args.persistence_hits)
            if len(results) != len(sources):
                continue
            # Stable tie order favors the established z-score model.
            priority = {"legacy_z": 0, "legacy_rankcal": 1, "robust": 2}
            anomaly_choice = max(results, key=lambda name: (results[name][0], -priority[name]))
            rca_choice = max(results, key=lambda name: (
                -np.inf if not np.isfinite(results[name][1]) else results[name][1],
                -priority[name],
            ))
            for objective, choice, metric_index, metric_name in (
                ("anomaly", anomaly_choice, 0, "validation_f1"),
                ("rca", rca_choice, 1, "validation_mrr"),
            ):
                target = out / objective / fold / system
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(paths[choice], target / "service_window_scores.csv")
                records.append({
                    "objective": objective, "fold": fold, "system": system,
                    "selected_variant": choice, metric_name: results[choice][metric_index],
                    "legacy_z_validation_f1": results["legacy_z"][0],
                    "legacy_rankcal_validation_f1": results["legacy_rankcal"][0],
                    "robust_validation_f1": results["robust"][0],
                    "legacy_z_validation_mrr": results["legacy_z"][1],
                    "legacy_rankcal_validation_mrr": results["legacy_rankcal"][1],
                    "robust_validation_mrr": results["robust"][1],
                    "selection_data": "validation only",
                })
    if missing:
        raise SystemExit("Missing candidate files: " + ", ".join(missing))
    table = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model_selection.csv", index=False)
    print(table.to_string(index=False))
    print(f"Wrote validation-only variant selections under {out}")


if __name__ == "__main__":
    main()
