"""Apply validation empirical-tail calibration to existing LSTM raw errors.

This creates a score-only ablation: weights and preprocessing stay fixed, so it
isolates calibration from the revised robust preprocessing experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="experiments/lstm")
    parser.add_argument("--out", default="experiments/lstm_rankcal")
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    missing: list[str] = []
    for fold in ("fold1", "fold2", "fold3"):
        for system in ("ob", "ss", "tt"):
            path = source / fold / system / "service_window_scores.csv"
            if not path.exists():
                missing.append(f"{fold}/{system}")
                continue
            scores = pd.read_csv(path)
            if "raw_anomaly_score" not in scores:
                raise ValueError(f"{path} is missing raw_anomaly_score")
            normal = scores[scores["split"].eq("validation") & ~scores["is_anomaly"].astype(bool)]
            calibration: dict[str, list[float]] = {}
            for service, group in normal.groupby("service"):
                calibration[str(service)] = np.sort(group["raw_anomaly_score"].to_numpy(dtype=float)).tolist()
            scores["anomaly_score"] = 0.0
            for service, indexes in scores.groupby("service").groups.items():
                ref = np.asarray(calibration.get(str(service), []), dtype=float)
                if not len(ref):
                    continue
                raw = scores.loc[indexes, "raw_anomaly_score"].to_numpy(dtype=float)
                tail = len(ref) - np.searchsorted(ref, raw, side="left")
                scores.loc[indexes, "anomaly_score"] = -np.log10((tail + 1) / (len(ref) + 1))
            target = out / fold / system
            target.mkdir(parents=True, exist_ok=True)
            scores.to_csv(target / "service_window_scores.csv", index=False)
            (target / "validation_score_calibration.json").write_text(json.dumps({
                service: {"normal_score_reference": values, "n": len(values)}
                for service, values in calibration.items()
            }), encoding="utf-8")
            print(f"Calibrated {fold}/{system}: {len(scores):,} rows")
    if missing:
        raise SystemExit("Missing source score files: " + ", ".join(missing))


if __name__ == "__main__":
    main()
