"""Evaluate temporal-graph root scores alongside forecast-error RCA baselines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def case_rankings(scores: pd.DataFrame, score_column: str, horizon: int) -> pd.DataFrame:
    post = scores[
        scores["split"].eq("test")
        & scores["seconds_from_injection"].ge(0)
        & scores["seconds_from_injection"].lt(horizon)
        & scores[score_column].notna()
    ]
    records: list[dict] = []
    keys = ["fold", "system", "case"]
    for key, group in post.groupby(keys):
        fold, system, case = key
        values = []
        for service, service_group in group.groupby("service"):
            scores_for_service = service_group[score_column].to_numpy(float)
            top_count = max(1, math.ceil(len(scores_for_service) * 0.25))
            values.append((str(service), float(np.sort(scores_for_service)[-top_count:].mean())))
        ranked = sorted(values, key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        root = str(group["root_cause_service"].iloc[0])
        root_rank = next((index for index, (service, _) in enumerate(ranked, start=1) if service == root), None)
        records.append({
            "fold": fold, "system": system, "case": case,
            "fault_status": str(group["fault_status"].iloc[0]),
            "root_cause_service": root, "score_source": score_column,
            "predicted_root": ranked[0][0], "root_rank": root_rank,
            "hit_at_1": int(root_rank == 1),
            "hit_at_3": int(root_rank is not None and root_rank <= 3),
            "mrr": 0.0 if root_rank is None else 1.0 / root_rank,
        })
    return pd.DataFrame(records)


def confidence_metrics(frame: pd.DataFrame, bins: int = 10) -> dict[str, float]:
    if frame.empty:
        return {"brier_score": float("nan"), "expected_calibration_error": float("nan")}
    probabilities = np.stack([
        np.asarray(list(json.loads(value).values()), dtype=float)
        for value in frame["root_probabilities"]
    ])
    services = list(json.loads(frame["root_probabilities"].iloc[0]).keys())
    labels = np.asarray([
        services.index(str(root)) if str(root) in services else -1
        for root in frame["root_cause_service"]
    ], dtype=int)
    valid = labels >= 0
    probabilities, labels = probabilities[valid], labels[valid]
    if not len(labels):
        return {"brier_score": float("nan"), "expected_calibration_error": float("nan")}
    one_hot = np.eye(probabilities.shape[1])[labels]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(confidence[in_bin].mean()) - float(correct[in_bin].mean()))
    return {
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "expected_calibration_error": float(ece),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--horizon", type=int, default=120)
    args = parser.parse_args()
    scores_path = Path(args.scores)
    scores = pd.read_csv(scores_path)
    required = {"fold", "system", "case", "split", "fault_status", "service",
                "seconds_from_injection", "root_cause_service", "root_score", "anomaly_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Score file missing required columns: {sorted(missing)}")
    root = case_rankings(scores, "root_score", args.horizon)
    metric = case_rankings(scores, "anomaly_score", args.horizon)
    both = pd.concat([root, metric], ignore_index=True)
    out = Path(args.out) if args.out else scores_path.parent / "root_head_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    both.to_csv(out / "root_head_rankings_by_case.csv", index=False)
    summary = both.groupby(["system", "fault_status", "score_source"], as_index=False).agg(
        cases=("case", "size"), hit_at_1=("hit_at_1", "mean"),
        hit_at_3=("hit_at_3", "mean"), mrr=("mrr", "mean"),
    )
    summary.to_csv(out / "root_head_metrics_summary.csv", index=False)
    case_path = scores_path.parent / "root_cause_case_predictions.csv"
    if case_path.exists():
        cases = pd.read_csv(case_path)
        test = cases[cases["split"].eq("test")].copy()
        coverage_rows = []
        for status, subset in [("all", test), *[(str(name), group) for name, group in test.groupby("fault_status")]]:
            for confidence_cutoff in (0.0, 0.25, 0.5, 0.75, 0.9):
                kept = subset[subset["confidence"].ge(confidence_cutoff)]
                coverage_rows.append({
                    "fault_status": status, "confidence_cutoff": confidence_cutoff,
                    "cases": int(len(subset)), "covered_cases": int(len(kept)),
                    "coverage": float(len(kept) / len(subset)) if len(subset) else float("nan"),
                    "risk_when_covered": float(1.0 - kept["hit_at_1"].mean()) if len(kept) else float("nan"),
                    "hit_at_1_when_covered": float(kept["hit_at_1"].mean()) if len(kept) else float("nan"),
                })
        coverage = pd.DataFrame(coverage_rows)
        coverage.to_csv(out / "selective_risk_coverage.csv", index=False)
        selected = test[~test["abstain"].astype(bool)]
        calibration_path = scores_path.parent / "root_calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
        selective_report = {
            "test_cases": int(len(test)),
            "validation_selected_confidence_threshold": calibration.get("validation_abstention_threshold"),
            "test_coverage": float(len(selected) / len(test)) if len(test) else float("nan"),
            "test_risk_when_covered": float(1.0 - selected["hit_at_1"].mean()) if len(selected) else float("nan"),
            **{f"test_{key}": value for key, value in confidence_metrics(test).items()},
            "test_used_to_select_threshold": False,
        }
        (out / "selective_evaluation.json").write_text(
            json.dumps(selective_report, indent=2, allow_nan=True), encoding="utf-8"
        )
    print(summary.to_string(index=False))
    print(f"Wrote per-case and aggregate root-head metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
