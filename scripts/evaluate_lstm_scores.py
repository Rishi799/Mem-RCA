"""Evaluate LSTM per-service scores using validation-selected thresholds.

The input is the ``service_window_scores.csv`` written by
``train_lstm_forecaster.py``. It can also combine multiple fold/system outputs
after concatenation because calibration is performed separately by fold/system.
"""
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def window_scores(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "system", "case", "split", "window_start"]
    return frame.groupby(keys, as_index=False).agg(
        anomaly_score=("anomaly_score", "max"),
        is_anomaly=("is_anomaly", "max"),
        seconds_from_injection=("seconds_from_injection", "max"),
        fault_status=("fault_status", "first"),
    )


def persistent_alarms(frame: pd.DataFrame, alarms: np.ndarray, hits: int) -> np.ndarray:
    """Require ``hits`` alarms in a rolling three-window neighborhood per case."""
    if hits <= 1:
        return alarms
    if hits > 3:
        raise ValueError("--persistence-hits must be between 1 and 3")
    result = np.zeros(len(frame), dtype=bool)
    ordered = frame.assign(_alarm=alarms, _pos=np.arange(len(frame))).sort_values(["case", "window_start"])
    smooth = ordered.groupby("case", sort=False)["_alarm"].transform(
        lambda values: values.rolling(3, min_periods=1).sum().ge(hits)
    )
    result[ordered["_pos"].to_numpy(dtype=int)] = smooth.to_numpy(dtype=bool)
    return result


def root_rankings(frame: pd.DataFrame, horizon: int, split: str = "test") -> pd.DataFrame:
    test = frame[
        frame["split"].eq(split)
        & frame["seconds_from_injection"].ge(0)
        & frame["seconds_from_injection"].lt(horizon)
        & frame["anomaly_score"].notna()
    ]
    output: list[dict[str, object]] = []
    for (fold, system, case), group in test.groupby(["fold", "system", "case"]):
        rows = []
        for service, service_group in group.groupby("service"):
            values = service_group["anomaly_score"].to_numpy(dtype=float)
            top_count = max(1, math.ceil(len(values) * 0.25))
            rows.append((str(service), float(np.sort(values)[-top_count:].mean())))
        ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        root = str(group["root_cause_service"].iloc[0])
        root_rank = next((index for index, (name, _) in enumerate(ranked, start=1) if name == root), None)
        output.append({
            "fold": fold,
            "system": system,
            "case": case,
            "fault_status": str(group["fault_status"].iloc[0]),
            "root_cause_service": root,
            "predicted_top1": ranked[0][0],
            "predicted_top3": ";".join(name for name, _ in ranked[:3]),
            "root_rank": root_rank,
            "hit_at_1": int(root_rank == 1),
            "hit_at_3": int(root_rank is not None and root_rank <= 3),
            "mrr": 0.0 if root_rank is None else 1.0 / root_rank,
        })
    return pd.DataFrame(output)


def case_service_scores(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    selected = frame[
        frame["split"].isin(["validation", "test"])
        & frame["seconds_from_injection"].ge(0)
        & frame["seconds_from_injection"].lt(horizon)
        & frame["anomaly_score"].notna()
    ]
    records: list[dict[str, object]] = []
    case_cols = ["fold", "system", "case", "split", "fault", "fault_status", "root_cause_service"]
    for key, group in selected.groupby(case_cols):
        fold, system, case, split, fault, status, root = key
        for service, service_group in group.groupby("service"):
            values = service_group["anomaly_score"].to_numpy(dtype=float)
            top_count = max(1, math.ceil(len(values) * 0.25))
            records.append({
                "fold": str(fold), "system": str(system), "case": str(case),
                "split": str(split), "fault": str(fault), "fault_status": str(status),
                "true_root": str(root), "service": str(service),
                "score": float(np.sort(values)[-top_count:].mean()),
            })
    return pd.DataFrame(records)


def graph_rankings(frame: pd.DataFrame, adjacency: dict[str, list[str]], alpha: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["fold", "system", "case", "split", "fault", "fault_status", "true_root"]
    for key, group in frame.groupby(keys, sort=False):
        fold, system, case, split, fault, status, root = key
        own = dict(zip(group["service"].astype(str), group["score"].astype(float)))
        for service, score in own.items():
            child_scores = [max(0.0, own[child]) for child in adjacency.get(service, []) if child in own]
            support = float(np.mean(child_scores)) if child_scores else 0.0
            records.append({
                "fold": fold, "system": system, "case": case, "split": split,
                "fault": fault, "fault_status": status, "true_root": root,
                "service": service, "metric_score": score, "graph_support": support,
                "graph_score": score + alpha * support,
            })
    ranked = pd.DataFrame(records)
    if ranked.empty:
        return ranked
    group_cols = ["fold", "system", "case"]
    ranked["metric_rank"] = ranked.groupby(group_cols)["metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    ranked["graph_rank"] = ranked.groupby(group_cols)["graph_score"].rank(
        method="first", ascending=False
    ).astype(int)
    ranked["is_root_cause"] = ranked["service"].eq(ranked["true_root"])
    return ranked


def rank_metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    roots = frame[frame["is_root_cause"]]
    ranks = roots[column].to_numpy(dtype=float)
    if not len(ranks):
        return {"cases": 0, "hit_at_1": float("nan"), "hit_at_3": float("nan"), "mrr": float("nan")}
    return {
        "cases": int(len(ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def evaluate_graph_reranker(
    scores: pd.DataFrame,
    manifest_path: Path,
    edges_path: Path,
    out_dir: Path,
    horizon: int,
) -> None:
    incidents = case_service_scores(scores, horizon)
    manifest = pd.read_csv(manifest_path)
    edges = pd.read_csv(edges_path)
    adjacency_by_group: dict[tuple[str, str], dict[str, list[str]]] = {}
    for (fold, system), split_rows in manifest.groupby(["fold", "system"]):
        train_cases = set(split_rows.loc[split_rows["split"].eq("train"), "case"].astype(str))
        topology = edges[
            edges["system"].astype(str).eq(str(system))
            & edges["case"].astype(str).isin(train_cases)
        ]
        adjacency: dict[str, list[str]] = {}
        for source, target in topology[["source", "target"]].drop_duplicates().itertuples(index=False, name=None):
            adjacency.setdefault(str(source), []).append(str(target))
        adjacency_by_group[(str(fold), str(system))] = adjacency

    tuning: list[dict[str, object]] = []
    selected: dict[tuple[str, str], float] = {}
    for (fold, system), group in incidents[incidents["split"].eq("validation")].groupby(["fold", "system"]):
        adjacency = adjacency_by_group[(str(fold), str(system))]
        best_alpha, best_mrr = 0.0, -1.0
        for alpha in ALPHAS:
            ranked = graph_rankings(group, adjacency, alpha)
            metrics = rank_metrics(ranked, "graph_rank")
            tuning.append({"fold": fold, "system": system, "alpha": alpha, **{f"validation_{k}": v for k, v in metrics.items()}})
            if metrics["mrr"] > best_mrr + 1e-12:
                best_alpha, best_mrr = alpha, float(metrics["mrr"])
        selected[(str(fold), str(system))] = best_alpha

    metric_rows: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []
    for (fold, system), alpha in sorted(selected.items()):
        group = incidents[
            incidents["fold"].eq(fold)
            & incidents["system"].eq(system)
            & incidents["split"].eq("test")
        ]
        ranked = graph_rankings(group, adjacency_by_group[(fold, system)], alpha)
        ranked["selected_alpha"] = alpha
        rankings.append(ranked)
        for status in ("all", "known", "unseen"):
            subset = ranked if status == "all" else ranked[ranked["fault_status"].eq(status)]
            metric = rank_metrics(subset, "metric_rank")
            graph = rank_metrics(subset, "graph_rank")
            metric_rows.append({
                "fold": fold, "system": system, "fault_status": status,
                "selected_alpha": alpha, "cases": graph["cases"],
                "metric_hit_at_1": metric["hit_at_1"], "metric_hit_at_3": metric["hit_at_3"],
                "metric_mrr": metric["mrr"], "graph_hit_at_1": graph["hit_at_1"],
                "graph_hit_at_3": graph["hit_at_3"], "graph_mrr": graph["mrr"],
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning).to_csv(out_dir / "graph_alpha_tuning.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "graph_metrics.csv", index=False)
    if rankings:
        pd.concat(rankings, ignore_index=True).to_csv(out_dir / "graph_rankings.csv", index=False)
    (out_dir / "graph_evaluation.json").write_text(json.dumps({
        "alpha_grid": list(ALPHAS),
        "selection": "validation MRR, selected separately by fold and system; ties select lower alpha",
        "graph_source": "training-case trace edges only",
        "trace_edges_are_causal": False,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--normal-fpr", type=float, default=0.01)
    parser.add_argument("--persistence-hits", type=int, default=1,
                        help="Require this many alarms in the last 3 windows (1 disables persistence filtering)")
    parser.add_argument("--rca-horizon", type=int, default=120)
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--out", default="experiments/lstm/evaluation")
    args = parser.parse_args()
    if not 0 < args.normal_fpr < 0.5:
        raise ValueError("--normal-fpr must be between 0 and 0.5")
    if not 1 <= args.persistence_hits <= 3:
        raise ValueError("--persistence-hits must be between 1 and 3")

    scores = pd.read_csv(args.scores)
    required = {
        "fold", "system", "case", "split", "window_start", "service",
        "anomaly_score", "is_anomaly", "seconds_from_injection", "root_cause_service",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Input scores are missing required columns: {missing}")
    windows = window_scores(scores)
    metric_rows: list[dict[str, object]] = []
    for (fold, system), group in windows.groupby(["fold", "system"]):
        validation_normal = group[
            group["split"].eq("validation") & ~group["is_anomaly"].astype(bool)
        ]
        test = group[group["split"].eq("test")].copy()
        if validation_normal.empty or test.empty:
            continue
        threshold = float(validation_normal["anomaly_score"].quantile(1.0 - args.normal_fpr))
        val = group[group["split"].eq("validation")].copy()
        val_y = val["is_anomaly"].astype(bool).to_numpy()
        # Empirical-tail scores are discrete. Strict exceedance avoids counting
        # every tie at the selected normal quantile as a false alarm.
        val_pred = persistent_alarms(val, val["anomaly_score"].gt(threshold).to_numpy(), args.persistence_hits)
        val_normal = ~val_y
        validation_fpr = float(val_pred[val_normal].mean()) if val_normal.any() else float("nan")
        validation_recall = float(recall_score(val_y, val_pred, zero_division=0))
        validation_f1 = float(f1_score(val_y, val_pred, zero_division=0))
        for status in ("all", "known", "unseen"):
            subset = test if status == "all" else test[test["fault_status"].eq(status)]
            if subset.empty:
                continue
            y = subset["is_anomaly"].astype(bool).to_numpy()
            values = subset["anomaly_score"].to_numpy(dtype=float)
            pred = persistent_alarms(subset, values > threshold, args.persistence_hits)
            normal = ~y
            post_horizon = subset[
                subset["is_anomaly"].astype(bool)
                & subset["seconds_from_injection"].ge(0)
                & subset["seconds_from_injection"].lt(args.rca_horizon)
            ]
            cases_with_target = post_horizon["case"].nunique()
            detected_cases = post_horizon.assign(
                alarm=post_horizon["anomaly_score"].gt(threshold)
            ).groupby("case")["alarm"].any().sum()
            metric_rows.append({
                "fold": fold, "system": system, "fault_status": status,
                "threshold": threshold, "test_windows": int(len(subset)),
                "persistence_hits_in_3": args.persistence_hits,
                "validation_normal_fpr": validation_fpr,
                "validation_known_recall": validation_recall,
                "validation_known_f1": validation_f1,
                "positive_test_windows": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, values)) if y.any() and normal.any() else float("nan"),
                "average_precision": float(average_precision_score(y, values)) if y.any() else float("nan"),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "false_positive_rate": float(pred[normal].mean()) if normal.any() else float("nan"),
                "incident_detection_rate_120s": float(detected_cases / cases_with_target) if cases_with_target else float("nan"),
            })

    validation_rankings = root_rankings(scores, args.rca_horizon, split="validation")
    for (fold, system), group in validation_rankings.groupby(["fold", "system"]):
        for row in metric_rows:
            if row["fold"] == fold and row["system"] == system:
                row["validation_root_cases"] = int(len(group))
                row["validation_root_hit_at_1"] = float(group["hit_at_1"].mean())
                row["validation_root_hit_at_3"] = float(group["hit_at_3"].mean())
                row["validation_root_mrr"] = float(group["mrr"].mean())

    rankings = root_rankings(scores, args.rca_horizon)
    for (fold, system), group in rankings.groupby(["fold", "system"]):
        for status in ("all", "known", "unseen"):
            subset = group if status == "all" else group[group["fault_status"].eq(status)]
            row = next((item for item in metric_rows if item["fold"] == fold and item["system"] == system and item["fault_status"] == status), None)
            if row is not None:
                row["rca_cases"] = int(len(subset))
                row["hit_at_1"] = float(subset["hit_at_1"].mean()) if len(subset) else float("nan")
                row["hit_at_3"] = float(subset["hit_at_3"].mean()) if len(subset) else float("nan")
                row["mrr"] = float(subset["mrr"].mean()) if len(subset) else float("nan")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    rankings.to_csv(out / "root_cause_rankings.csv", index=False)
    evaluate_graph_reranker(
        scores, Path(args.manifest), Path(args.edges), out / "graph", args.rca_horizon
    )
    print(metrics.to_string(index=False))
    print(f"Wrote metrics and root-cause rankings under {out}")


if __name__ == "__main__":
    main()

