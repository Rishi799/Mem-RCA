"""Tune and evaluate a graph-supported RCA reranker against metric-only RCA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def case_rankings(frame: pd.DataFrame, adjacency: dict[str, list[str]], alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["fold", "system", "case", "fault", "fault_status", "true_root"]
    for key, group in frame.groupby(keys, sort=False):
        fold, system, case, fault, status, true_root = key
        own = dict(zip(group["service"].astype(str), group["score"].astype(float)))
        for service, base_score in own.items():
            child_scores = [max(0.0, own[child]) for child in adjacency.get(service, []) if child in own]
            support = float(np.mean(child_scores)) if child_scores else 0.0
            graph_score = base_score + alpha * support
            rows.append({
                "fold": fold, "system": system, "case": case, "fault": fault,
                "fault_status": status, "true_root": true_root, "service": service,
                "metric_score": base_score, "graph_support": support,
                "graph_score": graph_score, "is_root_cause": service == true_root,
            })
    ranked = pd.DataFrame(rows)
    ranked["metric_rank"] = ranked.groupby(["fold", "system", "case"])["metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    ranked["graph_rank"] = ranked.groupby(["fold", "system", "case"])["graph_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return ranked


def root_metrics(ranked: pd.DataFrame, rank_column: str) -> dict[str, float | int]:
    roots = ranked[ranked["is_root_cause"]]
    ranks = roots[rank_column].to_numpy(dtype=float)
    if not len(ranks):
        return {"cases": 0, "hit_at_1": float("nan"), "hit_at_3": float("nan"),
                "mean_reciprocal_rank": float("nan")}
    return {
        "cases": int(len(ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="experiments/baseline")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--splits", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--out", default="experiments/graph_rca")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    args = parser.parse_args()

    baseline = Path(args.baseline)
    validation = pd.read_csv(baseline / "validation_root_rankings.csv")
    test = pd.read_csv(baseline / "test_root_rankings.csv")
    edges = pd.read_csv(args.edges)
    manifest = pd.read_csv(args.splits)
    adjacency_by_group: dict[tuple[str, str], dict[str, list[str]]] = {}
    for (fold, system), split_rows in manifest.groupby(["fold", "system"]):
        training_cases = set(split_rows.loc[split_rows["split"].eq("train"), "case"].astype(str))
        validation_cases = set(split_rows.loc[split_rows["split"].eq("validation"), "case"].astype(str))
        testing_cases = set(split_rows.loc[split_rows["split"].eq("test"), "case"].astype(str))
        if training_cases & (validation_cases | testing_cases):
            raise ValueError(f"{fold}/{system}: case leakage across split assignments")
        topology = edges[edges["system"].astype(str).eq(str(system))
                        & edges["case"].astype(str).isin(training_cases)]
        adjacency: dict[str, list[str]] = {}
        for source, target in topology[["source", "target"]].drop_duplicates().itertuples(index=False, name=None):
            adjacency.setdefault(str(source), []).append(str(target))
        topology_cases = set(topology["case"].astype(str))
        if topology_cases & (validation_cases | testing_cases):
            raise ValueError(f"{fold}/{system}: validation or test trace entered the topology")
        adjacency_by_group[(str(fold), str(system))] = adjacency

    tuning_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected: dict[tuple[str, str], float] = {}
    for fold, system in sorted(validation[["fold", "system"]].drop_duplicates().itertuples(index=False, name=None)):
        subset = validation[validation["fold"].eq(fold) & validation["system"].eq(system)]
        best_alpha, best_mrr = 0.0, -1.0
        for alpha in sorted(set(args.alphas)):
            ranked = case_rankings(subset, adjacency_by_group[(str(fold), str(system))], alpha)
            score = root_metrics(ranked, "graph_rank")
            tuning_rows.append({"fold": fold, "system": system, "alpha": alpha,
                                "validation_cases": score["cases"],
                                "validation_hit_at_1": score["hit_at_1"],
                                "validation_hit_at_3": score["hit_at_3"],
                                "validation_mrr": score["mean_reciprocal_rank"]})
            mrr = float(score["mean_reciprocal_rank"])
            if mrr > best_mrr + 1e-12:
                best_alpha, best_mrr = alpha, mrr
        selected[(str(fold), str(system))] = best_alpha
        selection_rows.append({"fold": fold, "system": system, "selected_alpha": best_alpha,
                               "validation_mrr": best_mrr})

    test_rank_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for (fold, system), alpha in sorted(selected.items()):
        subset = test[test["fold"].eq(fold) & test["system"].eq(system)]
        adjacency = adjacency_by_group[(fold, system)]
        ranked = case_rankings(subset, adjacency, alpha)
        ranked["selected_alpha"] = alpha
        ranked["has_training_graph"] = bool(adjacency)
        test_rank_frames.append(ranked)
        for status in ("all", "known", "unseen"):
            group = ranked if status == "all" else ranked[ranked["fault_status"].eq(status)]
            base = root_metrics(group, "metric_rank")
            graph = root_metrics(group, "graph_rank")
            metric_rows.append({
                "fold": fold, "system": system, "fault_status": status, "selected_alpha": alpha,
                "cases": graph["cases"], "metric_hit_at_1": base["hit_at_1"],
                "metric_hit_at_3": base["hit_at_3"], "metric_mrr": base["mean_reciprocal_rank"],
                "graph_hit_at_1": graph["hit_at_1"], "graph_hit_at_3": graph["hit_at_3"],
                "graph_mrr": graph["mean_reciprocal_rank"],
                "cases_with_training_graph": int(group[group["has_training_graph"]]["case"].nunique()),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tuning = pd.DataFrame(tuning_rows)
    selection = pd.DataFrame(selection_rows)
    rankings = pd.concat(test_rank_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    tuning.to_csv(out / "validation_alpha_tuning.csv", index=False)
    selection.to_csv(out / "validation_alpha_selection.csv", index=False)
    rankings.to_csv(out / "test_graph_rankings.csv", index=False)
    metrics.to_csv(out / "test_graph_metrics.csv", index=False)
    report = {
        "method": "metric RCA score plus alpha times mean positive direct-child anomaly score",
        "alpha_grid": sorted(set(args.alphas)),
        "selection": "validation MRR on known-fault cases, separately per fold and system; ties select lower alpha",
        "graph_source": f"training-case edges only from {args.edges}; split assignments from {args.splits}",
        "trace_edges_are_causal": False,
        "leakage_control": "validation/test case traces are excluded from the graph topology",
        "missing_graph_behavior": "no training graph support; falls back to metric-only score",
        "note": "Graph-informed ranking is an empirical reranking baseline using observed service-call edges, not a causal model.",
    }
    (out / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Validation-selected alpha by fold/system:")
    print(selection.to_string(index=False))
    print("\nHeld-out test comparison:")
    print(metrics[metrics["fault_status"].eq("all")].to_string(index=False))
    print(f"Saved graph RCA comparison under {out}")


if __name__ == "__main__":
    main()
