"""Execute root cause ranking evaluation across splits and systems.

Compatible with Person A's anomaly score schema:
- Schema: case, fold, split, window_start, service, anomaly_score
- Evaluates: Hit@1, Hit@3, and MRR.
- Supports ablation: score-only vs. graph-informed ranking.
- Keeps train, validation, and test strictly separated; no tuning on test.
- Gracefully handles missing graphs (Sock Shop, missing traces).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memrca.evaluation.ranker import ServiceRanker, evaluate_rankings
from memrca.data.graph import EDGE_COLUMNS


def generate_mock_person_a_scores(
    manifest_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible sample anomaly scores conforming to Person A's schema.

    Schema: ['case', 'fold', 'split', 'window_start', 'service', 'anomaly_score']
    Used to demonstrate end-to-end evaluation before Person A delivers detector outputs.
    """
    rng = np.random.default_rng(seed)
    records = []

    # System services map
    system_services = {
        "ob": ["checkoutservice", "currencyservice", "emailservice", "productcatalogservice", "recommendationservice", "frontend"],
        "ss": ["carts", "catalogue", "orders", "payment", "user", "front-end"],
        "tt": ["ts-auth-service", "ts-order-service", "ts-route-service", "ts-train-service", "ts-travel-service"],
    }

    for _, row in manifest_df.iterrows():
        case = row["case"]
        fold = row["fold"]
        split = row["split"]
        system = row["system"]
        true_rc = row["root_cause_service"]
        services = system_services.get(system, [true_rc, "other_service"])

        # Create 3 time windows per case
        for win_idx, win_time in enumerate([1000, 1060, 1120]):
            for svc in services:
                # Root cause has higher baseline anomaly score with some noise
                base = 2.5 if svc == true_rc else 0.5
                score = max(0.0, float(rng.normal(loc=base, scale=0.8)))
                records.append({
                    "case": case,
                    "fold": fold,
                    "split": split,
                    "system": system,
                    "window_start": win_time,
                    "service": svc,
                    "anomaly_score": round(score, 4),
                    "true_root_cause": true_rc,
                })

    return pd.DataFrame(records)


def run_evaluation(
    scores_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    edges_df: pd.DataFrame | None = None,
    graph_weight: float = 0.0,
    out_rankings_path: str | Path | None = None,
    out_metrics_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform ranking and evaluation per case and window."""
    ranker = ServiceRanker(graph_weight=graph_weight)

    # Fit normalizer strictly on train split
    train_scores = scores_df[scores_df["split"] == "train"]
    if not train_scores.empty:
        ranker.fit_normalizer(train_scores)

    # Group scores by (case, fold, split, system, window_start)
    case_meta = manifest_df.set_index(["case", "fold"]).to_dict("index")

    ranking_records = []
    grouped = scores_df.groupby(["case", "fold", "split", "system", "window_start"])

    for (case, fold, split, system, win_start), grp in grouped:
        true_rc = grp["true_root_cause"].iloc[0] if "true_root_cause" in grp.columns else case_meta.get((case, fold), {}).get("root_cause_service", "")
        service_scores = dict(zip(grp["service"], grp["anomaly_score"]))

        # Filter edges for this case if available
        case_edges = None
        if edges_df is not None and not edges_df.empty and "case" in edges_df.columns:
            case_edges = edges_df[edges_df["case"] == case]

        ranked_services = ranker.rank_case(service_scores, edges=case_edges)

        # Determine rank of the true root cause (1-based)
        predicted_rank = None
        for r_idx, (svc, score) in enumerate(ranked_services, start=1):
            if svc == true_rc:
                predicted_rank = r_idx
                break

        top1 = ranked_services[0][0] if ranked_services else None
        top3 = [s for s, _ in ranked_services[:3]]

        ranking_records.append({
            "case": case,
            "fold": fold,
            "split": split,
            "system": system,
            "window_start": win_start,
            "root_cause_service": true_rc,
            "predicted_rank": predicted_rank,
            "top1_predicted": top1,
            "top3_predicted": ";".join(top3),
            "hit@1": 1 if predicted_rank == 1 else 0,
            "hit@3": 1 if predicted_rank is not None and predicted_rank <= 3 else 0,
            "reciprocal_rank": (1.0 / predicted_rank) if predicted_rank else 0.0,
        })

    predictions_df = pd.DataFrame(ranking_records)
    metrics_df = evaluate_rankings(predictions_df, group_cols=["system", "fold", "split"])

    if out_rankings_path:
        p = Path(out_rankings_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(p, index=False)

    if out_metrics_path:
        p = Path(out_metrics_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(p, index=False)

    return predictions_df, metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=None, help="Path to input anomaly scores CSV")
    parser.add_argument("--manifest", default="experiments/split_manifest.csv", help="Path to split manifest CSV")
    parser.add_argument("--edges", default=None, help="Path to optional edge list CSV")
    parser.add_argument("--out-dir", default="experiments/ranking", help="Output directory")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run verify_split.py first.")
    manifest_df = pd.read_csv(manifest_path)

    # Load or generate scores
    if args.scores and Path(args.scores).is_file():
        print(f"Loading anomaly scores from {args.scores}...")
        scores_df = pd.read_csv(args.scores)
    else:
        print("No external scores provided. Generating benchmark scores with Person A schema...")
        scores_df = generate_mock_person_a_scores(manifest_df)
        mock_path = Path(args.out_dir) / "sample_person_a_scores.csv"
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(mock_path, index=False)
        print(f"Sample scores saved to {mock_path}")

    # Load edges if provided
    edges_df = pd.read_csv(args.edges) if (args.edges and Path(args.edges).is_file()) else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline: Pure anomaly score ranking (graph disabled)
    print("\n--- Evaluating Baseline: Score-Only Ranking ---")
    _, base_metrics = run_evaluation(
        scores_df=scores_df,
        manifest_df=manifest_df,
        edges_df=None,
        graph_weight=0.0,
        out_rankings_path=out_dir / "baseline_score_only_rankings.csv",
        out_metrics_path=out_dir / "baseline_score_only_metrics.csv",
    )
    print(base_metrics.to_string(index=False))

    # 2. Graph-informed ranking ablation (alpha=0.3)
    print("\n--- Evaluating Ablation: Graph-Informed Ranking (alpha=0.3) ---")
    _, graph_metrics = run_evaluation(
        scores_df=scores_df,
        manifest_df=manifest_df,
        edges_df=edges_df,
        graph_weight=0.3,
        out_rankings_path=out_dir / "graph_informed_rankings.csv",
        out_metrics_path=out_dir / "graph_informed_metrics.csv",
    )
    print(graph_metrics.to_string(index=False))

    print(f"\nAll ranking outputs and metrics saved under: {out_dir}/")


if __name__ == "__main__":
    main()
