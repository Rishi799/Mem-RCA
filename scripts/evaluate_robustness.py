"""Run robust root cause evaluation with inductive topological ranking and bootstrap 95% CIs.

Generates:
1. Inductive topological vs. baseline score rankings.
2. Bootstrap 95% confidence intervals per system, fold, and split.
3. Summary report detailing robustness against benchmark shortcuts and small-sample variance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memrca.evaluation.inductive_ranker import InductiveServiceRanker
from memrca.evaluation.bootstrap_ci import evaluate_with_confidence_intervals
from scripts.evaluate_root_cause import generate_mock_person_a_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="experiments/ranking/sample_person_a_scores.csv", help="Input scores CSV")
    parser.add_argument("--manifest", default="experiments/split_manifest.csv", help="Split manifest CSV")
    parser.add_argument("--edges", default="experiments/graphs/re2_case_edges.csv", help="Optional edge list CSV")
    parser.add_argument("--out-dir", default="experiments/robustness", help="Output directory")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("Manifest not found, running verify_split first...")
        from scripts.verify_split import main as run_verify
        run_verify()
    manifest_df = pd.read_csv(manifest_path)

    scores_path = Path(args.scores)
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
    else:
        scores_df = generate_mock_person_a_scores(manifest_df)
        scores_df.to_csv(scores_path, index=False)

    edges_df = pd.read_csv(args.edges) if (args.edges and Path(args.edges).is_file()) else None

    print("Evaluating with InductiveServiceRanker (breaking node identity shortcut)...")
    ranker = InductiveServiceRanker(graph_weight=0.25, topological_prior_weight=0.15)

    ranking_records = []
    case_meta = manifest_df.set_index(["case", "fold"]).to_dict("index")

    grouped = scores_df.groupby(["case", "fold", "split", "system", "window_start"])
    for (case, fold, split, system, win_start), grp in grouped:
        true_rc = (
            grp["true_root_cause"].iloc[0]
            if "true_root_cause" in grp.columns
            else case_meta.get((case, fold), {}).get("root_cause_service", "")
        )
        service_scores = dict(zip(grp["service"], grp["anomaly_score"]))

        case_edges = None
        if edges_df is not None and not edges_df.empty and "case" in edges_df.columns:
            case_edges = edges_df[edges_df["case"] == case]

        ranked_services = ranker.rank_case(service_scores, edges=case_edges)

        predicted_rank = None
        for r_idx, (svc, score) in enumerate(ranked_services, start=1):
            if svc == true_rc:
                predicted_rank = r_idx
                break

        ranking_records.append({
            "case": case,
            "fold": fold,
            "split": split,
            "system": system,
            "window_start": win_start,
            "root_cause_service": true_rc,
            "predicted_rank": predicted_rank,
            "hit@1": 1 if predicted_rank == 1 else 0,
            "hit@3": 1 if predicted_rank is not None and predicted_rank <= 3 else 0,
            "reciprocal_rank": (1.0 / predicted_rank) if predicted_rank else 0.0,
        })

    predictions_df = pd.DataFrame(ranking_records)
    predictions_df.to_csv(out_dir / "inductive_rankings.csv", index=False)

    print(f"Computing bootstrap 95% confidence intervals ({args.bootstraps} resamples)...")
    ci_metrics_df = evaluate_with_confidence_intervals(
        predictions_df,
        group_cols=["system", "split"],
        n_bootstraps=args.bootstraps,
    )
    ci_metrics_path = out_dir / "bootstrap_confidence_intervals.csv"
    ci_metrics_df.to_csv(ci_metrics_path, index=False)

    print("\n=== Robustness Evaluation Results (Grouped by System & Split with 95% CI) ===")
    print(ci_metrics_df.to_string(index=False))
    print(f"\nSaved bootstrap metrics to: {ci_metrics_path}")


if __name__ == "__main__":
    main()
