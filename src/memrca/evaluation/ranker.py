"""Root cause ranking baselines and evaluation metrics for Mem-RCA (Phase 1 Person B).

Supports:
- Transparent service ranking based on anomaly scores.
- Metric-only baseline vs. graph-informed ranking (aggregating downstream/upstream dependency scores).
- Metric evaluation: Hit@1, Hit@3, and Mean Reciprocal Rank (MRR).
- Strict split separation (Train / Validation / Test) without test tuning.
- Graceful handling of systems/cases with missing trace graphs (e.g. Sock Shop).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_mrr(ranks: List[int]) -> float:
    """Compute Mean Reciprocal Rank from 1-based ranks."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def compute_hit_at_k(ranks: List[int], k: int) -> float:
    """Compute Hit@k from 1-based ranks."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


class ServiceRanker:
    """Ranks candidate microservices using anomaly scores and optional graph dependencies."""

    def __init__(
        self,
        graph_weight: float = 0.0,
        normalize_scores: bool = True,
    ):
        """
        Args:
            graph_weight: Weight alpha in: final_score = (1 - alpha) * self_score + alpha * neighbor_score
            normalize_scores: Whether to normalize scores per window/case using min-max.
        """
        self.graph_weight = graph_weight
        self.normalize_scores = normalize_scores
        # Training normalization parameters if fitted
        self.service_scales_: Dict[str, float] = {}

    def fit_normalizer(self, train_scores: pd.DataFrame) -> None:
        """Calculate normalization statistics strictly from training data."""
        grouped = train_scores.groupby("service")["anomaly_score"].std()
        self.service_scales_ = {svc: (val if val > 1e-6 else 1.0) for svc, val in grouped.items()}

    def rank_case(
        self,
        service_scores: Dict[str, float],
        edges: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[str, float]]:
        """Rank services for a single incident or window.

        Args:
            service_scores: Dictionary of service -> anomaly score.
            edges: Optional DataFrame with ['source', 'target', 'span_count'].

        Returns:
            Sorted list of (service_name, final_score) in descending order.
        """
        if not service_scores:
            return []

        services = list(service_scores.keys())
        scores = {s: float(service_scores[s]) for s in services}

        # Apply graph propagation if graph evidence exists and graph_weight > 0
        if self.graph_weight > 0 and edges is not None and not edges.empty:
            # Build directed adjacency / call weights
            # If service A calls B, an anomaly in B can cause failure in A, or A initiated the fault.
            # In trace graph: source (caller) -> target (callee)
            # Propagate anomaly evidence across observed edges
            adj: Dict[str, List[Tuple[str, float]]] = {s: [] for s in services}
            total_spans = edges["span_count"].sum()
            weight_factor = total_spans if total_spans > 0 else 1.0

            for _, row in edges.iterrows():
                src = str(row["source"])
                tgt = str(row["target"])
                cnt = float(row.get("span_count", 1.0))
                norm_w = cnt / weight_factor
                if src in adj and tgt in scores:
                    adj[src].append((tgt, norm_w))
                if tgt in adj and src in scores:
                    adj[tgt].append((src, norm_w))

            combined_scores = {}
            for s in services:
                neighbors = adj.get(s, [])
                if neighbors:
                    neighbor_score = sum(scores[n] * w for n, w in neighbors) / sum(w for _, w in neighbors)
                    combined_scores[s] = (1.0 - self.graph_weight) * scores[s] + self.graph_weight * neighbor_score
                else:
                    # Missing edge fallback: transparently use pure anomaly score
                    combined_scores[s] = scores[s]
            ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Pure score ranking
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked


def evaluate_rankings(
    predictions_df: pd.DataFrame,
    group_cols: List[str] = ["system", "fold", "split"],
) -> pd.DataFrame:
    """Compute Hit@1, Hit@3, and MRR grouped by system, fold, and split.

    predictions_df must contain:
    ['case', 'system', 'fold', 'split', 'root_cause_service', 'predicted_rank']
    where predicted_rank is 1-based rank of the true root cause.
    """
    results = []
    for keys, group in predictions_df.groupby(group_cols):
        ranks = group["predicted_rank"].dropna().astype(int).tolist()
        total = len(group)
        hit1 = compute_hit_at_k(ranks, 1)
        hit3 = compute_hit_at_k(ranks, 3)
        mrr = compute_mrr(ranks)

        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        record["case_count"] = total
        record["Hit@1"] = round(hit1, 4)
        record["Hit@3"] = round(hit3, 4)
        record["MRR"] = round(mrr, 4)
        results.append(record)

    return pd.DataFrame(results)
