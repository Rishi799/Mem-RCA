"""Inductive topological ranking and features for microservice RCA.

Breaks the benchmark shortcut where neural localisers memorize fixed service identities
rather than learning invariant propagation physics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_structural_node_features(
    services: List[str],
    edges: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute inductive structural topological features independent of service identity.

    Features computed per service:
    - in_degree: number of services calling this service.
    - out_degree: number of services called by this service.
    - call_ratio: out_degree / (in_degree + out_degree + 1e-5) (root services typically initiate calls).
    - total_spans_in: weighted incoming trace span count.
    - total_spans_out: weighted outgoing trace span count.
    - is_leaf: 1.0 if out_degree == 0 else 0.0
    - is_entrypoint: 1.0 if in_degree == 0 else 0.0
    """
    feats = {
        s: {
            "in_degree": 0.0,
            "out_degree": 0.0,
            "call_ratio": 0.5,
            "total_spans_in": 0.0,
            "total_spans_out": 0.0,
            "is_leaf": 1.0,
            "is_entrypoint": 1.0,
        }
        for s in services
    }

    if edges is None or edges.empty:
        return feats

    for _, row in edges.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        spans = float(row.get("span_count", 1.0))

        if src in feats:
            feats[src]["out_degree"] += 1.0
            feats[src]["total_spans_out"] += spans
            feats[src]["is_leaf"] = 0.0

        if tgt in feats:
            feats[tgt]["in_degree"] += 1.0
            feats[tgt]["total_spans_in"] += spans
            feats[tgt]["is_entrypoint"] = 0.0

    for s, f in feats.items():
        total_deg = f["in_degree"] + f["out_degree"]
        if total_deg > 0:
            f["call_ratio"] = f["out_degree"] / total_deg

    return feats


class InductiveServiceRanker:
    """Ranks services using inductive topological priors and anomaly signal deviations.

    Unlike transductive models that overfit to fixed service labels (memorizing the 5 roots),
    this ranker evaluates:
    1. Local Anomaly Magnitude (z-score / forecast residual).
    2. Topological Directionality (incoming vs outgoing propagation ratio).
    3. Graph Neighborhood Upstream Propagation Evidence.
    """

    def __init__(
        self,
        graph_weight: float = 0.25,
        topological_prior_weight: float = 0.15,
    ):
        self.graph_weight = graph_weight
        self.topological_prior_weight = topological_prior_weight

    def rank_case(
        self,
        service_scores: Dict[str, float],
        edges: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[str, float]]:
        if not service_scores:
            return []

        services = list(service_scores.keys())
        scores = {s: float(service_scores[s]) for s in services}
        structural_feats = extract_structural_node_features(services, edges)

        # Baseline normalized anomaly score
        max_score = max(scores.values()) if scores and max(scores.values()) > 1e-6 else 1.0
        norm_scores = {s: scores[s] / max_score for s in services}

        # Neighborhood evidence propagation
        if self.graph_weight > 0 and edges is not None and not edges.empty:
            callees: Dict[str, List[Tuple[str, float]]] = {s: [] for s in services}
            for _, row in edges.iterrows():
                src = str(row["source"])
                tgt = str(row["target"])
                cnt = float(row.get("span_count", 1.0))
                if src in callees and tgt in norm_scores:
                    callees[src].append((tgt, cnt))

            final_scores = {}
            for s in services:
                # Base score component
                s_base = norm_scores[s]

                # Propagated downstream stress: root causes frequently push errors/delays down into callees
                callee_list = callees.get(s, [])
                if callee_list:
                    tot_w = sum(w for _, w in callee_list)
                    downstream_score = sum(norm_scores[c] * w for c, w in callee_list) / (tot_w if tot_w > 0 else 1.0)
                else:
                    downstream_score = s_base

                # Inductive topological prior: root causes are rarely pure passive leaves
                topo_bias = structural_feats[s]["call_ratio"]

                score = (
                    (1.0 - self.graph_weight - self.topological_prior_weight) * s_base
                    + self.graph_weight * downstream_score
                    + self.topological_prior_weight * (s_base * (0.8 + 0.4 * topo_bias))
                )
                final_scores[s] = float(score)
        else:
            final_scores = norm_scores

        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
