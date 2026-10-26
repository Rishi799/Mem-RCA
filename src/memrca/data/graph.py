"""Build observed service-to-service call edges from RCAEval traces."""
from __future__ import annotations

import pandas as pd


EDGE_COLUMNS = ["source", "target", "span_count"]


def service_edges_from_traces(
    traces: pd.DataFrame,
    metric_services: set[str] | None = None,
) -> pd.DataFrame:
    """Return observed parent-service -> child-service edges and span counts.

    Edges represent invocation structure found in traces, not proof of causality.
    When metric service names are supplied, a trace name ending in ``service`` is
    shortened only if that exact shortened name exists among the metric services
    (for example, Online Boutique's ``frontendservice`` maps to ``frontend``).
    Same-service parent/child spans are omitted from the inter-service graph.
    """
    required = {"traceID", "spanID", "parentSpanID", "serviceName"}
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError(f"Trace table is missing required columns: {missing}")

    spans = traces[["traceID", "spanID", "parentSpanID", "serviceName"]].copy()
    spans = spans.dropna(subset=["traceID", "spanID", "serviceName"])
    spans["traceID"] = spans["traceID"].astype(str)
    spans["spanID"] = spans["spanID"].astype(str)
    spans["parentSpanID"] = spans["parentSpanID"].fillna("").astype(str)
    spans["serviceName"] = spans["serviceName"].astype(str)
    spans = spans[spans["parentSpanID"].ne("")]

    parents = spans[["traceID", "spanID", "serviceName"]].rename(
        columns={"spanID": "parentSpanID", "serviceName": "parent_service"}
    )
    joined = spans.merge(
        parents,
        on=["traceID", "parentSpanID"],
        how="inner",
        validate="many_to_one",
    )

    known_metrics = metric_services or set()

    def align_to_metrics(name: str) -> str:
        suffix = "service"
        shortened = name[: -len(suffix)] if name.endswith(suffix) else name
        return shortened if shortened != name and shortened in known_metrics else name

    joined["source"] = joined["parent_service"].map(align_to_metrics)
    joined["target"] = joined["serviceName"].map(align_to_metrics)
    joined = joined[joined["source"].ne(joined["target"])]

    if joined.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    return (
        joined.groupby(["source", "target"], as_index=False)
        .size()
        .rename(columns={"size": "span_count"})
        .sort_values(["span_count", "source", "target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
