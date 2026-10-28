"""Build and validate observed service-to-service call edges from RCAEval traces.

Note:
- Edges represent observed invocation/dependency evidence within distributed traces,
  not conclusive proof of fault causality.
- Sock Shop ('ss') lacks trace instrumentation in RCAEval RE2; graph evidence must
  gracefully handle empty or missing trace cases.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd

EDGE_COLUMNS = ["source", "target", "span_count"]


def normalize_service_name(name: str, metric_services: Iterable[str] | None = None) -> str:
    """Normalize trace service name to align with metric service identifiers.

    For example, in Online Boutique, 'frontendservice' in traces corresponds to
    'frontend' in metric telemetry when metric_services contains 'frontend'.
    Matching is case-insensitive when an exact metric-service name is not found.
    """
    clean_name = str(name).strip()
    if not metric_services:
        return clean_name

    known = set(metric_services)
    if clean_name in known:
        return clean_name

    # Check for common suffix stripping (e.g. 'frontendservice' -> 'frontend')
    suffix = "service"
    if clean_name.lower().endswith(suffix):
        shortened = clean_name[: -len(suffix)]
        if shortened in known:
            return shortened

    # Case-insensitive match check
    lower_map = {s.lower(): s for s in known}
    if clean_name.lower() in lower_map:
        return lower_map[clean_name.lower()]

    return clean_name


def extract_trace_graph_diagnostics(
    traces: pd.DataFrame,
    metric_services: Iterable[str] | None = None,
    allow_self_edges: bool = False,
) -> dict:
    """Analyze trace data and extract graph edges alongside quality diagnostics.

    Checks:
    - Missing columns / empty traces
    - Total raw spans, root spans (missing/empty parentSpanID), and internal spans
    - Spans whose parentSpanID does not match any spanID within the same traceID
    - Same-service self-edges
    - Duplicate edge aggregation counts
    """
    if traces is None or traces.empty:
        return {
            "edges": pd.DataFrame(columns=EDGE_COLUMNS),
            "total_spans": 0,
            "root_spans": 0,
            "orphan_spans": 0,
            "matched_parent_spans": 0,
            "self_edges_omitted": 0,
            "unique_edges": 0,
        }

    required = {"traceID", "spanID", "parentSpanID", "serviceName"}
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError(f"Trace table is missing required columns: {missing}")

    spans = traces[["traceID", "spanID", "parentSpanID", "serviceName"]].copy()
    spans = spans.dropna(subset=["traceID", "spanID", "serviceName"])
    total_spans = len(spans)

    spans["traceID"] = spans["traceID"].astype(str)
    spans["spanID"] = spans["spanID"].astype(str)
    spans["parentSpanID"] = spans["parentSpanID"].fillna("").astype(str).str.strip()
    spans["serviceName"] = spans["serviceName"].astype(str).str.strip()

    # RCAEval contains repeated span rows. Keep the first exported record for
    # each trace/span so duplicates do not inflate parent-child edge counts.
    spans = spans.drop_duplicates(["traceID", "spanID"], keep="first")

    # Normalize service names
    spans["service_norm"] = spans["serviceName"].apply(lambda s: normalize_service_name(s, metric_services))

    # Identify root spans (empty parentSpanID or '0')
    is_root = spans["parentSpanID"].isin({"", "0", "None", "nan", "null"})
    root_spans_count = int(is_root.sum())
    non_root_spans = spans[~is_root].copy()

    # Match parents within the same traceID
    parents = spans[["traceID", "spanID", "service_norm"]].drop_duplicates(subset=["traceID", "spanID"])
    parents = parents.rename(columns={"spanID": "parentSpanID", "service_norm": "parent_service"})

    joined = non_root_spans.merge(
        parents,
        on=["traceID", "parentSpanID"],
        how="left",
    )

    matched = joined[joined["parent_service"].notna()].copy()
    orphan_count = int(joined["parent_service"].isna().sum())
    matched_count = len(matched)

    matched["source"] = matched["parent_service"]
    matched["target"] = matched["service_norm"]

    is_self = matched["source"] == matched["target"]
    self_edges_count = int(is_self.sum())

    if not allow_self_edges:
        matched = matched[~is_self]

    if matched.empty:
        edges_df = pd.DataFrame(columns=EDGE_COLUMNS)
    else:
        edges_df = (
            matched.groupby(["source", "target"], as_index=False)
            .size()
            .rename(columns={"size": "span_count"})
            .sort_values(["span_count", "source", "target"], ascending=[False, True, True])
            .reset_index(drop=True)
        )

    return {
        "edges": edges_df,
        "total_spans": total_spans,
        "root_spans": root_spans_count,
        "orphan_spans": orphan_count,
        "matched_parent_spans": matched_count,
        "self_edges_omitted": self_edges_count,
        "unique_edges": len(edges_df),
    }


def service_edges_from_traces(
    traces: pd.DataFrame,
    metric_services: Iterable[str] | None = None,
    allow_self_edges: bool = False,
) -> pd.DataFrame:
    """Return observed parent-service -> child-service edges and span counts.

    Edges represent invocation structure found in traces, not proof of causality.
    """
    diagnostics = extract_trace_graph_diagnostics(
        traces=traces,
        metric_services=metric_services,
        allow_self_edges=allow_self_edges,
    )
    return diagnostics["edges"]
