"""Per-service anomaly onset at 1-second resolution, and call-graph distances from the root.

For every metric of a service, the robust z-score against the incident's own normal
period is taken on a log scale, log10(1 + |z|); the service's anomaly trace is the
largest value over its metrics, smoothed by a trailing (causal) median. A service's
onset is the first second after the reference time at which the trace stays above
its threshold for ``persistence`` consecutive seconds. Thresholds come from the
normal period only: the largest smoothed value seen there plus a margin, with a floor.
Everything is causal, so the same detector can run online during a replay.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from memrca.data.loader import Case, metric_columns_by_service

CLIP_Z = 1e4


def anomaly_trace(case: Case, services: list[str], smooth: int = 5) -> pd.DataFrame:
    """(seconds relative to injection) x services; NaN for services without metrics."""
    metrics = case.metrics.sort_index()
    normal = metrics[metrics.index < case.inject_time]
    groups = metric_columns_by_service(metrics.columns)
    out = {}
    for service in services:
        columns = groups.get(service, [])
        if not columns or len(normal) < 10:
            out[service] = np.full(len(metrics), np.nan)
            continue
        values = metrics[columns].astype(float)
        base = normal[columns].astype(float)
        q25, q50, q75 = base.quantile(0.25), base.quantile(0.5), base.quantile(0.75)
        scale = (q75 - q25) / 1.349
        scale = scale.where(scale > 1e-9, base.std())
        scale = scale.where(scale > 1e-9, (q50.abs() * 0.01).clip(lower=1e-6))
        z = ((values - q50) / scale).clip(-CLIP_Z, CLIP_Z)
        score = np.log10(1.0 + z.abs()).max(axis=1, skipna=True)
        out[service] = score.rolling(smooth, min_periods=1).median().to_numpy()
    trace = pd.DataFrame(out, index=(metrics.index - case.inject_time).astype(int))
    return trace[~trace.index.duplicated()]


def thresholds(trace: pd.DataFrame, start: int, end: int, margin: float = 0.3, floor: float = 0.8) -> pd.Series:
    """Per-service threshold from the calibration window [start, end) (seconds rel. to injection)."""
    window = trace[(trace.index >= start) & (trace.index < end)]
    return (window.max() + margin).clip(lower=floor)


def detect_onsets(trace: pd.DataFrame, threshold: pd.Series, reference: int = 0,
                  horizon: int = 300, persistence: int = 5) -> pd.Series:
    """First second t in [reference, reference + horizon) where the trace stays above threshold
    for ``persistence`` seconds, reported relative to ``reference``; NaN if none."""
    window = trace[(trace.index >= reference) & (trace.index < reference + horizon + persistence)]
    above = window.ge(threshold, axis=1)
    onsets = {}
    for service in trace.columns:
        flags = above[service].to_numpy()
        run, onset = 0, np.nan
        for i, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run == persistence:
                start = window.index[i - persistence + 1] - reference
                onset = float(start) if start < horizon else np.nan
                break
        onsets[service] = onset
    return pd.Series(onsets)


def graph_distances(edges: pd.DataFrame, root: str) -> pd.DataFrame:
    """Hop distances from ``root`` over caller -> callee edges.

    ``up``: hops along callers (services that call the root, their callers, ...);
    ``down``: hops along callees; ``any``: ignoring direction. Missing = unreachable.
    """
    callers, callees = {}, {}
    for source, target in edges[["source", "target"]].itertuples(index=False):
        callees.setdefault(source, set()).add(target)
        callers.setdefault(target, set()).add(source)

    def bfs(neighbours) -> dict[str, int]:
        seen, queue = {root: 0}, deque([root])
        while queue:
            node = queue.popleft()
            for nxt in neighbours(node):
                if nxt not in seen:
                    seen[nxt] = seen[node] + 1
                    queue.append(nxt)
        return seen

    up = bfs(lambda n: callers.get(n, ()))
    down = bfs(lambda n: callees.get(n, ()))
    both = bfs(lambda n: callers.get(n, set()) | callees.get(n, set()))
    return pd.DataFrame({"up": pd.Series(up, dtype=float), "down": pd.Series(down, dtype=float),
                         "any": pd.Series(both, dtype=float)})
