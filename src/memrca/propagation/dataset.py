"""Incidents as time-to-impact samples for propagation forecasting.

The clock starts at t0, the root service's onset (injection time if the root is never
detected). Time after t0 is cut into bins of ``BIN`` seconds up to ``HORIZON``. For every
bin b and service, features describe only what is observable at t0 + b*BIN:

static  (per service)  call-graph hops from the root up the callers, down the callees and
                       ignoring direction (+ missing flags), whether the service is in the
                       trace graph, whether it is the root, the fault type (or "unknown")
dynamic (per bin)      the service's own anomaly level over the last BIN seconds relative to
                       its threshold, the share of those seconds above threshold, the shares
                       of its callers / callees already affected, the share of all services
                       affected, and the elapsed time

A service is "affected" once its onset has been confirmed (onset + persistence seconds).
The label is the bin in which a not-yet-affected service's onset falls (censored when
there is none within the horizon). Features are service-agnostic, so one model serves
every system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from memrca.data.loader import Case, metric_columns_by_service
from memrca.propagation.onsets import anomaly_trace, detect_onsets, graph_distances, thresholds

BIN = 10
HORIZON = 300
N_BINS = HORIZON // BIN
PERSISTENCE = 5
FAULTS = ["cpu", "mem", "disk", "delay", "loss", "socket", "unknown"]
STATIC_NAMES = ["hop_up", "hop_down", "hop_any", "no_up", "no_down", "no_any", "in_graph", "is_root"] + \
               [f"fault_{f}" for f in FAULTS]
DYNAMIC_NAMES = ["own_level", "own_above", "callers_hit", "callees_hit", "all_hit", "elapsed"]
ALREADY, CENSORED = -2, -1


@dataclass
class Incident:
    case: str
    system: str
    fault: str
    root: str
    repetition: int
    suite: str
    services: list[str]
    edge_index: np.ndarray      # (2, E) caller -> callee, service indices
    static: np.ndarray          # (N, len(STATIC_NAMES))
    dynamic: np.ndarray         # (N_BINS, N, len(DYNAMIC_NAMES))
    onset_bin: np.ndarray       # (N,) bin of onset after t0; ALREADY or CENSORED
    delay: np.ndarray           # (N,) seconds after t0; NaN when censored
    t0: float
    trace: pd.DataFrame | None = None  # anomaly trace, kept for replay
    threshold: pd.Series | None = None


def _hops(values: pd.Series, services: list[str]) -> np.ndarray:
    return np.array([values.get(s, np.nan) for s in services], dtype=float)


def build_incident(case: Case, meta: dict, edges: pd.DataFrame, keep_trace: bool = False,
                   fault_known: bool = True) -> Incident:
    services = sorted(metric_columns_by_service(case.metrics.columns))
    index = {s: i for i, s in enumerate(services)}
    trace = anomaly_trace(case, services)
    threshold = thresholds(trace, int(trace.index.min()), 0)
    onsets = detect_onsets(trace, threshold, reference=0, horizon=HORIZON + 120, persistence=PERSISTENCE)
    root = meta["root"]
    root_onset = onsets.get(root, np.nan)
    t0 = 0.0 if np.isnan(root_onset) else float(root_onset)

    pairs = edges[["source", "target"]].drop_duplicates()
    pairs = pairs[pairs["source"].isin(index) & pairs["target"].isin(index)]
    edge_index = np.array([[index[s] for s in pairs["source"]], [index[t] for t in pairs["target"]]], dtype=np.int64) \
        if len(pairs) else np.empty((2, 0), dtype=np.int64)
    dist = graph_distances(pairs, root)
    in_graph = set(pairs["source"]) | set(pairs["target"])
    up, down, both = (_hops(dist[k], services) for k in ("up", "down", "any"))
    fault = meta["fault"] if fault_known and meta["fault"] in FAULTS else "unknown"
    static = np.column_stack([
        np.nan_to_num(np.minimum(up, 6) / 3), np.nan_to_num(np.minimum(down, 6) / 3), np.nan_to_num(np.minimum(both, 6) / 3),
        np.isnan(up), np.isnan(down), np.isnan(both),
        [s in in_graph for s in services], [s == root for s in services],
        *[[fault == f] * len(services) for f in FAULTS],
    ]).astype(np.float32)

    delay = onsets.reindex(services).to_numpy(dtype=float) - t0
    onset_bin = np.full(len(services), CENSORED, dtype=np.int64)
    within = ~np.isnan(delay) & (delay < HORIZON)
    onset_bin[within] = np.floor(np.maximum(delay[within], 0) / BIN).astype(np.int64)
    onset_bin[within & (delay < 0)] = ALREADY
    if root in index:
        onset_bin[index[root]] = ALREADY
    delay[~within] = np.nan

    # dynamic features, observable at the start of every bin
    values = trace.reindex(columns=services).to_numpy(dtype=float)
    times = trace.index.to_numpy()
    confirmed = onsets.reindex(services).to_numpy(dtype=float) + PERSISTENCE  # time the onset becomes known
    callers = [[] for _ in services]
    callees = [[] for _ in services]
    for s, t in edge_index.T:
        callees[s].append(t)
        callers[t].append(s)
    dynamic = np.zeros((N_BINS, len(services), len(DYNAMIC_NAMES)), dtype=np.float32)
    thr = threshold.reindex(services).to_numpy(dtype=float)
    for b in range(N_BINS):
        now = t0 + b * BIN
        recent = (times >= now - BIN) & (times < now)
        window = values[recent] if recent.any() else np.full((1, len(services)), np.nan)
        level = np.nan_to_num(np.nanmax(window, axis=0) - thr, nan=0.0) if np.isfinite(window).any() else np.zeros(len(services))
        above = np.nan_to_num(np.nanmean(window >= thr, axis=0), nan=0.0)
        hit = np.nan_to_num(confirmed <= now, nan=0.0).astype(float)
        dynamic[b, :, 0] = np.clip(level, -2, 3)
        dynamic[b, :, 1] = above
        dynamic[b, :, 2] = [hit[c].mean() if c else 0.0 for c in callers]
        dynamic[b, :, 3] = [hit[c].mean() if c else 0.0 for c in callees]
        dynamic[b, :, 4] = hit.mean()
        dynamic[b, :, 5] = b / N_BINS
    return Incident(case=meta["case"], system=meta["system"], fault=meta["fault"], root=root,
                    repetition=int(meta["repetition"]), suite=meta["suite"], services=services,
                    edge_index=edge_index, static=static, dynamic=dynamic, onset_bin=onset_bin, delay=delay,
                    t0=t0, trace=trace if keep_trace else None, threshold=threshold if keep_trace else None)
