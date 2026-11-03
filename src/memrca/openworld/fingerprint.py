"""Incident fingerprints: global pooling vs root-anchored (contribution C2).

An incident is summarised from a per-service feature matrix ``F`` with shape
(services, dims). Two feature spaces are used:

* metric-shift profiles: for every service, the robust z-shift of each canonical
  metric type (cpu, mem, diskio, ...) in the first seconds after injection,
  relative to the same case's normal period. The dimensions mean the same thing
  for every service, so fingerprints are comparable across root services.
* learned embeddings: the temporal-graph encoder's per-service embeddings.

Pooling strategies:

* ``global_mean``  uniform mean over services (typical incident summary).
* ``global_max``   per-dimension value with the largest magnitude over services.
* ``anchor``       [root service row || symptom signature], where the symptom
                   signature is the other services' rows weighted by how strongly
                   each was affected. The root is the localiser's prediction (or
                   the true root for an oracle upper bound).
* ``hybrid``       [anchor || global_max]: the anchored view plus the strongest
                   symptom anywhere, for small systems where one service's
                   symptom already summarises the incident well.
"""
from __future__ import annotations

import numpy as np

from memrca.data.loader import Case, metric_columns_by_service

METRIC_TYPES = ["cpu", "mem", "diskio", "socket", "workload", "latency-50", "latency-90", "error"]
# RE1 Online Boutique names two metric types differently from the other suites
METRIC_ALIASES = {"latency": "latency-90", "load": "workload"}
POOLINGS = ["global_mean", "global_max", "anchor", "hybrid"]
# post-injection windows (seconds) of the time-resolved profile: onset, build-up, steady state
PHASES = [(0, 30), (30, 60), (60, 120)]
# feature spaces: 120 s metric shift, time-resolved shift, learned temporal-graph embeddings
FEATURE_KEYS = {"shift": "shift", "phase": "phase", "embed": "embedding"}


def metric_shift_profile(case: Case, services: list[str], horizon: int = 120, clip: float = 1e6,
                         start: int = 0) -> np.ndarray:
    """Signed log10 robust z-shift per (service, metric type); NaN where a service lacks that metric.

    z = (median of the faulty seconds in [start, horizon) after injection - normal median) /
    normal robust scale, using only this case's own pre-injection period as reference. Normal
    periods are often nearly constant, so z spans many orders of magnitude; the value returned
    is sign(z) * log10(1 + |z|) (z clipped at ``clip``), which keeps the ordering between
    metric types that a small clip would erase.
    """
    normal = case.normal()
    faulty = case.faulty()
    faulty = faulty[(faulty.index >= case.inject_time + start) & (faulty.index < case.inject_time + horizon)]
    profile = np.full((len(services), len(METRIC_TYPES)), np.nan, dtype=np.float64)
    if normal.empty or faulty.empty:
        return profile
    groups = metric_columns_by_service(case.metrics.columns)
    type_index = {name: i for i, name in enumerate(METRIC_TYPES)}
    for s, service in enumerate(services):
        for column in groups.get(service, []):
            metric = column.partition("_")[2]
            metric = METRIC_ALIASES.get(metric, metric)
            if metric not in type_index:
                continue
            before = normal[column].astype(float).dropna()
            after = faulty[column].astype(float).dropna()
            if len(before) < 5 or len(after) < 3:
                continue
            q25, q50, q75 = np.percentile(before, [25, 50, 75])
            scale = (q75 - q25) / 1.349
            if not np.isfinite(scale) or scale < 1e-9:
                scale = float(before.std())
            if not np.isfinite(scale) or scale < 1e-9:
                scale = max(abs(q50) * 0.01, 1e-6)
            z = float(np.clip((np.median(after) - q50) / scale, -clip, clip))
            value = float(np.sign(z) * np.log10(1.0 + abs(z)))
            current = profile[s, type_index[metric]]
            if np.isnan(current) or abs(value) > abs(current):
                profile[s, type_index[metric]] = value
    return profile


def root_sources(d: dict) -> dict[str, np.ndarray]:
    """Root used to anchor each incident's fingerprint: the learned localiser ("pred", when the
    incident set has ``root_logits``), the largest metric shift over all services ("shift",
    learning-free) or the true root ("oracle", upper bound)."""
    services = list(d["services"])
    magnitude = np.nan_to_num(np.abs(d["shift"]), nan=0.0).max(axis=2)
    roots = {"oracle": np.array([services.index(r) if r in services else 0 for r in d["root"]]),
             "shift": magnitude.argmax(axis=1)}
    if "root_logits" in d:
        roots["pred"] = d["root_logits"].argmax(axis=1)
    return roots


def fingerprints(d: dict, space: str, how: str, roots: dict[str, np.ndarray]) -> np.ndarray:
    """One fingerprint per incident of an incident set ``d`` (arrays keyed by FEATURE_KEYS).

    ``how`` is a pooling name; root-dependent poolings carry their root source as a suffix,
    e.g. "anchor_pred" or "hybrid_shift".
    """
    feats = d[FEATURE_KEYS[space]]
    magnitude = d["error"] if space == "embed" else np.nan_to_num(np.abs(d["shift"]), nan=0.0).max(axis=2)
    rows = []
    for i in range(len(feats)):
        if how.endswith(tuple(f"_{source}" for source in roots)):
            base, _, source = how.rpartition("_")
            rows.append(pool(feats[i], int(roots[source][i]), magnitude[i], base))
        else:
            rows.append(pool(feats[i], 0, magnitude[i], how))
    return np.stack(rows)


def phase_shift_profile(case: Case, services: list[str]) -> np.ndarray:
    """Time-resolved profile (services, phases * metric types): one shift profile per PHASES window."""
    return np.concatenate([metric_shift_profile(case, services, horizon=end, start=start)
                           for start, end in PHASES], axis=1)


def pool(features: np.ndarray, root: int, magnitude: np.ndarray, how: str) -> np.ndarray:
    """Turn a (services, dims) matrix into one incident fingerprint."""
    filled = np.nan_to_num(features, nan=0.0)
    if how == "global_mean":
        return filled.mean(axis=0)
    if how == "global_max":
        idx = np.abs(filled).argmax(axis=0)
        return filled[idx, np.arange(filled.shape[1])]
    if how == "anchor":
        others = np.ones(len(filled), dtype=bool)
        others[root] = False
        weights = np.nan_to_num(magnitude, nan=0.0).clip(min=0.0) * others
        if weights.sum() <= 0:
            weights = others.astype(float)
        signature = (filled * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
        return np.concatenate([filled[root], signature])
    if how == "hybrid":
        return np.concatenate([pool(features, root, magnitude, "anchor"),
                               pool(features, root, magnitude, "global_max")])
    raise ValueError(f"Unknown pooling {how!r}; choose from {POOLINGS}")
