"""Case-aligned window data for the temporal graph model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from memrca.data.loader import Case, load_case, metric_columns_by_service


@dataclass
class ServiceScaler:
    feature_names: list[str]
    center: np.ndarray
    scale: np.ndarray


@dataclass
class GraphWindow:
    case: str
    split: str
    fault: str
    fault_status: str
    root_service: str
    root_index: int
    window_start: int
    target_time: int
    seconds_from_injection: int
    is_anomaly: bool
    values: list[np.ndarray]
    masks: list[np.ndarray]
    targets: list[np.ndarray]
    target_masks: list[np.ndarray]


@dataclass
class TemporalGraphData:
    services: list[str]
    scalers: dict[str, ServiceScaler]
    windows: dict[str, list[GraphWindow]]
    edge_index: np.ndarray
    feature_sizes: list[int]


def fit_service_scalers(train_cases: list[Case], min_coverage: float = 0.8) -> tuple[list[str], dict[str, ServiceScaler]]:
    """Fit per-service robust transforms on normal rows from training cases only."""
    groups = [metric_columns_by_service(case.metrics.columns) for case in train_cases]
    services = sorted(set().union(*(set(group) for group in groups)))
    scalers: dict[str, ServiceScaler] = {}
    for service in services:
        candidates = sorted(set().union(*(set(group.get(service, [])) for group in groups)))
        frames = [case.normal().reindex(columns=candidates) for case in train_cases]
        raw = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        if raw.empty or not raw.notna().any().any():
            raise ValueError(f"No observed normal training metrics for service {service}")
        coverage = raw.notna().mean(axis=0)
        features = coverage[coverage >= min_coverage].index.tolist()
        if not features:
            features = [str(coverage.idxmax())]
        raw = raw[features]
        transformed = np.sign(raw) * np.log1p(np.abs(raw))
        center = transformed.median(axis=0).to_numpy(dtype=np.float32)
        q25 = transformed.quantile(0.25, axis=0).to_numpy(dtype=np.float32)
        q75 = transformed.quantile(0.75, axis=0).to_numpy(dtype=np.float32)
        scale = ((q75 - q25) / 1.349).astype(np.float32)
        filled = transformed.fillna(pd.Series(center, index=features)).to_numpy(dtype=np.float32)
        fallback = filled.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(scale) | (scale < 1e-6)
        scale[bad] = fallback[bad]
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        scalers[service] = ServiceScaler(features, center, scale)
    return services, scalers


def _transform_case(case: Case, scalers: dict[str, ServiceScaler]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the complete one-second time grid and normalized node series/masks."""
    times = case.metrics.index.to_numpy(dtype=np.int64)
    if not len(times):
        return times, {}, {}
    grid = np.arange(int(times.min()), int(times.max()) + 1, dtype=np.int64)
    frame = case.metrics.reindex(grid).replace([np.inf, -np.inf], np.nan)
    values_by_service: dict[str, np.ndarray] = {}
    masks_by_service: dict[str, np.ndarray] = {}
    for service, scaler in scalers.items():
        raw = frame.reindex(columns=scaler.feature_names).to_numpy(dtype=np.float32)
        observed = np.isfinite(raw)
        safe = np.where(observed, raw, 0.0)
        transformed = np.sign(safe) * np.log1p(np.abs(safe))
        normalized = np.clip((transformed - scaler.center) / scaler.scale, -10.0, 10.0)
        normalized[~observed] = 0.0
        values_by_service[service] = normalized.astype(np.float32, copy=False)
        masks_by_service[service] = observed.astype(np.float32, copy=False)
    return grid, values_by_service, masks_by_service


def build_temporal_graph_data(
    raw_dir: str | Path,
    manifest_path: str | Path,
    edge_path: str | Path,
    fold: str,
    system: str,
    window_length: int = 30,
    stride: int = 10,
    min_coverage: float = 0.8,
) -> TemporalGraphData:
    """Prepare all case windows, scalers, and training-only graph topology."""
    if window_length < 2 or stride < 1:
        raise ValueError("window_length must be at least 2 and stride must be positive")
    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest["fold"].astype(str).eq(fold)
        & manifest["system"].astype(str).eq(system)
    ].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows for fold={fold}, system={system}")
    train_names = selected.loc[selected["split"].eq("train"), "case"].astype(str).tolist()
    train_cases = [load_case(raw_dir, name) for name in train_names]
    services, scalers = fit_service_scalers(train_cases, min_coverage)
    service_indices = {service: i for i, service in enumerate(services)}
    windows: dict[str, list[GraphWindow]] = {name: [] for name in ("train", "validation", "test")}

    for row in selected.itertuples(index=False):
        case = load_case(raw_dir, str(row.case))
        grid, normalized, observed = _transform_case(case, scalers)
        if len(grid) < window_length:
            continue
        root_service = str(case.root_cause)
        root_index = service_indices.get(root_service, -1)
        for start in range(0, len(grid) - window_length + 1, stride):
            target_idx = start + window_length - 1
            window_start = int(grid[start])
            target_time = int(grid[target_idx])
            if window_start < case.inject_time <= target_time:
                continue
            node_values: list[np.ndarray] = []
            node_masks: list[np.ndarray] = []
            target_values: list[np.ndarray] = []
            target_masks: list[np.ndarray] = []
            for service in services:
                values = normalized[service]
                mask = observed[service]
                node_values.append(values[start:target_idx])
                node_masks.append(mask[start:target_idx])
                target_values.append(values[target_idx])
                target_masks.append(mask[target_idx])
            windows[str(row.split)].append(GraphWindow(
                case=str(row.case), split=str(row.split), fault=case.fault,
                fault_status=str(getattr(row, "fault_status", "unknown")),
                root_service=root_service, root_index=root_index,
                window_start=window_start, target_time=target_time,
                seconds_from_injection=window_start - int(case.inject_time),
                is_anomaly=window_start >= int(case.inject_time), values=node_values,
                masks=node_masks, targets=target_values, target_masks=target_masks,
            ))

    if not windows["train"] or not windows["validation"]:
        raise ValueError("Training or validation windows are empty")

    edges_file = Path(edge_path)
    if edges_file.exists():
        edges = pd.read_csv(edges_file)
        train_set = set(train_names)
        edges = edges[
            edges["system"].astype(str).eq(system)
            & edges["case"].astype(str).isin(train_set)
            & edges["source"].astype(str).isin(service_indices)
            & edges["target"].astype(str).isin(service_indices)
        ]
        pairs = edges[["source", "target"]].drop_duplicates()
        edge_index = np.asarray([
            [service_indices[str(source)] for source in pairs["source"]],
            [service_indices[str(target)] for target in pairs["target"]],
        ], dtype=np.int64) if len(pairs) else np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return TemporalGraphData(
        services=services, scalers=scalers, windows=windows, edge_index=edge_index,
        feature_sizes=[len(scalers[service].feature_names) for service in services],
    )
