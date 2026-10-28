"""Convert one regularly sampled RCAEval case into labeled metric windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from memrca.data.loader import Case


@dataclass
class WindowedCase:
    """Window features and timing labels for a single incident.

    ``is_anomaly`` labels windows wholly after fault injection. Windows that
    straddle the injection timestamp are omitted to avoid ambiguous labels.
    Feature values are left as-is (including any NaNs) for the downstream
    training pipeline to impute using training data only.
    """

    case_name: str
    features: pd.DataFrame
    metadata: pd.DataFrame


def make_windows(case: Case, window_seconds: int, stride_seconds: int | None = None) -> WindowedCase:
    """Aggregate metrics into non-overlapping or strided fixed-width windows.

    RCAEval metric samples are one second apart. The function checks this
    assumption instead of silently generating misleading windows from a case
    with irregular or missing timestamps. ``stride_seconds`` defaults to the
    window width; set it lower for overlapping windows.
    """
    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")

    times = case.metrics.index.to_numpy()
    if len(times) < 2 or not np.all(np.diff(times) == 1):
        raise ValueError(f"{case.name}: expected complete one-second metric sampling")

    feature_rows: list[pd.Series] = []
    records: list[dict] = []
    width = int(window_seconds)
    stride = int(stride_seconds)
    for start in range(0, len(case.metrics) - width + 1, stride):
        stop = start + width
        first_time, last_time = int(times[start]), int(times[stop - 1])
        # Exclude a boundary-crossing window so injection labels stay precise.
        if first_time < case.inject_time <= last_time:
            continue

        chunk = case.metrics.iloc[start:stop]
        feature_rows.append(chunk.mean(axis=0).rename(f"{first_time}"))
        records.append({
            "window_start": first_time,
            "window_end": last_time,
            "seconds_from_injection": first_time - case.inject_time,
            "is_anomaly": first_time >= case.inject_time,
        })

    features = pd.DataFrame(feature_rows).reset_index(drop=True)
    metadata = pd.DataFrame(records)
    return WindowedCase(case_name=case.name, features=features, metadata=metadata)
