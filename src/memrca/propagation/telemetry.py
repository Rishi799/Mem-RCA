"""Second-by-second service health for replaying a recorded incident as live monitoring.

The console's live mode streams this to look like a monitoring dashboard. It uses the
same causal anomaly trace, normal-period thresholds and onset rule as the time-to-impact
dataset, so a service turns anomalous in the replay exactly when the propagation labels
say it was affected, and the first confirmed onset is the moment the incident is detected.
Health values are the trace divided by the service's threshold: 1 is the alert line.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from memrca.data.loader import Case, load_case, metric_columns_by_service
from memrca.propagation.dataset import HORIZON, PERSISTENCE
from memrca.propagation.onsets import anomaly_trace, detect_onsets, thresholds


def replay_telemetry(raw_dir: str | Path, case_name: str, **kwargs) -> dict:
    return case_telemetry(load_case(raw_dir, case_name), **kwargs)


def case_telemetry(case: Case, pre: int = 150, post: int = 700, step: int = 2) -> dict:
    """Health of every service from ``pre`` seconds before the injection to ``post`` after it.

    ``confirmed`` is the second (relative to injection) at which each service's onset is
    confirmed, i.e. onset + persistence; ``detect_s`` is the earliest of them (None if the
    detector never fires). Nothing names the root service or the fault type.
    """
    services = sorted(metric_columns_by_service(case.metrics.columns))
    trace = anomaly_trace(case, services)
    threshold = thresholds(trace, int(trace.index.min()), 0)
    onsets = detect_onsets(trace, threshold, reference=0, horizon=HORIZON + 120, persistence=PERSISTENCE)
    ratio = trace.div(threshold).loc[-pre:post]
    ratio = ratio[ratio.index % step == 0].round(2)
    confirmed = onsets.reindex(services) + PERSISTENCE
    detect = float(confirmed.min()) if confirmed.notna().any() else None
    values = ratio.to_numpy(dtype=float).T
    return {
        "services": services, "step_s": step, "t": ratio.index.astype(int).tolist(),
        "ratio": [[None if np.isnan(v) else float(v) for v in row] for row in values],
        "confirmed": [None if np.isnan(v) else float(v) for v in confirmed.to_numpy(dtype=float)],
        "detect_s": detect, "inject_unix": int(case.inject_time), "end_s": int(ratio.index.max()),
    }
