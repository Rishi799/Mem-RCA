"""Read RCAEval cases (Hugging Face Parquet layout).

Layout under the raw data directory:
    cases.parquet                         index of all 735 cases
    <suite><system>_<service>_<fault>_<rep>/
        metrics.parquet                   'time' (unix s) + one column per '{service}_{metric}'
        logs.parquet                      timestamp, container_name, message   (RE2/RE3)
        traces.parquet                    time, traceID, spanID, serviceName, ..., parentSpanID
        inject_time.txt                   unix timestamp of the fault injection
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def load_index(raw_dir: str | Path) -> pd.DataFrame:
    """The case index: one row per case with root cause, fault type, inject time, ..."""
    return pd.read_parquet(Path(raw_dir) / "cases.parquet")


def list_cases(raw_dir: str | Path, prefix: str = "") -> list[str]:
    """Names of downloaded case folders, optionally filtered by prefix (e.g. 're2ob')."""
    raw = Path(raw_dir)
    return sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith(prefix))


def parse_case_name(name: str) -> dict:
    """'re2ob_checkoutservice_cpu_1' -> suite, system, service, fault, repetition.

    Service names may contain '-' (e.g. 'ts-auth-service'), so the fault and repetition
    are taken from the right-hand end of the name.
    """
    head, rest = name.split("_", 1)
    service, fault, rep = rest.rsplit("_", 2)
    return {"case": name, "suite": head[:3], "system": head[3:], "service": service,
            "fault": fault, "repetition": int(rep)}


@dataclass
class Case:
    name: str
    root_cause: str
    fault: str
    repetition: int
    inject_time: int
    metrics: pd.DataFrame          # indexed by unix time (s), columns '{service}_{metric}'

    def normal(self) -> pd.DataFrame:
        return self.metrics[self.metrics.index < self.inject_time]

    def faulty(self) -> pd.DataFrame:
        return self.metrics[self.metrics.index >= self.inject_time]


def load_case(raw_dir: str | Path, name: str) -> Case:
    folder = Path(raw_dir) / name
    meta = parse_case_name(name)
    metrics = pd.read_parquet(folder / "metrics.parquet")
    metrics = metrics.set_index("time").sort_index()
    inject_time = int((folder / "inject_time.txt").read_text().strip())
    return Case(name=name, root_cause=meta["service"], fault=meta["fault"],
                repetition=meta["repetition"], inject_time=inject_time, metrics=metrics)


def load_logs(raw_dir: str | Path, name: str) -> pd.DataFrame | None:
    path = Path(raw_dir) / name / "logs.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_traces(raw_dir: str | Path, name: str) -> pd.DataFrame | None:
    path = Path(raw_dir) / name / "traces.parquet"
    return pd.read_parquet(path) if path.exists() else None


def metric_columns_by_service(columns) -> dict[str, list[str]]:
    """Group '{service}_{metric}' columns by service.

    Assumes service names contain no underscore (true for the RCAEval systems); the
    exploration script prints the result so this can be checked per system.
    """
    groups: dict[str, list[str]] = {}
    for col in columns:
        service, _, _metric = col.partition("_")
        groups.setdefault(service, []).append(col)
    return groups
