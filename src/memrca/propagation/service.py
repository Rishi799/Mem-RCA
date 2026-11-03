"""Time-to-impact forecasts for one recorded incident, as served to the console.

The forecaster rebuilds the incident's features, runs PropagationNet from every bin
(each forecast uses only what was observable at that moment) and returns, per service,
the probability of being hit within the horizon with a 10-50-90 % onset band, next to
the onsets that actually happened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from memrca.data.loader import load_case
from memrca.propagation.architecture import documented_edges
from memrca.propagation.dataset import ALREADY, BIN, DYNAMIC_NAMES, N_BINS, PERSISTENCE, STATIC_NAMES, build_incident
from memrca.propagation.model import PropagationNet, forecast


def _quantile(cum: np.ndarray, level: float, start: int) -> list:
    reached = cum >= level
    first = np.argmax(reached, axis=0)
    return [float((start + f + 0.5) * BIN) if reached[:, i].any() else None for i, f in enumerate(first)]


class PropagationForecaster:
    def __init__(self, model_path: str | Path, edges_csv: str | Path, raw_dir: str | Path) -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = PropagationNet(len(STATIC_NAMES), len(DYNAMIC_NAMES))
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()
        self.edges = pd.read_csv(edges_csv)[["system", "source", "target"]].drop_duplicates()
        self.raw = Path(raw_dir)
        self._cache: dict = {}

    @staticmethod
    def _documented(system: str, services: list[str], edge_index: np.ndarray) -> list[list[int]]:
        """Documented links not already in the trace graph, as service-index pairs (drawing only)."""
        index = {s: i for i, s in enumerate(services)}
        traced = {frozenset(map(int, pair)) for pair in edge_index.T}
        return [[index[a], index[b]] for a, b in documented_edges(system, services)
                if frozenset((index[a], index[b])) not in traced]

    def incident(self, case: str, system: str, root: str, fault: str | None) -> dict:
        key = (case, root, fault)
        if key in self._cache:
            return self._cache[key]
        meta = {"case": case, "system": system, "fault": fault or "unknown", "root": root, "repetition": 0, "suite": ""}
        inc = build_incident(load_case(self.raw, case), meta, self.edges[self.edges["system"].eq(system)],
                             fault_known=fault is not None)
        elapsed = DYNAMIC_NAMES.index("elapsed")
        forecasts = []
        for start in range(N_BINS):
            cum = forecast(self.model, inc.static, inc.dynamic[start], inc.edge_index, start, N_BINS, elapsed)
            forecasts.append({"start_s": start * BIN, "p_hit": np.round(cum[-1], 3).tolist(),
                              "q10": _quantile(cum, 0.1, start), "q50": _quantile(cum, 0.5, start),
                              "q90": _quantile(cum, 0.9, start)})
        hop = lambda col: [None if np.isnan(v) else int(round(v)) for v in  # noqa: E731
                           np.where(inc.static[:, STATIC_NAMES.index(f"no_{col}")] > 0, np.nan,
                                    inc.static[:, STATIC_NAMES.index(f"hop_{col}")] * 3)]
        actual = []  # seconds after t0; 0 for the root and services hit before t0; None if not hit
        for d, b, s in zip(inc.delay, inc.onset_bin, inc.services):
            if s == root or b == ALREADY:
                actual.append(0.0)
            else:
                actual.append(None if np.isnan(d) else float(d))
        result = {
            "case": case, "root": root, "fault_used": meta["fault"] if fault else "unknown", "t0": inc.t0,
            "bin_s": BIN, "horizon_s": N_BINS * BIN, "confirm_s": PERSISTENCE,
            "services": inc.services, "edges": inc.edge_index.T.tolist(),
            "doc_edges": self._documented(system, inc.services, inc.edge_index),
            "in_graph": (inc.static[:, STATIC_NAMES.index("in_graph")] > 0).tolist(),
            "hop_up": hop("up"), "hop_down": hop("down"), "hop_any": hop("any"),
            "already": [bool(b == ALREADY and s != root) for b, s in zip(inc.onset_bin, inc.services)],
            "actual_s": actual, "forecasts": forecasts,
        }
        self._cache[key] = result
        return result
