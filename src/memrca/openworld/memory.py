"""Incident memory: content-addressed failure prototypes with open-set rejection.

Each known failure type is stored as a prototype (mean of standardised,
L2-normalised training fingerprints). A new incident is compared with every
prototype by cosine similarity; if the best similarity is below a threshold tau
chosen on validation data, the incident is declared NOVEL. New types are added by
writing a prototype (``enrol``), without any gradient update.

With ``scoring="knn"`` the memory keeps every stored incident instead of one mean
per type, and the similarity to a type is the mean of the ``k`` closest stored
incidents of that type; this suits types whose incidents look different depending
on which service failed. ``scoring="split"`` uses the k-NN similarity only for the
NOVEL decision and names accepted incidents by the nearest prototype, so a type
enrolled from a few incidents is not out-voted by types with many stored incidents.
"""
from __future__ import annotations

import numpy as np

NOVEL = "NOVEL"
SCORINGS = ["prototype", "knn", "split"]
# named memory configurations: name -> (scoring, k)
MEMORIES = {"prototype": ("prototype", 1), "knn1": ("knn", 1), "knn3": ("knn", 3),
            "split1": ("split", 1), "split3": ("split", 3)}


def parse_config(config: str) -> tuple[str, str, str]:
    """"space:pooling+memory" (e.g. "shift:anchor_pred+split1") -> (space, pooling, memory)."""
    space, _, rest = config.partition(":")
    pooling, _, memory = rest.partition("+")
    return space, pooling, memory or "prototype"


class IncidentMemory:
    def __init__(self, scoring: str = "prototype", k: int = 1) -> None:
        if scoring not in SCORINGS:
            raise ValueError(f"Unknown scoring {scoring!r}; choose from {SCORINGS}")
        self.scoring = scoring
        self.k = k
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.labels: list[str] = []
        self.prototypes: np.ndarray | None = None  # (types, dims), unit rows
        self.examples: dict[str, np.ndarray] = {}  # type -> (incidents, dims), unit rows
        self.tau: float = -np.inf

    def _embed(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "call fit() first"
        z = (np.atleast_2d(x) - self.mean) / self.std
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    def fit(self, x: np.ndarray, y: list[str]) -> "IncidentMemory":
        """Standardise with training statistics and store one prototype per known type."""
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std > 1e-9, std, 1.0)
        z = self._embed(x)
        self.labels = sorted(set(y))
        y_arr = np.asarray(y)
        protos = np.stack([z[y_arr == label].mean(axis=0) for label in self.labels])
        self.prototypes = protos / np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
        self.examples = {label: z[y_arr == label] for label in self.labels}
        return self

    def _knn(self, z: np.ndarray) -> np.ndarray:
        columns = []
        for label in self.labels:
            sims = z @ self.examples[label].T
            k = min(self.k, sims.shape[1])
            columns.append(np.sort(sims, axis=1)[:, -k:].mean(axis=1))
        return np.stack(columns, axis=1)

    def similarities(self, x: np.ndarray) -> np.ndarray:
        """Similarity of each incident to each known type, used for naming: (incidents, types)."""
        assert self.prototypes is not None
        z = self._embed(x)
        return self._knn(z) if self.scoring == "knn" else z @ self.prototypes.T

    def best_similarity(self, x: np.ndarray) -> np.ndarray:
        """Familiarity score used for the NOVEL decision (higher = more familiar)."""
        if self.scoring == "prototype":
            return self.similarities(x).max(axis=1)
        return self._knn(self._embed(x)).max(axis=1)

    def calibrate(self, x_val: np.ndarray, keep_fraction: float = 0.95) -> float:
        """tau = familiarity below which only (1 - keep_fraction) of known validation incidents fall."""
        self.tau = float(np.quantile(self.best_similarity(x_val), 1.0 - keep_fraction))
        return self.tau

    def predict(self, x: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Known type or NOVEL for each incident, plus its familiarity score."""
        names = self.similarities(x).argmax(axis=1)
        best = self.best_similarity(x)
        return [self.labels[i] if s >= self.tau else NOVEL for i, s in zip(names, best)], best

    def enrol(self, label: str, x_examples: np.ndarray) -> None:
        """Add (or refresh) a failure type from a few examples; no retraining."""
        assert self.prototypes is not None
        z = self._embed(x_examples)
        self.examples[label] = z
        proto = z.mean(axis=0)
        proto /= max(np.linalg.norm(proto), 1e-12)
        if label in self.labels:
            self.prototypes[self.labels.index(label)] = proto
        else:
            self.labels.append(label)
            self.prototypes = np.vstack([self.prototypes, proto])
