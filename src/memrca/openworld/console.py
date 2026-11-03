"""Interactive open-world session behind the Mem-RCA console (scripts/serve_console.py).

A session replays a stream of incidents against an incident memory:
  1. localise the root service and build the incident's fingerprint;
  2. compare it with every failure type in memory; below tau it is NOVEL;
  3. NOVEL incidents wait in a buffer; groups of similar ones (HDBSCAN) are proposed
     as candidate new failure types;
  4. the operator names a group and it is enrolled into memory, without retraining.
Everything shown comes from the same fingerprint, memory and threshold code that the
experiments use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from memrca.data.loader import load_case, metric_columns_by_service
from memrca.openworld.fingerprint import METRIC_ALIASES, METRIC_TYPES, fingerprints, root_sources
from memrca.openworld.memory import MEMORIES, NOVEL, IncidentMemory, parse_config

FORECAST_FAULTS = {"cpu", "mem", "disk", "delay", "loss", "socket"}
SEGMENT_NAMES = {"anchor": ["Root service", "Symptoms elsewhere (weighted)"],
                 "hybrid": ["Root service", "Symptoms elsewhere (weighted)", "Strongest symptom anywhere"],
                 "global_max": ["Strongest symptom anywhere"], "global_mean": ["Average over services"]}


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


class ConsoleSession:
    def __init__(self, d: dict, config: str, stream: np.ndarray, raw_dir: str | Path | None = None,
                 title: str = "", min_cluster: int = 3, system: str = "") -> None:
        self.d, self.config, self.stream, self.title, self.system = d, config, np.asarray(stream), title, system
        self.last: dict | None = None
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.min_cluster = min_cluster
        self.space, self.pooling, memory_name = parse_config(config)
        self.roots = root_sources(d)
        self.root_source = self.pooling.rpartition("_")[2] if self.pooling.startswith(("anchor", "hybrid")) else None
        self.x = fingerprints(d, self.space, self.pooling, self.roots)
        split, fault = d["split"], d["fault"]
        self.memory = IncidentMemory(*MEMORIES[memory_name]).fit(self.x[split == "train"], list(fault[split == "train"]))
        self.memory.calibrate(self.x[split == "validation"], keep_fraction=0.95)
        self.initial_types = list(self.memory.labels)
        # fixed 2-D map of the memory space (PCA of the stored incidents)
        z = self.memory._embed(self.x[split == "train"])
        self._centre = z.mean(axis=0)
        self._axes = np.linalg.svd(z - self._centre, full_matrices=False)[2][:2]
        self.position = 0
        self.buffer: list[int] = []
        self.candidates: list[dict] = []
        self.history: list[dict] = []
        self.enrolled: list[dict] = []
        self.aliases: dict[str, str] = {}  # enrolled name -> majority true type (for scoring only)
        self._next_candidate_id = 0

    # ---------- views ----------
    def _project(self, rows: np.ndarray) -> list[list[float]]:
        return ((self.memory._embed(rows) - self._centre) @ self._axes.T).round(4).tolist()

    def _ranking(self, i: int) -> dict:
        services = [str(s) for s in self.d["services"]]
        magnitude = np.nan_to_num(np.abs(self.d["shift"][i]), nan=0.0).max(axis=1)
        out = {"metric_shift": [{"service": services[j], "score": float(magnitude[j])}
                                for j in np.argsort(-magnitude)[:5]]}
        if "root_logits" in self.d:
            p = _softmax(np.asarray(self.d["root_logits"][i], dtype=float))
            out["learned"] = [{"service": services[j], "score": float(p[j])} for j in np.argsort(-p)[:5]]
        return out

    def _segments(self, i: int) -> list[dict]:
        base = self.pooling.rpartition("_")[0] if self.pooling.endswith(("_pred", "_shift", "_oracle")) else self.pooling
        names = SEGMENT_NAMES.get(base, [base])
        width = len(METRIC_TYPES)
        return [{"name": name, "values": np.nan_to_num(self.x[i][k * width:(k + 1) * width]).round(3).tolist()}
                for k, name in enumerate(names) if (k + 1) * width <= len(self.x[i])]

    def _series(self, i: int, service: str) -> dict | None:
        """The anchor service's most-shifted metric around the alert, for the incident chart."""
        if self.raw_dir is None or not (self.raw_dir / str(self.d["case"][i]) / "metrics.parquet").is_file():
            return None
        case = load_case(self.raw_dir, str(self.d["case"][i]))
        services = [str(s) for s in self.d["services"]]
        shift = np.nan_to_num(np.abs(self.d["shift"][i][services.index(service)]), nan=0.0)
        columns = metric_columns_by_service(case.metrics.columns).get(service, [])
        if not columns:
            return None
        wanted = METRIC_TYPES[int(shift.argmax())]
        column = next((c for c in columns if METRIC_ALIASES.get(c.partition("_")[2], c.partition("_")[2]) == wanted),
                      columns[0])
        window = case.metrics[column].astype(float)
        window = window[(window.index >= case.inject_time - 240) & (window.index < case.inject_time + 240)].dropna()
        step = max(1, len(window) // 240)
        window = window.iloc[::step]
        return {"metric": column, "t": (window.index - case.inject_time).astype(int).tolist(),
                "v": window.round(5).tolist()}

    def _truth_correct(self, true_type: str, verdict: str) -> bool:
        """Right type (an enrolled name counts as the type most of its incidents had), or NOVEL
        when no type in memory stands for the incident's true type."""
        meaning = {label: self.aliases.get(label, label) for label in self.memory.labels}
        if true_type in meaning.values():
            return meaning.get(verdict) == true_type
        return verdict == NOVEL

    # ---------- actions ----------
    def peek_case(self) -> str | None:
        """The recording the next incident comes from, for the live replay (not shown to the user)."""
        return str(self.d["case"][int(self.stream[self.position])]) if self.position < len(self.stream) else None

    def next_incident(self) -> dict | None:
        if self.position >= len(self.stream):
            return None
        i = int(self.stream[self.position])
        self.position += 1
        names, best = self.memory.predict(self.x[i][None, :])
        sims = self.memory.similarities(self.x[i][None, :])[0]
        services = [str(s) for s in self.d["services"]]
        anchor = services[int(self.roots[self.root_source][i])] if self.root_source else None
        true_type = str(self.d["fault"][i])
        verdict = names[0]
        record = {
            "step": self.position, "index": i, "case": str(self.d["case"][i]),
            "verdict": verdict, "familiarity": float(best[0]), "tau": self.memory.tau,
            "similar_types": sorted([{"type": t, "similarity": float(s)} for t, s in zip(self.memory.labels, sims)],
                                    key=lambda r: -r["similarity"]),
            "anchor": anchor, "ranking": self._ranking(i), "segments": self._segments(i),
            "metric_types": METRIC_TYPES, "map": self._project(self.x[i][None, :])[0],
            "series": self._series(i, anchor or self._ranking(i)["metric_shift"][0]["service"]),
            "truth": {"type": true_type, "root": str(self.d["root"][i]),
                      "known_at_start": true_type in self.initial_types,
                      "correct": self._truth_correct(true_type, verdict)},
        }
        self.history.append({k: record[k] for k in ("step", "index", "case", "verdict", "familiarity", "truth")})
        self.last = record
        if verdict == NOVEL:
            self.buffer.append(i)
            self._propose()
        return record

    def _propose(self) -> None:
        self.candidates = []
        if len(self.buffer) < self.min_cluster:
            return
        labels = HDBSCAN(min_cluster_size=self.min_cluster).fit_predict(self.memory._embed(self.x[self.buffer]))
        for c in sorted(set(labels) - {-1}):
            members = [self.buffer[j] for j in np.where(labels == c)[0]]
            truth = [str(self.d["fault"][m]) for m in members]
            centre = self.x[members].mean(axis=0)
            width = len(METRIC_TYPES)
            root_part = np.nan_to_num(centre[:width])
            top = [METRIC_TYPES[k] for k in np.argsort(-np.abs(root_part))[:2]]
            self.candidates.append({
                "id": self._next_candidate_id, "members": [str(self.d["case"][m]) for m in members],
                "indices": members, "size": len(members), "signature": top,
                "truth_majority": max(set(truth), key=truth.count), "map": self._project(centre[None, :])[0],
            })
            self._next_candidate_id += 1

    def propagation_context(self) -> dict | None:
        """Root and fault type to condition the time-to-impact forecast of the latest incident on:
        the anchored (or top-ranked) root, and the memory's verdict when it is a known fault type."""
        if self.last is None:
            return None
        ranking = self.last["ranking"]
        root = self.last["anchor"] or (ranking.get("learned") or ranking["metric_shift"])[0]["service"]
        verdict = self.last["verdict"]
        return {"case": self.last["case"], "system": self.system, "root": root,
                "fault": verdict if verdict in FORECAST_FAULTS else None, "verdict": verdict}

    def enrol(self, candidate_id: int, name: str) -> dict:
        candidate = next(c for c in self.candidates if c["id"] == candidate_id)
        name = name.strip() or f"new-type-{len(self.enrolled) + 1}"
        self.memory.enrol(name, self.x[candidate["indices"]])
        self.aliases[name] = candidate["truth_majority"]
        self.enrolled.append({"name": name, "size": candidate["size"], "at_step": self.position,
                              "truth_majority": candidate["truth_majority"]})
        self.buffer = [b for b in self.buffer if b not in candidate["indices"]]
        self._propose()
        return self.state()

    def state(self) -> dict:
        split, fault = self.d["split"], self.d["fault"]
        stored = np.where(split == "train")[0]
        verdicts = [h["verdict"] for h in self.history]
        correct = [h["truth"]["correct"] for h in self.history]
        types = [{"type": t, "stored": int(len(self.memory.examples[t])),
                  "enrolled": t not in self.initial_types} for t in self.memory.labels]
        return {
            "title": self.title, "config": self.config, "tau": self.memory.tau,
            "position": self.position, "total": int(len(self.stream)),
            "kpis": {"processed": len(self.history), "recognised": sum(v != NOVEL for v in verdicts),
                     "novel": sum(v == NOVEL for v in verdicts), "enrolled": len(self.enrolled),
                     "types": len(self.memory.labels),
                     "open_set_accuracy": float(np.mean(correct)) if correct else None},
            "types": types,
            "map": {"stored": [{"xy": xy, "type": str(fault[i])} for xy, i in zip(self._project(self.x[stored]), stored)],
                    "prototypes": [{"xy": xy, "type": t} for xy, t in
                                   zip(((self.memory.prototypes - self._centre) @ self._axes.T).round(4).tolist(),
                                       self.memory.labels)],
                    "buffer": [{"xy": xy, "case": str(self.d["case"][i])} for xy, i in
                               zip(self._project(self.x[self.buffer]) if self.buffer else [], self.buffer)]},
            "buffer": [{"case": str(self.d["case"][i]), "truth": str(fault[i])} for i in self.buffer],
            "candidates": [{k: v for k, v in c.items() if k != "indices"} for c in self.candidates],
            "enrolled": self.enrolled,
            "history": self.history[-60:],
        }
