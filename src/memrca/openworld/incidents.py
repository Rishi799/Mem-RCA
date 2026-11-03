"""Incident sets: aligned arrays describing a group of incidents for the open-world memory.

An incident set is a dict with one entry per incident in each array:
  case, fault, root          names
  services                   the system's services (shared by all incidents)
  shift                      (incidents, services, metric types) metric-shift profiles
  split                      "train" (memory), "validation" (threshold), "test", "test_extra"
  status                     "known" or "unseen" fault type
  root_logits (optional)     the learned localiser's scores per service
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RE1_TYPES = ["cpu", "mem", "disk", "delay", "loss"]


def holdout_type_split(fault: np.ndarray, repetition: np.ndarray, unseen: str) -> tuple[np.ndarray, np.ndarray]:
    """RE1 open-world fold: repetitions 1-2 of known types = memory, 3 = validation,
    4-5 = known test; every incident of the held-out type is unseen test."""
    known = np.asarray(fault) != unseen
    repetition = np.asarray(repetition)
    split = np.select([known & np.isin(repetition, [1, 2]), known & (repetition == 3)], ["train", "validation"], "test")
    return split, np.where(known, "known", "unseen")


def learned_root_logits(predictions_csv: str | Path, cases: np.ndarray, services: list[str],
                        fallback: np.ndarray) -> np.ndarray:
    """(incidents, services) log root probabilities from a trained localiser's
    root_cause_case_predictions.csv; an incident without a prediction points at ``fallback``."""
    predictions = pd.read_csv(predictions_csv)
    by_case = {row.case: json.loads(row.root_probabilities) for row in predictions.itertuples(index=False)}
    logits = np.full((len(cases), len(services)), np.log(1e-6))
    for i, case in enumerate(cases):
        probabilities = by_case.get(case)
        if probabilities is None:
            logits[i, fallback[i]] = 0.0
            continue
        for j, service in enumerate(services):
            logits[i, j] = np.log(max(probabilities.get(service, 0.0), 1e-6))
    return logits


def load_profiles(profiles_dir: str | Path, suite: str, system: str) -> dict:
    """Profiles written by scripts/extract_shift_profiles.py for one suite and system."""
    return dict(np.load(Path(profiles_dir) / f"{suite.lower()}_{system}.npz", allow_pickle=False))


def align_services(shift: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    """Re-index (incidents, services, dims) profiles onto ``target`` services; absent ones are NaN."""
    aligned = np.full((shift.shape[0], len(target), shift.shape[2]), np.nan)
    for j, service in enumerate(target):
        if service in source:
            aligned[:, j] = shift[:, source.index(service)]
    return aligned


def holdout_type_incident_set(profiles: dict, unseen: str, predictions_csv: str | Path | None = None) -> dict:
    """One RE1 open-world fold; with ``predictions_csv`` the learned localiser's scores are attached."""
    split, status = holdout_type_split(profiles["fault"], profiles["repetition"], unseen)
    d = {"case": profiles["case"], "fault": profiles["fault"], "root": profiles["root"],
         "services": profiles["services"], "shift": profiles["shift"], "split": split, "status": status}
    if predictions_csv is not None:
        shift_root = np.nan_to_num(np.abs(d["shift"]), nan=0.0).max(axis=2).argmax(axis=1)
        d["root_logits"] = learned_root_logits(predictions_csv, d["case"], [str(s) for s in d["services"]], shift_root)
    return d


def cross_suite_incident_set(memory_suite: dict, new_suite: dict) -> dict:
    """Memory from one suite (repetition 1 = memory, 2 = validation, 3 = known test); every
    incident of the other suite is an unseen-type test incident, mapped onto the memory's services."""
    services = list(memory_suite["services"])
    new_shift = align_services(new_suite["shift"], list(new_suite["services"]), services)
    rep = memory_suite["repetition"]
    n_new = len(new_suite["case"])
    return {
        "case": np.r_[memory_suite["case"], new_suite["case"]],
        "fault": np.r_[memory_suite["fault"], new_suite["fault"]],
        "root": np.r_[memory_suite["root"], new_suite["root"]],
        "services": memory_suite["services"],
        "shift": np.concatenate([memory_suite["shift"], new_shift]),
        "split": np.r_[np.select([rep == 1, rep == 2], ["train", "validation"], "test"), ["test"] * n_new],
        "status": np.r_[["known"] * len(rep), ["unseen"] * n_new],
    }
