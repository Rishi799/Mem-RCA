"""Discover-and-enrol simulation (contribution C3) and few-shot enrolment (E4).

Stream simulation, per seed/fold/system and per random arrival order:
  * the memory starts with the known fault types (train prototypes, tau from validation);
  * test incidents (known repetition 3 + all repetitions of the fold's unseen types)
    arrive one by one; an incident below tau is labelled NOVEL and buffered;
  * whenever the buffer holds at least --min-cluster incidents, HDBSCAN groups it;
    each group is shown to a simulated engineer, who names it with the group's
    majority true label, and its centre is written to memory (no retraining);
  * we record when each unseen type was first enrolled (occurrences seen so far),
    cluster purity, accuracy on later incidents of that type, and known-type accuracy.

Few-shot enrolment (E4): each unseen type is enrolled directly from k examples
(k = 1, 3, 5) and tested on its remaining incidents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.openworld.fingerprint import fingerprints, root_sources  # noqa: E402
from memrca.openworld.memory import MEMORIES, NOVEL, IncidentMemory, parse_config  # noqa: E402
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_open_world import attach_phase  # noqa: E402


def setup(d, x, memory_name: str = "prototype"):
    split = d["split"]
    memory = IncidentMemory(*MEMORIES[memory_name]).fit(x[split == "train"], list(d["fault"][split == "train"]))
    memory.calibrate(x[split == "validation"], keep_fraction=0.95)
    stream = np.where(((split == "test") & (d["status"] == "known"))
                      | ((d["status"] == "unseen") & np.isin(split, ["test", "test_extra"])))[0]
    return memory, stream


def run_stream(d, x, order, min_cluster: int, memory_name: str = "prototype") -> dict:
    memory, _ = setup(d, x, memory_name)
    fault, status = d["fault"], d["status"]
    unseen_types = sorted(set(fault[status == "unseen"]))
    buffer: list[int] = []
    seen_count = {t: 0 for t in unseen_types}
    enrolled_at: dict[str, int] = {}
    purities, after_enrol, known_ok = [], {t: [] for t in unseen_types}, []
    for i in order:
        true = str(fault[i])
        if true in seen_count:
            seen_count[true] += 1
        names, _ = memory.predict(x[i][None, :])
        pred = names[0]
        if status[i] == "known":
            known_ok.append(pred == true)
        elif true in enrolled_at:
            after_enrol[true].append(pred == true)
        if pred != NOVEL:
            continue
        buffer.append(i)
        if len(buffer) < min_cluster:
            continue
        z = memory._embed(x[buffer])
        labels = HDBSCAN(min_cluster_size=min_cluster).fit_predict(z)
        clustered = set()
        for c in sorted(set(labels) - {-1}):
            members = [buffer[j] for j in np.where(labels == c)[0]]
            truth = pd.Series([str(fault[m]) for m in members])
            name = truth.mode().iloc[0]
            purities.append(float((truth == name).mean()))
            # Only genuinely new labels are written; a group the engineer recognises as an
            # existing type is cleared from the buffer without overwriting that prototype.
            if name not in memory.labels:
                memory.enrol(name, x[members])
                if name in unseen_types:
                    enrolled_at[name] = seen_count[name]
            clustered.update(members)
        buffer = [b for b in buffer if b not in clustered]
    return {
        "types_discovered": f"{len(enrolled_at)}/{len(unseen_types)}",
        "discovered_fraction": len(enrolled_at) / max(len(unseen_types), 1),
        "occurrences_to_discovery": float(np.mean(list(enrolled_at.values()))) if enrolled_at else np.nan,
        "cluster_purity": float(np.mean(purities)) if purities else np.nan,
        "acc_after_enrolment": float(np.mean([v for vs in after_enrol.values() for v in vs]))
        if any(after_enrol.values()) else np.nan,
        "known_accuracy": float(np.mean(known_ok)) if known_ok else np.nan,
        "left_in_buffer": len(buffer),
    }


def few_shot(d, x, k: int, rng, memory_name: str = "prototype") -> list[dict]:
    rows = []
    fault, status = d["fault"], d["status"]
    _, stream = setup(d, x, memory_name)
    for t in sorted(set(fault[status == "unseen"])):
        pool_idx = np.where((fault == t) & (status == "unseen"))[0]
        if len(pool_idx) <= k:
            continue
        memory, _ = setup(d, x, memory_name)
        shots = rng.choice(pool_idx, size=k, replace=False)
        memory.enrol(t, x[shots])
        rest = np.setdiff1d(pool_idx, shots)
        names, _ = memory.predict(x[rest])
        known = stream[status[stream] == "known"]
        known_names, _ = memory.predict(x[known])
        rows.append({"type": t, "k": k, "new_type_accuracy": float(np.mean([n == t for n in names])),
                     "known_accuracy": float(np.mean([n == f for n, f in zip(known_names, fault[known])]))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="experiments/open_world/features")
    ap.add_argument("--out", default="experiments/open_world/results")
    ap.add_argument("--profiles", default="experiments/open_world/profiles")
    ap.add_argument("--variants", nargs="+", default=["shift:anchor_pred+prototype", "shift:global_mean+prototype"],
                    help="space:pooling+memory, memory one of " + ", ".join(MEMORIES))
    ap.add_argument("--orders", type=int, default=5, help="random arrival orders per run")
    ap.add_argument("--min-cluster", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stream_rows, shot_rows = [], []
    for path in sorted(Path(args.features).glob("seed*/fold*/*.npz")):
        d = dict(np.load(path, allow_pickle=False))
        attach_phase(d, Path(args.profiles), path.stem)
        run = {"seed": path.parent.parent.name, "fold": path.parent.name, "system": path.stem}
        roots = root_sources(d)
        for variant in args.variants:
            space, how, memory_name = parse_config(variant)
            x = fingerprints(d, space, how, roots)
            _, stream = setup(d, x, memory_name)
            for o in range(args.orders):
                order = np.random.default_rng(o).permutation(stream)
                stream_rows.append({**run, "variant": variant, "order": o,
                                    **run_stream(d, x, order, args.min_cluster, memory_name)})
            rng = np.random.default_rng(0)
            for k in (1, 3, 5):
                for r in few_shot(d, x, k, rng, memory_name):
                    shot_rows.append({**run, "variant": variant, **r})

    streams = pd.DataFrame(stream_rows)
    shots = pd.DataFrame(shot_rows)
    streams.to_csv(out / "discovery_by_run.csv", index=False)
    shots.to_csv(out / "few_shot_by_run.csv", index=False)
    cols = ["discovered_fraction", "occurrences_to_discovery", "cluster_purity", "acc_after_enrolment", "known_accuracy"]
    pd.set_option("display.width", 200)
    print("\n=== C3: discover -> enrol stream (mean over seeds x folds x orders) ===")
    print(streams.groupby(["system", "variant"])[cols].mean().round(3).to_string())
    print("\n=== E4: few-shot enrolment (mean) ===")
    print(shots.groupby(["system", "variant", "k"])[["new_type_accuracy", "known_accuracy"]].mean().round(3).to_string())
    print(f"\nCSV outputs in {out}/")


if __name__ == "__main__":
    main()
