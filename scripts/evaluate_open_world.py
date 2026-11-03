"""Open-world evaluation: NOVEL detection with an incident memory (E2) and root-cause shortcut checks (C4).

Inputs are the per-incident feature files from scripts/extract_incident_features.py.

E2 (key experiment). For every seed/fold/system and every fingerprint variant:
  memory prototypes   <- train incidents (repetition 1, known fault types)
  threshold tau       <- validation incidents (repetition 2, known), 95% retained
  known test          <- repetition 3 of known types
  unseen test         <- the fold's held-out fault types: repetition 3 ("test") plus
                         repetitions 1-2 ("test_extra", never used by any earlier run)
Reported: novelty AUROC (all unseen, and fresh test_extra-only), OSCR, closed-set
accuracy on known types, and accept/reject rates at tau.

C4 (shortcut check). Root ranking on test incidents by: the learned root head; a
no-telemetry prior that only knows which services were roots in training; a
learning-free telemetry ranking (largest metric shift), over all services and
restricted to the training-root candidates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.openworld.fingerprint import fingerprints, root_sources  # noqa: E402
from memrca.openworld.memory import IncidentMemory  # noqa: E402
from memrca.openworld.metrics import novelty_auroc, oscr  # noqa: E402

VARIANTS = [(space, how) for space in ("shift", "embed")
            for how in ("global_mean", "global_max", "anchor_pred", "anchor_oracle")]


def attach_phase(d: dict, profiles: Path, system: str, suite: str = "re2") -> None:
    """Add the time-resolved profiles (scripts/extract_shift_profiles.py) aligned to d's cases and services."""
    p = np.load(profiles / f"{suite}_{system}.npz", allow_pickle=False)
    row = {c: i for i, c in enumerate(p["case"])}
    source = list(p["services"])
    phase = p["phase"][[row[c] for c in d["case"]]]
    aligned = np.full((len(d["case"]), len(d["services"]), phase.shape[2]), np.nan)
    for j, service in enumerate(d["services"]):
        if service in source:
            aligned[:, j] = phase[:, source.index(service)]
    d["phase"] = aligned


def open_set_row(d, x: np.ndarray, scoring: str = "prototype", k: int = 1) -> dict:
    split, status, fault = d["split"], d["status"], d["fault"]
    train = split == "train"
    val = split == "validation"
    known = (split == "test") & (status == "known")
    unseen = (status == "unseen") & np.isin(split, ["test", "test_extra"])
    fresh = (status == "unseen") & (split == "test_extra")
    memory = IncidentMemory(scoring, k).fit(x[train], list(fault[train]))
    tau = memory.calibrate(x[val], keep_fraction=0.95)
    best = memory.best_similarity(x)
    typed = np.array([memory.labels[j] for j in memory.similarities(x).argmax(axis=1)])
    correct = typed == fault
    return {
        "tau": tau,
        "known_test": int(known.sum()), "unseen_test": int(unseen.sum()), "unseen_fresh": int(fresh.sum()),
        "auroc": novelty_auroc(best[known], best[unseen]),
        "auroc_fresh": novelty_auroc(best[known], best[fresh]),
        "oscr": oscr(best[known], correct[known], best[unseen]),
        "closed_set_acc": float(correct[known].mean()),
        "known_accepted": float((best[known] >= tau).mean()),
        "unseen_flagged_novel": float((best[unseen] < tau).mean()),
    }


def rank_of(scores: np.ndarray, truth: int) -> int:
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == truth)[0][0]) + 1


def root_rows(d, run: dict) -> list[dict]:
    services = list(d["services"])
    candidates = [services.index(s) for s in d["train_roots"] if s in services]
    magnitude = np.nan_to_num(np.abs(d["shift"]), nan=0.0).max(axis=2)
    n_c = len(candidates)
    prior_mrr = float(np.mean([1.0 / k for k in range(1, n_c + 1)])) if n_c else 0.0
    rows = []
    for i in np.where(np.isin(d["split"], ["test", "test_extra"]))[0]:
        if d["root"][i] not in services:
            continue
        truth = services.index(d["root"][i])
        is_candidate = np.zeros(len(services))
        is_candidate[candidates] = 1.0
        ranks = {
            "root_head": rank_of(d["root_logits"][i], truth),
            "shift_all_services": rank_of(magnitude[i], truth),
            # candidates first (ordered by shift), then the remaining services by shift
            "shift_training_roots_only": rank_of(magnitude[i] + 1e6 * is_candidate, truth),
        }
        for method, r in ranks.items():
            rows.append({**run, "case": d["case"][i], "split": d["split"][i], "status": d["status"][i],
                         "method": method, "hit_at_1": float(r == 1), "hit_at_3": float(r <= 3), "mrr": 1.0 / r})
        in_c = truth in candidates
        rows.append({**run, "case": d["case"][i], "split": d["split"][i], "status": d["status"][i],
                     "method": "no_telemetry_prior",
                     "hit_at_1": (1.0 / n_c) if in_c else 0.0,
                     "hit_at_3": (min(3, n_c) / n_c) if in_c else 0.0,
                     "mrr": prior_mrr if in_c else 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="experiments/open_world/features")
    ap.add_argument("--out", default="experiments/open_world/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    open_rows, roots = [], []
    for path in sorted(Path(args.features).glob("seed*/fold*/*.npz")):
        d = dict(np.load(path, allow_pickle=False))
        run = {"seed": path.parent.parent.name, "fold": path.parent.name, "system": path.stem}
        anchors = root_sources(d)
        for space, how in VARIANTS:
            x = fingerprints(d, space, how, anchors)
            open_rows.append({**run, "space": space, "pooling": how, **open_set_row(d, x)})
        roots += root_rows(d, run)

    open_df = pd.DataFrame(open_rows)
    open_df.to_csv(out / "open_set_by_run.csv", index=False)
    metrics = ["auroc", "auroc_fresh", "oscr", "closed_set_acc", "known_accepted", "unseen_flagged_novel"]
    summary = open_df.groupby(["system", "space", "pooling"])[metrics].agg(["mean", "std"])
    summary.to_csv(out / "open_set_summary.csv")

    # Paired comparison: anchor (predicted root) vs each global pooling, same run.
    paired = []
    for space in ("shift", "embed"):
        base = open_df[open_df["space"].eq(space)].pivot_table(
            index=["seed", "fold", "system"], columns="pooling", values="auroc")
        for g in ("global_mean", "global_max"):
            delta = base["anchor_pred"] - base[g]
            for system, values in delta.groupby(level="system"):
                paired.append({"space": space, "versus": g, "system": system,
                               "mean_delta_auroc": float(values.mean()),
                               "runs_anchor_better": f"{int((values > 0).sum())}/{len(values)}"})
    pd.DataFrame(paired).to_csv(out / "e2_paired_auroc.csv", index=False)

    root_df = pd.DataFrame(roots)
    root_df.to_csv(out / "root_by_case.csv", index=False)
    root_summary = (root_df.groupby(["seed", "fold", "system", "status", "method"])[["hit_at_1", "hit_at_3", "mrr"]]
                    .mean().groupby(["system", "status", "method"]).agg(["mean", "std"]))
    root_summary.to_csv(out / "root_shortcut_summary.csv")

    pd.set_option("display.width", 200)
    print("\n=== E2: NOVEL detection (mean over seeds x folds) ===")
    print(open_df.groupby(["system", "space", "pooling"])[["auroc", "auroc_fresh", "oscr", "closed_set_acc",
                                                         "unseen_flagged_novel"]].mean().round(3).to_string())
    print("\n=== E2 paired: anchor_pred minus global (AUROC) ===")
    print(pd.DataFrame(paired).to_string(index=False))
    print("\n=== C4: root ranking on test incidents (mean over runs) ===")
    print(root_df.groupby(["system", "status", "method"])[["hit_at_1", "hit_at_3", "mrr"]].mean().round(3).to_string())
    print(f"\nCSV outputs in {out}/")


if __name__ == "__main__":
    main()
