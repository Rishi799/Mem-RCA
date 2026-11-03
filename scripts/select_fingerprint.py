"""Validation-only selection of the incident fingerprint and memory (follow-up to E2).

Test incidents were already looked at in E2, so every further design choice is made
without them. Protocol, per seed/fold/system:
  * pseudo-open-set on known types only: each known type t in turn plays "unseen";
    the memory is built from train incidents (repetition 1) of the other known types,
    known evaluation incidents are their validation incidents (repetition 2), and the
    pseudo-unseen incidents are all train + validation incidents of t;
  * validation score = novelty AUROC, averaged over t, seeds and folds (OSCR breaks ties).
Rule fixed before any test number is computed: per system, the candidate with the
best validation score is selected (an operator tunes on their own system's past
incidents); a single candidate is also selected by the mean score over systems, and a
learning-free candidate (no trained localiser; used for suites the model never saw).
Only then are the selected candidates compared on the real test incidents with the E2
protocol (evaluate_open_world.open_set_row). Test numbers for every candidate are
written to CSV for transparency, together with how well validation ranks predict them.
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
from memrca.openworld.memory import MEMORIES, IncidentMemory, parse_config  # noqa: E402
from memrca.openworld.metrics import novelty_auroc, oscr  # noqa: E402
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_open_world import attach_phase, open_set_row  # noqa: E402

SPACES = ["shift", "phase"]
POOLINGS = ["global_mean", "global_max", "anchor_pred", "hybrid_pred", "anchor_shift", "hybrid_shift"]
BASELINES = {"E2 anchor": "shift:anchor_pred+prototype", "E2 global_max": "shift:global_max+prototype",
             "E2 global_mean": "shift:global_mean+prototype"}
TEST_METRICS = ["auroc", "auroc_fresh", "oscr", "closed_set_acc", "known_accepted", "unseen_flagged_novel"]


def candidates(learning_free: bool = False) -> list[str]:
    return [f"{space}:{pooling}+{memory}" for space in SPACES for pooling in POOLINGS for memory in MEMORIES
            if not (learning_free and pooling.endswith("_pred"))]


def mem_rca_candidates(root: str) -> list[str]:
    """Candidates of the frozen Mem-RCA procedure (docs/EXTERNAL_VALIDATION_PROTOCOL.md);
    ``root`` is the anchor source: "pred" (learned localiser) or "shift" (learning-free)."""
    return [f"shift:{view}+{memory}" for view in (f"anchor_{root}", "global_max", f"hybrid_{root}")
            for memory in ("split1", "split3")]


def baseline_configs(root: str) -> dict[str, str]:
    return {"global_mean": "shift:global_mean+prototype", "global_max": "shift:global_max+prototype",
            "anchor": f"shift:anchor_{root}+prototype"}


def pseudo_open_set(fault: np.ndarray, split: np.ndarray, x: np.ndarray, scoring: str, k: int,
                    train_split: str = "train", val_split: str = "validation") -> dict:
    """Leave-one-known-type-out check using only train/validation incidents of known types."""
    aurocs, oscrs = [], []
    for t in sorted(set(fault[split == train_split])):
        train = (split == train_split) & (fault != t)
        known = (split == val_split) & (fault != t)
        unseen = np.isin(split, [train_split, val_split]) & (fault == t)
        memory = IncidentMemory(scoring, k).fit(x[train], list(fault[train]))
        best = memory.best_similarity(x)
        correct = np.array([memory.labels[j] for j in memory.similarities(x).argmax(axis=1)]) == fault
        aurocs.append(novelty_auroc(best[known], best[unseen]))
        oscrs.append(oscr(best[known], correct[known], best[unseen]))
    return {"val_auroc": float(np.mean(aurocs)), "val_oscr": float(np.mean(oscrs))}


def pick(scores: pd.DataFrame) -> str:
    return scores.sort_values(["val_auroc", "val_oscr"], ascending=False).index[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="experiments/open_world/features")
    ap.add_argument("--profiles", default="experiments/open_world/profiles")
    ap.add_argument("--out", default="experiments/open_world/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(Path(args.features).glob("seed*/fold*/*.npz")):
        d = dict(np.load(path, allow_pickle=False))
        attach_phase(d, Path(args.profiles), path.stem)
        run = {"seed": path.parent.parent.name, "fold": path.parent.name, "system": path.stem}
        roots = root_sources(d)
        for config in candidates():
            space, pooling, memory = parse_config(config)
            x = fingerprints(d, space, pooling, roots)
            scoring, k = MEMORIES[memory]
            rows.append({**run, "config": config, **pseudo_open_set(d["fault"], d["split"], x, scoring, k),
                         **open_set_row(d, x, scoring, k)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "selection_by_run.csv", index=False)

    val = df.groupby(["system", "config"])[["val_auroc", "val_oscr"]].mean()
    systems = list(val.index.get_level_values("system").unique())
    chosen = {s: pick(val.loc[s]) for s in systems}
    chosen_memrca = {s: pick(val.loc[s].loc[mem_rca_candidates("pred")]) for s in systems}
    chosen_free = {s: pick(val.loc[s].loc[mem_rca_candidates("shift")]) for s in systems}
    overall = pick(val.groupby("config").mean())

    pd.set_option("display.width", 220)
    print("=== Validation top 8 per system (pseudo-open-set on known types; no test data) ===")
    for s in systems:
        print(f"-- {s}")
        print(val.loc[s].sort_values(["val_auroc", "val_oscr"], ascending=False).head(8).round(3).to_string())
    print(f"\nSearch over all {len(candidates())}: {chosen}\nMem-RCA: {chosen_memrca}\n"
          f"Mem-RCA learning-free: {chosen_free}\nOne config for all: {overall}")

    test = df.groupby(["system", "config"])[TEST_METRICS].mean()
    summary = []
    for s in systems:
        runs = df[df["system"].eq(s)].set_index(["seed", "fold"])
        e2 = runs[runs["config"].eq(BASELINES["E2 anchor"])]["auroc"]
        roles = {**BASELINES, f"search over {len(candidates())}": chosen[s], "Mem-RCA": chosen_memrca[s],
                 "Mem-RCA learning-free": chosen_free[s], "one config for all": overall}
        for role, config in roles.items():
            delta = runs[runs["config"].eq(config)]["auroc"] - e2
            summary.append({"system": s, "role": role, "config": config,
                            "val_auroc": val.loc[(s, config), "val_auroc"], **test.loc[(s, config)].to_dict(),
                            "runs_better_than_e2_anchor": f"{int((delta > 0).sum())}/{len(delta)}"})
        rho = val.loc[s, "val_auroc"].rank().corr(test.loc[s, "auroc"].rank())
        print(f"{s}: Spearman(validation AUROC, test AUROC) over {len(val.loc[s])} candidates = {rho:.2f}")
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "selection_summary.csv", index=False)
    test.join(val).to_csv(out / "selection_all_candidates.csv")
    pd.Series({**{f"search_{s}": c for s, c in chosen.items()},
               **{f"memrca_{s}": c for s, c in chosen_memrca.items()},
               **{f"memrca_learning_free_{s}": c for s, c in chosen_free.items()}, "across_systems": overall}
              ).to_json(out / "selected_configs.json", indent=2)
    print("\n=== Test (E2 protocol), selected candidates vs E2 fingerprints ===")
    print(summary.round(3).to_string(index=False))
    print(f"\nCSV outputs in {out}/")


if __name__ == "__main__":
    main()
