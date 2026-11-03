"""Confirmatory external validation (docs/EXTERNAL_VALIDATION_PROTOCOL.md).

E-RE1: within-suite open world on RE1, five leave-one-fault-type-out folds per system, with the
       learned localiser (scripts/train_external_localisers.py) and the learning-free root.
E-RE3: memory built from RE2 resource faults; RE3 code-level incidents must look unfamiliar.

The method, candidates, splits and hypotheses were frozen before this script was run; it
prints the hypothesis checks and writes every number to <out>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.openworld.fingerprint import fingerprints, root_sources  # noqa: E402
from memrca.openworld.incidents import (RE1_TYPES, cross_suite_incident_set, holdout_type_incident_set,  # noqa: E402
                                        load_profiles)
from memrca.openworld.memory import MEMORIES, IncidentMemory, parse_config  # noqa: E402
from memrca.openworld.metrics import novelty_auroc  # noqa: E402
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_open_world import open_set_row, rank_of  # noqa: E402
from select_fingerprint import baseline_configs, mem_rca_candidates, pick, pseudo_open_set  # noqa: E402
from simulate_discovery import few_shot, run_stream, setup  # noqa: E402
from train_external_localisers import FOLD  # noqa: E402

TEST_METRICS = ["auroc", "oscr", "closed_set_acc", "known_accepted", "unseen_flagged_novel"]


def evaluate_configs(d: dict, roots: dict, configs: list[str]) -> list[dict]:
    rows = []
    for config in configs:
        space, pooling, memory = parse_config(config)
        x = fingerprints(d, space, pooling, roots)
        scoring, k = MEMORIES[memory]
        rows.append({"config": config, **pseudo_open_set(d["fault"], d["split"], x, scoring, k),
                     **open_set_row(d, x, scoring, k)})
    return rows


def all_configs(roots: list[str]) -> list[str]:
    configs = []
    for root in roots:
        configs += mem_rca_candidates(root) + list(baseline_configs(root).values())
    return list(dict.fromkeys(configs))


def roles(val: pd.DataFrame, root: str) -> dict[str, str]:
    """Mem-RCA (selected on validation) and the fixed baselines for one root source."""
    return {"Mem-RCA": pick(val.loc[mem_rca_candidates(root)]),
            **{f"baseline {name}": c for name, c in baseline_configs(root).items()}}


def run_re1(profiles: Path, models: Path, systems: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    open_rows, root_rows, data = [], [], {}
    for system in systems:
        p = load_profiles(profiles, "re1", system)
        services = list(p["services"])
        for unseen in RE1_TYPES:
            # the learned root is the localiser's top service; the shift root where it has no prediction
            predictions = models / f"{system}_{unseen}" / FOLD / system / "root_cause_case_predictions.csv"
            d = holdout_type_incident_set(p, unseen, predictions)
            split, status = d["split"], d["status"]
            roots = root_sources(d)
            missing = len(set(d["case"]) - set(pd.read_csv(predictions)["case"]))
            data[(system, unseen)] = (d, roots)
            run = {"system": system, "unseen_type": unseen, "learned_root_missing": missing}
            open_rows += [{**run, **r} for r in evaluate_configs(d, roots, all_configs(["pred", "shift"]))]
            for i in np.where(split == "test")[0]:
                if p["root"][i] not in services:
                    continue
                truth = services.index(p["root"][i])
                magnitude = np.nan_to_num(np.abs(p["shift"][i]), nan=0.0).max(axis=1)
                for method, hit in (("learned", roots["pred"][i] == truth), ("shift_all_services", roots["shift"][i] == truth)):
                    root_rows.append({**run, "status": status[i], "method": method, "hit_at_1": float(hit),
                                      "rank": rank_of(magnitude, truth) if method != "learned" else np.nan})
    return pd.DataFrame(open_rows), pd.DataFrame(root_rows), data


def run_re3(profiles: Path, systems: list[str]) -> pd.DataFrame:
    rows = []
    for system in systems:
        # RE3 is mapped onto the services the RE2 memory knows; others are absent (NaN)
        d = cross_suite_incident_set(load_profiles(profiles, "re2", system), load_profiles(profiles, "re3", system))
        roots = {"shift": root_sources(d)["shift"]}
        results = pd.DataFrame(evaluate_configs(d, roots, all_configs(["shift"]))).set_index("config")
        for role, config in roles(results, "shift").items():
            space, pooling, memory = parse_config(config)
            x = fingerprints(d, space, pooling, roots)
            memory_obj = IncidentMemory(*MEMORIES[memory]).fit(x[d["split"] == "train"],
                                                               list(d["fault"][d["split"] == "train"]))
            best = memory_obj.best_similarity(x)
            known = (d["split"] == "test") & (d["status"] == "known")
            unseen = d["status"] == "unseen"
            p_value = mannwhitneyu(best[known], best[unseen], alternative="greater").pvalue
            rows.append({"system": system, "role": role, "config": config,
                         "val_auroc": results.loc[config, "val_auroc"], **results.loc[config, TEST_METRICS].to_dict(),
                         "auroc_check": novelty_auroc(best[known], best[unseen]), "mann_whitney_p": p_value,
                         "known_test": int(known.sum()), "unseen_test": int(unseen.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default="experiments/open_world/profiles")
    ap.add_argument("--models", default="experiments/external/re1/runs")
    ap.add_argument("--out", default="experiments/external/results")
    ap.add_argument("--systems", nargs="+", default=["ob", "ss", "tt"])
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--skip-re1", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    re3 = run_re3(Path(args.profiles), args.systems)
    re3.to_csv(out / "re3_novel_types.csv", index=False)
    print("=== E-RE3: RE2 memory vs RE3 code-level faults (learning-free root) ===")
    print(re3.round(4).to_string(index=False))
    h3 = re3[re3["role"].eq("Mem-RCA")]
    print(f"H3 (AUROC > 0.5 and p < 0.05 in every system): "
          f"{'SUPPORTED' if ((h3['auroc'] > 0.5) & (h3['mann_whitney_p'] < 0.05)).all() else 'NOT SUPPORTED'}")
    if args.skip_re1:
        return

    open_df, root_df, data = run_re1(Path(args.profiles), Path(args.models), args.systems)
    open_df.to_csv(out / "re1_open_set_by_fold.csv", index=False)
    root_df.to_csv(out / "re1_root_by_case.csv", index=False)

    summary, selected = [], {}
    for root in ("pred", "shift"):
        for system in args.systems:
            sub = open_df[open_df["system"].eq(system)]
            val = sub.groupby("config")[["val_auroc", "val_oscr"]].mean()
            for role, config in roles(val, root).items():
                folds = sub[sub["config"].eq(config)]
                summary.append({"root": root, "system": system, "role": role, "config": config,
                                "val_auroc": val.loc[config, "val_auroc"], **folds[TEST_METRICS].mean().to_dict()})
                if role == "Mem-RCA":
                    selected[(root, system)] = config
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "re1_summary.csv", index=False)
    print("\n=== E-RE1: open world on RE1 (mean over 5 leave-one-type-out folds) ===")
    print(summary.round(3).to_string(index=False))

    tt = open_df[open_df["system"].eq("tt")].pivot_table(index="unseen_type", columns="config", values="auroc")
    base = baseline_configs("pred")
    anchor = tt[base["anchor"]]
    wins = {g: int((anchor > tt[base[g]]).sum()) for g in ("global_mean", "global_max")}
    h1 = all(anchor.mean() > tt[base[g]].mean() and wins[g] >= 4 for g in wins)
    print(f"\nH1 (TT, learned root): anchor {anchor.mean():.3f} vs global_mean {tt[base['global_mean']].mean():.3f} "
          f"({wins['global_mean']}/5 folds), global_max {tt[base['global_max']].mean():.3f} "
          f"({wins['global_max']}/5) -> {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    learned = summary[summary["root"].eq("pred")].groupby("role")["auroc"].mean()
    h2 = all(learned["Mem-RCA"] > learned[f"baseline {g}"] for g in base)
    print("H2 (mean AUROC over systems, learned root): "
          + ", ".join(f"{r} {v:.3f}" for r, v in learned.items()) + f" -> {'SUPPORTED' if h2 else 'NOT SUPPORTED'}")
    print("\nRoot Hit@1 on RE1 test incidents:")
    print(root_df.groupby(["system", "status", "method"])["hit_at_1"].mean().unstack("method").round(3).to_string())

    stream_rows, shot_rows = [], []
    for (system, unseen), (d, roots) in data.items():
        for name, config in (("Mem-RCA", selected[("pred", system)]), ("baseline global_mean", base["global_mean"])):
            space, pooling, memory = parse_config(config)
            x = fingerprints(d, space, pooling, roots)
            _, stream = setup(d, x, memory)
            for o in range(args.orders):
                order = np.random.default_rng(o).permutation(stream)
                stream_rows.append({"system": system, "unseen_type": unseen, "role": name, "order": o,
                                    **run_stream(d, x, order, 3, memory)})
            rng = np.random.default_rng(0)
            for k in (1, 3, 5):
                shot_rows += [{"system": system, "unseen_type": unseen, "role": name, **r}
                              for r in few_shot(d, x, k, rng, memory)]
    streams, shots = pd.DataFrame(stream_rows), pd.DataFrame(shot_rows)
    streams.to_csv(out / "re1_discovery.csv", index=False)
    shots.to_csv(out / "re1_few_shot.csv", index=False)
    print("\n=== E-RE1 discover -> enrol (mean over folds x orders) ===")
    print(streams.groupby(["system", "role"])[["discovered_fraction", "occurrences_to_discovery", "cluster_purity",
                                               "acc_after_enrolment", "known_accuracy"]].mean().round(3).to_string())
    print("\n=== E-RE1 k-shot enrolment ===")
    print(shots.groupby(["system", "role", "k"])[["new_type_accuracy", "known_accuracy"]].mean().round(3).to_string())
    print(f"\nCSV outputs in {out}/")


if __name__ == "__main__":
    main()
