"""Validate case-level RE2 folds and write a reproducible split manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw", default=None, help="Override raw_dir from the config")
    parser.add_argument("--out", default="experiments/splits")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw or config["raw_dir"])
    index = pd.read_parquet(raw_dir / "cases.parquet")
    suite = str(config.get("suite", "re2")).upper()
    index = index[index["suite"].astype(str).str.upper() == suite].copy()
    if index.empty:
        raise ValueError(f"No {suite} cases in {raw_dir / 'cases.parquet'}")

    repetition_sets = {
        "train": set(map(int, config["train_repetitions"])),
        "validation": set(map(int, config["validation_repetitions"])),
        "test": set(map(int, config["test_repetitions"])),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = repetition_sets[left] & repetition_sets[right]
        if overlap:
            raise ValueError(f"Repetition leakage between {left} and {right}: {sorted(overlap)}")

    if "root_cause_service" not in index.columns:
        raise ValueError("Expected root_cause_service label in RCAEval case index")

    rows: list[dict[str, object]] = []
    for fold_name, fold in config["folds"].items():
        known, unseen = set(fold["known"]), set(fold["unseen"])
        if known & unseen:
            raise ValueError(f"{fold_name}: known and unseen fault sets overlap")
        expected_faults = set(map(str, config["fault_types"]))
        if known | unseen != expected_faults:
            raise ValueError(
                f"{fold_name}: known and unseen sets must cover configured faults; "
                f"missing={sorted(expected_faults - (known | unseen))}, "
                f"extra={sorted((known | unseen) - expected_faults)}"
            )
        for case in index.to_dict("records"):
            rep, fault = int(case["repetition"]), str(case["fault"])
            if rep in repetition_sets["train"] and fault in known:
                split = "train"
            elif rep in repetition_sets["validation"] and fault in known:
                split = "validation"
            elif rep in repetition_sets["test"]:
                split = "test"
            else:
                continue
            rows.append({
                "fold": fold_name,
                "split": split,
                "case": str(case["case"]),
                "system": str(case["system"]),
                "fault": fault,
                "root_cause_service": str(case["root_cause_service"]),
                "fault_status": "unseen" if fault in unseen else "known",
                "repetition": rep,
                "has_logs": bool(case.get("has_logs", False)),
                "has_traces": bool(case.get("has_traces", False)),
            })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("Split manifest is empty")
    if manifest.duplicated(["fold", "case"]).any():
        raise ValueError("A case appears more than once within a fold")
    if manifest[manifest["split"].eq("train")]["fault_status"].eq("unseen").any():
        raise ValueError("Unseen fault leaked into training")
    if manifest[manifest["split"].eq("validation")]["fault_status"].eq("unseen").any():
        raise ValueError("Unseen fault leaked into validation")
    expected_systems = set(map(str, config["systems"]))
    if set(manifest["system"]) != expected_systems:
        raise ValueError(f"System mismatch: expected {sorted(expected_systems)}, got {sorted(set(manifest['system']))}")
    groups = index.groupby(["system", "fault", "repetition"]).size()
    if not groups.eq(5).all():
        bad = groups[groups.ne(5)].to_dict()
        raise ValueError(f"Expected five root-cause services per system/fault/repetition group: {bad}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{suite.lower()}_folds.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "suite": suite,
        "case_rows_per_fold": {
            fold: int(len(manifest[manifest["fold"].eq(fold)].drop_duplicates("case")))
            for fold in config["folds"]
        },
        "counts": {
            "/".join(map(str, key if isinstance(key, tuple) else (key,))): int(count)
            for key, count in manifest.groupby(
                ["fold", "split", "system", "fault_status", "fault", "repetition"], dropna=False
            ).size().items()
        },
        "split_totals": {
            f"{fold}/{split}": int(len(group))
            for (fold, split), group in manifest.groupby(["fold", "split"])
        },
        "case_overlap_between_splits": False,
        "unseen_in_train_or_validation": False,
    }
    summary_path = out_dir / f"{suite.lower()}_folds_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(manifest)} fold-case rows)")
    print(manifest.groupby(["fold", "split", "system"]).size().to_string())
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
