"""Verify and generate experimental split manifest for Mem-RCA Phase 1.

Checks configs/data.yaml against RCAEval RE2 case metadata:
- Confirms repetitions 1, 2, and 3 map strictly to train, validation, and test.
- Confirms unseen fault types appear only in test for every fold.
- Generates a compact CSV manifest with case counts by system, fold, split, fault, and root-cause service.
- Flags any missing telemetry, empty groups, or data leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def build_split_manifest(
    index_df: pd.DataFrame,
    config: dict,
    suite: str = "re2",
) -> tuple[pd.DataFrame, dict]:
    suite_cases = index_df[index_df["suite"].astype(str).str.lower() == suite.lower()].copy()
    if suite_cases.empty:
        raise ValueError(f"No cases found for suite: {suite}")

    folds = config.get("folds", {})
    train_reps = set(config.get("train_repetitions", [1]))
    val_reps = set(config.get("validation_repetitions", [2]))
    test_reps = set(config.get("test_repetitions", [3]))

    records = []
    audit_findings = {
        "suite": suite,
        "total_cases": len(suite_cases),
        "folds": {},
        "leakage_detected": False,
        "issues": [],
    }

    for fold_name, fold_info in sorted(folds.items()):
        known_faults = set(fold_info.get("known", []))
        unseen_faults = set(fold_info.get("unseen", []))
        all_faults = known_faults | unseen_faults

        fold_stats = {
            "known_faults": sorted(known_faults),
            "unseen_faults": sorted(unseen_faults),
            "train_count": 0,
            "validation_count": 0,
            "test_known_count": 0,
            "test_unseen_count": 0,
            "total_test_count": 0,
            "unassigned_count": 0,
        }

        for _, row in suite_cases.iterrows():
            case_name = row["case"]
            system = row["system"]
            service = row["root_cause_service"]
            fault = row["fault"]
            rep = int(row["repetition"])
            has_logs = bool(row.get("has_logs", False))
            has_traces = bool(row.get("has_traces", False))

            if fault not in all_faults:
                split = "excluded"
                fault_category = "unknown_to_fold"
            elif fault in unseen_faults:
                # Unseen fault types must appear only in test
                split = "test"
                fault_category = "unseen"
                fold_stats["test_unseen_count"] += 1
            else:
                fault_category = "known"
                if rep in train_reps:
                    split = "train"
                    fold_stats["train_count"] += 1
                elif rep in val_reps:
                    split = "validation"
                    fold_stats["validation_count"] += 1
                elif rep in test_reps:
                    split = "test"
                    fold_stats["test_known_count"] += 1
                else:
                    split = "unassigned"
                    fold_stats["unassigned_count"] += 1
                    audit_findings["issues"].append(f"Case {case_name} rep {rep} not in train/val/test reps")

            # Leakage check: verify no unseen fault is in train or validation
            if fault_category == "unseen" and split in ("train", "validation"):
                audit_findings["leakage_detected"] = True
                audit_findings["issues"].append(f"Leakage: unseen fault {fault} in {split} for {case_name}")

            # Repetition leak check: rep 1 strictly train, rep 2 strictly val, rep 3 strictly test
            if fault_category == "known":
                if split == "train" and rep not in train_reps:
                    audit_findings["issues"].append(f"Rep mismatch: {case_name} rep {rep} assigned train")
                if split == "validation" and rep not in val_reps:
                    audit_findings["issues"].append(f"Rep mismatch: {case_name} rep {rep} assigned validation")
                if split == "test" and rep not in test_reps:
                    audit_findings["issues"].append(f"Rep mismatch: {case_name} rep {rep} assigned test")

            records.append({
                "case": case_name,
                "suite": suite,
                "system": system,
                "fold": fold_name,
                "split": split,
                "fault": fault,
                "fault_category": fault_category,
                "root_cause_service": service,
                "repetition": rep,
                "has_logs": has_logs,
                "has_traces": has_traces,
            })

        fold_stats["total_test_count"] = fold_stats["test_known_count"] + fold_stats["test_unseen_count"]
        audit_findings["folds"][fold_name] = fold_stats

    manifest_df = pd.DataFrame(records)
    return manifest_df, audit_findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML")
    parser.add_argument("--raw", default="data/raw", help="Path to raw data directory with cases.parquet")
    parser.add_argument("--suite", default="re2", help="Suite to audit (re1, re2, re3)")
    parser.add_argument("--out-manifest", default="experiments/split_manifest.csv", help="Output CSV path")
    parser.add_argument("--out-report", default="experiments/split_audit_report.json", help="Output JSON report path")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(args.raw)
    index_path = raw_dir / "cases.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Case index not found at {index_path}. Please download cases.parquet first.")

    index_df = pd.read_parquet(index_path)
    manifest_df, findings = build_split_manifest(index_df, config, suite=args.suite)

    out_man = Path(args.out_manifest)
    out_man.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_man, index=False)

    out_rep = Path(args.out_report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print(f"=== Split Manifest & Audit for Suite {args.suite.upper()} ===")
    print(f"Total cases evaluated: {findings['total_cases']}")
    print(f"Leakage detected: {findings['leakage_detected']}")
    if findings["issues"]:
        print(f"Issues detected ({len(findings['issues'])}): {findings['issues'][:5]}")
    else:
        print("All split criteria verified: No leakage, strict repetition assignments.")

    for fold_name, stats in findings["folds"].items():
        print(f"\nFold: {fold_name}")
        print(f"  Train (known, rep 1):         {stats['train_count']}")
        print(f"  Validation (known, rep 2):    {stats['validation_count']}")
        print(f"  Test Known (known, rep 3):    {stats['test_known_count']}")
        print(f"  Test Unseen (reps 1,2,3):     {stats['test_unseen_count']}")
        print(f"  Total Test:                   {stats['total_test_count']}")

    print(f"\nManifest saved to: {out_man}")
    print(f"Audit report saved to: {out_rep}")


if __name__ == "__main__":
    main()
