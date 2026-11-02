"""Create an RE2 train/validation to independent-suite test manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--base-manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--fold", default="fold1", choices=["fold1", "fold2", "fold3"])
    parser.add_argument("--suite", default="RE3", choices=["RE1", "RE3"],
                        help="Independent final test suite")
    parser.add_argument("--validation-suite", default="RE1", choices=["RE1", "RE3"],
                        help="Independent validation/calibration suite")
    parser.add_argument("--system", choices=["ob", "ss", "tt"], default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw = Path(args.raw)
    base = pd.read_csv(args.base_manifest)
    keep = base[
        base["fold"].astype(str).eq(args.fold)
        & base["split"].eq("train")
    ].copy()
    if args.system:
        keep = keep[keep["system"].astype(str).eq(args.system)]
    index = pd.read_parquet(raw / "cases.parquet")
    suite_rows = []
    for suite_name, split_name, status_name in (
        (args.validation_suite, "validation", "external_validation"),
        (args.suite, "test", "external_test"),
    ):
        external = index[index["suite"].astype(str).str.upper().eq(suite_name)].copy()
        if args.system:
            external = external[external["system"].astype(str).eq(args.system)]
        if external.empty:
            raise ValueError(f"No {suite_name} cases in {raw / 'cases.parquet'}")
        missing = [str(row.case) for row in external.itertuples(index=False)
                   if not (raw / str(row.case) / "metrics.parquet").is_file()
                   or not (raw / str(row.case) / "inject_time.txt").is_file()]
        if missing:
            examples = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} {suite_name} cases are not downloaded under {raw}; examples: {examples}. "
                f"Run `python scripts/download.py --suite {suite_name.lower()}` first."
            )
        suite_rows.append(pd.DataFrame({
            "fold": args.fold,
            "split": split_name,
            "case": external["case"].astype(str),
            "system": external["system"].astype(str),
            "fault": external["fault"].astype(str),
            "fault_status": status_name,
            "root_cause_service": external["root_cause_service"].astype(str),
            "repetition": external["repetition"].astype(int),
            "suite": suite_name,
        }))
    manifest = pd.concat([keep, *suite_rows], ignore_index=True, sort=False)
    if manifest.duplicated(["fold", "case"]).any():
        raise ValueError("A case occurs more than once in the external manifest")
    out = Path(args.out) if args.out else Path("experiments/splits") / f"{args.fold}_re2_train_{args.validation_suite.lower()}_val_{args.suite.lower()}_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"Wrote {len(keep)} RE2 training rows, {len(suite_rows[0])} {args.validation_suite} validation rows, and {len(suite_rows[1])} {args.suite} test rows to {out.resolve()}")
    print("Fit training transforms on RE2 train only, select/calibrate on the validation suite, and do not tune on the final test suite.")


if __name__ == "__main__":
    main()
