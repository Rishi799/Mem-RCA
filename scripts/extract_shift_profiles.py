"""Learning-free metric-shift profiles for every downloaded RCAEval case (RE1, RE2, RE3).

For each suite/system, services are the union of services with metric columns. Per case:
  shift  (services, 8)       robust log-z shift of each metric type in the first 120 s
  phase  (services, 3 * 8)   the same shift in the 0-30 s, 30-60 s and 60-120 s windows
Written to <out>/<suite>_<system>.npz with the case metadata from cases.parquet.
These profiles need no trained model, so they also cover suites the model never saw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, load_index, metric_columns_by_service  # noqa: E402
from memrca.openworld.fingerprint import metric_shift_profile, phase_shift_profile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="experiments/open_world/profiles")
    ap.add_argument("--suites", nargs="+", default=["RE1", "RE2", "RE3"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = load_index(args.raw)

    for suite in args.suites:
        for system in ("ob", "ss", "tt"):
            rows = index[index["suite"].eq(suite) & index["system"].eq(system)].sort_values("case")
            rows = rows[[(Path(args.raw) / c / "metrics.parquet").is_file() for c in rows["case"]]]
            if rows.empty:
                continue
            cases = [load_case(args.raw, c) for c in rows["case"]]
            services = sorted(set().union(*(metric_columns_by_service(c.metrics.columns) for c in cases)))
            shift = np.stack([metric_shift_profile(c, services) for c in cases])
            phase = np.stack([phase_shift_profile(c, services) for c in cases])
            np.savez_compressed(
                out / f"{suite.lower()}_{system}.npz",
                case=rows["case"].to_numpy(str), fault=rows["fault"].to_numpy(str),
                root=rows["root_cause_service"].to_numpy(str), repetition=rows["repetition"].to_numpy(int),
                services=np.array(services), shift=shift, phase=phase)
            print(f"{suite} {system}: {len(cases)} cases, {len(services)} services", flush=True)


if __name__ == "__main__":
    main()
