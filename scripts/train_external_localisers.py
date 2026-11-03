"""Train the temporal-graph localiser on each RE1 open-world fold (docs/EXTERNAL_VALIDATION_PROTOCOL.md).

Per system and held-out fault type: memory/train = repetitions 1-2 of the four known types,
validation = repetition 3, test = repetitions 4-5 (known) + every incident of the held-out
type (unseen). RE1 has no traces, so the graph has only self-loops. Training reuses
scripts/train_temporal_graph.py unchanged; its per-incident root probabilities
(root_cause_case_predictions.csv) are read by scripts/evaluate_external.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_index  # noqa: E402
from memrca.openworld.incidents import RE1_TYPES, holdout_type_split  # noqa: E402

FOLD = "fold1"  # the trainer expects a fold name; each manifest holds exactly one split
ATTEMPTS = 3  # CUDA/cuDNN occasionally fails transiently on Windows when several runs share the GPU


def build_manifest(index: pd.DataFrame, system: str, unseen: str) -> pd.DataFrame:
    rows = index[index["suite"].eq("RE1") & index["system"].eq(system)].copy()
    split, status = holdout_type_split(rows["fault"].to_numpy(), rows["repetition"].to_numpy(), unseen)
    return pd.DataFrame({
        "fold": FOLD, "split": split, "case": rows["case"], "system": system, "fault": rows["fault"],
        "root_cause_service": rows["root_cause_service"], "fault_status": status,
        "repetition": rows["repetition"], "has_logs": rows["has_logs"], "has_traces": rows["has_traces"],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="experiments/external/re1")
    ap.add_argument("--systems", nargs="+", default=["ob", "ss", "tt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    index = load_index(args.raw)
    for system in args.systems:
        for unseen in RE1_TYPES:
            manifest_path = out / "manifests" / f"{system}_{unseen}.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            build_manifest(index, system, unseen).to_csv(manifest_path, index=False)
            run_root = out / "runs" / f"{system}_{unseen}"
            if (run_root / FOLD / system / "root_cause_case_predictions.csv").exists():
                continue
            command = [sys.executable, str(PROJECT / "scripts" / "train_temporal_graph.py"),
                       "--raw", args.raw, "--manifest", str(manifest_path), "--fold", FOLD,
                       "--system", system, "--out", str(run_root), "--edges", str(out / "no_trace_edges.csv"),
                       "--seed", str(args.seed), "--device", args.device, "--force"]
            for attempt in range(1, ATTEMPTS + 1):
                print(f"training RE1 {system}, unseen type {unseen} (attempt {attempt}) ...", flush=True)
                if subprocess.run(command, cwd=PROJECT, stdout=subprocess.DEVNULL).returncode == 0:
                    break
            else:
                raise RuntimeError(f"training failed {ATTEMPTS} times for RE1 {system} / {unseen}")
            print(f"done RE1 {system} / {unseen}", flush=True)


if __name__ == "__main__":
    main()
