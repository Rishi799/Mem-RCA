"""Service hold-out for root-cause ranking (contribution C4).

In RE2 every system has the same five root-cause services in train and test, so a
ranker can score well by learning which services are usually roots. Here each root
service is held out in turn: the temporal-graph model is retrained without any incident
whose root is that service, and then ranks
  * seen_root   - repetition-3 incidents of the four remaining root services;
  * unseen_root - all incidents of the held-out service (a root it never saw as root).
All six fault types are known, so the only change is the root service.

The learned head is compared with learning-free metric-shift rankings over all services
and over the training-root candidates only (the candidate shortcut), and with the
no-telemetry prior. Training reuses scripts/train_temporal_graph.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, load_index  # noqa: E402
from memrca.openworld.fingerprint import metric_shift_profile  # noqa: E402

FOLD = "fold1"  # the trainer expects a fold name; each manifest holds exactly one split


def build_manifest(index: pd.DataFrame, system: str, held_out: str) -> pd.DataFrame:
    cases = index[index["suite"].eq("RE2") & index["system"].eq(system)].copy()
    seen = cases["root_cause_service"].ne(held_out)
    split = np.select(
        [seen & cases["repetition"].eq(1), seen & cases["repetition"].eq(2), seen & cases["repetition"].eq(3)],
        ["train", "validation", "test"], default="test")
    return pd.DataFrame({
        "fold": FOLD, "split": split, "case": cases["case"], "system": system, "fault": cases["fault"],
        "root_cause_service": cases["root_cause_service"],
        "fault_status": np.where(seen, "seen_root", "unseen_root"),
        "repetition": cases["repetition"], "has_logs": cases["has_logs"], "has_traces": cases["has_traces"],
    })


def rank_of(scores: np.ndarray, truth: int) -> int:
    return int(np.where(np.argsort(-scores, kind="stable") == truth)[0][0]) + 1


def evaluate_run(raw: str, run_dir: Path, manifest: pd.DataFrame, held_out: str) -> list[dict]:
    predictions = pd.read_csv(run_dir / "root_cause_case_predictions.csv")
    predictions = predictions[predictions["split"].eq("test")]
    train_roots = sorted(manifest.loc[manifest["split"].eq("train"), "root_cause_service"].unique())
    rows = []
    for row in predictions.itertuples(index=False):
        probabilities = json.loads(row.root_probabilities)
        services = list(probabilities)
        truth = services.index(row.root_cause_service)
        shift = metric_shift_profile(load_case(raw, row.case), services)
        magnitude = np.nan_to_num(np.abs(shift), nan=0.0).max(axis=1)
        candidate = np.isin(services, train_roots).astype(float)
        in_candidates = row.root_cause_service in train_roots
        ranks = {
            "learned_head": int(row.root_rank),
            "shift_all_services": rank_of(magnitude, truth),
            "shift_training_roots_first": rank_of(magnitude + 1e6 * candidate, truth),
        }
        base = {"system": row.system, "held_out": held_out, "case": row.case, "status": row.fault_status}
        for method, r in ranks.items():
            rows.append({**base, "method": method, "hit_at_1": float(r == 1), "hit_at_3": float(r <= 3),
                         "mrr": 1.0 / r})
        n = len(train_roots)
        rows.append({**base, "method": "no_telemetry_prior",
                     "hit_at_1": 1.0 / n if in_candidates else 0.0,
                     "hit_at_3": min(3, n) / n if in_candidates else 0.0,
                     "mrr": float(np.mean(1.0 / np.arange(1, n + 1))) if in_candidates else 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="experiments/service_holdout")
    ap.add_argument("--systems", nargs="+", default=["ob", "ss", "tt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    index = load_index(args.raw)

    rows = []
    for system in args.systems:
        roots = sorted(index.loc[index["suite"].eq("RE2") & index["system"].eq(system), "root_cause_service"].unique())
        for held_out in roots:
            manifest = build_manifest(index, system, held_out)
            manifest_path = out / "manifests" / f"{system}_{held_out}.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(manifest_path, index=False)
            run_root = out / "runs" / f"{system}_{held_out}"
            run_dir = run_root / FOLD / system
            if args.force or not (run_dir / "root_cause_case_predictions.csv").exists():
                print(f"training {system} without root {held_out} ...", flush=True)
                command = [sys.executable, str(PROJECT / "scripts" / "train_temporal_graph.py"),
                           "--raw", args.raw, "--manifest", str(manifest_path), "--fold", FOLD,
                           "--system", system, "--out", str(run_root), "--seed", str(args.seed),
                           "--device", args.device] + (["--force"] if args.force else [])
                subprocess.run(command, cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)
            rows += evaluate_run(args.raw, run_dir, manifest, held_out)
            print(f"done {system} / {held_out}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "service_holdout_by_case.csv", index=False)
    summary = (df.groupby(["system", "held_out", "status", "method"])[["hit_at_1", "hit_at_3", "mrr"]].mean()
               .groupby(["system", "status", "method"]).mean())
    summary.to_csv(out / "service_holdout_summary.csv")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
