"""Run the LSTM trainer for multiple system/fold combinations sequentially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=["ob", "ss", "tt"], default=["ob", "ss", "tt"])
    parser.add_argument("--folds", nargs="+", choices=["fold1", "fold2", "fold3"], default=["fold1", "fold2", "fold3"])
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--preprocessing", choices=["legacy", "robust"], default="robust")
    parser.add_argument("--score-calibration", choices=["robust_z", "empirical_tail"], default="empirical_tail")
    parser.add_argument("--force", action="store_true", help="Retrain even when a score file already exists")
    parser.add_argument("--out", default="experiments/lstm")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    trainer = Path(__file__).with_name("train_lstm_forecaster.py")
    failures: list[str] = []
    jobs = [(fold, system) for fold in args.folds for system in args.systems]
    for index, (fold, system) in enumerate(jobs, start=1):
        run_dir = Path(args.out) / fold / system
        score_file = run_dir / "service_window_scores.csv"
        if score_file.exists() and not args.force:
            print(f"[{index}/{len(jobs)}] Skip {fold}/{system}: {score_file} already exists", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training.log"
        command = [
            sys.executable, str(trainer), "--system", system, "--fold", fold,
            "--device", args.device, "--epochs", str(args.epochs), "--out", args.out,
            "--preprocessing", args.preprocessing, "--score-calibration", args.score_calibration,
        ]
        print(f"[{index}/{len(jobs)}] Train {fold}/{system}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            failures.append(f"{fold}/{system} (exit {result.returncode}; see {log_path})")
            print(f"  FAILED: {failures[-1]}", flush=True)
        else:
            print(f"  Complete: {score_file}", flush=True)

    if failures:
        raise SystemExit("Training grid had failures: " + "; ".join(failures))
    print("Training grid complete.", flush=True)


if __name__ == "__main__":
    main()
