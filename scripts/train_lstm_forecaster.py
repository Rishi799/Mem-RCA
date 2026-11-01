"""Train normal-only per-service LSTM forecasters and write held-out scores.

Example (from the repository root, with RE2 downloaded and a CUDA PyTorch build):
    python scripts/train_lstm_forecaster.py --system ob --fold fold1 --device cuda

Each model observes the first ``window_length - 1`` seconds and predicts the
last metric vector in the same window. Training uses only normal target samples.
Validation normal targets select the checkpoint; validation and test outputs
are written as per-service anomaly scores for the existing RCA evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised on systems without optional torch
    raise SystemExit(
        "PyTorch is required for Phase 2 training. Install the CUDA-compatible "
        "build for your laptop from https://pytorch.org/get-started/locally/"
    ) from exc

from memrca.data.loader import load_case, metric_columns_by_service
from memrca.models.lstm_forecaster import LSTMForecaster


def windows_for_case(
    case,
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    length: int,
    stride: int,
    preprocessing: str = "robust",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forecast the last sample in each width-sized input window.

    The model sees the first ``length - 1`` samples and predicts the final
    sample, keeping window labels aligned with the Phase 1 baseline.
    """
    frame = case.metrics.reindex(columns=feature_names)
    raw = frame.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    if preprocessing == "robust":
        transformed = np.sign(np.where(observed, raw, 0.0)) * np.log1p(np.abs(np.where(observed, raw, 0.0)))
        values = np.where(observed, transformed, center)
        values = np.clip((values - center) / scale, -10.0, 10.0).astype(np.float32, copy=False)
    else:
        values = np.where(observed, raw, center)
        values = ((values - center) / scale).astype(np.float32, copy=False)
    times = frame.index.to_numpy(dtype=np.int64)
    if len(times) > 1 and not np.all(np.diff(times) == 1):
        raise ValueError(f"{case.name}: expected complete one-second metric sampling")
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    window_starts: list[int] = []
    target_times: list[int] = []
    target_missing: list[float] = []
    for start in range(0, len(values) - length + 1, stride):
        target_index = start + length - 1
        window_start = int(times[start])
        target_time = int(times[target_index])
        if window_start < case.inject_time <= target_time:
            continue
        sequences.append(values[start:target_index])
        targets.append(values[target_index])
        window_starts.append(window_start)
        target_times.append(target_time)
        target_missing.append(float(1.0 - observed[target_index].mean()))
    if not sequences:
        return (
            np.empty((0, length - 1, len(feature_names)), dtype=np.float32),
            np.empty((0, len(feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(sequences), np.stack(targets),
        np.asarray(window_starts, dtype=np.int64),
        np.asarray(target_times, dtype=np.int64),
        np.asarray(target_missing, dtype=np.float32),
    )


def fit_scaler(cases: list, feature_names: list[str], preprocessing: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit imputation and standardization using normal training rows only."""
    frames = [case.normal().reindex(columns=feature_names) for case in cases]
    values = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    valid = values.columns[values.notna().any(axis=0)].tolist()
    if not valid:
        raise ValueError("No observed normal training metrics for this service")
    if preprocessing == "robust":
        coverage = values.notna().mean(axis=0)
        valid = coverage[coverage >= 0.8].index.tolist() or valid
        values = values[valid]
        values = np.sign(values) * np.log1p(np.abs(values))
        center = values.median(axis=0).to_numpy(dtype=np.float32)
        q25 = values.quantile(0.25, axis=0).to_numpy(dtype=np.float32)
        q75 = values.quantile(0.75, axis=0).to_numpy(dtype=np.float32)
        scale = ((q75 - q25) / 1.349).astype(np.float32)
        filled = values.fillna(pd.Series(center, index=valid)).to_numpy(dtype=np.float32)
        fallback = filled.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(scale) | (scale < 1e-6)
        scale[bad] = fallback[bad]
    else:
        values = values[valid]
        center = values.median(axis=0).to_numpy(dtype=np.float32)
        filled = values.fillna(pd.Series(center, index=valid)).to_numpy(dtype=np.float32)
        scale = filled.std(axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return center, scale, valid


def sequences_from_cases(
    cases: list,
    feature_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    length: int,
    stride: int,
    normal_only: bool,
    preprocessing: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[object, int]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    refs: list[tuple[object, int]] = []
    for case in cases:
        if not case.metrics.reindex(columns=feature_names).notna().any().any():
            continue
        x, y, _window_starts, target_times, _target_missing = windows_for_case(
            case, feature_names, center, scale, length, stride, preprocessing
        )
        keep = target_times < case.inject_time if normal_only else np.ones(len(target_times), dtype=bool)
        if keep.any():
            xs.append(x[keep])
            ys.append(y[keep])
            refs.extend((case, int(ts)) for ts in target_times[keep])
    if not xs:
        return (np.empty((0, length - 1, len(feature_names)), dtype=np.float32),
                np.empty((0, len(feature_names)), dtype=np.float32), refs)
    return np.concatenate(xs), np.concatenate(ys), refs


def train_service(
    train_cases: list,
    val_cases: list,
    service: str,
    metric_columns: list[str],
    args,
    device: torch.device,
) -> tuple[LSTMForecaster, dict]:
    center, scale, feature_names = fit_scaler(train_cases, metric_columns, args.preprocessing)
    train_x, train_y, _ = sequences_from_cases(
        train_cases, feature_names, center, scale, args.window, args.stride, normal_only=True,
        preprocessing=args.preprocessing,
    )
    val_x, val_y, _ = sequences_from_cases(
        val_cases, feature_names, center, scale, args.window, args.stride, normal_only=True,
        preprocessing=args.preprocessing,
    )
    if len(train_x) == 0 or len(val_x) == 0:
        raise ValueError(f"Insufficient normal sequences for {service}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = LSTMForecaster(len(feature_names), args.hidden_size, args.layers, args.dropout).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_inputs = torch.from_numpy(val_x).to(device)
    val_targets = torch.from_numpy(val_y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_inputs), val_targets).cpu())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"  {service} epoch {epoch}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {service}")
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "service": service,
        "feature_names": feature_names,
        "center": center,
        "scale": scale,
        "window_length": args.window,
        "stride": args.stride,
        "hidden_size": args.hidden_size,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "preprocessing": (
            "signed_log1p_train_median_iqr_clip10_80pct_coverage"
            if args.preprocessing == "robust" else "legacy_train_median_std"
        ),
        "score_calibration": args.score_calibration,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "history": history,
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--system", required=True, choices=["ob", "ss", "tt"])
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2", "fold3"])
    parser.add_argument("--window", type=int, default=None, help="Input sequence length in seconds")
    parser.add_argument("--stride", type=int, default=None, help="Prediction stride in seconds")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--preprocessing", choices=["legacy", "robust"], default="robust")
    parser.add_argument("--score-calibration", choices=["robust_z", "empirical_tail"], default="empirical_tail")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default="experiments/lstm")
    args = parser.parse_args()
    if args.window is not None and args.window < 2:
        raise ValueError("--window must be at least 2")
    if args.stride is not None and args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw or config["raw_dir"])
    args.window = int(args.window or config.get("window_length", 30))
    args.stride = int(args.stride or config.get("window_stride", 10))
    if args.window < 2:
        raise ValueError("Configured window length must be at least 2")
    manifest = pd.read_csv(args.manifest, dtype={"case": str, "fold": str, "split": str})
    selected = manifest[(manifest["system"] == args.system) & (manifest["fold"] == args.fold)]
    train_names = selected.loc[selected["split"] == "train", "case"].tolist()
    val_names = selected.loc[selected["split"] == "validation", "case"].tolist()
    score_rows = selected[selected["split"].isin(["validation", "test"])].copy()
    if not train_names or not val_names or score_rows.empty:
        raise ValueError(f"Missing expected split cases for {args.system}/{args.fold}")

    print(f"Device: {device}; system={args.system}; fold={args.fold}", flush=True)
    train_cases = [load_case(raw_dir, name) for name in train_names]
    val_cases = [load_case(raw_dir, name) for name in val_names]
    scoring_cases = {name: load_case(raw_dir, name) for name in score_rows["case"].tolist()}
    services: dict[str, set[str]] = {}
    for case in train_cases:
        for service, columns in metric_columns_by_service(case.metrics.columns).items():
            services.setdefault(service, set()).update(columns)

    out_dir = Path(args.out) / args.fold / args.system
    out_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[dict[str, object]] = []
    trained_services: list[str] = []
    skipped_services: dict[str, str] = {}
    for service, columns in sorted(services.items()):
        print(f"Training {service} ({len(columns)} metrics)", flush=True)
        try:
            model, metadata = train_service(train_cases, val_cases, service, sorted(columns), args, device)
        except ValueError as exc:
            skipped_services[service] = str(exc)
            print(f"  skipped {service}: {exc}", flush=True)
            continue
        trained_services.append(service)
        checkpoint_path = out_dir / f"{service}.pt"
        payload = {
            "model_state": model.state_dict(),
            "metadata": {
                **{k: v for k, v in metadata.items() if k not in {"history", "center", "scale"}},
                "center": metadata["center"].tolist(),
                "scale": metadata["scale"].tolist(),
            },
            "system": args.system,
            "fold": args.fold,
            "seed": args.seed,
        }
        torch.save(payload, checkpoint_path)
        (out_dir / f"{service}_history.json").write_text(
            json.dumps(metadata["history"], indent=2), encoding="utf-8"
        )

        for row in score_rows.itertuples(index=False):
            case = scoring_cases[row.case]
            feature_names = metadata["feature_names"]
            if not case.metrics.reindex(columns=feature_names).notna().any().any():
                continue
            x, y, window_starts, target_times, missing_fraction = windows_for_case(
                case, feature_names, metadata["center"], metadata["scale"], args.window, args.stride,
                args.preprocessing,
            )
            if len(x) == 0:
                continue
            model.eval()
            with torch.no_grad():
                predictions = model(torch.from_numpy(x).to(device)).cpu().numpy()
            errors = np.mean(np.square(predictions - y), axis=1)
            if args.preprocessing == "robust":
                errors = errors + missing_fraction
            fault_row = row._asdict()
            for window_start, time, score, missing in zip(window_starts, target_times, errors, missing_fraction):
                all_scores.append({
                    "fold": args.fold,
                    "system": args.system,
                    "split": fault_row["split"],
                    "case": case.name,
                    "fault": case.fault,
                    "fault_status": fault_row.get("fault_status", "unknown"),
                    "root_cause_service": case.root_cause,
                    "service": service,
                    "window_start": int(window_start),
                    "target_time": int(time),
                    "seconds_from_injection": int(window_start - case.inject_time),
                    "is_anomaly": bool(window_start >= case.inject_time),
                    "anomaly_score": float(score),
                    "missing_fraction": float(missing),
                })
        print(f"  saved {checkpoint_path}", flush=True)

    if not all_scores:
        raise RuntimeError("No validation/test scores were produced")
    score_path = out_dir / "service_window_scores.csv"
    scores = pd.DataFrame(all_scores)
    scores["raw_anomaly_score"] = scores["anomaly_score"]
    normal_validation = scores[
        scores["split"].eq("validation") & ~scores["is_anomaly"].astype(bool)
    ]
    calibration: dict[str, dict[str, object]] = {}
    if args.score_calibration == "empirical_tail":
        for service, group in normal_validation.groupby("service"):
            reference = np.sort(group["raw_anomaly_score"].to_numpy(dtype=float)).tolist()
            calibration[str(service)] = {"normal_score_reference": reference, "n": len(reference)}
        scores["anomaly_score"] = 0.0
        for service, indexes in scores.groupby("service").groups.items():
            reference = np.asarray(calibration.get(str(service), {}).get("normal_score_reference", []), dtype=float)
            if not len(reference):
                continue
            raw = scores.loc[indexes, "raw_anomaly_score"].to_numpy(dtype=float)
            upper_tail = len(reference) - np.searchsorted(reference, raw, side="left")
            p_value = (upper_tail + 1.0) / (len(reference) + 1.0)
            scores.loc[indexes, "anomaly_score"] = -np.log10(p_value)
    else:
        scores["calibration_center"] = 0.0
        scores["calibration_scale"] = 1.0
        for service, group in normal_validation.groupby("service"):
            values = group["raw_anomaly_score"].to_numpy(dtype=float)
            center = float(np.median(values))
            scale = float(1.4826 * np.median(np.abs(values - center)))
            if not np.isfinite(scale) or scale < 1e-9:
                scale = float(np.std(values))
            if not np.isfinite(scale) or scale < 1e-9:
                scale = 1.0
            calibration[str(service)] = {"center": center, "scale": scale}
        scores["calibration_center"] = scores["service"].map(
            {service: values["center"] for service, values in calibration.items()}
        ).fillna(0.0)
        scores["calibration_scale"] = scores["service"].map(
            {service: values["scale"] for service, values in calibration.items()}
        ).fillna(1.0)
        scores["anomaly_score"] = (
            scores["raw_anomaly_score"] - scores["calibration_center"]
        ) / scores["calibration_scale"]
    (out_dir / "validation_score_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    scores.to_csv(score_path, index=False)
    (out_dir / "training_summary.json").write_text(json.dumps({
        "system": args.system,
        "fold": args.fold,
        "device": str(device),
        "preprocessing": args.preprocessing,
        "score_calibration": args.score_calibration,
        "trained_services": trained_services,
        "skipped_services": skipped_services,
        "score_rows": len(all_scores),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {score_path} ({len(all_scores)} score rows)", flush=True)


if __name__ == "__main__":
    main()

