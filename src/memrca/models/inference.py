"""Load saved LSTM forecasters and score service telemetry windows."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from memrca.models.lstm_forecaster import LSTMForecaster


def load_service_forecaster(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[LSTMForecaster, dict]:
    """Load a trainer checkpoint and return the model plus preprocessing metadata."""
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    metadata = payload["metadata"]
    calibration_path = Path(checkpoint).with_name("validation_score_calibration.json")
    if calibration_path.exists():
        metadata["score_calibration"] = json.loads(calibration_path.read_text(encoding="utf-8"))
    model = LSTMForecaster(
        input_size=len(metadata["feature_names"]),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, {**metadata, "system": payload["system"], "fold": payload["fold"]}


def score_service_windows(
    model: LSTMForecaster,
    metadata: dict,
    metrics: pd.DataFrame,
    stride: int | None = None,
    device: str | torch.device = "cpu",
    inject_time: int | None = None,
) -> pd.DataFrame:
    """Return prediction errors for aligned service telemetry windows.

    ``metrics`` is indexed by integer Unix-second timestamps and contains the
    feature columns named in the checkpoint. Scores are higher when the model's
    prediction is less consistent with normal training telemetry.
    """
    device = torch.device(device)
    feature_names = list(metadata["feature_names"])
    length = int(metadata["window_length"])
    stride = int(stride or metadata["stride"])
    center = np.asarray(metadata["center"], dtype=np.float32)
    scale = np.asarray(metadata["scale"], dtype=np.float32)
    frame = metrics.reindex(columns=feature_names)
    raw = frame.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    if metadata.get("preprocessing", "").startswith("signed_log1p"):
        transformed = np.sign(np.where(observed, raw, 0.0)) * np.log1p(np.abs(np.where(observed, raw, 0.0)))
        values = np.where(observed, transformed, center)
        values = np.clip((values - center) / scale, -10.0, 10.0).astype(np.float32, copy=False)
    else:
        values = np.where(observed, raw, center)
        values = ((values - center) / scale).astype(np.float32, copy=False)
    times = frame.index.to_numpy(dtype=np.int64)
    if len(times) > 1 and not np.all(np.diff(times) == 1):
        raise ValueError("Expected complete one-second metric sampling")
    if not frame.notna().any().any():
        return pd.DataFrame(columns=["window_start", "target_time", "anomaly_score"])
    records: list[dict[str, float | int]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values) - length + 1, stride):
            target_index = start + length - 1
            window_start = int(times[start])
            target_time = int(times[target_index])
            if inject_time is not None and window_start < inject_time <= target_time:
                continue
            x = torch.from_numpy(values[start:target_index][None, ...]).to(device)
            y = torch.from_numpy(values[target_index]).to(device)
            prediction = model(x)[0]
            raw_tensor = torch.mean(torch.square(prediction - y))
            if metadata.get("preprocessing", "").startswith("signed_log1p"):
                raw_tensor = raw_tensor + (1.0 - observed[target_index].mean())
            raw_score = float(raw_tensor.cpu())
            calibration = metadata.get("score_calibration", {}).get(metadata.get("service", ""), {})
            reference = np.asarray(calibration.get("normal_score_reference", []), dtype=float)
            if len(reference):
                tail_count = len(reference) - np.searchsorted(reference, raw_score, side="left")
                score = float(-np.log10((tail_count + 1.0) / (len(reference) + 1.0)))
            elif "center" in calibration:
                scale = float(calibration.get("scale", 1.0)) or 1.0
                score = float((raw_score - float(calibration["center"])) / scale)
            else:
                score = raw_score
            records.append({
                "window_start": window_start,
                "target_time": target_time,
                "anomaly_score": score,
                "raw_anomaly_score": raw_score,
            })
    return pd.DataFrame(records, columns=[
        "window_start", "target_time", "raw_anomaly_score", "anomaly_score"
    ])

