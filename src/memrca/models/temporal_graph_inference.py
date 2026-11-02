"""Preprocessing-consistent inference API for the temporal graph checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from memrca.models.temporal_graph import MissingnessAwareTemporalGraph


class TemporalGraphPredictor:
    """Score one aligned service telemetry window with a saved graph model.

    ``window_by_service`` maps service names to DataFrames with ``window_length``
    synchronized samples. Each DataFrame uses the original metric column names.
    The final sample is the forecast target; preceding samples are model history.
    Missing services/features are permitted and represented by observation masks.
    """

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.services = list(payload["services"])
        self.config = dict(payload["config"])
        self.window_length = int(self.config.get("window", 30))
        self.feature_names = {
            service: list(payload["scalers"][service]["feature_names"])
            for service in self.services
        }
        self.center = {
            service: np.asarray(payload["scalers"][service]["center"], dtype=np.float32)
            for service in self.services
        }
        self.scale = {
            service: np.asarray(payload["scalers"][service]["scale"], dtype=np.float32)
            for service in self.services
        }
        self.edge_index = torch.as_tensor(payload["edge_index"], dtype=torch.long, device=self.device)
        self.model = MissingnessAwareTemporalGraph(
            payload["feature_sizes"],
            hidden_size=int(self.config["hidden_size"]),
            num_layers=int(self.config["layers"]),
            attention_heads=int(self.config["heads"]),
            dropout=float(self.config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

        calibration_path = self.checkpoint_path.parent / "validation_score_calibration.json"
        self.anomaly_references = json.loads(calibration_path.read_text(encoding="utf-8"))["normal_score_reference"]
        root_path = self.checkpoint_path.parent / "root_calibration.json"
        root_calibration = json.loads(root_path.read_text(encoding="utf-8"))
        self.temperature = float(root_calibration["temperature"])
        self.abstention_threshold = float(root_calibration["validation_abstention_threshold"])

    def score_window(self, window_by_service: dict[str, pd.DataFrame]) -> dict:
        if not window_by_service:
            raise ValueError("Provide at least one service window")
        lengths = {len(frame) for frame in window_by_service.values()}
        if lengths != {self.window_length}:
            raise ValueError(f"Every service window must contain exactly {self.window_length} aligned samples")

        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        targets: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        missing_fraction: dict[str, float] = {}
        for service in self.services:
            features = self.feature_names[service]
            frame = window_by_service.get(service)
            if frame is None:
                raw = np.full((self.window_length, len(features)), np.nan, dtype=np.float32)
            else:
                raw = frame.reindex(columns=features).replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
            observed = np.isfinite(raw)
            safe = np.where(observed, raw, 0.0)
            transformed = np.sign(safe) * np.log1p(np.abs(safe))
            normalized = np.clip((transformed - self.center[service]) / self.scale[service], -10.0, 10.0)
            normalized[~observed] = 0.0
            sequences.append(torch.from_numpy(normalized[:-1][None, ...]).to(self.device))
            masks.append(torch.from_numpy(observed[:-1].astype(np.float32)[None, ...]).to(self.device))
            targets.append(normalized[-1])
            target_masks.append(observed[-1].astype(np.float32))
            missing_fraction[service] = float(1.0 - observed[-1].mean())

        with torch.no_grad():
            output = self.model(sequences, masks, self.edge_index)
        forecasts = output["forecast"]
        root_logits = output["root_cause_logits"]
        assert isinstance(forecasts, list) and isinstance(root_logits, torch.Tensor)
        raw_root_logits = root_logits[0].detach().cpu().numpy()
        # Per-window values are only for ranking in the live view. Calibration
        # parameters were fitted on incident aggregates, so do not apply them here.
        logits = raw_root_logits.copy()
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        order = np.argsort(probabilities)[::-1]
        predicted = self.services[int(order[0])]
        confidence = float(probabilities[order[0]])

        service_scores = []
        for index, service in enumerate(self.services):
            mask = target_masks[index]
            observed_count = float(mask.sum())
            raw_error = None
            calibrated_score = None
            if observed_count > 0:
                prediction = forecasts[index][0].detach().cpu().numpy()
                raw_error = float(np.sum(np.square(prediction - targets[index]) * mask) / observed_count
                                  + missing_fraction[service])
                reference = np.asarray(self.anomaly_references.get(service, []), dtype=float)
                if len(reference):
                    tail = len(reference) - np.searchsorted(reference, raw_error, side="left")
                    calibrated_score = float(-np.log10((tail + 1.0) / (len(reference) + 1.0)))
                else:
                    calibrated_score = raw_error
            service_scores.append({
                "service": service,
                "raw_anomaly_score": raw_error,
                "anomaly_score": calibrated_score,
                "missing_fraction": missing_fraction[service],
                "root_score": float(probabilities[index]),
                "root_rank": int(np.where(order == index)[0][0]) + 1,
            })
        return {
            "fold": self.config.get("fold"),
            "system": self.config.get("system"),
            "predicted_root": predicted,
            "root_score": confidence,
            "abstain": None,
            "decision_unit": "single_window; incident-level calibration requires aggregation",
            "services": service_scores,
            "graph_edges": self.edge_index.detach().cpu().numpy().T.tolist(),
            "root_logits": raw_root_logits.tolist(),
            "interpretation": "Observed dependency evidence; not proof of causal fault propagation.",
        }

    def predict_incident(self, aligned_windows: list[dict[str, pd.DataFrame]]) -> dict:
        """Aggregate windows and apply the validation-calibrated incident decision.

        The temperature and abstention threshold were fitted to a top-quartile
        aggregate over the first 120 post-injection seconds. Callers must provide
        windows from that same incident horizon for the decision to be aligned.
        """
        results = [self.score_window(window) for window in aligned_windows]
        return self.predict_incident_from_window_scores(results)

    def predict_incident_from_window_scores(self, results: list[dict]) -> dict:
        """Calibrate an incident from already-scored post-injection windows."""
        if not results:
            raise ValueError("Provide at least one scored window for incident diagnosis")
        if len(results) > 12:
            raise ValueError("Incident calibration is defined for at most 12 windows in the first 120 seconds")
        logits = np.asarray([result["root_logits"] for result in results], dtype=float)
        top_count = max(1, int(np.ceil(len(logits) * 0.25)))
        aggregate = np.sort(logits, axis=0)[-top_count:].mean(axis=0)
        scaled = aggregate / max(self.temperature, 1e-8)
        scaled -= scaled.max()
        probabilities = np.exp(scaled) / np.exp(scaled).sum()
        ranking = probabilities.argsort()[::-1]
        predicted_index = int(ranking[0])
        confidence = float(probabilities[predicted_index])
        return {
            "fold": self.config.get("fold"),
            "system": self.config.get("system"),
            "predicted_root": self.services[predicted_index],
            "root_confidence": confidence,
            "abstain": bool(confidence < self.abstention_threshold),
            "abstention_threshold": self.abstention_threshold,
            "root_probabilities": {service: float(probabilities[i]) for i, service in enumerate(self.services)},
            "windows_aggregated": len(results),
            "decision_unit": "incident-level top-quartile logit aggregation",
            "service_scores_by_window": [result["services"] for result in results],
        }
