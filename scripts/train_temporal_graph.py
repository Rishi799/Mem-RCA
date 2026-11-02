"""Train one missingness-aware temporal graph model for a fold/system.

This is the Phase 3 candidate model. It trains forecasting on normal windows,
root-cause ranking on training incidents, and consistency across masked views.
Do not use the already-inspected RE2 test outputs as confirmatory evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.temporal_graph import GraphWindow, build_temporal_graph_data  # noqa: E402
from memrca.models.temporal_graph import (  # noqa: E402
    MissingnessAwareTemporalGraph,
    missingness_consistency_loss,
)


def collate(windows: list[GraphWindow]):
    node_count = len(windows[0].values)
    sequences = [torch.from_numpy(np.stack([row.values[node] for row in windows])) for node in range(node_count)]
    masks = [torch.from_numpy(np.stack([row.masks[node] for row in windows])) for node in range(node_count)]
    targets = [torch.from_numpy(np.stack([row.targets[node] for row in windows])) for node in range(node_count)]
    target_masks = [torch.from_numpy(np.stack([row.target_masks[node] for row in windows])) for node in range(node_count)]
    normal = torch.tensor([not row.is_anomaly for row in windows], dtype=torch.bool)
    root_index = torch.tensor([row.root_index for row in windows], dtype=torch.long)
    metadata = [{
        "case": row.case, "split": row.split, "fault": row.fault,
        "fault_status": row.fault_status, "root_service": row.root_service,
        "window_start": row.window_start, "target_time": row.target_time,
        "seconds_from_injection": row.seconds_from_injection,
        "is_anomaly": row.is_anomaly,
    } for row in windows]
    return sequences, masks, targets, target_masks, normal, root_index, metadata


def to_device(values: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [value.to(device, non_blocking=True) for value in values]


def forecast_loss(outputs, targets, masks, normal: torch.Tensor) -> torch.Tensor:
    forecasts = outputs["forecast"]
    assert isinstance(forecasts, list)
    losses = []
    for prediction, target, observed in zip(forecasts, targets, masks):
        valid = observed * normal[:, None].to(observed.dtype)
        losses.append((((prediction - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def root_loss(outputs, root_index: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    logits = outputs["root_cause_logits"]
    assert isinstance(logits, torch.Tensor)
    faulted = (~normal) & root_index.ge(0)
    if not faulted.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[faulted], root_index[faulted])


def run_epoch(model, loader, edge_index, device, args, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    root_rr: list[float] = []
    for batch in loader:
        sequences, masks, targets, target_masks, normal, root_index, _meta = batch
        sequences = to_device(sequences, device)
        masks = to_device(masks, device)
        targets = to_device(targets, device)
        target_masks = to_device(target_masks, device)
        normal = normal.to(device)
        root_index = root_index.to(device)
        model_masks = [torch.ones_like(mask) for mask in masks] if args.disable_mask_input else masks
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            clean = model(sequences, model_masks, edge_index)
            loss_forecast = forecast_loss(clean, targets, target_masks, normal)
            loss_root = root_loss(clean, root_index, normal)
            if training and args.mask_probability > 0:
                masked_sequences: list[torch.Tensor] = []
                masked_masks: list[torch.Tensor] = []
                for sequence, observed in zip(sequences, masks):
                    keep = (torch.rand_like(observed) >= args.mask_probability).to(observed.dtype)
                    masked_masks.append(torch.ones_like(observed) if args.disable_mask_input else observed * keep)
                    masked_sequences.append(sequence * keep)
                masked = model(masked_sequences, masked_masks, edge_index)
                loss_consistency = missingness_consistency_loss(clean, masked)
            else:
                loss_consistency = loss_forecast * 0.0
            loss = loss_forecast + args.root_weight * loss_root + args.consistency_weight * loss_consistency
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))

        logits = clean["root_cause_logits"]
        assert isinstance(logits, torch.Tensor)
        valid = (~normal) & root_index.ge(0)
        if valid.any():
            ranks = logits[valid].argsort(dim=1, descending=True).argsort(dim=1) + 1
            rr = 1.0 / ranks.gather(1, root_index[valid, None]).squeeze(1).float()
            root_rr.extend(rr.detach().cpu().tolist())
    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(root_rr)) if root_rr else float("nan")


def score_windows(model, windows, services, scalers, edge_index, device, batch_size, disable_mask_input=False):
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False, collate_fn=collate)
    records: list[dict] = []
    model.eval()
    with torch.no_grad():
        for sequences, masks, targets, target_masks, _normal, _root, metadata in loader:
            sequences = to_device(sequences, device)
            masks = to_device(masks, device)
            targets = to_device(targets, device)
            target_masks = to_device(target_masks, device)
            model_masks = [torch.ones_like(mask) for mask in masks] if disable_mask_input else masks
            output = model(sequences, model_masks, edge_index)
            forecasts = output["forecast"]
            root_logits = output["root_cause_logits"]
            assert isinstance(forecasts, list) and isinstance(root_logits, torch.Tensor)
            root_logit_values = root_logits.cpu().numpy()
            per_service_errors: list[np.ndarray] = []
            per_service_missing: list[np.ndarray] = []
            for prediction, target, observed in zip(forecasts, targets, target_masks):
                observed_count = observed.sum(dim=1)
                missing = 1.0 - observed.mean(dim=1)
                errors = (((prediction - target) ** 2) * observed).sum(dim=1) / observed_count.clamp_min(1.0)
                errors = errors + missing
                errors = errors.masked_fill(observed_count.eq(0), float("nan"))
                per_service_errors.append(errors.cpu().numpy())
                per_service_missing.append(missing.cpu().numpy())
            for batch_index, meta in enumerate(metadata):
                for node, service in enumerate(services):
                    raw_error = float(per_service_errors[node][batch_index])
                    records.append({
                        **meta,
                        "service": service,
                        "raw_anomaly_score": raw_error,
                        "anomaly_score": raw_error,
                        "missing_fraction": float(per_service_missing[node][batch_index]),
                        "root_logit": float(root_logit_values[batch_index, node]),
                        "root_cause_service": meta["root_service"],
                    })
    return pd.DataFrame(records)


def aggregate_case_root_logits(scores: pd.DataFrame, services: list[str], horizon: int) -> pd.DataFrame:
    """Aggregate per-window logits into one service score vector per incident."""
    selected = scores[
        scores["is_anomaly"].astype(bool)
        & scores["seconds_from_injection"].ge(0)
        & scores["seconds_from_injection"].lt(horizon)
    ]
    output: list[dict] = []
    for (split, fold, system, case), group in selected.groupby(["split", "fold", "system", "case"]):
        vector = np.zeros(len(services), dtype=np.float64)
        for service, service_group in group.groupby("service"):
            values = service_group["root_logit"].to_numpy(dtype=float)
            top = max(1, int(np.ceil(len(values) * 0.25)))
            if service in services:
                vector[services.index(service)] = float(np.sort(values)[-top:].mean())
        row = {"split": split, "fold": fold, "system": system, "case": case,
               "root_cause_service": str(group["root_cause_service"].iloc[0]),
               "fault_status": str(group["fault_status"].iloc[0]),
               "root_logits": vector.tolist()}
        output.append(row)
    return pd.DataFrame(output)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar temperature on validation incidents only."""
    if len(logits) == 0:
        return 1.0
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def selective_threshold(probabilities: np.ndarray, labels: np.ndarray, target_risk: float) -> tuple[float, float, float]:
    """Choose the lowest validation confidence threshold meeting target error."""
    if len(probabilities) == 0:
        return float("inf"), float("nan"), 0.0
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    best_threshold = float("inf")
    best_coverage = 0.0
    best_risk = float("nan")
    for threshold in np.unique(np.r_[0.0, confidence]):
        kept = confidence >= threshold
        if not kept.any():
            continue
        coverage = float(kept.mean())
        risk = float(1.0 - correct[kept].mean())
        if risk <= target_risk and coverage > best_coverage:
            best_threshold, best_coverage, best_risk = float(threshold), coverage, risk
    return best_threshold, best_risk, best_coverage


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict[str, float]:
    if len(probabilities) == 0:
        return {"brier_score": float("nan"), "expected_calibration_error": float("nan")}
    one_hot = np.eye(probabilities.shape[1])[labels]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(confidence[in_bin].mean()) - float(correct[in_bin].mean()))
    return {
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "expected_calibration_error": float(ece),
    }


def calibrate_root_head(scores: pd.DataFrame, services: list[str], target_risk: float, horizon: int):
    """Calibrate root probabilities and select a validation-only abstention rule."""
    cases = aggregate_case_root_logits(scores, services, horizon)
    validation = cases[cases["split"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("No post-injection validation cases available for root calibration")
    validation_logits = np.stack(validation["root_logits"].to_numpy())
    labels = np.asarray([
        services.index(name) if name in services else -1
        for name in validation["root_cause_service"].astype(str)
    ], dtype=int)
    valid = labels >= 0
    validation_logits, labels = validation_logits[valid], labels[valid]
    temperature = fit_temperature(validation_logits, labels)
    calibrated = torch.softmax(torch.tensor(validation_logits / temperature), dim=1).numpy()
    threshold, val_risk, val_coverage = selective_threshold(calibrated, labels, target_risk)
    val_calibration_metrics = calibration_metrics(calibrated, labels)

    rows: list[dict] = []
    for row in cases.to_dict("records"):
        logits = np.asarray(row["root_logits"], dtype=float)
        probability = torch.softmax(torch.tensor(logits / temperature), dim=0).numpy()
        prediction = int(probability.argmax())
        truth = str(row["root_cause_service"])
        rank_order = probability.argsort()[::-1]
        rank = next((index for index, node in enumerate(rank_order, start=1)
                     if services[int(node)] == truth), None)
        confidence = float(probability[prediction])
        rows.append({
            "split": row["split"], "fold": row["fold"], "system": row["system"], "case": row["case"],
            "fault_status": row["fault_status"],
            "root_cause_service": truth, "predicted_root": services[prediction],
            "root_rank": rank, "hit_at_1": int(rank == 1),
            "hit_at_3": int(rank is not None and rank <= 3),
            "mrr": 0.0 if rank is None else 1.0 / rank,
            "confidence": confidence, "abstain": bool(confidence < threshold),
            "root_probabilities": json.dumps({service: float(probability[i]) for i, service in enumerate(services)}),
        })
    predictions = pd.DataFrame(rows)
    return predictions, {
        "temperature": temperature,
        "target_validation_risk": target_risk,
        "validation_abstention_threshold": threshold,
        "validation_risk_at_threshold": val_risk,
        "validation_coverage_at_threshold": val_coverage,
        "validation_brier_score": val_calibration_metrics["brier_score"],
        "validation_expected_calibration_error": val_calibration_metrics["expected_calibration_error"],
        "validation_incidents": int(len(validation_logits)),
        "aggregation": f"mean top quartile of root logits in first {horizon}s after injection",
        "test_used_for_calibration": False,
    }


def apply_window_root_probabilities(scores: pd.DataFrame, services: list[str], temperature: float,
                                     abstention_threshold: float) -> pd.DataFrame:
    """Attach relative per-window ranking scores; incident calibration is separate."""
    output = scores.copy()
    output["root_score"] = np.nan
    output["root_rank"] = np.nan
    output["root_confidence"] = np.nan
    output["root_predicted"] = ""
    output["root_abstain"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    for _key, group in output.groupby(["case", "window_start"], sort=False):
        logits_by_service = dict(zip(group["service"].astype(str), group["root_logit"].astype(float)))
        logits = np.asarray([logits_by_service.get(service, 0.0) for service in services])
        logits -= logits.max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        order = probabilities.argsort()[::-1]
        rank_lookup = {services[node]: rank for rank, node in enumerate(order, start=1)}
        predicted = services[int(order[0])]
        confidence = float(probabilities[order[0]])
        output.loc[group.index, "root_score"] = group["service"].astype(str).map(
            {service: float(probabilities[i]) for i, service in enumerate(services)}
        )
        output.loc[group.index, "root_rank"] = group["service"].astype(str).map(rank_lookup)
        output.loc[group.index, "root_confidence"] = np.nan
        output.loc[group.index, "root_predicted"] = predicted
        output.loc[group.index, "root_abstain"] = pd.NA
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--fold", required=True, choices=["fold1", "fold2", "fold3"])
    parser.add_argument("--system", required=True, choices=["ob", "ss", "tt"])
    parser.add_argument("--out", default="experiments/temporal_graph")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.2)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--disable-graph", action="store_true", help="Ablation: use self-loops only")
    parser.add_argument("--disable-mask-input", action="store_true",
                        help="Ablation: hide observation masks from the model")
    parser.add_argument("--target-root-risk", type=float, default=0.2,
                        help="Maximum validation root error for abstention threshold selection")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if not 0.0 <= args.target_root_risk < 1.0:
        raise ValueError("--target-root-risk must be in [0, 1)")

    run_dir = Path(args.out) / args.fold / args.system
    model_path = run_dir / "model.pt"
    score_path = run_dir / "service_window_scores.csv"
    if score_path.exists() and model_path.exists() and not args.force:
        print(f"Outputs exist under {run_dir}; pass --force to retrain")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    data = build_temporal_graph_data(
        args.raw, args.manifest, args.edges, args.fold, args.system,
        window_length=args.window, stride=args.stride,
    )
    edge_index = torch.from_numpy(data.edge_index).to(device=device, dtype=torch.long)
    if args.disable_graph:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    model = MissingnessAwareTemporalGraph(
        data.feature_sizes, hidden_size=args.hidden_size, num_layers=args.layers,
        attention_heads=args.heads, dropout=args.dropout,
    ).to(device)
    if model_path.exists() and not args.force:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if checkpoint.get("services") != data.services or checkpoint.get("feature_sizes") != data.feature_sizes:
            raise ValueError("Saved checkpoint does not match the rebuilt service/feature schema; use --force")
        model.load_state_dict(checkpoint["model_state"])
        best_mrr = float(checkpoint.get("best_validation_root_mrr", float("nan")))
        if not np.isfinite(best_mrr):
            history_path = run_dir / "training_history.json"
            if history_path.exists():
                saved_history = json.loads(history_path.read_text(encoding="utf-8"))
                values = [float(row["validation_root_mrr"]) for row in saved_history
                          if np.isfinite(float(row["validation_root_mrr"]))]
                if values:
                    best_mrr = max(values)
        print(f"Scoring with existing checkpoint: {model_path}")
    else:
        train_loader = DataLoader(data.windows["train"], batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate)
        validation_loader = DataLoader(data.windows["validation"], batch_size=args.batch_size,
                                       shuffle=False, collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best_state = None
        best_mrr = -1.0
        best_loss = float("inf")
        stale = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_mrr = run_epoch(model, train_loader, edge_index, device, args, optimizer)
            val_loss, val_mrr = run_epoch(model, validation_loader, edge_index, device, args)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_root_mrr": train_mrr,
                            "validation_loss": val_loss, "validation_root_mrr": val_mrr})
            print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_root_mrr={val_mrr:.4f}", flush=True)
            improved = (val_mrr > best_mrr + 1e-5) or (abs(val_mrr - best_mrr) <= 1e-5 and val_loss < best_loss)
            if improved:
                best_mrr, best_loss = val_mrr, val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError("Training did not produce a validation checkpoint")
        model.load_state_dict(best_state)
        torch.save({"model_state": model.state_dict(), "services": data.services,
                    "feature_sizes": data.feature_sizes, "edge_index": data.edge_index,
                    "scalers": {service: {"feature_names": scaler.feature_names,
                                           "center": scaler.center.tolist(), "scale": scaler.scale.tolist()}
                                for service, scaler in data.scalers.items()},
                    "fold": args.fold, "system": args.system, "best_validation_root_mrr": best_mrr,
                    "config": vars(args)}, model_path)
        (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    scored = pd.concat([
        score_windows(model, data.windows["validation"], data.services, data.scalers, edge_index, device, args.batch_size,
                      args.disable_mask_input),
        score_windows(model, data.windows["test"], data.services, data.scalers, edge_index, device, args.batch_size,
                      args.disable_mask_input),
    ], ignore_index=True)
    normal_val = scored[(scored["split"].eq("validation")) & (~scored["is_anomaly"].astype(bool))]
    calibration: dict[str, list[float]] = {}
    for service, group in normal_val.groupby("service"):
        raw = group["raw_anomaly_score"].dropna().to_numpy(float)
        calibration[str(service)] = np.sort(raw).tolist()
    for service, indexes in scored.groupby("service").groups.items():
        reference = np.asarray(calibration.get(str(service), []), dtype=float)
        raw = scored.loc[indexes, "raw_anomaly_score"].to_numpy(float)
        valid = np.isfinite(raw)
        if len(reference):
            tail_count = len(reference) - np.searchsorted(reference, raw[valid], side="left")
            scored.loc[np.asarray(indexes)[valid], "anomaly_score"] = -np.log10(
                (tail_count + 1.0) / (len(reference) + 1.0)
            )
        scored.loc[np.asarray(indexes)[~valid], "anomaly_score"] = np.nan
    scored.insert(0, "fold", args.fold)
    scored.insert(1, "system", args.system)
    root_predictions, root_calibration = calibrate_root_head(
        scored, data.services, args.target_root_risk, horizon=120
    )
    scored = apply_window_root_probabilities(
        scored, data.services, root_calibration["temperature"],
        root_calibration["validation_abstention_threshold"],
    )
    scored.to_csv(score_path, index=False)
    root_predictions.to_csv(run_dir / "root_cause_case_predictions.csv", index=False)
    (run_dir / "root_calibration.json").write_text(
        json.dumps(root_calibration, indent=2, allow_nan=True), encoding="utf-8"
    )
    (run_dir / "validation_score_calibration.json").write_text(
        json.dumps({"method": "empirical upper tail from normal validation forecast errors",
                    "normal_score_reference": calibration}, indent=2), encoding="utf-8"
    )
    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved validation/test scores: {score_path.resolve()}")
    print(f"Best validation root MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
