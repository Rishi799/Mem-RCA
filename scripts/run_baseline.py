"""Train and evaluate a normal-only metric anomaly and RCA baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from memrca.baselines.isolation_forest import ServiceIsolationForest
from memrca.data.loader import load_case, metric_columns_by_service
from memrca.data.windows import WindowedCase, make_windows


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return center, scale


def eval_binary(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = y.astype(bool)
    pred = scores >= threshold
    normal = ~y
    return {
        "windows": int(len(y)),
        "positive_windows": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(pred[normal].mean()) if normal.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if y.any() and normal.any() else float("nan"),
        "average_precision": float(average_precision_score(y, scores)) if y.any() else float("nan"),
    }


def load_windows_for_system(raw_dir: Path, system: str, rows: pd.DataFrame,
                            window: int, stride: int) -> tuple[dict[str, WindowedCase], dict[str, object]]:
    names = sorted(set(rows["case"].astype(str)))
    windows: dict[str, WindowedCase] = {}
    labels: dict[str, object] = {}
    for name in names:
        case = load_case(raw_dir, name)
        windows[name] = make_windows(case, window_seconds=window, stride_seconds=stride)
        labels[name] = {"root": case.root_cause, "fault": case.fault, "repetition": case.repetition,
                        "inject_time": case.inject_time}
    return windows, labels


def score_cases(names: list[str], windows: dict[str, WindowedCase],
                models: dict[str, ServiceIsolationForest]) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        results[name] = {}
        for service, model in models.items():
            results[name][service] = model.anomaly_score(windows[name].features)
    return results


def service_zscores(raw_scores: dict[str, dict[str, np.ndarray]],
                    windows: dict[str, WindowedCase],
                    normal_cases: list[str], services: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[float, float]]]:
    calibration: dict[str, tuple[float, float]] = {}
    for service in services:
        vals = [raw_scores[name][service][~windows[name].metadata["is_anomaly"].to_numpy()]
                for name in normal_cases if service in raw_scores.get(name, {})]
        combined = np.concatenate(vals) if vals else np.array([], dtype=float)
        if not len(combined):
            continue
        calibration[service] = robust_location_scale(combined)
    return apply_calibration(raw_scores, calibration), calibration


def apply_calibration(raw_scores: dict[str, dict[str, np.ndarray]],
                      calibration: dict[str, tuple[float, float]]) -> dict[str, dict[str, np.ndarray]]:
    z: dict[str, dict[str, np.ndarray]] = {}
    for name, by_service in raw_scores.items():
        z[name] = {}
        for service, values in by_service.items():
            if service not in calibration:
                continue
            center, scale = calibration[service]
            z[name][service] = (values - center) / scale
    return z


def combined_window_scores(service_scores: dict[str, np.ndarray]) -> np.ndarray:
    if not service_scores:
        return np.array([], dtype=float)
    matrix = np.vstack(list(service_scores.values()))
    return np.nanmax(matrix, axis=0)


def root_ranking(zscores: dict[str, np.ndarray], metadata: pd.DataFrame,
                 horizon: int) -> list[tuple[str, float]]:
    starts = metadata["seconds_from_injection"].to_numpy()
    post = metadata["is_anomaly"].to_numpy(dtype=bool) & (starts < horizon)
    if not post.any():
        post = metadata["is_anomaly"].to_numpy(dtype=bool)
    scores: list[tuple[str, float]] = []
    for service, values in zscores.items():
        selected = values[post]
        selected = selected[np.isfinite(selected)]
        if not len(selected):
            continue
        top_n = max(1, math.ceil(len(selected) * 0.25))
        score = float(np.mean(np.sort(selected)[-top_n:]))
        scores.append((service, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--normal-fpr", type=float, default=0.01)
    parser.add_argument("--rca-horizon", type=int, default=120,
                        help="Seconds after injection used to score root-cause ranks")
    parser.add_argument("--out", default="experiments/baseline")
    args = parser.parse_args()
    if not 0 < args.normal_fpr < 0.5:
        raise ValueError("--normal-fpr must be between 0 and 0.5")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw or config["raw_dir"])
    window = int(args.window or config.get("window_length") or 30)
    stride = int(args.stride or config.get("window_stride") or window)
    manifest = pd.read_csv(args.manifest, dtype={"case": str, "fold": str, "split": str})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    service_window_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    validation_ranking_rows: list[dict[str, object]] = []

    for system in config["systems"]:
        system_rows = manifest[manifest["system"].astype(str).eq(str(system))]
        case_windows, case_labels = load_windows_for_system(raw_dir, str(system), system_rows,
                                                            window=window, stride=stride)
        for fold in config["folds"]:
            print(f"[{fold}/{system}] fitting normal-only service models", flush=True)
            fold_rows = system_rows[system_rows["fold"].eq(fold)]
            train_names = fold_rows.loc[fold_rows["split"].eq("train"), "case"].astype(str).tolist()
            val_names = fold_rows.loc[fold_rows["split"].eq("validation"), "case"].astype(str).tolist()
            test_rows = fold_rows[fold_rows["split"].eq("test")]
            test_names = test_rows["case"].astype(str).tolist()

            service_columns: dict[str, set[str]] = {}
            for name in train_names:
                groups = metric_columns_by_service(case_windows[name].features.columns)
                for service, columns in groups.items():
                    service_columns.setdefault(service, set()).update(columns)
            models: dict[str, ServiceIsolationForest] = {}
            for service, columns_set in sorted(service_columns.items()):
                columns = sorted(columns_set)
                normal_frames = []
                for name in train_names:
                    win = case_windows[name]
                    normal = ~win.metadata["is_anomaly"].to_numpy(dtype=bool)
                    frame = win.features.reindex(columns=columns).loc[normal]
                    normal_frames.append(frame)
                train_matrix = pd.concat(normal_frames, ignore_index=True)
                if train_matrix.empty or not train_matrix.notna().any().any():
                    continue
                model = ServiceIsolationForest(random_state=17)
                models[service] = model.fit(train_matrix)

            val_raw = score_cases(val_names, case_windows, models)
            val_z, calibration = service_zscores(val_raw, case_windows, val_names, sorted(models))
            val_global = []
            for name in val_names:
                scores = combined_window_scores(val_z.get(name, {}))
                normal = ~case_windows[name].metadata["is_anomaly"].to_numpy(dtype=bool)
                val_global.extend(scores[normal].tolist())
            if not val_global:
                raise ValueError(f"{fold}/{system}: no normal validation scores to calibrate threshold")
            threshold = float(np.quantile(np.asarray(val_global), 1 - args.normal_fpr))

            for name in val_names:
                ranks = root_ranking(val_z.get(name, {}), case_windows[name].metadata, args.rca_horizon)
                labels = case_labels[name]
                for position, (service, score) in enumerate(ranks, start=1):
                    validation_ranking_rows.append({
                        "fold": fold, "system": system, "case": name, "fault": labels["fault"],
                        "fault_status": "known", "true_root": str(labels["root"]),
                        "horizon_seconds": args.rca_horizon, "service": service,
                        "score": float(score), "rank": position,
                        "is_root_cause": service == str(labels["root"]),
                    })

            test_raw = score_cases(test_names, case_windows, models)
            test_z = apply_calibration(test_raw, calibration)
            delays: list[float] = []
            detected = 0
            root_cases: list[dict[str, object]] = []

            for name in test_names:
                win = case_windows[name]
                meta = win.metadata
                service_scores = test_z.get(name, {})
                system_scores = combined_window_scores(service_scores)
                y = meta["is_anomaly"].to_numpy(dtype=bool)
                alarm = system_scores >= threshold
                post = y
                hit_positions = np.flatnonzero(alarm & post)
                if len(hit_positions):
                    detected += 1
                    delays.append(float(meta.iloc[hit_positions[0]]["seconds_from_injection"]))
                base = fold_rows[fold_rows["case"].eq(name)].iloc[0]
                labels = case_labels[name]
                for idx, row in meta.iterrows():
                    window_rows.append({
                        "fold": fold, "system": system, "case": name, "fault": labels["fault"],
                        "fault_status": base["fault_status"], "window_start": int(row["window_start"]),
                        "seconds_from_injection": int(row["seconds_from_injection"]),
                        "is_anomaly": bool(row["is_anomaly"]),
                        "anomaly_score": float(system_scores[idx]), "alarm": bool(alarm[idx]),
                    })
                for service, scores in service_scores.items():
                    for idx, row in meta.iterrows():
                        service_window_rows.append({
                            "fold": fold, "system": system, "case": name,
                            "fault": labels["fault"], "fault_status": base["fault_status"],
                            "window_start": int(row["window_start"]),
                            "seconds_from_injection": int(row["seconds_from_injection"]),
                            "is_anomaly": bool(row["is_anomaly"]), "service": service,
                            "anomaly_score": float(scores[idx]),
                        })
                ranks = root_ranking(service_scores, meta, args.rca_horizon)
                true_root = str(labels["root"])
                rank = next((i + 1 for i, (service, _) in enumerate(ranks) if service == true_root), None)
                root_cases.append({"case": name, "fault_status": str(base["fault_status"]),
                                   "fault": labels["fault"], "true_root": true_root, "rank": rank})
                for position, (service, score) in enumerate(ranks, start=1):
                    ranking_rows.append({
                        "fold": fold, "system": system, "case": name, "fault": labels["fault"],
                        "fault_status": base["fault_status"], "true_root": true_root,
                        "horizon_seconds": args.rca_horizon, "service": service,
                        "score": float(score), "rank": position, "is_root_cause": service == true_root,
                    })

            for status in ("all", "known", "unseen"):
                if status == "all":
                    select_cases = set(test_names)
                else:
                    select_cases = set(test_rows.loc[test_rows["fault_status"].eq(status), "case"].astype(str))
                score_subset = [row for row in window_rows if row["fold"] == fold and row["system"] == system
                                and row["case"] in select_cases]
                yy = np.asarray([row["is_anomaly"] for row in score_subset], dtype=bool)
                ss = np.asarray([row["anomaly_score"] for row in score_subset], dtype=float)
                values = eval_binary(yy, ss, threshold) if len(yy) else {}
                cases_subset = [item for item in root_cases if item["case"] in select_cases]
                ranks = [item["rank"] for item in cases_subset if item["rank"] is not None]
                values.update({
                    "fold": fold, "system": system, "fault_status": status,
                    "threshold_calibration": f"validation normal {1-args.normal_fpr:.1%} quantile",
                    "rca_cases": len(cases_subset),
                    "rca_cases_with_rank": len(ranks),
                    "hit_at_1": float(np.mean([r <= 1 for r in ranks])) if ranks else float("nan"),
                    "hit_at_3": float(np.mean([r <= 3 for r in ranks])) if ranks else float("nan"),
                    "mean_reciprocal_rank": float(np.mean([1 / r for r in ranks])) if ranks else float("nan"),
                    "incident_detection_rate": detected / len(test_names) if status == "all" and test_names else float("nan"),
                    "mean_detection_delay_seconds": float(np.mean(delays)) if delays and status == "all" else float("nan"),
                    "window_seconds": window, "stride_seconds": stride,
                    "rca_horizon_seconds": args.rca_horizon,
                })
                metric_rows.append(values)
            print(f"[{fold}/{system}] scored {len(test_names)} test incidents", flush=True)

    metrics = pd.DataFrame(metric_rows)
    windows_out = pd.DataFrame(window_rows)
    service_windows_out = pd.DataFrame(service_window_rows)
    rankings_out = pd.DataFrame(ranking_rows)
    validation_rankings_out = pd.DataFrame(validation_ranking_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    windows_out.to_csv(out_dir / "test_window_scores.csv", index=False)
    service_windows_out.to_csv(out_dir / "test_service_window_scores.csv", index=False)
    rankings_out.to_csv(out_dir / "test_root_rankings.csv", index=False)
    validation_rankings_out.to_csv(out_dir / "validation_root_rankings.csv", index=False)
    report = {
        "method": "per-service Isolation Forest trained only on pre-injection training windows",
        "feature": "30-second (configurable), mean-aggregated service metrics",
        "window_seconds": window,
        "stride_seconds": stride,
        "normal_fpr_target": args.normal_fpr,
        "threshold_tuning": "validation pre-injection windows only",
        "root_cause_score": f"mean of top 25% standardized service scores in first {args.rca_horizon} seconds after injection",
        "graph_used_in_ranker": False,
        "note": "This is a transparent baseline. Fault injection time is used as the anomaly label boundary, not as a model feature.",
        "metrics_rows": len(metrics),
        "window_score_rows": len(windows_out),
        "service_window_score_rows": len(service_windows_out),
        "root_ranking_rows": len(rankings_out),
        "validation_root_ranking_rows": len(validation_rankings_out),
    }
    (out_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Saved baseline outputs under {out_dir}")


if __name__ == "__main__":
    main()
