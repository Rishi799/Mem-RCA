"""Estimate observed anomaly propagation delays on held-out RCAEval cases.

This is an event-time analysis over model alarms and training-fold trace edges;
it does not establish causal propagation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def persistent_alarm_times(frame: pd.DataFrame, threshold: float, hits: int) -> dict[str, int]:
    """First timestamp in a run of `hits` threshold exceedances, by service."""
    alarms: dict[str, int] = {}
    for service, group in frame.sort_values("target_time").groupby("service"):
        flags = group["anomaly_score"].to_numpy(float) > threshold
        times = group["target_time"].to_numpy(np.int64)
        if len(flags) < hits:
            continue
        # A consecutive-window rule avoids counting isolated score spikes.
        for i in range(hits - 1, len(flags)):
            if bool(np.all(flags[i - hits + 1:i + 1])):
                alarms[str(service)] = int(times[i - hits + 1])
                break
    return alarms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="experiments/lstm_selected/anomaly")
    parser.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    parser.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    parser.add_argument("--normal-fpr", type=float, default=0.01)
    parser.add_argument("--persistence-hits", type=int, default=2)
    parser.add_argument("--out", default="experiments/propagation")
    args = parser.parse_args()
    if not 0 < args.normal_fpr < 0.5 or args.persistence_hits not in (1, 2, 3):
        raise ValueError("normal-fpr must be in (0, .5) and persistence-hits in 1..3")

    root = Path(args.scores)
    paths = sorted(root.glob("fold*/**/service_window_scores.csv"))
    if not paths:
        raise FileNotFoundError(f"No service_window_scores.csv files under {root}")
    score_frames = [pd.read_csv(path) for path in paths]
    scores = pd.concat(score_frames, ignore_index=True)
    required = {"fold", "system", "case", "split", "service", "target_time", "anomaly_score", "is_anomaly", "root_cause_service"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Score files are missing columns: {sorted(missing)}")

    manifest = pd.read_csv(args.manifest)
    edges = pd.read_csv(args.edges)
    threshold_rows = []
    thresholds: dict[tuple[str, str], float] = {}
    for (fold, system), group in scores.groupby(["fold", "system"]):
        normal = group[group["split"].eq("validation") & ~group["is_anomaly"].astype(bool)]
        if normal.empty:
            continue
        threshold = float(normal["anomaly_score"].quantile(1 - args.normal_fpr))
        thresholds[(str(fold), str(system))] = threshold
        threshold_rows.append({"fold": fold, "system": system, "threshold": threshold,
                               "normal_validation_windows": len(normal), "target_normal_fpr": args.normal_fpr})

    event_rows: list[dict] = []
    case_rows: list[dict] = []
    for (fold, system, case), group in scores[scores["split"].eq("test")].groupby(["fold", "system", "case"]):
        threshold = thresholds.get((str(fold), str(system)))
        if threshold is None:
            continue
        post_injection = group[group["seconds_from_injection"].ge(0)].copy()
        alarms = persistent_alarm_times(post_injection, threshold, args.persistence_hits)
        root_service = str(group["root_cause_service"].iloc[0])
        root_time = alarms.get(root_service)
        injection_time = int((group["target_time"] - group["seconds_from_injection"] - 29).iloc[0])
        manifest_rows = manifest[(manifest["fold"].astype(str) == str(fold)) &
                                 (manifest["system"].astype(str) == str(system)) &
                                 (manifest["split"] == "train")]
        train_cases = set(manifest_rows["case"].astype(str))
        topology = edges[(edges["system"].astype(str) == str(system)) & edges["case"].astype(str).isin(train_cases)]
        pairs = topology[["source", "target"]].drop_duplicates()
        observed_pairs = 0
        for source, target in pairs.itertuples(index=False, name=None):
            if str(source) in alarms and str(target) in alarms:
                delay = alarms[str(target)] - alarms[str(source)]
                if delay >= 0:
                    event_rows.append({"fold": fold, "system": system, "case": case,
                                       "source": str(source), "target": str(target),
                                       "source_alarm_time": alarms[str(source)],
                                       "target_alarm_time": alarms[str(target)], "delay_seconds": delay,
                                       "root_alarm_time": root_time})
                    observed_pairs += 1
        case_rows.append({"fold": fold, "system": system, "case": case,
                          "root_service": root_service, "root_detected": root_time is not None,
                          "root_detection_delay_seconds": None if root_time is None else root_time - injection_time,
                          "services_with_persistent_alarm": len(alarms), "trace_edges_with_ordered_alarm_pair": observed_pairs,
                          "fault_status": group["fault_status"].iloc[0] if "fault_status" in group else "unknown"})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(out / "validation_thresholds.csv", index=False)
    event_frame = pd.DataFrame(event_rows)
    case_frame = pd.DataFrame(case_rows)
    event_frame.to_csv(out / "edge_delay_events.csv", index=False)
    case_frame.to_csv(out / "case_detection_summary.csv", index=False)
    if event_frame.empty:
        summary = pd.DataFrame(columns=["system", "source", "target", "events", "median_delay_seconds", "mean_delay_seconds", "p90_delay_seconds"])
    else:
        summary = event_frame.groupby(["system", "source", "target"], as_index=False).agg(
            events=("delay_seconds", "size"), median_delay_seconds=("delay_seconds", "median"),
            mean_delay_seconds=("delay_seconds", "mean"), p90_delay_seconds=("delay_seconds", lambda x: x.quantile(.9)))
    summary.to_csv(out / "edge_delay_summary.csv", index=False)
    print("Observed propagation timing summary (held-out cases; training-fold graph; noncausal):")
    print(summary.to_string(index=False) if len(summary) else "No trace edge pairs had ordered persistent alarms.")
    if len(case_frame):
        print(f"\nRoot detection coverage: {case_frame.root_detected.mean():.3f} across {len(case_frame)} test cases")
    print(f"Wrote event analysis to {out.resolve()}")


if __name__ == "__main__":
    main()
