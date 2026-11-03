"""Feasibility check for time-to-impact forecasting (RE2, development data).

1. Label each service's anomaly onset at 1-second resolution (memrca.propagation.onsets),
   relative to the root service's own onset, within 300 s.
2. Sanity: root detection rate, and false onsets when the detector is run on the normal
   period (calibrated on its first half, tested on its second half).
3. Structure: is the affected set and its timing related to call-graph distance, direction
   and fault type, and is it reproducible across repetitions of the same (root, fault)?
4. Predictability: leave-one-repetition-out forecast from the other repetitions of the same
   (root, fault) - what a memory of past incidents could know - against all-at-once and
   graph-distance baselines.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, load_index, metric_columns_by_service  # noqa: E402
from memrca.propagation.onsets import anomaly_trace, detect_onsets, graph_distances, thresholds  # noqa: E402

HORIZON = 300


def label_onsets(raw: str, index: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in index.itertuples(index=False):
        case = load_case(raw, row.case)
        services = sorted(metric_columns_by_service(case.metrics.columns))
        trace = anomaly_trace(case, services)
        start = int(trace.index.min())
        onsets = detect_onsets(trace, thresholds(trace, start, 0), reference=0, horizon=HORIZON)
        middle = start // 2  # pseudo-injection halfway through the normal period
        false = detect_onsets(trace, thresholds(trace, start, middle), reference=middle, horizon=min(HORIZON, -middle - 5))
        root_onset = onsets.get(row.root_cause_service, np.nan)
        dist = graph_distances(edges[edges["system"].eq(row.system)], row.root_cause_service)
        for service in services:
            rows.append({
                "case": row.case, "system": row.system, "fault": row.fault, "root": row.root_cause_service,
                "rep": row.repetition, "service": service, "is_root": service == row.root_cause_service,
                "onset": onsets[service], "root_onset": root_onset, "false_onset": not np.isnan(false[service]),
                "hop_up": dist["up"].get(service, np.nan), "hop_down": dist["down"].get(service, np.nan),
                "hop_any": dist["any"].get(service, np.nan),
            })
        print(f"{row.case}: {int(onsets.notna().sum())}/{len(services)} services with onset", flush=True)
    df = pd.DataFrame(rows)
    # delay relative to the root's onset (injection time when the root itself is not detected)
    df["delay"] = df["onset"] - df["root_onset"].fillna(0.0)
    df["affected"] = df["onset"].notna()
    return df


def f1(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    return 2 * tp / (len(pred) + len(true)) if (pred or true) else 0.0


def forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-repetition-out: forecast each incident from other incidents only."""
    out = []
    others = df[~df["is_root"]]
    for case, inc in others.groupby("case"):
        system, root, fault = inc["system"].iloc[0], inc["root"].iloc[0], inc["fault"].iloc[0]
        true = set(inc.loc[inc["affected"], "service"])
        truth = inc.set_index("service")["delay"]
        same = others[others["system"].eq(system) & others["root"].eq(root) & others["fault"].eq(fault) & others["case"].ne(case)]
        # per-hop median delays learned from incidents with a different root (no leakage of this config)
        pool = others[others["system"].eq(system) & others["root"].ne(root) & others["affected"]]
        hop_delay = pool.groupby(pool["hop_any"].fillna(-1))["delay"].median()
        predictions = {
            "memory: same root+fault, other reps": (
                set(same.groupby("service")["affected"].mean().loc[lambda s: s >= 0.5].index),
                same[same["affected"]].groupby("service")["delay"].median()),
            "all services at root time": (set(inc["service"]), pd.Series(0.0, index=inc["service"])),
            "graph: 1-hop neighbours, per-hop median": (
                set(inc.loc[inc["hop_any"].le(1), "service"]),
                inc.set_index("service")["hop_any"].fillna(-1).map(hop_delay)),
        }
        for method, (pred, delay) in predictions.items():
            both = sorted(pred & true)
            err = (delay.reindex(both) - truth.reindex(both)).abs().dropna()
            rho = (spearmanr(delay.reindex(both), truth.reindex(both)).statistic
                   if len(both) >= 3 and delay.reindex(both).nunique() > 1 and truth.reindex(both).nunique() > 1 else np.nan)
            out.append({"case": case, "system": system, "fault": fault, "method": method, "n_true": len(true),
                        "n_pred": len(pred), "f1": f1(pred, true), "mae_s": err.mean() if len(err) else np.nan,
                        "spearman": rho})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    ap.add_argument("--out", default="experiments/propagation_forecast")
    ap.add_argument("--reuse", action="store_true", help="reuse saved onset labels")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = out / "onsets_re2.csv"
    if args.reuse and labels.exists():
        df = pd.read_csv(labels)
    else:
        index = load_index(args.raw)
        index = index[index["suite"].eq("RE2")]
        edges = pd.read_csv(args.edges)[["system", "source", "target"]].drop_duplicates()
        df = label_onsets(args.raw, index, edges)
        df.to_csv(labels, index=False)

    pd.set_option("display.width", 200)
    roots = df[df["is_root"]]
    print("\n=== Detector sanity ===")
    print(pd.DataFrame({
        "root detected": roots.groupby("system")["affected"].mean(),
        "root onset median s": roots.groupby("system")["onset"].median(),
        "false onsets in normal period": df.groupby("system")["false_onset"].mean(),
    }).round(3).to_string())

    others = df[~df["is_root"]]
    print("\n=== Share of non-root services affected within 300 s, by fault ===")
    print(others.pivot_table(index="system", columns="fault", values="affected").round(2).to_string())

    graph = others[others["system"].isin(["ob", "tt"])].copy()
    graph["hop"] = graph["hop_any"].map(lambda h: "not in graph" if np.isnan(h) else ("3+" if h >= 3 else str(int(h))))
    print("\n=== By call-graph distance from the root (OB, TT) ===")
    print(graph.groupby(["system", "hop"]).agg(services=("affected", "size"), affected=("affected", "mean"),
                                               delay_median=("delay", "median"),
                                               delay_q25=("delay", lambda s: s.quantile(0.25)),
                                               delay_q75=("delay", lambda s: s.quantile(0.75))).round(2).to_string())
    graph["direction"] = np.select([graph["hop_up"].notna() & graph["hop_down"].isna(),
                                    graph["hop_down"].notna() & graph["hop_up"].isna(),
                                    graph["hop_up"].notna() & graph["hop_down"].notna()],
                                   ["caller side (upstream)", "callee side (downstream)", "both"], "other")
    print("\n=== By direction relative to the root ===")
    print(graph.groupby(["system", "direction"]).agg(services=("affected", "size"), affected=("affected", "mean"),
                                                     delay_median=("delay", "median")).round(2).to_string())
    print("\n=== Delay of affected services by fault type (median s) ===")
    print(others[others["affected"]].pivot_table(index="system", columns="fault", values="delay", aggfunc="median").round(1).to_string())

    print("\n=== Reproducibility across repetitions of the same (root, fault) ===")
    repro = []
    for (system, root, fault), g in others.groupby(["system", "root", "fault"]):
        sets = {c: set(x.loc[x["affected"], "service"]) for c, x in g.groupby("case")}
        delays = {c: x.set_index("service")["delay"] for c, x in g.groupby("case")}
        for a, b in itertools.combinations(sorted(sets), 2):
            union = sets[a] | sets[b]
            common = sorted(sets[a] & sets[b])
            rho = (spearmanr(delays[a][common], delays[b][common]).statistic
                   if len(common) >= 3 and delays[a][common].nunique() > 1 and delays[b][common].nunique() > 1 else np.nan)
            repro.append({"system": system, "fault": fault, "jaccard": len(common) / len(union) if union else 1.0,
                          "delay_spearman": rho, "abs_delay_diff": (delays[a][common] - delays[b][common]).abs().median()
                          if common else np.nan})
    repro = pd.DataFrame(repro)
    print(repro.groupby("system")[["jaccard", "delay_spearman", "abs_delay_diff"]].median().round(2).to_string())
    print(repro.groupby("fault")[["jaccard", "delay_spearman", "abs_delay_diff"]].median().round(2).to_string())

    fc = forecasts(df)
    fc.to_csv(out / "feasibility_forecasts.csv", index=False)
    print("\n=== Leave-one-repetition-out forecast (mean over incidents) ===")
    print(fc.groupby(["system", "method"])[["f1", "mae_s", "spearman"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
