"""Look inside one RCAEval case: what metrics exist, sampling interval, and what the fault does.

    python scripts/explore_case.py                                  # first RE2 Online Boutique case
    python scripts/explore_case.py --case re2tt_ts-order-service_cpu_1

Prints a summary and saves two figures to experiments/explore/:
  <case>_root_metrics.png   every metric of the root-cause service, fault period shaded
  <case>_service_shift.png  how much each service's metrics shift after the fault (z-score)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from memrca.data.loader import list_cases, load_case, load_logs, load_traces, metric_columns_by_service

KNOWN, FAULT, REF, INK, MUTED, GRID = "#2F6FED", "#C7791C", "#16244C", "#222A39", "#6B7280", "#E4E9F2"


def service_shift(case) -> dict[str, float]:
    """Largest |z-score| jump of any metric of a service: mean(faulty) vs normal mean/std."""
    normal, faulty = case.normal(), case.faulty()
    mu, sd = normal.mean(), normal.std().replace(0, np.nan)
    z = ((faulty.mean() - mu) / sd).abs()
    scores = {}
    for service, cols in metric_columns_by_service(case.metrics.columns).items():
        vals = z[cols].dropna()
        scores[service] = float(vals.max()) if len(vals) else 0.0
    return scores


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C0CC")
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", default="experiments/explore")
    args = ap.parse_args()

    name = args.case or list_cases(args.raw, "re2ob")[0]
    case = load_case(args.raw, name)
    m = case.metrics
    groups = metric_columns_by_service(m.columns)
    dt = np.diff(m.index.values)

    print(f"\n=== {name} ===")
    print(f"root cause: {case.root_cause}   fault: {case.fault}   repetition: {case.repetition}")
    print(f"timesteps: {len(m)}  (normal {len(case.normal())}, faulty {len(case.faulty())})")
    print(f"sampling interval: median {np.median(dt):.1f} s  (min {dt.min():.0f}, max {dt.max():.0f})")
    print(f"metric columns: {m.shape[1]}   services: {len(groups)}")
    print(f"missing values: {m.isna().mean().mean() * 100:.1f}% of cells")
    metric_types = sorted({c.partition('_')[2] for c in m.columns})
    print(f"metric types ({len(metric_types)}): {metric_types}")
    print("metrics per service:")
    for s, cols in sorted(groups.items()):
        print(f"  {s:32s} {len(cols)}")
    logs, traces = load_logs(args.raw, name), load_traces(args.raw, name)
    print(f"logs:   {'none' if logs is None else f'{len(logs)} rows, columns {list(logs.columns)}'}")
    print(f"traces: {'none' if traces is None else f'{len(traces)} rows, columns {list(traces.columns)}'}")

    shift = service_shift(case)
    ranked = sorted(shift.items(), key=lambda kv: kv[1], reverse=True)
    rank = [s for s, _ in ranked].index(case.root_cause) + 1 if case.root_cause in shift else None
    print(f"\nservices ranked by metric shift after the fault (top 5):")
    for s, v in ranked[:5]:
        print(f"  {s:32s} z={v:8.2f}{'   <-- root cause' if s == case.root_cause else ''}")
    print(f"root cause rank by simple z-score shift: {rank} of {len(shift)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = (m.index.values - case.inject_time) / 60.0  # minutes relative to injection

    # Figure 1: small multiples, one panel per metric of the root-cause service.
    cols = groups.get(case.root_cause, [])
    if cols:
        n = len(cols)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow), squeeze=False)
        for ax, col in zip(axes.flat, cols):
            ax.axvspan(0, t.max(), color=FAULT, alpha=0.08, linewidth=0)
            ax.plot(t, m[col].values, color=KNOWN, linewidth=1.6)
            ax.axvline(0, color=REF, linewidth=1.2, linestyle="--")
            ax.set_title(col.partition("_")[2], fontsize=9, color=INK, loc="left")
            style(ax)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"{name}: metrics of the root-cause service ({case.root_cause}); "
                     f"dashed line = fault injection, shaded = faulty period",
                     fontsize=10, color=INK, x=0.01, ha="left")
        fig.supxlabel("minutes relative to fault injection", fontsize=9, color=MUTED)
        fig.tight_layout()
        fig.savefig(out / f"{name}_root_metrics.png", dpi=130)
        plt.close(fig)

    # Figure 2: per-service shift, root cause highlighted and labelled.
    top = ranked[:15]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.2))
    labels = [s for s, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    colors = [FAULT if s == case.root_cause else KNOWN for s in labels]
    ax.barh(labels, vals, color=colors, height=0.6)
    for y, (s, v) in enumerate(zip(labels, vals)):
        ax.text(v, y, f"  {v:.1f}" + ("  root cause" if s == case.root_cause else ""),
                va="center", fontsize=8, color=INK)
    ax.set_xlabel("largest |z-score| shift of any metric after the fault", fontsize=9, color=MUTED)
    ax.set_title(f"{name}: which services changed most", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"{name}_service_shift.png", dpi=130)
    plt.close(fig)
    print(f"\nfigures saved to {out}/")


if __name__ == "__main__":
    main()
