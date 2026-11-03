"""Result figures (PNG + PDF) from the experiment outputs, for the report, slides and paper.

    python scripts/make_figures.py            # writes docs/figures/

Figures whose inputs are missing are skipped with a note. Colours are the validated
categorical slots used by the console (Mem-RCA is always slot 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
SYSTEMS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def grouped_bars(ax, values: dict[str, list[float]], label_series: str | None = None, ylim=(0, 1.0),
                 chance: float | None = None, fmt: str = "{:.2f}", legend: bool = True,
                 colors: list[str] | None = None) -> None:
    """values: series name -> one value per system (in SYSTEMS order)."""
    n = len(values)
    width = 0.8 / n
    x = np.arange(len(SYSTEMS))
    for k, (name, vals) in enumerate(values.items()):
        bars = ax.bar(x + (k - (n - 1) / 2) * width, vals, width * 0.92, color=(colors or SLOTS)[k], label=name, zorder=2)
        if name == label_series:
            for bar, v in zip(bars, vals):
                if v == v:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, fmt.format(v), ha="center", va="bottom",
                            fontsize=9, color=INK, fontweight="bold")
    if chance is not None:
        ax.axhline(chance, color=INK_2, linestyle=(0, (4, 4)), linewidth=1, zorder=1)
        ax.text(len(SYSTEMS) - 0.5, chance + 0.01, "chance", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, list(SYSTEMS.values()))
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(ncol=n, loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=9, handlelength=1, columnspacing=1.2)


def save(fig, out: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def by_role(df: pd.DataFrame, roles: dict[str, str], metric: str = "auroc") -> dict[str, list[float]]:
    return {name: [float(df[(df["system"] == s) & (df["role"] == role)][metric].mean()) for s in SYSTEMS]
            for name, role in roles.items()}


def summary_json(e: Path, out: Path) -> None:
    """Key numbers for the slide deck's native charts (docs/figures/results_summary.json)."""
    def per_system(df: pd.DataFrame, key: str, value: str, metric: str) -> dict[str, float]:
        # full precision: the deck's chart labels do their own rounding
        return {s: float(df[(df["system"] == s) & (df[key] == value)][metric].mean()) for s in SYSTEMS}

    roles_e2 = {"Mem-RCA": "Mem-RCA", "anchor": "E2 anchor", "global max": "E2 global_max", "global mean": "E2 global_mean"}
    roles_ext = {"Mem-RCA": "Mem-RCA", "anchor": "baseline anchor", "global max": "baseline global_max",
                 "global mean": "baseline global_mean"}
    summary: dict = {}
    path = e / "open_world" / "results" / "selection_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        summary["re2"] = {name: per_system(df, "role", role, "auroc") for name, role in roles_e2.items()}
    path = e / "external" / "results" / "re3_novel_types.csv"
    if path.exists():
        df = pd.read_csv(path)
        summary["re3"] = {name: per_system(df, "role", role, "auroc") for name, role in roles_ext.items()}
        mem = df[df["role"] == "Mem-RCA"].set_index("system")
        summary["re3_detail"] = {s: {"flagged": round(float(mem.loc[s, "unseen_flagged_novel"]), 3),
                                     "known_kept": round(float(mem.loc[s, "known_accepted"]), 3),
                                     "p": float(mem.loc[s, "mann_whitney_p"])} for s in SYSTEMS}
    path = e / "external" / "results" / "re1_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["root"].eq("pred")]
        summary["re1"] = {name: per_system(df, "role", role, "auroc") for name, role in roles_ext.items()}
        folds = pd.read_csv(e / "external" / "results" / "re1_open_set_by_fold.csv")
        tt = folds[folds["system"].eq("tt")].pivot_table(index="unseen_type", columns="config", values="auroc")
        anchor = tt["shift:anchor_pred+prototype"]
        rivals = {"global mean": tt["shift:global_mean+prototype"], "global max": tt["shift:global_max+prototype"]}
        wins = {g: int((anchor > v).sum()) for g, v in rivals.items()}
        h1 = all(anchor.mean() > v.mean() and wins[g] >= 4 for g, v in rivals.items())  # rule frozen in the protocol
        means = {name: float(np.mean(list(vals.values()))) for name, vals in summary["re1"].items()}
        h2 = all(means["Mem-RCA"] > means[g] for g in ("anchor", "global max", "global mean"))
        summary["re1_hypotheses"] = {
            "h1": h1, "h2": h2,
            "h1_text": f"anchor {anchor.mean():.2f} vs global mean {rivals['global mean'].mean():.2f} "
                       f"({wins['global mean']}/5 folds) and global max {rivals['global max'].mean():.2f} "
                       f"({wins['global max']}/5) → {'supported' if h1 else 'not supported'}",
            "h2_text": f"Mem-RCA {means['Mem-RCA']:.2f} vs anchor {means['anchor']:.2f}, global max "
                       f"{means['global max']:.2f}, global mean {means['global mean']:.2f} → "
                       f"{'supported' if h2 else 'not supported'}",
        }
    path = e / "open_world" / "results" / "root_by_case.csv"
    if path.exists():
        df = pd.read_csv(path)
        summary["shortcut"] = {name: per_system(df, "method", m, "hit_at_1") for name, m in {
            "Learned localiser": "root_head", "Shift, 5 training roots": "shift_training_roots_only",
            "Shift, all services": "shift_all_services", "No telemetry": "no_telemetry_prior"}.items()}
    path = e / "service_holdout" / "service_holdout_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        summary["holdout"] = {status: {name: per_system(df[df["status"] == status], "method", m, "hit_at_1")
                                       for name, m in {"Learned localiser": "learned_head",
                                                       "Training roots first": "shift_training_roots_first",
                                                       "Shift, all services": "shift_all_services",
                                                       "No telemetry": "no_telemetry_prior"}.items()}
                              for status in ("seen_root", "unseen_root")}
    shots, streams = e / "open_world" / "results" / "few_shot_by_run.csv", e / "open_world" / "results" / "discovery_by_run.csv"
    sel = e / "open_world" / "results" / "selected_configs.json"
    if shots.exists() and streams.exists() and sel.exists():
        chosen = pd.read_json(sel, typ="series")
        f, st = pd.read_csv(shots), pd.read_csv(streams)
        summary["few_shot"] = {s: {name: [round(float(v), 3) for v in f[(f["system"] == s) & (f["variant"] == variant)]
                                          .groupby("k")["new_type_accuracy"].mean()]
                                   for name, variant in {"Mem-RCA": chosen[f"memrca_{s}"],
                                                         "global mean": "shift:global_mean+prototype"}.items()}
                               for s in SYSTEMS}
        summary["discovery"] = {s: {k: round(float(v), 3) for k, v in
                                    st[(st["system"] == s) & (st["variant"] == chosen[f"memrca_{s}"])]
                                    [["discovered_fraction", "cluster_purity", "acc_after_enrolment"]].mean().items()}
                                for s in SYSTEMS}
    for suite, split in (("re2", "unseen-root"), ("re3", "RE3 (confirmatory)")):
        path = e / "propagation_forecast" / f"forecast_summary_{suite}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["split"].eq(split) & df["start_s"].eq(0)]
            summary[f"propagation_{suite}"] = {
                method: {s: {m: float(g.loc[g["system"].eq(s), m].mean()) for m in ("auroc", "c_index")}
                         for s in [*SYSTEMS, "all"]}
                for method, g in df.groupby("method")}
        path = e / "propagation_forecast" / f"forecast_update_gain_{suite}.csv"
        if path.exists():
            g = pd.read_csv(path)
            g = g[g["system"].eq("all") & g["update_at_s"].eq(60) & g["split"].eq(split)]
            if len(g):
                summary[f"propagation_gain_{suite}"] = {"t0": float(g["auroc_t0_forecast"].iloc[0]),
                                                        "updated_60s": float(g["auroc_updated"].iloc[0])}

    def clean(value):  # NaN (e.g. nothing discovered) becomes null
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return None if isinstance(value, float) and value != value else value

    (out / "results_summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out / 'results_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", default="experiments")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    e, out = Path(args.experiments), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_json(e, out)

    path = e / "open_world" / "results" / "selection_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        grouped_bars(ax, by_role(df, {"Mem-RCA": "Mem-RCA", "anchor": "E2 anchor", "global max": "E2 global_max",
                                      "global mean": "E2 global_mean"}), "Mem-RCA", chance=0.5)
        ax.set_ylabel("Novelty AUROC")
        save(fig, out, "fig_re2_novelty")
    else:
        print(f"skip RE2 figure: {path} missing")

    path = e / "external" / "results" / "re3_novel_types.csv"
    if path.exists():
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        grouped_bars(ax, by_role(df, {"Mem-RCA": "Mem-RCA", "anchor": "baseline anchor",
                                      "global max": "baseline global_max", "global mean": "baseline global_mean"}),
                     "Mem-RCA", chance=0.5)
        ax.set_ylabel("Novelty AUROC (RE3 code faults)")
        save(fig, out, "fig_re3_code_faults")
    else:
        print(f"skip RE3 figure: {path} missing")

    path = e / "external" / "results" / "re1_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["root"].eq("pred")]
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        grouped_bars(ax, by_role(df, {"Mem-RCA": "Mem-RCA", "anchor": "baseline anchor",
                                      "global max": "baseline global_max", "global mean": "baseline global_mean"}),
                     "Mem-RCA", chance=0.5)
        ax.set_ylabel("Novelty AUROC (RE1)")
        save(fig, out, "fig_re1_novelty")
    else:
        print(f"skip RE1 figure: {path} missing")

    path = e / "open_world" / "results" / "root_by_case.csv"
    if path.exists():
        df = pd.read_csv(path)
        methods = {"Learned localiser": "root_head", "Shift, 5 training roots": "shift_training_roots_only",
                   "Shift, all services": "shift_all_services", "No telemetry": "no_telemetry_prior"}
        values = {name: [float(df[(df["system"] == s) & (df["method"] == m)]["hit_at_1"].mean()) for s in SYSTEMS]
                  for name, m in methods.items()}
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        grouped_bars(ax, values, "Learned localiser")
        ax.set_ylabel("Root-cause Hit@1 (RE2)")
        save(fig, out, "fig_c4_shortcut")

    path = e / "service_holdout" / "service_holdout_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
        methods = {"Learned localiser": "learned_head", "Training roots first": "shift_training_roots_first",
                   "Shift, all services": "shift_all_services", "No telemetry": "no_telemetry_prior"}
        for ax, status, title in zip(axes, ("seen_root", "unseen_root"), ("Root seen in training", "Root held out")):
            sub = df[df["status"].eq(status)]
            values = {name: [float(sub[(sub["system"] == s) & (sub["method"] == m)]["hit_at_1"].mean()) for s in SYSTEMS]
                      for name, m in methods.items()}
            grouped_bars(ax, values, "Learned localiser", ylim=(0, 1.08), legend=False)
            ax.set_title(title, fontsize=10, color=INK, pad=8)
        axes[0].set_ylabel("Root-cause Hit@1")
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07),
                   fontsize=9, handlelength=1, columnspacing=1.5)
        save(fig, out, "fig_service_holdout")
    else:
        print(f"skip service hold-out figure: {path} missing")

    path = e / "open_world" / "results" / "few_shot_by_run.csv"
    sel = e / "open_world" / "results" / "selected_configs.json"
    if path.exists() and sel.exists():
        df = pd.read_csv(path)
        chosen = pd.read_json(sel, typ="series")
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
        for ax, s in zip(axes, SYSTEMS):
            for k, (name, variant) in enumerate({"Mem-RCA": chosen[f"memrca_{s}"],
                                                 "global mean": "shift:global_mean+prototype"}.items()):
                sub = df[(df["system"] == s) & (df["variant"] == variant)].groupby("k")["new_type_accuracy"].mean()
                ax.plot(sub.index, sub.values, marker="o", color=SLOTS[0] if k == 0 else SLOTS[3], linewidth=2,
                        markersize=6, label=name)
            ax.set_title(SYSTEMS[s], fontsize=10, color=INK)
            ax.set_xticks([1, 3, 5])
            ax.set_xlabel("examples of the new type (k)")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("New-type accuracy")
        axes[0].legend(loc="lower right", fontsize=9)
        save(fig, out, "fig_few_shot")

    path = e / "propagation_forecast" / "forecast_summary_re3.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["start_s"].eq(0)]
        methods = {"PropagationNet": SLOTS[0], "graph table": SLOTS[2], "all at once": SLOTS[3]}
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
        for ax, metric, title in zip(axes, ("auroc", "c_index"), ("Who is hit within 300 s (AUROC)",
                                                                   "Who is hit first (C-index)")):
            values = {m: [float(df[(df["system"] == s) & (df["method"] == m)][metric].mean()) for s in SYSTEMS]
                      for m in methods}
            grouped_bars(ax, values, "PropagationNet", chance=0.5, legend=False, colors=list(methods.values()))
            ax.set_title(title, fontsize=10, color=INK, pad=8)
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07),
                   fontsize=9, handlelength=1, columnspacing=1.5)
        save(fig, out, "fig_propagation_re3")


if __name__ == "__main__":
    main()
