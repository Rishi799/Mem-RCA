"""Time-to-impact forecasting: train and evaluate PropagationNet against simple baselines.

Development (RE2):
  rep-CV        test each repetition with a model trained on the other two (all faults known)
  unseen-root   test each root service with a model trained without any incident of that root
Confirmatory (RE3, --confirm; protocol in docs/PROPAGATION_PROTOCOL.md):
  train on all RE2 incidents, forecast RE3 code-level incidents with the fault type unknown

Forecasts are made at the root's onset (t0) and updated 30 s and 60 s later. Metrics over
services not yet hit: AUROC of P(hit within 300 s), F1 at P >= 0.5, MAE (s) of the predicted
onset for services correctly predicted to be hit, and the C-index of the risk ordering.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, load_index  # noqa: E402
from memrca.propagation.dataset import (ALREADY, BIN, CENSORED, DYNAMIC_NAMES, FAULTS, N_BINS,  # noqa: E402
                                        STATIC_NAMES, build_incident)
from memrca.propagation.model import PropagationNet, forecast  # noqa: E402

ELAPSED = DYNAMIC_NAMES.index("elapsed")
FAULT_COLUMNS = [STATIC_NAMES.index(f"fault_{f}") for f in FAULTS]
UNKNOWN = STATIC_NAMES.index("fault_unknown")
START_BINS = [0, 3, 6]  # forecast at t0, t0 + 30 s, t0 + 60 s


# ---------------- data ----------------
def load_incidents(raw: str, suite: str, edges: pd.DataFrame, cache: Path) -> list:
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    index = load_index(raw)
    index = index[index["suite"].eq(suite)]
    incidents = []
    for row in index.itertuples(index=False):
        if not (Path(raw) / row.case / "metrics.parquet").is_file():
            continue
        meta = {"case": row.case, "system": row.system, "fault": row.fault, "root": row.root_cause_service,
                "repetition": row.repetition, "suite": suite}
        incidents.append(build_incident(load_case(raw, row.case), meta, edges[edges["system"].eq(row.system)],
                                        fault_known=suite != "RE3"))
        print(f"built {row.case}", flush=True)
    cache.write_bytes(pickle.dumps(incidents))
    return incidents


def variant(inc, drop_fault: bool = False, drop_dynamic: bool = False):
    static, dynamic = inc.static.copy(), inc.dynamic.copy()
    if drop_fault:
        static[:, FAULT_COLUMNS] = 0.0
        static[:, UNKNOWN] = 1.0
    if drop_dynamic:
        keep = dynamic[:, :, ELAPSED].copy()
        dynamic[:] = 0.0
        dynamic[:, :, ELAPSED] = keep
    return static, dynamic


# ---------------- PropagationNet ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FAULT_DROPOUT = 0.3  # share of training steps with the fault type hidden, as when the memory says NOVEL


def train_net(train: list, seed: int, epochs: int = 20, use_graph: bool = True, **ablation) -> PropagationNet:
    """Fixed settings, chosen before any result: 20 epochs, Adam 3e-3, weight decay 1e-4, fault-type
    dropout 0.3 (so the model also learns to forecast when the fault type is unknown)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PropagationNet(len(STATIC_NAMES), len(DYNAMIC_NAMES), use_graph=use_graph).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensors = []
    for inc in train:
        static, dynamic = variant(inc, **ablation)
        ob = inc.onset_bin
        bins = np.arange(N_BINS)[:, None]
        at_risk = (ob[None, :] != ALREADY) & ((ob[None, :] == CENSORED) | (ob[None, :] >= bins))
        target = (ob[None, :] == bins).astype(np.float32)
        hidden, _ = variant(inc, drop_fault=True)
        tensors.append(tuple(torch.from_numpy(a).to(DEVICE) for a in (
            np.repeat(static[None], N_BINS, axis=0), np.repeat(hidden[None], N_BINS, axis=0),
            dynamic, inc.edge_index, at_risk, target)))
    model.train()
    for _ in range(epochs):
        for i in rng.permutation(len(tensors)):
            known, unknown, dynamic, edges, at_risk, target = tensors[i]
            static = unknown if rng.random() < FAULT_DROPOUT else known
            if not at_risk.any():
                continue
            logits = model(static, dynamic, edges)
            loss = loss_fn(logits[at_risk], target[at_risk])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.cpu()


def net_predict(model, inc, start: int, **ablation) -> tuple[np.ndarray, np.ndarray]:
    static, dynamic = variant(inc, **ablation)
    cum = forecast(model, static, dynamic[start], inc.edge_index, start, N_BINS, ELAPSED)
    p_hit = cum[-1]
    first = np.argmax(cum >= 0.5, axis=0).astype(float)
    onset = np.where(cum.max(axis=0) >= 0.5, (start + first + 0.5) * BIN, np.nan)
    return p_hit, onset


# ---------------- baselines ----------------
def hop_key(inc) -> list:
    up = np.where(inc.static[:, STATIC_NAMES.index("no_up")] > 0, -1, np.round(inc.static[:, 0] * 3))
    down = np.where(inc.static[:, STATIC_NAMES.index("no_down")] > 0, -1, np.round(inc.static[:, 1] * 3))
    return list(zip(np.minimum(up, 3).astype(int), np.minimum(down, 3).astype(int)))


def table_rows(incidents: list) -> pd.DataFrame:
    rows = []
    for inc in incidents:
        for i, key in enumerate(hop_key(inc)):
            if inc.onset_bin[i] == ALREADY:
                continue
            rows.append({"system": inc.system, "fault": inc.fault, "root": inc.root, "service": inc.services[i],
                         "key": key, "hit": inc.onset_bin[i] >= 0, "delay": inc.delay[i]})
    return pd.DataFrame(rows)


class Baselines:
    def __init__(self, train: list) -> None:
        self.rows = table_rows(train)

    def _from(self, rows: pd.DataFrame, by: str, keys: list, fallback=(np.nan, np.nan)):
        stats = rows.groupby(by).agg(p=("hit", "mean"), d=("delay", "median"))
        p = np.array([stats["p"].get(k, fallback[0]) for k in keys], dtype=float)
        d = np.array([stats["d"].get(k, fallback[1]) for k in keys], dtype=float)
        return p, d

    def predict(self, inc, method: str) -> tuple[np.ndarray, np.ndarray]:
        n = len(inc.services)
        if method == "all at once":
            return np.ones(n), np.zeros(n)
        system_rows = self.rows[self.rows["system"].eq(inc.system)]
        keys = hop_key(inc)
        overall = (system_rows["hit"].mean(), system_rows["delay"].median())
        p, d = self._from(system_rows, "key", keys, overall)
        if method == "graph table":
            return p, np.where(p >= 0.5, d, np.nan)
        # memory: same fault and root (per service), else same fault (per graph key), else graph table
        fault_rows = system_rows[system_rows["fault"].eq(inc.fault)]
        if len(fault_rows):
            p2, d2 = self._from(fault_rows, "key", keys, (np.nan, np.nan))
            p, d = np.where(np.isnan(p2), p, p2), np.where(np.isnan(d2), d, d2)
            same = fault_rows[fault_rows["root"].eq(inc.root)]
            if len(same):
                p3, d3 = self._from(same, "service", inc.services, (np.nan, np.nan))
                p, d = np.where(np.isnan(p3), p, p3), np.where(np.isnan(d3), d, d3)
        return p, np.where(p >= 0.5, d, np.nan)


# ---------------- metrics ----------------
def score_rows(inc, start: int, p: np.ndarray, onset: np.ndarray, method: str, split: str) -> list[dict]:
    ob = inc.onset_bin
    risk = (ob != ALREADY) & ((ob == CENSORED) | (ob >= start))
    rows = []
    for i in np.where(risk)[0]:
        rows.append({"split": split, "method": method, "start_s": start * BIN, "system": inc.system,
                     "fault": inc.fault, "case": inc.case, "service": inc.services[i], "hit": bool(ob[i] >= 0),
                     "true_delay": inc.delay[i], "p_hit": float(p[i]), "pred_onset": float(onset[i])})
    return rows


def update_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Does updating the forecast later help? Compare the t0 forecast and the forecast made at
    t0 + s on exactly the services still not hit at t0 + s."""
    net = df[df["method"].eq("PropagationNet")]
    net = pd.concat([net, net.assign(system="all")], ignore_index=True)
    out = []
    for (split, system), g in net.groupby(["split", "system"]):
        first = g[g["start_s"].eq(0)].set_index(["case", "service"])
        for s in sorted(g["start_s"].unique()):
            if s == 0:
                continue
            later = g[g["start_s"].eq(s)].set_index(["case", "service"])
            early = first.loc[later.index]
            if later["hit"].nunique() < 2:
                continue
            out.append({"split": split, "system": system, "update_at_s": s, "services": len(later),
                        "auroc_t0_forecast": roc_auc_score(early["hit"], early["p_hit"]),
                        "auroc_updated": roc_auc_score(later["hit"], later["p_hit"])})
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df, df.assign(system="all")], ignore_index=True)  # per system and pooled
    out = []
    for key, g in df.groupby(["split", "method", "start_s", "system"]):
        auroc = roc_auc_score(g["hit"], g["p_hit"]) if g["hit"].nunique() == 2 else np.nan
        f1s, cidx = [], []
        for _, inc in g.groupby("case"):
            pred, true = set(inc.loc[inc["p_hit"] >= 0.5, "service"]), set(inc.loc[inc["hit"], "service"])
            f1s.append(1.0 if not pred and not true else 2 * len(pred & true) / max(len(pred) + len(true), 1))
            hit = inc[inc["hit"]]
            conc = tot = 0
            for _, a in hit.iterrows():  # a is hit before b (or b never hit within the horizon)
                later = inc[(~inc["hit"]) | (inc["true_delay"] > a["true_delay"])]
                tot += len(later)
                conc += (a["p_hit"] > later["p_hit"]).sum() + 0.5 * (a["p_hit"] == later["p_hit"]).sum()
            if tot:
                cidx.append(conc / tot)
        tp = g[g["hit"] & (g["p_hit"] >= 0.5) & g["pred_onset"].notna()]
        out.append(dict(zip(["split", "method", "start_s", "system"], key), services=len(g), hit_rate=g["hit"].mean(),
                        auroc=auroc, f1=np.mean(f1s), mae_s=(tp["pred_onset"] - tp["true_delay"]).abs().mean(),
                        c_index=np.mean(cidx) if cidx else np.nan))
    return pd.DataFrame(out)


def evaluate(train: list, test: list, split: str, seeds: list[int], ablations: bool) -> list[dict]:
    rows = []
    baselines = Baselines(train)
    for inc in test:
        for method in ("all at once", "graph table", "memory"):
            p, onset = baselines.predict(inc, method)
            rows += score_rows(inc, 0, p, onset, method, split)
    configs = {"PropagationNet": {}}
    if ablations:
        configs.update({"PropagationNet - graph": {"use_graph": False}, "PropagationNet - fault type": {"drop_fault": True},
                        "PropagationNet - live evidence": {"drop_dynamic": True}})
    for name, cfg in configs.items():
        use_graph = cfg.get("use_graph", True)
        ablation = {k: v for k, v in cfg.items() if k != "use_graph"}
        run_seeds = seeds if name == "PropagationNet" else seeds[:1]
        models = [train_net(train, seed, use_graph=use_graph, **ablation) for seed in run_seeds]
        for inc in test:
            for start in (START_BINS if name == "PropagationNet" else [0]):
                preds = [net_predict(m, inc, start, **ablation) for m in models]
                p = np.mean([q[0] for q in preds], axis=0)
                onset = np.nanmedian(np.stack([q[1] for q in preds]), axis=0) if len(preds) > 1 else preds[0][1]
                rows += score_rows(inc, start, p, onset, name, split)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--edges", default="experiments/graph/re2_graph_all_edges.csv")
    ap.add_argument("--out", default="experiments/propagation_forecast")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--confirm", action="store_true", help="run the frozen RE3 confirmatory test")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(args.edges)[["system", "source", "target"]].drop_duplicates()
    re2 = load_incidents(args.raw, "RE2", edges, out / "incidents_re2.pkl")
    pd.set_option("display.width", 220)

    if args.confirm:
        re3 = load_incidents(args.raw, "RE3", edges, out / "incidents_re3.pkl")
        rows = evaluate(re2, re3, "RE3 (confirmatory)", args.seeds, ablations=False)
        detail = pd.DataFrame(rows)
        detail.to_csv(out / "forecast_rows_re3.csv", index=False)
        summary = summarise(detail)
        summary.to_csv(out / "forecast_summary_re3.csv", index=False)
        gain = update_gain(detail)
        gain.to_csv(out / "forecast_update_gain_re3.csv", index=False)
        print(summary.round(3).to_string(index=False))
        print(gain.round(3).to_string(index=False))
        return

    rows = []
    for rep in (1, 2, 3):
        train = [i for i in re2 if i.repetition != rep]
        test = [i for i in re2 if i.repetition == rep]
        rows += evaluate(train, test, "rep-CV", args.seeds, ablations=True)
        print(f"rep-CV fold {rep} done", flush=True)
    for system in ("ob", "ss", "tt"):
        for root in sorted({i.root for i in re2 if i.system == system}):
            train = [i for i in re2 if not (i.system == system and i.root == root)]
            test = [i for i in re2 if i.system == system and i.root == root]
            rows += evaluate(train, test, "unseen-root", args.seeds[:1], ablations=False)
        print(f"unseen-root {system} done", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "forecast_rows_re2.csv", index=False)
    summary = summarise(detail)
    summary.to_csv(out / "forecast_summary_re2.csv", index=False)
    print(summary[summary["start_s"].eq(0)].round(3).to_string(index=False))
    print(summary[summary["method"].eq("PropagationNet")].round(3).to_string(index=False))
    gain = update_gain(detail)
    gain.to_csv(out / "forecast_update_gain_re2.csv", index=False)
    print(gain.round(3).to_string(index=False))

    model = train_net(re2, args.seeds[0])  # final model for the console
    torch.save({"state": model.state_dict(), "static": STATIC_NAMES, "dynamic": DYNAMIC_NAMES}, out / "propagation_net.pt")
    print(f"saved {out / 'propagation_net.pt'}")


if __name__ == "__main__":
    main()
