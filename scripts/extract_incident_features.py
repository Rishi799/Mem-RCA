"""Extract per-incident features from trained temporal-graph checkpoints (open-world study).

For every case of a fold/system it runs the frozen checkpoint on the fault windows in
the first ``--horizon`` seconds after injection and stores, per service:
  * the case-level root logit (mean of the top quartile of window logits; same rule
    as the Phase 3 calibration code) -> predicted root cause,
  * the mean learned embedding and mean forecast error,
  * the metric-shift profile over canonical metric types (model-independent).

Cases: the fold manifest (train/validation/test) plus "test_extra" cases, i.e.
repetitions 1-2 of the fold's unseen fault types. Those cases are not in any split of
the manifest, so no model was trained, tuned, or evaluated on them before.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from memrca.data.loader import load_case, parse_case_name  # noqa: E402
from memrca.data.temporal_graph import ServiceScaler, _transform_case  # noqa: E402
from memrca.models.temporal_graph import MissingnessAwareTemporalGraph  # noqa: E402
from memrca.openworld.fingerprint import METRIC_TYPES, metric_shift_profile  # noqa: E402


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MissingnessAwareTemporalGraph(
        ckpt["feature_sizes"], hidden_size=cfg["hidden_size"], num_layers=cfg["layers"],
        attention_heads=cfg["heads"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scalers = {s: ServiceScaler(v["feature_names"], np.asarray(v["center"], np.float32),
                                np.asarray(v["scale"], np.float32)) for s, v in ckpt["scalers"].items()}
    edge = torch.as_tensor(np.asarray(ckpt["edge_index"]), dtype=torch.long, device=device)
    if cfg.get("disable_graph"):
        edge = torch.empty((2, 0), dtype=torch.long, device=device)
    return model, ckpt["services"], scalers, edge, cfg


@torch.no_grad()
def case_features(model, services, scalers, edge, cfg, case, horizon, device):
    window, stride = int(cfg["window"]), int(cfg["stride"])
    grid, values, masks = _transform_case(case, scalers)
    starts = [s for s in range(0, len(grid) - window + 1, stride)
              if 0 <= int(grid[s]) - case.inject_time < horizon]
    if not starts:
        return None
    seq, msk, tgt, tmsk = [], [], [], []
    for service in services:
        v, m = values[service], masks[service]
        seq.append(torch.from_numpy(np.stack([v[s:s + window - 1] for s in starts])).to(device))
        msk.append(torch.from_numpy(np.stack([m[s:s + window - 1] for s in starts])).to(device))
        tgt.append(torch.from_numpy(np.stack([v[s + window - 1] for s in starts])).to(device))
        tmsk.append(torch.from_numpy(np.stack([m[s + window - 1] for s in starts])).to(device))
    model_masks = [torch.ones_like(m) for m in msk] if cfg.get("disable_mask_input") else msk
    out = model(seq, model_masks, edge)
    logits = out["root_cause_logits"].cpu().numpy()                 # (windows, services)
    emb = out["service_embeddings"].cpu().numpy().mean(axis=0)       # (services, hidden)
    errors = []
    for pred, target, observed in zip(out["forecast"], tgt, tmsk):
        count = observed.sum(dim=1)
        err = (((pred - target) ** 2) * observed).sum(dim=1) / count.clamp_min(1.0) + (1.0 - observed.mean(dim=1))
        errors.append(float(err.mean().cpu()))
    top = max(1, int(np.ceil(len(starts) * 0.25)))
    case_logits = np.sort(logits, axis=0)[-top:].mean(axis=0)
    return {"root_logits": case_logits, "embedding": emb, "error": np.asarray(errors),
            "shift": metric_shift_profile(case, services, horizon), "windows": len(starts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--manifest", default="experiments/splits/re2_folds.csv")
    ap.add_argument("--grid", default="experiments/temporal_graph_re2_grid/full")
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--folds", nargs="+", default=["fold1", "fold2", "fold3"])
    ap.add_argument("--systems", nargs="+", default=["ob", "ss", "tt"])
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out", default="experiments/open_world/features")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    manifest = pd.read_csv(args.manifest)
    all_cases = sorted(p.name for p in Path(args.raw).iterdir() if p.is_dir() and p.name.startswith("re2"))

    for seed in args.seeds:
        for fold in args.folds:
            for system in args.systems:
                model_path = Path(args.grid) / f"seed{seed}" / fold / system / "model.pt"
                if not model_path.exists():
                    print(f"skip (no checkpoint): {model_path}")
                    continue
                model, services, scalers, edge, cfg = load_model(model_path, device)
                rows = manifest[manifest["fold"].eq(fold) & manifest["system"].eq(system)]
                unseen = sorted(rows.loc[rows["fault_status"].eq("unseen"), "fault"].unique())
                listed = set(rows["case"])
                extra = [c for c in all_cases if c.startswith(f"re2{system}_") and c not in listed
                         and parse_case_name(c)["fault"] in unseen]
                table = [(r.case, r.split, r.fault, r.fault_status, str(r.root_cause_service))
                         for r in rows.itertuples(index=False)]
                table += [(c, "test_extra", parse_case_name(c)["fault"], "unseen", parse_case_name(c)["service"])
                          for c in extra]
                keep, feats = [], []
                for name, split, fault, status, root in table:
                    f = case_features(model, services, scalers, edge, cfg, load_case(args.raw, name),
                                      args.horizon, device)
                    if f is None:
                        print(f"  no fault windows: {name}")
                        continue
                    keep.append((name, split, fault, status, root))
                    feats.append(f)
                train_roots = sorted({r for _, s, _, _, r in keep if s == "train"})
                out = Path(args.out) / f"seed{seed}" / fold
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out / f"{system}.npz",
                    case=np.array([k[0] for k in keep]), split=np.array([k[1] for k in keep]),
                    fault=np.array([k[2] for k in keep]), status=np.array([k[3] for k in keep]),
                    root=np.array([k[4] for k in keep]),
                    root_logits=np.stack([f["root_logits"] for f in feats]),
                    embedding=np.stack([f["embedding"] for f in feats]),
                    error=np.stack([f["error"] for f in feats]),
                    shift=np.stack([f["shift"] for f in feats]),
                    services=np.array(services), train_roots=np.array(train_roots),
                    metric_types=np.array(METRIC_TYPES),
                )
                counts = pd.Series([k[1] for k in keep]).value_counts().to_dict()
                print(f"seed{seed} {fold} {system}: {len(keep)} cases {json.dumps(counts)} -> {out / (system + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
