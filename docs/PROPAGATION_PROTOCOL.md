# Time-to-impact forecasting: protocol (frozen before any PropagationNet result)

Written on 2026-11-03.

**What had been seen before writing this:** the RE2 feasibility analysis
(`scripts/propagation_feasibility.py`), which compared simple forecasts (memory of the same
root and fault, all at once, one-hop neighbours) on RE2. No PropagationNet result on RE2 or
RE3 had been seen, and nothing on RE3 propagation had been computed.

## Task

From the moment the root service's anomaly is confirmed (t0), forecast for every other
service whether its anomaly onset will occur within 300 s and when. Forecasts are made
at t0 and updated at t0 + 30 s and t0 + 60 s, each using only telemetry observed up to
that moment.

## Labels

- **Onset detector** (`memrca.propagation.onsets`):
  - per metric, the robust z against the incident's own normal period, on a log scale;
  - per service, the maximum over its metrics, then a 5 s trailing median;
  - threshold = the normal-period maximum + 0.3, floor 0.8;
  - onset = the first 5 consecutive seconds above the threshold.
- **t0** is the root's onset. It is the injection time if the root is never detected.

## Method

PropagationNet (`memrca.propagation.model`) is a discrete-time hazard model:

- **Per-service input:** static call-graph and fault features, plus live features (`memrca.propagation.dataset`).
- **Network:** an MLP encoder, then two directed graph-attention layers (evidence from callers and from callees), then a hazard head.
- **Training settings:** 20 epochs, Adam (lr 3e-3, weight decay 1e-4), fault-type dropout 0.3.
- **Seeds:** 42, 43 and 44; their forecasts are averaged.
- **Shared model:** one model for all systems, since every feature is service-agnostic.

## Baselines

- **All at once:** every service is hit at t0.
- **Graph table:** hit rate and median delay per (hops up, hops down) from the training incidents.
- **Memory:** the same root and fault in training incidents, else the same fault, else the graph table.

## Development (RE2; reported, not confirmatory)

- **rep-CV:** each repetition is tested with a model trained on the other two.
- **Unseen root:** each root service is tested with a model trained without it.
- **Ablations:** no graph, no fault type, no live evidence.

## Confirmatory test (RE3, run once with `python scripts/train_propagation.py --confirm`)

- **Training:** all 270 RE2 incidents.
- **Test:** all 90 RE3 code-level incidents. The fault type is unknown (the memory has never seen code faults).
- **Call graph:** the RE2 trace topology of each system.

Pre-registered hypotheses, judged on services not yet hit at the forecast time:

- **H4 (who is hit).** At t0, PropagationNet's AUROC of P(hit within 300 s), pooled over all RE3 services, is higher than the graph-table baseline's.
- **H5 (order).** At t0, PropagationNet's mean per-incident C-index is higher than the graph-table baseline's.
- **H6 (live update).** On the services still not hit at t0 + 60 s, the forecast updated at t0 + 60 s has a higher AUROC than the t0 forecast.

All metrics are reported per system and pooled, whatever they show.

## Caveats fixed in advance

- **Approximate labels:** the onset labels come from a detector, not ground truth. Detector false onsets in the normal period were 4–7 % on RE2.
- **Small and partial graphs:** RE3 has 30 incidents per system. Sock Shop has no trace graph, and some services lie outside the traced call graph.
- **Different campaign:** RE3 was recorded separately from RE2, and its faults are code-level, not resource faults.

## Outcome (added after the single RE3 run; the protocol above is unchanged)

- **H4: not supported.** Pooled AUROC at t0: PropagationNet 0.70 vs graph table 0.74.
  - Per system, PropagationNet is better on SS (0.66 vs 0.50) and TT (0.69 vs 0.61) and worse on OB (0.58 vs 0.61).
  - The pooled metric mixes systems with very different hit rates (SS 14 %, TT 2.6 %).
- **H5: supported.** C-index at t0: 0.64 vs 0.55 pooled, and higher in every system.
- **H6: supported pooled.** AUROC on services not hit by t0 + 60 s rises from 0.68 (t0 forecast) to 0.73 (updated forecast). It rose on OB and SS but not on TT (0.67 → 0.65).

RE2 development numbers:
- **rep-CV:** pooled AUROC PropagationNet 0.68, memory 0.69, graph table 0.65.
- **Unseen root:** 0.70 vs 0.66 / 0.63.
- **Ablations:** removing live evidence costs the most (C-index 0.69 → 0.63); removing graph attention changes nothing, because the hop features already carry the graph.

Outputs are in `experiments/propagation_forecast/`.
