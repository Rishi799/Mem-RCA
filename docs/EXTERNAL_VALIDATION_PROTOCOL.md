# External validation protocol (frozen before any RE1/RE3 result)

Written on 2026-11-03, after RE1 and RE3 were downloaded and before any open-world or
root-cause number was computed on them. Only the case index and metric column names were
inspected (needed to map metric types). RE2 is the development set: every design choice
below was made on RE2, including RE2 test incidents, so RE2 numbers are optimistic.
The RE1/RE3 numbers produced by `scripts/evaluate_external.py` are the confirmatory ones
and are reported whatever they show.

## Frozen method: Mem-RCA

- **Fingerprint input:** per-service metric-shift profile over the first 120 s after the
  alert/injection time (`metric_shift_profile`, 8 metric types; RE1 Online Boutique's
  `latency`/`load` are mapped to `latency-90`/`workload`). Time-resolved profiles were
  dropped: they were worse on RE2 for every system.
- **Root anchor:** (a) *learned*: the temporal-graph model (LSTM encoders + directed graph
  attention, `scripts/train_temporal_graph.py`, default settings, seed 42) retrained from
  scratch on each external fold's training incidents only; (b) *learning-free*: the service
  with the largest metric shift.
- **View and memory, chosen per system by the pseudo-open-set check:** view ∈ {anchor,
  global_max, hybrid}, memory = split (k-NN familiarity for the NOVEL decision, prototypes for
  naming and enrolment) with k ∈ {1, 3}; six candidates. Score = mean novelty AUROC when each
  known type in turn is treated as unseen, using memory and validation incidents only
  (OSCR breaks ties).
- **Threshold:** τ = 5th percentile of familiarity on validation incidents (95 % of known
  incidents kept).

## Baselines (fixed, prototype memory)

`global_mean`, `global_max` and `anchor` (with the same root source), i.e. the E2 set.

## E-RE1: within-suite open world on RE1 (375 incidents, metrics only)

- Per system, five folds: each of cpu, mem, disk, delay, loss is the unseen type once.
- Memory = repetitions 1–2 of the four known types (40 incidents), validation = repetition 3
  (20), known test = repetitions 4–5 (40), unseen test = all 25 incidents of the held-out type.
- The learned localiser is trained on the memory incidents and early-stopped/calibrated on the
  validation incidents of the same fold (no traces in RE1, so the graph has only self-loops).

## E-RE3: code-level faults as genuinely new failure types

- Per system, the memory is built from RE2 (all six resource fault types): repetition 1 =
  memory, repetition 2 = validation (τ and the pseudo-open-set check), repetition 3 = known
  test (30). Unseen test = all 30 RE3 incidents (code-level faults F1–F5, a different recording
  campaign). Learning-free root only (no model is trained for this experiment).

## Hypotheses and decision rules

- **H1 (C2 replication, primary).** On RE1 Train Ticket, anchor + prototype with the learned
  root has higher novelty AUROC than both global_mean + prototype and global_max + prototype:
  mean over the five folds higher, and higher in at least 4 of 5 folds.
- **H2 (procedure).** Averaged over the three RE1 systems, Mem-RCA (learned root) has higher
  novelty AUROC than each fixed baseline.
- **H3 (new failure types).** For each system, Mem-RCA flags RE3 code-level incidents as less
  familiar than RE2 known incidents: AUROC > 0.5, with a one-sided Mann–Whitney p < 0.05.
- **Secondary, descriptive:** OSCR, known-type accuracy, NOVEL rate at τ; learned vs
  learning-free root (Hit@1) on RE1 including the unseen type; the discover-and-enrol stream
  and k-shot enrolment on RE1 for Mem-RCA vs global_mean.

Caveats fixed in advance: RE3 known and unseen incidents come from different recording
campaigns, so H3 can partly reflect a campaign difference; each fingerprint is normalised by
its own pre-injection period to limit this. Folds share root services and load generators,
so the five folds per system are not independent samples.

## Outcome (added after the single run; the protocol above is unchanged)

- **H1: not supported.** RE1 Train Ticket, learned root:
  - anchor 0.59 vs global mean 0.60 (anchor better in 2/5 folds);
  - anchor 0.59 vs global max 0.62 (2/5 folds).
- **H2: supported.** Mean novelty AUROC over the RE1 systems: Mem-RCA 0.74 vs global max 0.72, anchor 0.67 and global mean 0.66.
- **H3: supported.** RE3 code-level faults vs RE2 memory, Mem-RCA AUROC:
  - OB 0.84 (p = 4e-6)
  - SS 0.83 (p = 5e-6)
  - TT 0.96 (p = 4e-10)

Details: [OPEN_WORLD_RESULTS.md](OPEN_WORLD_RESULTS.md), sections 4–5. Raw outputs: `experiments/external/results/`.
