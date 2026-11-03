# Mem-RCA results: open-world root cause analysis with an incident memory

**Status.** RE2 is the development set: every design choice was made on it, including after
looking at its test incidents, so RE2 numbers are optimistic. RE1 and RE3 were downloaded
afterwards and evaluated once under a protocol frozen beforehand
([EXTERNAL_VALIDATION_PROTOCOL.md](EXTERNAL_VALIDATION_PROTOCOL.md)); those numbers are confirmatory.

## 1. Summary

| | Online Boutique | Sock Shop | Train Ticket |
|---|---:|---:|---:|
| Services seen by the RE2 localiser | 12 | 15 | 69 |
| **RE2, two fault types unseen (development)**: Mem-RCA novelty AUROC | 0.78 | 0.60 | 0.79 |
| RE2: best E2 fingerprint (global max / anchor) | 0.71 | 0.63 | 0.75 |
| RE2: average over services (global mean) | 0.66 | 0.56 | 0.55 |
| **RE1, one fault type unseen (confirmatory)**: Mem-RCA | 0.82 | 0.76 | 0.65 |
| RE1: best fixed fingerprint | 0.84 | 0.71 | 0.62 |
| RE1: average over services | 0.82 | 0.54 | 0.60 |
| **RE3 code-level faults vs RE2 memory (confirmatory)**: Mem-RCA | 0.84 | 0.83 | 0.96 |
| RE3: average over services | 0.50 | 0.59 | 0.35 |

- **Pre-registered hypotheses.** H1 (anchoring beats global pooling on RE1 Train Ticket) is **not supported**. H2 (Mem-RCA beats every fixed fingerprint on average over RE1 systems) is **supported**, 0.74 vs at most 0.72. H3 (RE3 code faults look unfamiliar) is **supported** in every system.
- **C2, fingerprint view.**
  - Averaging a fingerprint over all services is unreliable. On Train Ticket it gives AUROC 0.55 (RE2) and 0.35 (RE3), and on Sock Shop 0.56 (RE2) and 0.54 (RE1).
  - Root anchoring fixed this on RE2 and RE3 Train Ticket, but the gain **did not replicate on RE1 Train Ticket** (H1).
  - No single view wins everywhere. Choosing the view per system from known incidents only (Mem-RCA) is the robust option: it is best or close to best on every suite.
- **C3, discover and enrol.** On Train Ticket, 47 % of unseen fault types are discovered from the stream, with 98 %-pure clusters. After enrolment, 90 % of later incidents of the new type are named correctly. With k = 3 examples, a new type is recognised 87 % of the time (global mean: 51 %).
- **C4, benchmark shortcut.** With five root services per system, a learning-free ranking restricted to those five nearly matches the learned localiser. When a root service is held out of training, the learned localiser drops from Hit@1 ≥ 0.94 to 0.20 / 0.13 / 0.09 (OB / SS / TT), while the learning-free largest-shift rule keeps 0.93 / 0.84 / 0.79. It has largely learned *which services are usually roots*.
- **Weak spots.** Sock Shop stays hard on RE2 (AUROC ≈ 0.6), and RE1 Train Ticket is hard for every fingerprint (≤ 0.65). On RE1, discovered types are often named wrongly after enrolment (30–38 %). At the 95 %-known operating point, many unseen RE2 incidents are still accepted as known. The learned temporal-graph embeddings are not useful as fingerprints.

## 2. Final method

1. **Localiser.** Per-service LSTM encoders and directed graph attention (`MissingnessAwareTemporalGraph`, Phase 3), trained on the fold's training incidents. Each incident's root scores are the mean of the top quarter of window logits in the first 120 s. A learning-free alternative ranks services by their largest metric shift.
2. **Metric-shift profile.** For every service and each of 8 metric types (cpu, mem, diskio, socket, workload, latency-50, latency-90, error), the robust shift of the first 120 s after the alert is measured against the incident's own pre-fault period: sign · log10(1 + |z|).
3. **Fingerprint views.**
   - *anchor* = [root row ‖ symptom signature], where the symptom signature is the other services' rows weighted by how strongly each shifted;
   - *global max* = the strongest shift per metric type over all services;
   - *hybrid* = both views concatenated.
4. **Memory (split).** Fingerprints are standardised with memory statistics and L2-normalised.
   - **NOVEL decision:** familiarity is the cosine similarity to the k nearest stored incidents of each type. Below τ, the incident is NOVEL. τ is the 5th percentile of familiarity on validation incidents.
   - **Naming:** an accepted incident is named by the nearest type prototype. Enrolment writes a new prototype and its incidents, with no gradient step.
5. **Configuration chosen per system without test data.** The view (anchor / global max / hybrid) and k ∈ {1, 3} are picked by a *pseudo-open-set check*: each known type in turn plays "unseen" among memory and validation incidents, and the candidate with the best mean novelty AUROC is chosen.

## 3. Development on RE2

Setup per fold and system:

- **Memory:** 20 known incidents (repetition 1).
- **Validation:** 20 incidents (repetition 2).
- **Known test:** 20 incidents (repetition 3).
- **Unseen test:** 10 incidents (repetition 3) plus 20 "fresh" incidents (repetitions 1–2 of the unseen types, never used by any earlier run).

There are 3 seeds × 3 folds; the localisers are the 27 Phase 3 checkpoints.

### E2: root-anchored vs global fingerprints (prototype memory, fixed before results)

| System | global mean | global max | anchor (learned root) | anchor, paired vs global mean |
|---|---:|---:|---:|---|
| OB | 0.66 | **0.71** | 0.64 | −0.02 (3/9 runs) |
| SS | 0.56 | 0.62 | **0.63** | +0.07 (6/9) |
| TT | 0.55 | 0.62 | **0.75** | **+0.20 (9/9)** |

The fresh unseen incidents give the same picture (TT anchor 0.76). Anchoring on the true root instead of the predicted one changes little, so localisation errors are not the bottleneck. The learned per-service embeddings are near chance (0.50–0.59) with every pooling. The encoders are per service and trained for forecasting, so their spaces are not comparable across services.

### Choosing view and memory without test data

| System | Mem-RCA choice | AUROC (fresh) | OSCR | Known-type acc. | Better than E2 anchor |
|---|---|---:|---:|---:|---:|
| OB | global max, k = 1 | **0.78** (0.78) | 0.68 | 0.82 | 9/9 runs |
| SS | global max, k = 3 | 0.60 (0.61) | 0.41 | 0.60 | 4/9 |
| TT | anchor (learned root), k = 1 | **0.79** (0.80) | 0.73 | 0.90 | 9/9 |

Lessons from the development search (60 candidates, documented in `selection_summary.csv`):

- **Too many candidates overfit the small validation sets.** Searching all 60 combinations picked a learning-free TT variant that was worse on test (0.72). The frozen procedure therefore keeps six candidates.
- **The learned localiser is worth having for anchoring.** On TT, anchor with the learned root reaches 0.756 mean AUROC; anchoring on the largest metric shift reaches 0.722.
- **Time-resolved profiles (0–30 / 30–60 / 60–120 s) were worse for every system** and were dropped.
- **k-NN familiarity improves detection but hurts enrolment** when used for naming too: a type enrolled from 3 incidents is out-voted. Splitting the two jobs (k-NN for NOVEL, prototypes for naming) fixes this. TT accuracy after enrolment is 0.90 with split memory, versus 0.79 with k-NN for both.

### Discover → enrol stream (C3) and k-shot enrolment (E4), Mem-RCA configuration

The stream simulation works as follows:

- **Arrivals:** test incidents arrive in a random order (5 orders per run).
- **Grouping:** NOVEL incidents are buffered and grouped by HDBSCAN (minimum 3 incidents).
- **Naming:** a simulated engineer names each group by its majority label, and the group is enrolled.

| System | Unseen types discovered | Occurrences before discovery | Cluster purity | Accuracy after enrolment | k-shot new-type acc. (k = 1 / 3 / 5) |
|---|---:|---:|---:|---:|---|
| OB | 50 % | 7.9 | 0.78 | 0.49 | 0.43 / 0.69 / 0.60 |
| SS | 0 % | – | – | – | 0.48 / 0.65 / 0.65 |
| TT | **47 %** | 9.1 | **0.98** | **0.90** | **0.68 / 0.87 / 0.87** |
| TT, global mean | 0 % | – | – | – | 0.31 / 0.51 / 0.52 |

## 4. Confirmatory: RE1 (independent suite, metrics only)

Five leave-one-fault-type-out folds per system:

- **Memory:** repetitions 1–2 of the four known types (40 incidents).
- **Validation:** repetition 3 (20 incidents).
- **Known test:** repetitions 4–5 (40 incidents).
- **Unseen test:** all 25 incidents of the held-out type.

The temporal-graph localiser was retrained from scratch on each fold's memory incidents (15 models). RE1 has no traces, so the graph has only self-loops, and no disk-I/O or socket metrics.

### Novelty detection (learned root)

| System | Mem-RCA choice | AUROC | OSCR | Known-type acc. | Unseen flagged at τ | anchor | global max | global mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OB | global max, k = 1 | 0.82 | 0.75 | 0.90 | 35 % | 0.72 | **0.84** | 0.82 |
| SS | global max, k = 3 | **0.76** | 0.64 | 0.82 | 9 % | 0.71 | 0.69 | 0.54 |
| TT | hybrid (learned root), k = 1 | **0.65** | 0.50 | 0.72 | 33 % | 0.59 | 0.62 | 0.60 |
| mean | | **0.74** | | | | 0.67 | 0.72 | 0.66 |

- **H1, not supported.** On Train Ticket, anchor + prototype (learned root) has AUROC 0.59, against 0.60 for global mean and 0.62 for global max. It is better in only 2 of 5 folds against each.
  - Possible reasons, not tested: RE1 lacks the disk-I/O and socket metrics that make the root row distinctive in RE2/RE3, and every fingerprint is weak on RE1 Train Ticket.
- **H2, supported.** Mem-RCA averages 0.74 against 0.72 (global max), 0.67 (anchor) and 0.66 (global mean). The per-system choice is the best or close to the best in every system: on SS +0.04 over the best fixed view, on OB −0.02.
- **Learning-free root.** With the learning-free root, the Train Ticket choice (hybrid, shift root) reaches 0.59. OB and SS do not use a root.

### Root localisation on RE1

| System | Learned, known types | Learned, unseen type | Largest shift, known | Largest shift, unseen |
|---|---:|---:|---:|---:|
| OB | 0.93 | 0.66 | 0.80 | 0.82 |
| SS | 0.99 | 0.72 | 0.90 | 0.92 |
| TT | 0.95 | 0.83 | 0.76 | 0.65 |

The learned localiser is strong on fault types it was trained on but loses accuracy on the held-out type (OB 0.93 → 0.66).

### Discover → enrol and k-shot enrolment on RE1

| System | Types discovered (Mem-RCA / global mean) | Cluster purity | Accuracy after enrolment | k = 3 new-type acc. (Mem-RCA / global mean) |
|---|---|---|---|---|
| OB | 40 % / 20 % | 0.99 / 0.88 | 0.38 / 0.64 | 0.71 / 0.66 |
| SS | 0 % / 56 % | – / 0.63 | – / 0.26 | 0.66 / 0.42 |
| TT | 56 % / 20 % | 0.73 / 0.49 | 0.30 / 0.34 | 0.61 / 0.46 |

The results are mixed:

- **k-shot enrolment:** Mem-RCA beats the averaged fingerprint in every system.
- **Stream discovery:** Mem-RCA finds more types on OB and TT, but enrolled types are often named wrongly afterwards. On SS, the averaged fingerprint discovers more, but with impure groups.

## 5. Confirmatory: RE3 code-level faults (H3)

The memory holds the six RE2 resource fault types. The known test incidents are RE2 repetition 3 (30 per system); the unseen ones are all 30 RE3 incidents per system (code-level faults F1–F5, a separate recording campaign). The root is learning-free.

| System | Mem-RCA choice | AUROC | One-sided Mann–Whitney p | Flagged NOVEL at τ | Known kept at τ | global mean AUROC |
|---|---|---:|---:|---:|---:|---:|
| OB | global max, k = 1 | 0.84 | 4e-6 | 53 % | 83 % | 0.50 |
| SS | global max, k = 3 | 0.83 | 5e-6 | 40 % | 90 % | 0.59 |
| TT | anchor, k = 1 | **0.96** | 4e-10 | **97 %** | 90 % | 0.35 |

**H3 is supported.** On TT, global max has a slightly higher AUROC (0.99) but flags only 30 % of code faults at τ, against 97 % for Mem-RCA. Caveat fixed in advance: known and unseen incidents come from different recording campaigns. Each fingerprint is normalised by its own pre-fault period, and the averaged fingerprint is near chance, so a pure campaign effect is unlikely to explain the result, but it cannot be ruled out.

## 6. Root-cause localisation and the benchmark shortcut (C4)

Hit@1 on RE2 test incidents (27 runs):

| System | Learned localiser | Shift, 5 training roots first | Shift, all services | No telemetry (prior) |
|---|---:|---:|---:|---:|
| OB | 0.99 | 0.92 | 0.92 | 0.20 |
| SS | 0.92 | 0.85 | 0.84 | 0.20 |
| TT | **0.97** | **0.95** | 0.79 | 0.20 |

### Service hold-out: roots never seen as roots in training

Each root service is held out in turn (15 retrained models, seed 42):

- **Training:** the localiser never sees an incident at the held-out service.
- **Test:** seen-root incidents are repetition 3 of the other four roots; unseen-root incidents are every incident at the held-out service.

All six fault types are known in this experiment.

| System | Learned, seen root | Learned, **held-out root** (Hit@3) | Largest shift, held-out root | Training roots first / no telemetry, held-out root |
|---|---:|---:|---:|---:|
| OB | 1.00 | **0.20** (0.60) | 0.93 | 0.00 / 0.00 |
| SS | 0.94 | **0.13** (0.54) | 0.84 | 0.00 / 0.00 |
| TT | 0.98 | **0.09** (0.24) | 0.79 | 0.00 / 0.00 |

- **Roots seen in training:** the learned localiser is near perfect.
- **Held-out roots:** it collapses, while the learning-free largest-shift rule hardly changes. The root head has mostly learned the identity of RE2's five root services, so RE2 Hit@1 overstates real localisation skill.
- **Consequence for Mem-RCA:** if the root may be a service that has never failed before, anchor on the largest-shift root, or use the global-max view, rather than trust the learned head. A localiser that combines both is future work.

## 7. What did not work

- **Learned temporal-graph embeddings as fingerprints:** near chance (AUROC 0.50–0.59).
- **Time-resolved shift profiles:** worse than the 120 s profile everywhere.
- **Searching 60 configurations on small validation sets:** it overfits (winner's curse), so the procedure was reduced to six candidates.
- **Root anchoring as a universal default:** it helped on RE2/RE3 Train Ticket but not on RE1 Train Ticket (H1).
- **Sock Shop:** no fingerprint separates unseen types well (AUROC ≈ 0.6), and no unseen type is ever discovered there.

## 8. Caveats

- **RE2 has been looked at.** Its test incidents were inspected in earlier phases and in this study, so RE2 numbers are optimistic. RE1 and RE3 are the confirmatory evidence.
- **Small samples.** Folds and seeds reuse the same root services and load patterns, so runs are not independent samples.
- **Simulated engineer.** The engineer in C3 is simulated with true labels.
- **Replay, not production.** All results come from replayed recordings, not production telemetry.

## 9. Reproduce

See the README section "Reproducing the open-world experiments". Outputs:

- `experiments/open_world/` (RE2);
- `experiments/external/` (RE1, RE3);
- `experiments/service_holdout/`;
- figures in `docs/figures/`.

## 10. C5: time-to-impact forecasting (PropagationNet)

**Method.** Once the root's anomaly is confirmed (t0), a graph neural network forecasts, for every other service, the probability of being hit in each 10 s bin, up to 300 s:

- an MLP encoder;
- two directed graph-attention layers, one passing evidence from callers and one from callees;
- a hazard head.

Forecasts are updated with the evidence observed so far. Protocol and outcome: [PROPAGATION_PROTOCOL.md](PROPAGATION_PROTOCOL.md).

**Feasibility findings (RE2):**
- Faults travel mainly to callers (upstream): 67 % of OB callers are hit, with a median of 11 s, against 36 % of callees at 70 s.
- Delay grows with the number of hops.
- Delay and loss faults spread fastest and most reproducibly.

**Results.**

| | AUROC hit within 300 s (pooled) | C-index |
|---|---:|---:|
| RE2 rep-CV: PropagationNet / memory / graph table | 0.68 / 0.69 / 0.65 | 0.69 / 0.70 / 0.63 |
| RE2 unseen root: PropagationNet / memory / graph table | **0.70** / 0.66 / 0.63 | **0.68** / 0.61 / 0.62 |
| RE3 confirmatory: PropagationNet / graph table | 0.70 / **0.74** (H4 not supported) | **0.64** / 0.55 (H5 supported) |

Updating the forecast with live evidence improves it consistently: on RE3, AUROC rises from 0.68 to 0.73 at t0 + 60 s (H6). Graph attention adds nothing over the hop-distance features (ablation), and absolute accuracy is moderate.
