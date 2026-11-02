# Phase 3 calibration status (RE2, exploratory)

## Current validation-fit diagnostics

The full temporal-graph model was retrained with seed 42 for five epochs on all three folds and all three systems. Temperature scaling and the abstention threshold were fitted on each run's 20 validation incidents. The table reports mean and standard deviation across the three folds; Brier score and ECE are measured on the same validation incidents used to fit temperature, so these are fit diagnostics, not an independent calibration evaluation.

| System | Validation incidents per fold | Brier score (mean ± SD) | ECE (mean ± SD) |
|---|---:|---:|---:|
| Online Boutique | 20 | 0.000011 ± 0.000013 | 0.000432 ± 0.000361 |
| Sock Shop | 20 | 0.059283 ± 0.060634 | 0.044031 ± 0.042970 |
| Train Ticket | 20 | 0.044397 ± 0.040315 | 0.028768 ± 0.024579 |

These values vary by system. Online Boutique's near-zero scores may reflect very easy, saturated predictions; they do not establish reliable confidence under distribution shift. Fold validation sets also overlap in incident IDs, so the three-fold aggregate is not 60 independent incidents per system.

## Evaluation limits and next runs

- The RE2 test cases were inspected during earlier development. Keep all RE2 results exploratory.
- The full-model three-seed grid across all folds/systems completed under `experiments/temporal_graph_re2_grid/` (27 runs total).
- No-graph, no-mask, and no-consistency-loss ablations completed for all folds/systems with seed 42 under `experiments/temporal_graph_re2_ablation_grid/` (27 runs total). This is still only one seed per ablation, so component effects need replication.
- RE1 and RE3 case folders are not present locally. The earlier RE3 download failed because this machine could not resolve the Hugging Face host. RE1 calibration plus untouched RE3 test remains necessary for independent validation.
- Across folds, some validation incident IDs repeat. For an independent calibration result, report the number of unique cases and use an external suite or a strictly separated calibration set.

Refresh run-level calibration summaries with:

```powershell
python scripts/summarize_temporal_graph_calibration.py --root experiments/temporal_graph_re2_grid
python scripts/summarize_temporal_graph_calibration.py --root experiments/temporal_graph_re2_ablation_grid
```
