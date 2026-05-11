# CatBoost Model

A gradient-boosted classifier for 3-class football outcome prediction, built on the same design as the XGBoost model with two key additions: native categorical feature support and a tuned draw weight.

## Design

Identical to XGBoost in structure — see [`xgboost/README.md`](../xgboost/README.md) for the full explanation of flip augmentation and symmetrized inference. Differences:

**Categorical features:** `confederation_home` and `confederation_away` are passed directly to CatBoost without encoding. CatBoost builds ordered target statistics for each category internally, which is more robust than one-hot encoding for low-frequency categories (e.g. OFC).

**Flipping categorical features:** When augmenting with flipped matches, `confederation_home` and `confederation_away` are swapped (not negated, since they are categorical).

## Feature Set

| Group | Features |
| --- | --- |
| Ratings | `points_dif`, `elo_diff`, `pi_diff` |
| Weighted points won | `pww_ma20_diff`, `pww_ma5_diff` |
| Weighted goals scored | `gw_ma20_diff`, `gw_ma5_diff` |
| Weighted goals suffered | `gsw_ma20_diff`, `gsw_ma5_diff` |
| Goal difference | `gd_ma20_diff`, `gd_ma5_diff` |
| Confederation (categorical) | `confederation_home`, `confederation_away` |

## Tuned Parameters

- `draw_weight = 0.75` (tuned on WC 2022 — best RPS)
- `iterations = 500`, `learning_rate = 0.05`, `depth = 6`

## Files

| File | Description |
| --- | --- |
| `model.py` | `CatBoostPredictor` class |
| `backtest.py` | WC 2022 backtest |
| `tune_draw_weight.py` | Sweeps `draw_weight` values |
| `wc2022_backtest_results.csv` | Per-match backtest results |

## WC 2022 Results

Accuracy: **53.1% (34/64)** — Log-Loss: **1.050** — RPS: **0.214**

See [models/README.md](../README.md) for full comparison.
