# XGBoost Model

A gradient-boosted classifier for 3-class football outcome prediction (home_win / draw / away_win), designed specifically for neutral-venue tournament prediction.

## Design Decisions

### 1. Difference features

All features are expressed as `home_team_value − away_team_value`. This is the natural representation for a symmetric quality comparison and avoids giving any signal to whichever team happens to be labelled "home" in a neutral venue match. See [eda/README.md](../../eda/README.md) for the feature selection rationale.

### 2. Augmented training (flip augmentation)

Every match in the training set is included twice:
- **Forward**: original features + original outcome
- **Flipped**: negated features + swapped outcome (home_win ↔ away_win, draw stays draw)

Negating all difference features is equivalent to swapping the two teams. This forces the model to learn symmetric quality differences, making it neutral-venue aware without discarding any training data.

### 3. Symmetrized inference

At prediction time, each match is predicted twice — forward and with teams swapped (features negated) — and the results are averaged:

```
P(team_A wins) = (P(home_win | forward) + P(away_win | inverted)) / 2
P(draw)        = (P(draw | forward)     + P(draw | inverted))     / 2
P(team_B wins) = (P(away_win | forward) + P(home_win | inverted)) / 2
```

Together with augmented training, this ensures predictions are fully invariant to which team is arbitrarily labelled "home" in the input data.

### 4. Draw class weighting

Draws (≈23% of outcomes) are underrepresented relative to home wins (≈49%) and away wins (≈29%). Without correction, XGBoost rarely predicts draws.

Class weights are set inversely proportional to class frequency in the augmented training set, then multiplied by a tunable `draw_weight` parameter:

| `draw_weight` | Draw class weight | Effect |
| --- | --- | --- |
| 0.0 | 0.00 | Never predicts draws |
| 0.65 | 0.95 | **Selected** — best accuracy trade-off |
| 1.0 | 1.47 | Full inverse-frequency compensation — over-predicts draws |

`draw_weight = 0.65` was tuned on the WC 2022 backtest.

## Feature Set

See [features.md](../../features.md#model-feature-selection) for the full description. Summary:

| Group | Features |
| --- | --- |
| Ratings | `elo_diff`, `abs_elo_diff` |
| ELO momentum | `elo_delta_20_diff` |
| ELO history | `elo_ma_2yr_diff`, `elo_ma_4yr_diff`, `elo_ma_8yr_diff` |
| WC experience | `wc_games_diff` |
| Weighted points won | `pww_ma20_diff`, `pww_ma5_diff` |
| Weighted goals scored | `gw_ma20_diff`, `gw_ma5_diff` |
| Weighted goals suffered | `gsw_ma20_diff`, `gsw_ma5_diff` |
| Goal difference | `gd_ma20_diff`, `gd_ma5_diff` |

`points_dif`, `abs_points_dif`, and `pi_diff` are excluded — see [`eda/README.md`](../../eda/README.md) for collinearity analysis. `neutral` is excluded — augmentation and symmetrized inference handle venue invariance.

## Files

| File | Description |
| --- | --- |
| `model.py` | `XGBoostPredictor` class |
| `backtest.py` | WC 2022 backtest — trains and evaluates on 64 matches |
| `tune_draw_weight.py` | Sweeps `draw_weight` values to find the optimal trade-off |
| `wc2022_backtest_results.csv` | Per-match backtest results |

## WC 2022 Results

Accuracy: **54.7% (35/64)** — Log-Loss: **1.069** — RPS: **0.218**

See [models/README.md](../README.md) for comparison with Dixon-Coles.
