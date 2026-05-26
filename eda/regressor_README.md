# Regressor Feature Analysis (ML-Poisson)

## Context

The ML-Poisson model predicts match outcomes in two stages:

1. **XGBoost goal regressor** predicts expected goals scored by the attacking team (`lambda`)
2. **Dixon-Coles score matrix** converts `lambda_home` and `lambda_away` into outcome probabilities

The regressor is trained from the **attacker's perspective** — a single model is trained on all matches pooling both home and away observations (2× data), with features constructed symmetrically:

- When the home team attacks: positive rating differences, home form features
- When the away team attacks: negated rating differences, away form features

Predicting `lambda_home` uses the home team as attacker; predicting `lambda_away` re-uses the same model with teams swapped. This EDA informs which features to include in the regressor.

## Feature Abbreviations

| Abbreviation | Full name | Description |
| --- | --- | --- |
| `gw` | goals weighted | Goals scored, weighted **up** by opponent FIFA points — scoring against stronger opponents counts more |
| `gsw` | goals suffered weighted | Goals conceded, weighted **down** by opponent FIFA points — conceding against stronger opponents is penalised less |
| `gd` | goal difference | Average goal difference (scored − conceded) |
| `pww` | points won weighted | Points won, weighted by opponent FIFA points |
| `ma5` / `ma20` | moving average | Average over last 5 or 20 games before the match |

## Script

[`regressor_feature_analysis.py`](regressor_feature_analysis.py)

Runs Spearman correlation, Mutual Information (regression), and Pearson multicollinearity checks separately for `home_goals` and `away_goals` targets.

## Target Distributions

| Target | Mean | Std | 0 goals | 1 goal | 2 goals | 3+ goals |
| --- | --- | --- | --- | --- | --- | --- |
| `home_goals` | 1.58 | 1.63 | 27.3% | 30.9% | 20.7% | 21.1% |
| `away_goals` | 1.07 | 1.30 | 40.9% | 32.4% | 15.7% | 11.0% |

Home teams score more and have fewer blanks — home advantage is real in the training data (which includes non-neutral matches). The regressor learns this from the data; for neutral venue prediction, symmetrized inference removes the bias at prediction time.

## Key Findings

### Predictive strength

- Rating features (`elo_diff`, `pi_neutral_diff`) are the strongest predictors for both targets (Spearman r ≈ 0.40–0.50, MI ≈ 0.09–0.16), well ahead of all form features.
- Weighted attacking features (`gw_ma20`, `gw_ma5`) outperform raw goals MAs — opponent adjustment adds information.
- Opponent defensive features (`gsw_ma20`, `gsw_ma5`) are consistently informative — how many goals the defending team typically concedes matters as much as how many the attacking team scores.
- `neutral` and `tournament_weight` have near-zero MI for both targets — excluded.

### Collinearity

Same pattern as the classifier EDA:

- Raw and weighted versions of the same metric are nearly identical (r ≈ 0.95–0.97) — keep only weighted versions.
- `elo_diff` ↔ `points_dif` (r = 0.923), `elo_diff` ↔ `pi_neutral_diff` (r ≈ 0.90) — both have partial r ≈ −0.05 after conditioning on ELO. Keep only `elo_diff`; drop `points_dif` and `pi_neutral_diff`.
- `ma5`, `ma10`, `ma20` windows are highly collinear (r > 0.86) — keeping two windows (ma5 + ma20) is sufficient.

## Selected Feature Set

All features are expressed from the **attacker's perspective** — a single regressor trained on both home and away observations.

| Group | Features | Notes |
| --- | --- | --- |
| Ratings | `elo_diff`, `abs_elo_diff` | `elo_diff` negated for away attacker; `abs_elo_diff` unchanged (magnitude is symmetric) |
| WC experience | `wc_games_diff` | Negated for away attacker |
| Attacker form | `att_gw_ma20`, `att_gw_ma5`, `att_pww_ma20` | Attacker's weighted goals scored and points won |
| Defender form | `def_gsw_ma20`, `def_gsw_ma5`, `def_gd_ma20` | Defender's weighted goals conceded and goal difference |

`points_dif` and `pi_neutral_diff` excluded — collinear with `elo_diff` (partial r ≈ −0.05). For away attacker: rating diffs are negated; `att_*` features use away team stats; `def_*` features use home team stats.
