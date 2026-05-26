# Exploratory Data Analysis

## Script

[`feature_analysis.py`](feature_analysis.py) — evaluates feature predictiveness and redundancy using the training set (all matches before WC 2022).

All continuous features are expressed as **difference features** (home team value − away team value), matching the format used during model inference where the home/away assignment at a neutral venue is arbitrary.

## Methodology

| Section | Method | Purpose |
| --- | --- | --- |
| Spearman correlation | Monotonic rank correlation with result (0=home win, 1=draw, 2=away win) | Measures linear and monotonic predictive strength |
| Mutual information | Model-free MI between feature and result | Captures non-linear relationships; complementary to Spearman |
| Pearson correlation matrix | Pairwise correlation between features | Detects redundant/collinear features |
| Chi-squared + Cramér's V | Association between categorical confederation and result | Appropriate test for categorical features |

## Key Findings

### Predictive strength

- **Rating features** (`elo_diff`) are the strongest individual predictors (Spearman r ≈ −0.51 to −0.56, MI ≈ 0.16–0.20).

### FIFA points vs ELO collinearity (see `collinearity_points_elo.py`)

`points_dif` and `elo_diff` have Pearson r = 0.92 (R² = 0.85 shared variance). A focused analysis found that:

- `elo_diff` alone: log-loss = 0.863, accuracy = 60.4%
- `points_dif` alone: log-loss = 0.887, accuracy = 59.1%
- both together: log-loss = 0.863 (no improvement over ELO alone)
- **Partial r(points_dif | elo_diff) = −0.05** — near-zero independent signal once ELO is known
- **Partial r(elo_diff | points_dif) = −0.21** — ELO adds substantial signal beyond FIFA points

**Decision: `points_dif` and `abs_points_dif` removed from all models.** ELO subsumes FIFA points; keeping both inflates VIF without adding predictive value.

### ELO multi-horizon means collinearity (see `elo_features_analysis.py`)

`elo_ma_2yr_diff` and `elo_ma_4yr_diff` have Pearson r = 0.97–0.98 with `elo_diff` — near-identical to current ELO because ratings change slowly. Partial r ≈ −0.04 (near-zero). VIF = 120–175 (severe).

`elo_ma_8yr_diff` is the most distinct (r = 0.94, VIF = 39, partial r = −0.079) and has the best individual log-loss improvement (−0.003 over elo_diff baseline).

**Decision: drop all ELO rolling means.** All three (r = 0.94–0.98) are too collinear with current ELO to add reliable independent signal. `elo_ma_8yr_diff` was initially kept (partial r = −0.079) but subsequently removed as it didn't perform as expected in practice.

- **Long-window moving averages (20 games)** are consistently more predictive than short-window ones (3 games), since they capture a team's true level rather than short-term noise.
- **`form_trend`** (linear slope of recent
 points) and **`days_since_last_match`** show near-zero MI and weak Spearman correlation — excluded from the model feature set.
- **`tournament_weight`** has negligible predictive power on its own — excluded.

### Weighted vs raw features

Raw and opponent-weighted versions of the same metric are nearly identical (Pearson r ≈ 0.97). The weighted version subsumes the raw one while adding opponent-strength context, so only weighted features are retained:

- `goals_weighted_ma` preferred over `goals_ma`
- `goals_suffered_weighted_ma` preferred over `goals_suffered_ma`
- `points_weighted_ma` preferred over `points_won_ma`

### Pi ratings vs ELO collinearity (see `collinearity_pi_elo.py`)

`pi_diff` and `elo_diff` have Pearson r = 0.90 (same pattern as points_dif). A focused analysis found:

- **Partial r(pi_diff | elo_diff) = −0.05** — near-zero independent signal once ELO is known
- Pi-ratings additionally conflate home/away venue context (`pi_h` built from home results, `pi_a` from away results), which is misleading at neutral WC venues

**Decision: `pi_diff` / `pi_neutral_diff` not added back.** ELO subsumes the predictive signal without the venue bias problem.

### Confederation

Confederation is statistically associated with outcome (chi-squared p < 0.001) but the effect size is very small (Cramér's V ≈ 0.07–0.09). **CatBoost includes confederation as a native categorical feature** — it handles low-frequency categories (e.g. OFC) robustly via ordered target statistics. XGBoost and ML-Poisson do not use confederation.

## Selected Feature Set

See the [Model Feature Selection](../features.md#model-feature-selection) section in `features.md`.
