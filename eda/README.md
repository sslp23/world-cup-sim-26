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

- **Rating features** (`elo_diff`, `pi_diff`, `points_dif`) are the strongest individual predictors (Spearman r ≈ −0.51 to −0.56, MI ≈ 0.16–0.20). All three retain independent signal despite being mutually correlated (Pearson r ≈ 0.90–0.93).
- **Long-window moving averages (20 games)** are consistently more predictive than short-window ones (3 games), since they capture a team's true level rather than short-term noise.
- **`form_trend`** (linear slope of recent
 points) and **`days_since_last_match`** show near-zero MI and weak Spearman correlation — excluded from the model feature set.
- **`tournament_weight`** has negligible predictive power on its own — excluded.

### Weighted vs raw features

Raw and opponent-weighted versions of the same metric are nearly identical (Pearson r ≈ 0.97). The weighted version subsumes the raw one while adding opponent-strength context, so only weighted features are retained:

- `goals_weighted_ma` preferred over `goals_ma`
- `goals_suffered_weighted_ma` preferred over `goals_suffered_ma`
- `points_weighted_ma` preferred over `points_won_ma`

### Confederation

Confederation is statistically associated with outcome (chi-squared p < 0.001) but the effect size is very small (Cramér's V ≈ 0.07–0.09). The information it carries is already encoded in the rating features — a UEFA team playing a CONMEBOL team will have a characteristic ELO/FIFA points gap that captures the quality differential implicitly. Confederation features are excluded from the model feature set.

## Selected Feature Set

See the [Model Feature Selection](../features.md#model-feature-selection) section in `features.md`.
