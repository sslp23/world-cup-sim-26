# Models

## Model Descriptions

### XGBoost

[`xgboost/`](xgboost/)

A gradient-boosted classifier predicting 3-class outcome probabilities (home_win / draw / away_win). Uses the full engineered feature set from the data pipeline.

See [`xgboost/README.md`](xgboost/README.md) for design details.

**Key parameters:**

- `draw_weight = 0.6` — draw class weight multiplier (tuned on WC 2022)
- `n_estimators = 500`, `learning_rate = 0.05`, `max_depth = 4`

**Features:** ELO-based ratings (`elo_diff`, `abs_elo_diff`), market value (`mv_sum_diff`, `mv_log_ratio`), WC history (`wc_games_diff`, `wc_best_round_diff`, `wc_gpg_diff`), and weighted form features (points won, goals scored/suffered, goal difference over 20 and 5 games). `points_dif` and `pi_diff` excluded after collinearity analysis.

**Training data:** All ranked matches (2018–2022) with ELO features available.

**Strengths:** Incorporates all rating systems (ELO, FIFA points) and form features simultaneously. Learns non-linear interactions between features.

**Limitations:** Black-box — harder to interpret individual predictions. Does not model score distributions, only outcome probabilities.

---

### Ordered Logit

[`ordered_logit/`](ordered_logit/)

A statistical model that treats outcomes (away_win < draw < home_win) as ordered thresholds on a latent "strength gap" variable. The strength gap is a weighted sum of four rating/market-value features; two learned thresholds separate the three outcome zones.

**Key parameters:**

- 4 feature coefficients (`elo_diff`, `points_dif`, `mv_sum_diff`, `mv_log_ratio`) + 2 cutpoints = 6 parameters total
- Fitted via BFGS maximum likelihood (statsmodels `OrderedModel`)

**Training data:** All ranked matches (2018–2022) with complete rating features. Rows with any NaN feature are dropped.

**Strengths:** Fully interpretable — coefficients show exactly how much each rating contributes. Strong baseline that proves rating features alone carry most of the predictive signal.

**Limitations:** Linear in features — cannot capture interactions or non-linearities. No form or context features.

---

### ML-Poisson

[`ml_poisson/`](ml_poisson/)

A hybrid model combining an XGBoost goal regressor with a Dixon-Coles score matrix. Stage 1 predicts expected goals per team (`lambda_home`, `lambda_away`) using an XGBoost regressor trained from the attacker's perspective. Stage 2 converts the lambdas into a full score probability matrix using the Dixon-Coles correction, from which outcome probabilities are derived.

**Key design choices:**

- **Single attacker-perspective regressor**: one XGBoost model is trained on both home and away observations simultaneously (attacker features: their recent offensive output; defender features: opponent's recent defensive record). Predicting `lambda_home` uses the home team as attacker; `lambda_away` negates the rating signs and swaps the form features.
- **Symmetrized inference**: each match is predicted forward (home as home) and inverted (teams swapped), then the two score matrices are averaged. Removes residual home/away label bias at neutral WC venues.
- **Match-conditional dynamic rho**: `rho_eff = rho_max × (1 − |λ_home − λ_away| / (λ_home + λ_away))`. Evenly matched games receive the full draw correction; lopsided games receive little/none. `rho_max = −0.40` (tuned on WC 2022).

**Key parameters:**

- `rho_max = −0.40` — maximum Dixon-Coles low-score correction (tuned on WC 2022)
- `n_estimators = 500`, `learning_rate = 0.05`, `max_depth = 4`, objective: `count:poisson`

**Training data:** All matches before 2022-11-20 with complete rating and form features.

**Strengths:** Explicit score distribution model — produces outcome probabilities through a principled generative process rather than direct classification. Better-calibrated probabilities than XGBoost (lower Log-Loss). The attacker-perspective single regressor guarantees symmetric lambda predictions.

**Limitations:** Two-stage model — errors from the goal regressor propagate into the score matrix. The XGBoost regressor is still a black box in stage 1. The Dixon-Coles correction only adjusts low-score cells (0-0, 1-0, 0-1, 1-1); high-scoring outcomes are standard independent Poisson.

---

### CatBoost

[`catboost/`](catboost/)

A gradient-boosted classifier with native categorical feature support. Same design as XGBoost (flip augmentation, symmetrized inference, draw weight tuning) with the addition of `confederation_home` and `confederation_away` as categorical features.

See [`catboost/README.md`](catboost/README.md) for design details.

**Key parameters:**

- `draw_weight = 0.8` — draw class weight multiplier (tuned on WC 2022)
- `iterations = 500`, `learning_rate = 0.05`, `depth = 6`

**Features:** ELO-based ratings (`elo_diff`, `abs_elo_diff`), market value (`mv_sum_diff`, `mv_log_ratio`), WC history (`wc_games_diff`, `wc_best_round_diff`, `wc_gpg_diff`), weighted form features (points won, goals scored/suffered, goal difference over 20 and 5 games), and `confederation_home` / `confederation_away` as native categorical features. `points_dif` and `pi_diff` excluded after collinearity analysis.

**Training data:** All ranked matches (2018–2022) with ELO features available.

**Strengths:** Best RPS in the cross-WC evaluation — best probability calibration. Handles categorical confederation features natively without one-hot encoding. `abs_elo_diff` allows the model to learn that large quality gaps suppress draws regardless of which team is stronger.

**Limitations:** Black-box. Does not model score distributions, only outcome probabilities.

---

### Ensemble (XGBoost + CatBoost + ML-Poisson)

[`ensemble/`](ensemble/)

Equal-weight average of the three best-performing ML models. No meta-learning — purely a test of whether diversity across modelling approaches improves over any single model.

**Strengths:** More robust than any individual model — the score-distribution signal from ML-Poisson complements the classification signal from XGBoost and CatBoost. Matches CatBoost on accuracy while being less sensitive to training set composition.

**Limitations:** Slower to run (trains three models). Averaging softens probability mass, which can hurt accuracy on decisive matches while helping RPS/Log-Loss calibration.
