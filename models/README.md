# Models

All models are backtested against the 64 matches of the FIFA World Cup 2022 (2022-11-20 → 2022-12-18).

## Metrics

| Metric | Description | Better |
| --- | --- | --- |
| **Accuracy** | % of matches where `argmax(probs) == actual outcome` | Higher |
| **Log-Loss** | `−mean(log p_correct)` — penalises confident wrong predictions | Lower |
| **RPS** | Ranked Probability Score (ordered home_win / draw / away_win) — penalises probability mass placed far from the true outcome | Lower |

Baseline (uniform 1/3 probabilities): Log-Loss = 1.099, RPS = 0.239.

---

## WC 2022 Backtest Results

| Model | Params | Accuracy | Log-Loss | RPS |
| --- | --- | --- | --- | --- |
| **Ordered Logit** | — | **56.2% (36/64)** | 1.044 | 0.218 |
| ML-Poisson | rho=−0.40 | 56.2% (36/64) | 1.072 | 0.221 |
| Ensemble (XGB+CB+MLP) | — | 51.6% (33/64) | 1.073 | 0.219 |
| XGBoost | draw_weight=0.65 | 50.0% (32/64) | 1.118 | 0.226 |
| Dixon-Coles† | xi=0.0005 | 50.0% (32/64) | **1.047** | **0.214** |
| CatBoost | draw_weight=0.72 | 48.4% (31/64) | 1.061 | 0.218 |

† Dixon-Coles is excluded from the cross-WC evaluation (`past_wc_backtest.py`) due to runtime cost. The WC 2022 result above is from its individual backtest.

Training set: all ranked matches before 2022-11-20 in `data/past_wc/wc2022/`.

**Takeaways:**

- WC 2022 is the parameter-tuning set, not an out-of-sample test. Results here reflect fitted hyperparameters.
- Ordered Logit and ML-Poisson share the best accuracy (56.2%) — rating features alone carry most of the predictive signal.
- Dixon-Coles achieves the best Log-Loss (1.047) and RPS (0.214) on WC 2022, driven by its well-calibrated score distribution model.
- CatBoost's WC 2022 accuracy (48.4%) is below its cross-WC average (56.6%) — WC 2022 upsets (Saudi Arabia, Japan, Morocco) disproportionately hurt the ML classifiers.
- Group-stage upsets are unpredictable from any model — all models fail on the same extreme mismatches.

---

## Model Descriptions

### Dixon-Coles

[`dixon_coles/`](dixon_coles/)

A Poisson-based statistical model. Estimates attack and defense parameters per team via Maximum Likelihood Estimation (MLE) with time-decay weighting — older matches contribute less. Corrects the known Poisson underestimation of 0-0 and 1-0 / 0-1 scorelines (Dixon & Coles, 1997).

**Key parameters:**

- `xi = 0.0005` — time decay rate (tuned on WC 2022; very slow decay gives essentially uniform weight across the training window)
- Low-score correction `ρ` fitted jointly with attack/defense parameters

**Training data:** Full international match history (1872–present). Time decay handles the recency emphasis.

**Strengths:** Explicit probability model — produces well-calibrated score distributions that can be used for simulation. Interpretable team parameters (attack/defense strength).

**Limitations:** Cannot incorporate auxiliary features (ELO, pi-ratings, form MAs). Assumes independence between home and away goal-scoring processes (partially relaxed by ρ correction).

---

### XGBoost

[`xgboost/`](xgboost/)

A gradient-boosted classifier predicting 3-class outcome probabilities (home_win / draw / away_win). Uses the full engineered feature set from the data pipeline.

See [`xgboost/README.md`](xgboost/README.md) for design details.

**Key parameters:**

- `draw_weight = 0.65` — draw class weight multiplier (tuned on WC 2022)
- `n_estimators = 500`, `learning_rate = 0.05`, `max_depth = 4`

**Features:** ELO-based ratings (`elo_diff`, `abs_elo_diff`), WC experience (`wc_games_diff`), and weighted form features (points won, goals scored/suffered, goal difference over 20 and 5 games). `points_dif`, `pi_diff`, `elo_delta_20_diff`, and ELO MA variants excluded after collinearity/predictive-power analysis.

**Training data:** All ranked matches (2018–2022) with ELO features available.

**Strengths:** Incorporates all rating systems (ELO, FIFA points) and form features simultaneously. Learns non-linear interactions between features.

**Limitations:** Black-box — harder to interpret individual predictions. Does not model score distributions, only outcome probabilities.

---

### Ordered Logit

[`ordered_logit/`](ordered_logit/)

A statistical model that treats outcomes (away_win < draw < home_win) as ordered thresholds on a latent "strength gap" variable. The strength gap is a weighted sum of the three rating features; two learned thresholds separate the three outcome zones.

**Key parameters:**

- 3 feature coefficients + 2 cutpoints = 5 parameters total
- Fitted via BFGS maximum likelihood (statsmodels `OrderedModel`)

**Training data:** All ranked matches (2018–2022) with complete rating features. No NaN rows dropped.

**Strengths:** Fully interpretable — coefficients show exactly how much each rating contributes. Best Log-Loss of all models. Strong baseline that proves rating features alone carry most of the predictive signal.

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

- `draw_weight = 0.72` — draw class weight multiplier (tuned on WC 2022)
- `iterations = 500`, `learning_rate = 0.05`, `depth = 6`

**Features:** ELO-based ratings (`elo_diff`, `abs_elo_diff`), WC experience (`wc_games_diff`), weighted form features (points won, goals scored/suffered, goal difference over 20 and 5 games), and `confederation_home` / `confederation_away` as native categorical features. `points_dif`, `pi_diff`, `elo_delta_20_diff`, and ELO MA variants excluded after collinearity/predictive-power analysis.

**Training data:** All ranked matches (2018–2022) with ELO features available.

**Strengths:** Best RPS in the cross-WC evaluation — best probability calibration. Handles categorical confederation features natively without one-hot encoding. `abs_elo_diff` allows the model to learn that large quality gaps suppress draws regardless of which team is stronger.

**Limitations:** Black-box. Does not model score distributions, only outcome probabilities.

---

### Ensemble (XGBoost + CatBoost + ML-Poisson)

[`ensemble/`](ensemble/)

Equal-weight average of the three best-performing ML models. No meta-learning — purely a test of whether diversity across modelling approaches improves over any single model.

**Strengths:** More robust than any individual model — the score-distribution signal from ML-Poisson complements the classification signal from XGBoost and CatBoost. Matches CatBoost on accuracy while being less sensitive to training set composition.

**Limitations:** Slower to run (trains three models). Averaging softens probability mass, which can hurt accuracy on decisive matches while helping RPS/Log-Loss calibration.
