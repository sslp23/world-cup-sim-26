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
| CatBoost | draw_weight=0.72 | 54.7% (35/64) | **1.042** | **0.214** |
| ML-Poisson | rho=−0.40 | 54.7% (35/64) | 1.074 | 0.220 |
| Ensemble (XGB+CB+MLP) | — | 54.7% (35/64) | 1.056 | 0.216 |
| XGBoost | draw_weight=0.65 | 53.1% (34/64) | 1.082 | 0.221 |
| Dixon-Coles | xi=0.0005 | 50.0% (32/64) | 1.047 | 0.214 |

Training set: all ranked matches before 2022-11-20 in `data/past_wc/wc2022/`.

**Takeaways:**

- WC 2022 is the parameter-tuning set, not an out-of-sample test. Results here reflect fitted hyperparameters.
- Ordered Logit leads on accuracy (56.2%) with just three rating features — a strong reminder that ELO/points are the dominant signal.
- CatBoost achieves the best Log-Loss (1.042) and RPS (0.214), showing better probability calibration than the other classifiers even on the tuning WC.
- ML-Poisson's explicit score model produces well-calibrated probabilities but its accuracy is dragged down by the WC 2022 upsets (Saudi Arabia, Japan, Morocco).
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

**Training data:** All ranked matches (2018–2022) with ELO and pi-ratings available.

**Strengths:** Incorporates all rating systems (ELO, pi-ratings, FIFA points) and form features simultaneously. Learns non-linear interactions between features.

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
- **`pi_neutral_diff`**: instead of the raw `pi_diff = pi_h_home − pi_a_away` (which bakes in home/away venue context), uses `avg(pi_h, pi_a) of attacker − avg(pi_h, pi_a) of defender`. This is neutral-venue aware — critical for WC prediction where the host's inflated `pi_h` would otherwise dominate.
- **Match-conditional dynamic rho**: `rho_eff = rho_max × (1 − |λ_home − λ_away| / (λ_home + λ_away))`. Evenly matched games receive the full draw correction; lopsided games receive little/none. `rho_max = −0.30` (tuned on WC 2022).

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

**Training data:** All ranked matches (2018–2022) with ELO and pi-ratings available.

**Strengths:** Best overall performer on accuracy. Handles categorical confederation features natively without one-hot encoding, which avoids information loss and reduces the risk of overfitting sparse dummy variables.

**Limitations:** Black-box. Does not model score distributions, only outcome probabilities.

---

### Ensemble (XGBoost + CatBoost + ML-Poisson)

[`ensemble/`](ensemble/)

Equal-weight average of the three best-performing ML models. No meta-learning — purely a test of whether diversity across modelling approaches improves over any single model.

**Strengths:** More robust than any individual model — the score-distribution signal from ML-Poisson complements the classification signal from XGBoost and CatBoost. Matches CatBoost on accuracy while being less sensitive to training set composition.

**Limitations:** Slower to run (trains three models). Averaging softens probability mass, which can hurt accuracy on decisive matches while helping RPS/Log-Loss calibration.
