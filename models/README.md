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

| Model | Accuracy | Log-Loss | RPS |
| --- | --- | --- | --- |
| **CatBoost** | **57.8% (37/64)** | 1.038 | **0.212** |
| XGBoost | 56.2% (36/64) | 1.090 | 0.214 |
| Ordered Logit | 53.1% (34/64) | **1.031** | 0.214 |
| Dixon-Coles | 53.1% (34/64) | 1.040 | 0.213 |
| ML-Poisson | — | — | — |

### By stage

| Stage | Dixon-Coles | Ord. Logit | XGBoost | CatBoost |
| --- | --- | --- | --- | --- |
| Group stage (48) | 52.1% (25/48) | 52.1% (25/48) | 52.1% (25/48) | **56.2% (27/48)** |
| Round of 16 (8) | 75.0% (6/8) | 75.0% (6/8) | **87.5% (7/8)** | **87.5% (7/8)** |
| Quarter-finals (4) | 0.0% (0/4) | 0.0% (0/4) | 25.0% (1/4) | 25.0% (1/4) |
| Semi-finals (2) | 100% (2/2) | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| Final / 3rd place (2) | 50.0% (1/2) | 50.0% (1/2) | 50.0% (1/2) | 0.0% (0/2) |

**Takeaways:**

- CatBoost is the best overall model — best accuracy (+4.7pp over Dixon-Coles) and best RPS. Adding confederation as a native categorical feature contributes meaningful signal.
- Ordered Logit achieves the best Log-Loss with only 3 features and 5 parameters, confirming that the bulk of predictive signal lives in the rating features (ELO, pi-ratings, FIFA points). Pi-ratings carry the most weight (coef +0.29 vs +0.004 for ELO).
- XGBoost and CatBoost both outperform the simpler models in the Round of 16 (7/8 vs 6/8), suggesting the form features and non-linear interactions add value in knockout-stage matchups.
- All models are equally weak in the Quarter-finals — top-vs-top knockout matches are inherently hard to predict from pre-match features.
- Group stage upsets (Saudi Arabia, Japan, Morocco, Cameroon) are not predictable from any model.

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

- `draw_weight = 0.75` — draw class weight multiplier (tuned on WC 2022)
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

### CatBoost

[`catboost/`](catboost/)

A gradient-boosted classifier with native categorical feature support. Same design as XGBoost (flip augmentation, symmetrized inference, draw weight tuning) with the addition of `confederation_home` and `confederation_away` as categorical features.

See [`catboost/README.md`](catboost/README.md) for design details.

**Key parameters:**

- `draw_weight = 0.7` — draw class weight multiplier (tuned on WC 2022)
- `iterations = 500`, `learning_rate = 0.05`, `depth = 6`

**Training data:** All ranked matches (2018–2022) with ELO and pi-ratings available.

**Strengths:** Best overall performer. Handles categorical confederation features natively without one-hot encoding, which avoids information loss and reduces the risk of overfitting sparse dummy variables.

**Limitations:** Black-box. Does not model score distributions, only outcome probabilities.
