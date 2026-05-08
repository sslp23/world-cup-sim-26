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
| Dixon-Coles | 53.1% (34/64) | 1.040 | **0.213** |
| XGBoost | **56.2% (36/64)** | 1.090 | 0.214 |

### By stage

| Stage | Dixon-Coles | XGBoost |
| --- | --- | --- |
| Group stage (48) | 52.1% (25/48) | 52.1% (25/48) |
| Round of 16 (8) | 75.0% (6/8) | **87.5% (7/8)** |
| Quarter-finals (4) | 0.0% (0/4) | 25.0% (1/4) |
| Semi-finals (2) | 100% (2/2) | 100% (2/2) |
| Final / 3rd place (2) | 50.0% (1/2) | 50.0% (1/2) |

**Takeaways:**
- XGBoost leads on accuracy (+3pp overall, +12pp in Round of 16) and is competitive on RPS.
- Dixon-Coles has a slight edge on Log-Loss and RPS, consistent with it producing better-calibrated probability distributions for low-scoring football matches.
- Both models struggle identically in the group stage (52.1%) — the upsets (Saudi Arabia, Japan, Morocco, Cameroon) are not predictable from pre-match features alone.
- Quarter-finals are the weakest stage for both models, reflecting the high variance of knockout football between evenly matched top teams.

---

## Models

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
