# World Cup 2026 Forecast

A quantitative pipeline to forecast the results of the 2026 FIFA World Cup using machine learning and statistical models.

## Methodology

The project follows a three-stage workflow:

```text
1. Data & Features      →    2. Model + Tune (WC 2022)    →    3. Evaluate (WC 2006–2022)
   data_pipeline/                models/  +  backtest            backtest/past_wc_backtest.py
```

**Stage 1 — Data & Features:** Build the full feature dataset from raw match results and FIFA rankings. ELO and pi-ratings are computed from the full match history (1872–present); form features use only the post-WC2018 window.

**Stage 2 — Model development & tuning:** Each model is developed and backtested against WC 2022 (64 matches). WC 2022 is the tuning set — hyperparameters (draw weight, rho, decay rate) are selected here. See [`models/README.md`](models/README.md) for the per-model results.

**Stage 3 — Cross-WC evaluation:** All tuned models are evaluated on each past World Cup (2006, 2010, 2014, 2018, 2022) with no further parameter changes. This measures true out-of-sample generalisability across different football eras. See [`backtest/`](backtest/) for the scripts and [`backtest/output/`](backtest/output/) for the per-match Excel results.

---

## Data Sources

- **FIFA Rankings**: Scraped from Transfermarkt via the [fifa-ranking-scraper](https://github.com/sslp23/fifa-ranking-scraper), which generates `data/resulting_data.csv`.
- **International Match Results**: Full history of international football results from 1872 to present, downloaded from [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) via `get_data.py`.

## Pipeline

Run the full pipeline with a single command:

```bash
python pipeline.py
```

This executes the following steps in order:

| Step | Script | Input | Output |
| --- | --- | --- | --- |
| 1 | `data_pipeline/get_data.py` | Kaggle API | `data/international_results.csv` |
| 2 | `data_pipeline/db_builder.py` | `international_results.csv` + `resulting_data.csv` | `data/ranked_database.csv` |
| 3 | `data_pipeline/elo_calculator.py` | `international_results.csv` | `data/elo_ratings.csv` |
| 4 | `data_pipeline/pi_ratings_calculator.py` | `international_results.csv` | `data/pi_ratings.csv` |
| 5 | `data_pipeline/features_creator.py` | `ranked_database.csv` + `elo_ratings.csv` + `pi_ratings.csv` + `resulting_data.csv` | `data/ranked_database_with_features.csv` |

## Features & EDA

See [features.md](features.md) for full documentation of all engineered features and the selected model feature set.

See [eda/README.md](eda/README.md) for the exploratory data analysis — feature predictiveness, multicollinearity findings, and the rationale behind feature selection.

### Summary

#### Static (per match)

- `tournament_weight` — competition importance weight (1.0 WC, 0.8 qualifiers, 0.9 other competitive, 0.5 friendlies)
- `points_dif` — FIFA points difference between home and away team (positive = home stronger)
- `home_elo` / `away_elo` / `elo_diff` — ELO ratings computed from full match history (eloratings.net formula)
- `confederation_home` / `confederation_away` — football confederation per team
- `neutral` / `tournament` — carried through from source data

#### Rolling (per team, last 20, 10, 5 and 3 games, leak-free)

- `points_won_ma` — average points earned
- `points_weighted_ma` — points weighted by opponent FIFA points (normalized by 1400)
- `goals_ma` / `goals_suffered_ma` — goals scored and conceded
- `goals_weighted_ma` / `goals_suffered_weighted_ma` — goals weighted by opponent strength
- `goal_diff_ma` — average goal difference
- `form_trend` — linear slope of points_won (positive = improving, negative = declining)
- `days_since_last_match` — rest/fatigue indicator

## ELO Rating System

Implements the [eloratings.net](https://www.eloratings.net) formula:

- **5-tier K-factor**:
  - K=60: FIFA World Cup finals
  - K=50: Continental championships (Euro, Copa América, AFCON, Asian Cup, Gold Cup, Olympics, …)
  - K=40: World Cup qualifiers + UEFA/CONCACAF Nations League (competitive continental leagues)
  - K=30: All other tournaments
  - K=20: Friendly matches
- **Goal difference multiplier**: ×1.5 for 2-goal wins, ×1.75 for 3, ×(1.75 + (N−3)/8) for N≥4
- **Home advantage**: +100 to home team's effective rating (skipped on neutral venues)

## Cross-WC Evaluation Summary

Averaged across WC 2006, 2010, 2014, 2018, 2022 (64 matches each):

| Model | Avg Accuracy | Avg RPS | Avg Log-Loss |
| --- | --- | --- | --- |
| **CatBoost** | 55.6% | **0.1960** | 0.9676 |
| Dixon-Coles | 55.6% | 0.1963 | **0.9641** |
| Ensemble (XGB + CB + MLP) | **58.4%** | 0.1965 | 0.9725 |
| ML-Poisson | 56.3% | 0.1974 | 0.9813 |
| Ordered Logit | 57.8% | 0.1995 | 0.9714 |
| XGBoost | 55.3% | 0.2023 | 1.0019 |

CatBoost leads on RPS (best probability calibration); Dixon-Coles leads on Log-Loss; Ensemble leads on accuracy. Dixon-Coles is not re-run in the automated batch (too slow) but its output from a previous run is included above.

See [`backtest/output/`](backtest/output/) for per-match Excel results per model and WC edition.

## Repository Structure

```text
wc_26_ml/
├── pipeline.py              # Master orchestrator — runs all sub-pipelines
├── features.md              # Feature documentation and model feature selection
├── data_pipeline/           # Stage 1: builds the features dataset
│   ├── pipeline.py          # Data sub-pipeline (called by root pipeline.py)
│   ├── get_data.py          # Downloads match history from Kaggle
│   ├── db_builder.py        # Merges matches with FIFA rankings
│   ├── elo_calculator.py    # Computes ELO ratings
│   ├── pi_ratings_calculator.py  # Computes pi-ratings (Constantinou & Fenton)
│   ├── features_creator.py  # Engineers all model features
│   └── validate_features.py # Sanity-checks computed features against raw data
├── eda/                     # Exploratory data analysis
│   ├── README.md            # EDA findings and feature selection rationale
│   └── feature_analysis.py  # Feature predictiveness and multicollinearity analysis
├── models/                  # Stage 2: model development and WC 2022 tuning
│   ├── README.md            # Model comparison and WC 2022 backtest results
│   ├── dixon_coles/         # Dixon-Coles Poisson model
│   ├── xgboost/             # XGBoost 3-class classifier
│   ├── catboost/            # CatBoost 3-class classifier
│   ├── ordered_logit/       # Ordered logistic regression (3-feature baseline)
│   ├── ml_poisson/          # ML-Poisson hybrid (XGBoost regressor + Dixon-Coles matrix)
│   └── ensemble/            # Equal-weight ensemble of XGBoost + CatBoost + ML-Poisson
├── backtest/                # Stage 3: cross-WC evaluation (WC 2006–2022)
│   ├── past_wc_backtest.py  # Runs all models on all past WC editions
│   ├── avg_performance.py   # Reads output/*.xlsx → avg_performance.xlsx summary
│   └── output/              # Per-match Excel results (one file per model)
├── simulation/
│   ├── dataset.py           # Builds post-WC22 training set + WC26 group stage rows
│   ├── predict.py           # Trains CatBoost on post-WC22 data, predicts WC26
│   ├── explain.py           # SHAP explanation for a single WC26 match (CatBoost)
│   └── output/              # wc_26_sim.xlsx — full group stage predictions
├── experiments/             # Ad-hoc tests (training window size, dataset builder)
└── data/
    ├── international_results.csv
    ├── resulting_data.csv
    ├── ranked_database.csv
    ├── elo_ratings.csv
    ├── pi_ratings.csv
    ├── ranked_database_with_features.csv
    └── past_wc/             # Per-edition datasets for cross-WC backtest
        ├── wc2006/
        ├── wc2010/
        ├── wc2014/
        ├── wc2018/
        └── wc2022/
```
