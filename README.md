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

- `home_elo` / `away_elo` / `elo_diff` / `abs_elo_diff` — ELO ratings computed from full match history (eloratings.net formula)
- `confederation_home` / `confederation_away` — football confederation per team (CatBoost only)
- `home_wc_best_round` / `away_wc_best_round` — best WC round ever reached (0=never qualified … 6=champion)
- `home_wc_goals_per_game` / `away_wc_goals_per_game` — goals scored per game in WC history (0 if never qualified)
- `home_wc_games` / `away_wc_games` — total WC matches played
- `neutral` / `tournament` — carried through from source data

#### Rolling (per team, last 20 and 5 games, leak-free)

- `points_weighted_ma` — points weighted by opponent FIFA points (normalized by 1400)
- `goals_weighted_ma` / `goals_suffered_weighted_ma` — goals weighted by opponent strength
- `goal_diff_ma` — average goal difference

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
| Ordered Logit | **57.8%** | 0.1995 | 0.9714 |
| XGBoost | 57.2% | 0.1977 | 0.9959 |
| Ensemble (XGB + CB + MLP) | 56.9% | 0.1918 | 0.9564 |
| **CatBoost** | 56.2% | **0.1914** | **0.9520** |
| ML-Poisson | 55.9% | 0.1925 | 0.9574 |

CatBoost leads on both RPS and Log-Loss (best probability calibration); Ordered Logit leads on accuracy.

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
│   ├── dataset.py           # Builds post-WC22 training set + WC26 group stage rows (shared)
│   ├── catboost/            # CatBoost simulation (run.py → group_tables.py → playoff.py)
│   │   ├── run.py           # Entry point: python -m simulation.catboost.run
│   │   ├── predict.py       # Group stage predictions
│   │   ├── group_tables.py  # Group tabs + best 3rd-place ranking
│   │   ├── playoff.py       # Knockout bracket simulation
│   │   └── explain.py       # SHAP explanation for a single match
│   ├── ml_poisson/          # ML-Poisson simulation (same structure, uses MLPoissonModel)
│   │   ├── run.py           # Entry point: python -m simulation.ml_poisson.run
│   │   ├── predict.py       # Group stage predictions
│   │   ├── group_tables.py  # Group tabs + best 3rd-place ranking
│   │   └── playoff.py       # Knockout bracket simulation
│   └── output/              # wc_26_catboost.xlsx / wc_26_ml_poisson.xlsx
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
