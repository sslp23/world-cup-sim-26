# World Cup 2026 Forecast

A quantitative pipeline to forecast the results of the 2026 FIFA World Cup using machine learning and statistical models.

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

- **5-tier K-factor**: 60 (WC finals) → 50 (continental finals) → 40 (qualifiers) → 30 (other) → 20 (friendlies)
- **Goal difference multiplier**: ×1.5 for 2-goal wins, ×1.75 for 3, ×(1.75 + (N−3)/8) for N≥4
- **Home advantage**: +100 to home team's effective rating (skipped on neutral venues)

## Repository Structure

```text
wc_26_ml/
├── pipeline.py              # Master orchestrator — runs all sub-pipelines
├── features.md              # Feature documentation and model feature selection
├── data_pipeline/           # Builds the source features dataset
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
├── models/                  # One subfolder per model
│   ├── README.md            # Model comparison and WC 2022 backtest results
│   ├── dixon_coles/         # Dixon-Coles Poisson model
│   ├── xgboost/             # XGBoost 3-class classifier
│   ├── catboost/            # CatBoost 3-class classifier (best overall)
│   ├── ordered_logit/       # Ordered logistic regression (3-feature baseline)
│   └── ml_poisson/          # ML-Poisson hybrid (XGBoost regressor + Dixon-Coles matrix)
└── data/
    ├── international_results.csv
    ├── resulting_data.csv
    ├── ranked_database.csv
    ├── elo_ratings.csv
    ├── pi_ratings.csv
    └── ranked_database_with_features.csv
```
