# World Cup 2026 Forecast

This repository contains a quantitative method to forecast the results of the 2026 FIFA World Cup.

## Data Sources

The model is built upon two main data sources:

- **FIFA Rankings**: Scraped from the web using the [fifa-ranking-scraper](https://github.com/sslp23/fifa-ranking-scraper) which generates the `resulting_data.csv` file. This file should be placed in the `data` folder.
- **International Match Results**: A comprehensive dataset of international football results from 1872 to the present. The `get_data.py` script automatically downloads this data from [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) and saves it as `international_results.csv` in the `data` folder.

## Usage

1.  **Get the data**:
    - Run `python get_data.py` to download the international match results.
    - Use the [fifa-ranking-scraper](httpss://github.com/sslp23/fifa-ranking-scraper) to generate `resulting_data.csv` and place it in the `data` folder.

2.  **Build the database**:
    - Run `python db_builder.py` to process the raw data and create the final database for analysis.
3.  **Create features**:
    - Run python features_creator.py to generate advanced features from the ranked database.

## Features Creator

The features_creator.py script generates a comprehensive set of features for machine learning models based on team performance metrics.

### How It Works

The script reads the ranked_database.csv file and creates features by analyzing each team's historical performance **before each match**. For every match in the dataset:

1. **Prior Games Analysis**: For both the home and away team, it retrieves all games played **before** the current match date
2. **Feature Calculation**: It calculates performance metrics from these prior games:
   - **Points Features**: Points won (3/1/0 for win/draw/loss) and their moving averages (5-game and 3-game windows)
   - **Weighted Points**: Points adjusted by opponent ranking strength
   - **Goal Metrics**: Average goals scored and conceded, with both raw and opponent-rank-weighted versions
3. **Column Creation**: Features are created with home_ and away_ prefixes, so each match row has uniform column names regardless of which teams are playing
4. **Efficiency**: Uses vectorized operations to process all rows efficiently

### Output Features

The script generates features for both home and away teams:
- points_won_ma_5, points_won_ma_3: 5-game and 3-game moving averages of points won
- points_weighted_ma_5, points_weighted_ma_3: Weighted points moving averages
- goals_ma_5, goals_ma_3: Goal scoring averages
- goals_suffered_ma_5, goals_suffered_ma_3: Goal conceding averages
- goals_weighted_ma_5, goals_weighted_ma_3: Goal scoring weighted by opponent rank
- goals_suffered_weighted_ma_5, goals_suffered_weighted_ma_3: Goal conceding weighted by opponent rank
- rank_dif: Difference between home team rank and away team rank
