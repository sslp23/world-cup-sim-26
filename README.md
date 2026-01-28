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
