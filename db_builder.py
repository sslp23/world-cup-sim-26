import pandas as pd
import re
from datetime import datetime

df =  pd.read_csv("data/international_results.csv")

df["date"] = pd.to_datetime(df["date"])

df = df[(df["date"] >= "2023-1-1")].reset_index(drop=True)

rank = pd.read_csv("data/resulting_data.csv")

rank = rank[['rank', 'nation_full_name', 'points', 'rank_date']]

rank["rank_date"] = pd.to_datetime(rank["rank_date"])
rank = rank[(rank["rank_date"] >= "2023-1-1")].reset_index(drop=True)
rank["nation_full_name"] = rank["nation_full_name"].str.replace("Czechia", "Czech Republic").str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")

rank = rank.set_index(['rank_date']).groupby(['nation_full_name'], group_keys=False).resample('D').first().ffill().reset_index()

df_ranked = df.merge(rank, left_on=["date", "home_team"], right_on=["rank_date", "nation_full_name"]).drop(["rank_date", "nation_full_name"], axis=1)

df_ranked = df_ranked.merge(rank, left_on=["date", "away_team"], right_on=["rank_date", "nation_full_name"], suffixes=("_home", "_away")).drop(["rank_date", "nation_full_name"], axis=1)

df_ranked.to_csv('data/ranked_database.csv', index=False)