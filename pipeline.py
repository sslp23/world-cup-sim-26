"""
Full pipeline — run this to produce the complete features dataset from scratch.

Steps:
  1. get_data        — download match history from Kaggle
  2. db_builder      — merge match history with FIFA rankings
  3. elo_calculator  — compute ELO ratings from full match history
  4. features_creator — build all model features
"""

import pandas as pd
import kagglehub
import os
import re
from elo_calculator import compute_elo
from features_creator import FeaturesCreator


# ── Step 1: Download data ──────────────────────────────────────────────────────
print("=== Step 1: Downloading data ===")
path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
print(os.listdir(path))

file_path = os.path.join(path, "results.csv")
df_results = pd.read_csv(file_path)
df_results.to_csv("data/international_results.csv", index=False)
print(f"Saved {len(df_results)} matches to data/international_results.csv")


# ── Step 2: Build ranked database ─────────────────────────────────────────────
print("\n=== Step 2: Building ranked database ===")
df = pd.read_csv("data/international_results.csv")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2018-8-1"].reset_index(drop=True)

rank = pd.read_csv("data/resulting_data.csv")
rank = rank[['rank', 'nation_full_name', 'points', 'rank_date']]
rank["rank_date"] = pd.to_datetime(rank["rank_date"])
rank = rank[rank["rank_date"] >= "2018-8-1"].reset_index(drop=True)
rank["nation_full_name"] = (
    rank["nation_full_name"]
    .str.replace("Czechia", "Czech Republic")
    .str.replace("IR Iran", "Iran")
    .str.replace("Korea Republic", "South Korea")
    .str.replace("USA", "United States")
)

rank = rank.set_index(['rank_date']).groupby(['nation_full_name'], group_keys=False).resample('D').first().ffill().reset_index()

df_ranked = df.merge(rank, left_on=["date", "home_team"], right_on=["rank_date", "nation_full_name"]).drop(["rank_date", "nation_full_name"], axis=1)
df_ranked = df_ranked.merge(rank, left_on=["date", "away_team"], right_on=["rank_date", "nation_full_name"], suffixes=("_home", "_away")).drop(["rank_date", "nation_full_name"], axis=1)
df_ranked.to_csv("data/ranked_database.csv", index=False)
print(f"Saved {len(df_ranked)} ranked matches to data/ranked_database.csv")


# ── Step 3: ELO ratings ────────────────────────────────────────────────────────
print("\n=== Step 3: Computing ELO ratings ===")
results_full = pd.read_csv("data/international_results.csv")
results_full['date'] = pd.to_datetime(results_full['date'])
results_full = results_full.sort_values('date').reset_index(drop=True)

elo_df, final_elo = compute_elo(results_full)
elo_df.to_csv("data/elo_ratings.csv", index=False)
print(f"ELO ratings saved: {len(elo_df)} matches processed")

top_teams = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 teams by final ELO:")
for team, elo_val in top_teams:
    print(f"  {team}: {elo_val:.0f}")


# ── Step 4: Feature engineering ───────────────────────────────────────────────
print("\n=== Step 4: Creating features ===")
creator = FeaturesCreator(
    csv_path="data/ranked_database.csv",
    conf_path="data/resulting_data.csv",
)
df_features = creator.create_all_features(elo_df=elo_df)

output_path = "data/ranked_database_with_features.csv"
creator.save_to_csv(output_path)

print("\nSample of created features:")
feature_cols = [col for col in df_features.columns if any(
    x in col for x in ['ma_', 'weighted', 'elo', 'trend', 'days_since', 'points_dif', 'confederation', 'tournament_weight']
)]
print(df_features[['date', 'home_team', 'away_team'] + feature_cols[:12]].head(10).to_string())
