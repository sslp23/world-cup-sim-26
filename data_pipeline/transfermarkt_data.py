import kagglehub
import os
import pandas as pd
from tqdm import tqdm

TOP_N_PLAYERS = 20
SUM_COL = f'top{TOP_N_PLAYERS}_sum'
MEDIAN_COL = f'top{TOP_N_PLAYERS}_median'

# Download latest version
path = kagglehub.dataset_download("davidcariboo/player-scores")

print(os.listdir(path))

file_path = os.path.join(path, "players.csv")
players = pd.read_csv(file_path)

file_path = os.path.join(path, "player_valuations.csv")
market_values = pd.read_csv(file_path)

market_values_full = market_values.merge(players[['player_id', 'name','country_of_citizenship']], left_on='player_id', right_on='player_id', how='left')

market_values_full['date'] = pd.to_datetime(market_values_full['date'])
market_values_sorted = market_values_full.sort_values('date')
today = pd.Timestamp('today').normalize() + pd.Timedelta(days=60)
dates = pd.date_range('2005-01-07', '2026-08-01', freq='D')

player_country = market_values_full[['player_id', 'country_of_citizenship']].drop_duplicates('player_id')

results = []
for d in tqdm(dates, desc="Building snapshots"):
    snapshot = (
        market_values_sorted[market_values_sorted['date'] <= d]
        .groupby('player_id', as_index=False)['market_value_in_eur']
        .last()
        .merge(player_country, on='player_id')
    )
    agg = (
        snapshot.sort_values('market_value_in_eur', ascending=False)
                .groupby('country_of_citizenship')
                .head(TOP_N_PLAYERS)
                .groupby('country_of_citizenship')['market_value_in_eur']
                .agg(**{SUM_COL: 'sum', MEDIAN_COL: 'median'})
                .reset_index()
    )
    agg['date'] = d
    results.append(agg)

final_df = pd.concat(results, ignore_index=True)

output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_values_info.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# --- Sanity checks ---

def top30_at_date(df, ref_date, label):
    available = df[df['date'] < ref_date]['date']
    if available.empty:
        print(f"\n{label}: no data before {ref_date.date()}")
        return
    last = available.max()
    top30 = (
        df[df['date'] == last]
        .sort_values(SUM_COL, ascending=False)
        .head(30)[['country_of_citizenship', SUM_COL, MEDIAN_COL]]
    )
    print(f"\n{label} — snapshot as of {last.date()}:")
    print(top30.to_string(index=False))

wc_dates = {
    'WC 2006': pd.Timestamp('2006-06-09'),
    'WC 2010': pd.Timestamp('2010-06-11'),
    'WC 2014': pd.Timestamp('2014-06-12'),
    'WC 2018': pd.Timestamp('2018-06-14'),
    'WC 2022': pd.Timestamp('2022-11-20'),
    'WC 2026 (today)': today + pd.Timedelta(days=60),
}
for label, wc_date in wc_dates.items():
    top30_at_date(final_df, wc_date, f"Top 30 prior to {label}")
