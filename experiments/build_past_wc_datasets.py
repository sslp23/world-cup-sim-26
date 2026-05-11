"""
Build ranked_database_with_features.csv for each past World Cup edition.

For each WC, the dataset covers:
  - Training window: from the day after the previous WC final
  - Through: end of the WC being built (so the WC matches are included
    and their features can be computed from prior history)

Output: data/past_wc/wc{year}/ranked_database_with_features.csv

ELO and pi-ratings are always computed from full match history and
reused from the existing CSVs (data/elo_ratings.csv, data/pi_ratings.csv).

Run from the project root:
    python -m experiments.build_past_wc_datasets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from data_pipeline.features_creator import FeaturesCreator

# (year, wc_start, wc_end, training_start)
# training_start = day after the previous WC final
EDITIONS = [
    (2006, '2006-06-09', '2006-07-09', '2002-07-01'),
    (2010, '2010-06-11', '2010-07-11', '2006-07-10'),
    (2014, '2014-06-12', '2014-07-13', '2010-07-12'),
    (2018, '2018-06-14', '2018-07-15', '2014-07-14'),
    (2022, '2022-11-20', '2022-12-18', '2018-07-16'),
]


def build_dataset(train_start: str, wc_end: str) -> pd.DataFrame:
    """
    Build ranked_database_with_features for [train_start, wc_end].
    """
    train_start_ts = pd.Timestamp(train_start)
    wc_end_ts      = pd.Timestamp(wc_end)

    # --- Match results ---
    results = pd.read_csv('data/international_results.csv')
    results['date'] = pd.to_datetime(results['date'])
    df = results[
        (results['date'] >= train_start_ts) &
        (results['date'] <= wc_end_ts)
    ].reset_index(drop=True)

    # --- FIFA rankings ---
    rank = pd.read_csv('data/resulting_data.csv', low_memory=False)
    rank = rank[['rank', 'nation_full_name', 'points', 'rank_date']]
    rank['rank_date'] = pd.to_datetime(rank['rank_date'])
    rank['points'] = pd.to_numeric(rank['points'], errors='coerce')
    # No upper bound on rankings — FIFA stops updating during WC, so the last
    # ranking before the WC must forward-fill to cover WC match dates.
    rank = rank[rank['rank_date'] >= train_start_ts].reset_index(drop=True)
    rank['nation_full_name'] = (
        rank['nation_full_name']
        .str.replace('Czechia', 'Czech Republic')
        .str.replace('IR Iran', 'Iran')
        .str.replace('Korea Republic', 'South Korea')
        .str.replace('USA', 'United States')
        .str.replace('Bosnia-Herzegovina', 'Bosnia and Herzegovina')
        .str.replace('Turkiye', 'Turkey')
        .str.replace('China', 'China PR', regex=False)
    )
    rank = (rank.set_index('rank_date')
                .groupby('nation_full_name', group_keys=False)
                .resample('D').first().ffill().reset_index())

    # --- Merge matches with rankings ---
    ranked = df.merge(
        rank, left_on=['date', 'home_team'],
        right_on=['rank_date', 'nation_full_name']
    ).drop(['rank_date', 'nation_full_name'], axis=1)
    ranked = ranked.merge(
        rank, left_on=['date', 'away_team'],
        right_on=['rank_date', 'nation_full_name'],
        suffixes=('_home', '_away')
    ).drop(['rank_date', 'nation_full_name'], axis=1)

    print(f'    Ranked matches after join: {len(ranked)}')

    # --- Features (ELO + pi-ratings reused from full-history CSVs) ---
    tmp_path = f'data/_tmp_ranked_{train_start}_{wc_end}.csv'
    ranked.to_csv(tmp_path, index=False)

    elo_df = pd.read_csv('data/elo_ratings.csv')
    pi_df  = pd.read_csv('data/pi_ratings.csv')

    creator = FeaturesCreator(csv_path=tmp_path, conf_path='data/resulting_data.csv')
    df_feat = creator.create_all_features(elo_df=elo_df, pi_df=pi_df)

    os.remove(tmp_path)
    return df_feat


def run():
    for year, wc_start, wc_end, train_start in EDITIONS:
        out_dir  = f'data/past_wc/wc{year}'
        out_path = f'{out_dir}/ranked_database_with_features.csv'

        os.makedirs(out_dir, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'  WC{year}  |  training: {train_start} -> {wc_end}')
        print(f'{"="*60}')

        df = build_dataset(train_start, wc_end)

        wc_start_ts = pd.Timestamp(wc_start)
        wc_end_ts   = pd.Timestamp(wc_end)
        train_rows  = (df['date'] < wc_start_ts).sum()
        wc_rows     = (
            (df['tournament'] == 'FIFA World Cup') &
            (df['date'] >= wc_start_ts) &
            (df['date'] <= wc_end_ts)
        ).sum()

        df.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')
        print(f'  Total rows: {len(df)}  |  Training rows: {train_rows}  |  WC matches: {wc_rows}')


if __name__ == '__main__':
    run()
