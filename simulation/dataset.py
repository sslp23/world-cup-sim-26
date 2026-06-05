"""
Builds the simulation dataset: post-WC22 training matches + WC26 group stage matches.

Training : 2022-12-19 → 2026-06-10 (ranked matches with real scores)
WC26     : 2026-06-11 → 2026-06-27 (NaN scores — future matches)

ELO and pi-ratings reused from full-history CSVs.
features_creator handles NaN scores — form features for WC26 matches
are computed using only pre-tournament rows (strict date < match_date filter).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from data_pipeline.features_creator import FeaturesCreator
from data_pipeline.market_values import merge_market_values

TRAIN_START = pd.Timestamp('2022-12-19')
WC26_END    = pd.Timestamp('2026-06-27')

NAME_FIXES = [
    ('Czechia',                          'Czech Republic'),
    ('IR Iran',                          'Iran'),
    ('Korea Republic',                   'South Korea'),
    ('USA',                              'United States'),
    ('Bosnia-Herzegovina',               'Bosnia and Herzegovina'),
    ('Turkiye',                          'Turkey'),
    ('Democratic Republic of the Congo', 'DR Congo'),
]


def fix_names(series: pd.Series) -> pd.Series:
    for old, new in NAME_FIXES:
        series = series.str.replace(old, new, regex=False)
    series = series.str.replace('China', 'China PR', regex=False)
    return series


def build() -> pd.DataFrame:
    print('Building simulation dataset...')

    results = pd.read_csv('data/international_results.csv')
    results['date'] = pd.to_datetime(results['date'])
    df = results[
        (results['date'] >= TRAIN_START) &
        (results['date'] <= WC26_END)
    ].reset_index(drop=True)

    rank = pd.read_csv('data/resulting_data.csv', low_memory=False)
    rank = rank[['rank', 'nation_full_name', 'points', 'rank_date']]
    rank['rank_date'] = pd.to_datetime(rank['rank_date'])
    rank['points'] = pd.to_numeric(rank['points'], errors='coerce')
    rank = rank[rank['rank_date'] >= TRAIN_START].reset_index(drop=True)
    rank['nation_full_name'] = fix_names(rank['nation_full_name'])

    # Add a sentinel row at WC26_END per team so that resample + ffill
    # extends each team's last known ranking through the tournament dates
    # (FIFA stops updating rankings during the WC).
    last_per_team = (rank.sort_values('rank_date')
                        .groupby('nation_full_name').last()
                        .reset_index())
    last_per_team['rank_date'] = WC26_END
    rank = pd.concat([rank, last_per_team], ignore_index=True)

    rank = (rank.set_index('rank_date')
                .groupby('nation_full_name', group_keys=False)
                .resample('D').first().ffill().reset_index())

    ranked = df.merge(rank, left_on=['date', 'home_team'],
                      right_on=['rank_date', 'nation_full_name']
                      ).drop(['rank_date', 'nation_full_name'], axis=1)
    ranked = ranked.merge(rank, left_on=['date', 'away_team'],
                          right_on=['rank_date', 'nation_full_name'],
                          suffixes=('_home', '_away')
                          ).drop(['rank_date', 'nation_full_name'], axis=1)

    ranked = merge_market_values(ranked)

    wc26_start = pd.Timestamp('2026-06-11')
    n_train = (ranked['date'] < wc26_start).sum()
    n_wc26  = (ranked['date'] >= wc26_start).sum()
    print(f'  Ranked rows — training: {n_train}  WC26: {n_wc26}')

    tmp = 'data/_tmp_sim.csv'
    ranked.to_csv(tmp, index=False)

    elo_df = pd.read_csv('data/elo_ratings.csv')
    pi_df  = pd.read_csv('data/pi_ratings.csv')

    creator = FeaturesCreator(csv_path=tmp, conf_path='data/resulting_data.csv')
    df_feat = creator.create_all_features(elo_df=elo_df, pi_df=pi_df)
    os.remove(tmp)

    return df_feat
