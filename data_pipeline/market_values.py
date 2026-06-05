import pandas as pd

_COUNTRY_MAP = {
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Brunei Darussalam': 'Brunei',
    'China': 'China PR',
    'Chinese Taipei': 'Taiwan',
    "Cote d'Ivoire": 'Ivory Coast',
    'Curacao': 'Curaçao',
    'Ireland': 'Republic of Ireland',
    'Korea, North': 'North Korea',
    'Korea, South': 'South Korea',
    'Macao': 'Macau',
    'Neukaledonien': 'New Caledonia',
    'Saint-Martin': 'Saint Martin',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Southern Sudan': 'South Sudan',
    'St. Kitts & Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'The Gambia': 'Gambia',
    'Türkiye': 'Turkey',
}


def merge_market_values(df: pd.DataFrame, mv_path: str = 'data/market_values_info.csv') -> pd.DataFrame:
    """Merge Transfermarkt top-20 squad market values onto df for home and away teams.

    Adds: home_mv_top20_sum, home_mv_top20_median, away_mv_top20_sum, away_mv_top20_median.
    Uses merge_asof(direction='backward') so the most recent snapshot on or before each
    match date is used — no sentinel rows needed, works regardless of how far the match
    date is beyond the last snapshot.
    """
    mv = pd.read_csv(mv_path)
    mv['date'] = pd.to_datetime(mv['date'])
    mv['country_of_citizenship'] = mv['country_of_citizenship'].replace(_COUNTRY_MAP)
    mv = mv.sort_values('date').reset_index(drop=True)

    # Stable positional index so we can map results back to original df order
    df = df.reset_index(drop=True).copy()
    df_sorted = df[['date', 'home_team', 'away_team']].copy()
    df_sorted['_row'] = df_sorted.index
    df_sorted = df_sorted.sort_values('date').reset_index(drop=True)

    for side, team_col in [('home', 'home_team'), ('away', 'away_team')]:
        result = pd.merge_asof(
            df_sorted[['date', team_col, '_row']],
            mv.rename(columns={
                'country_of_citizenship': team_col,
                'top20_sum': f'{side}_mv_top20_sum',
                'top20_median': f'{side}_mv_top20_median',
            })[['date', team_col, f'{side}_mv_top20_sum', f'{side}_mv_top20_median']],
            on='date',
            by=team_col,
            direction='backward',
        ).set_index('_row')
        df[f'{side}_mv_top20_sum'] = result[f'{side}_mv_top20_sum']
        df[f'{side}_mv_top20_median'] = result[f'{side}_mv_top20_median']

    return df
