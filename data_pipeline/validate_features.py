"""
Feature validation script.

For each team in TEAMS, picks one match from the features CSV and recomputes
every feature from raw data, comparing against the stored values.

Run from the project root:
    python validate_features.py
"""

import pandas as pd
import numpy as np

TEAMS = ['Brazil', 'Germany', 'Japan']
TOLERANCE = 1e-4   # absolute tolerance for float comparisons
MIN_PRIOR_GAMES = 10   # minimum prior games so all rolling features are non-NaN

# ── Helpers (mirror features_creator.py logic exactly) ────────────────────────

def get_tournament_weight(tournament):
    t = str(tournament)
    if 'FIFA World Cup' in t:
        return 1.0
    t_lower = t.lower()
    if 'qualifier' in t_lower or 'qualification' in t_lower:
        return 0.8
    if 'friendly' in t_lower:
        return 0.5
    return 0.9


def calc_points_won(team_score, opp_score):
    if team_score > opp_score:
        return 3
    elif team_score == opp_score:
        return 1
    return 0


def get_team_games_before_date(raw_df, team, date, q90_map):
    """Recomputes the same game history used by features_creator._get_team_games_before_date."""
    keep = ['date', 'points_won', 'points_weighted', 'goals', 'goals_suffered',
            'goals_weighted', 'goals_suffered_weighted', 'goal_diff']

    home = raw_df[(raw_df['home_team'] == team) & (raw_df['date'] < date)].copy()
    home['q90'] = home['date'].map(q90_map)
    home['points_won'] = home.apply(lambda r: calc_points_won(r['home_score'], r['away_score']), axis=1)
    home['points_weighted'] = home['points_won'] * (home['points_away'] / home['q90'])
    home['goals'] = home['home_score']
    home['goals_suffered'] = home['away_score']
    home['goals_weighted'] = home['goals'] * (home['points_away'] / home['q90'])
    home['goals_suffered_weighted'] = home['goals_suffered'] / (home['points_away'] / home['q90'])
    home['goal_diff'] = home['goals'] - home['goals_suffered']

    away = raw_df[(raw_df['away_team'] == team) & (raw_df['date'] < date)].copy()
    away['q90'] = away['date'].map(q90_map)
    away['points_won'] = away.apply(lambda r: calc_points_won(r['away_score'], r['home_score']), axis=1)
    away['points_weighted'] = away['points_won'] * (away['points_home'] / away['q90'])
    away['goals'] = away['away_score']
    away['goals_suffered'] = away['home_score']
    away['goals_weighted'] = away['goals'] * (away['points_home'] / away['q90'])
    away['goals_suffered_weighted'] = away['goals_suffered'] / (away['points_home'] / away['q90'])
    away['goal_diff'] = away['goals'] - away['goals_suffered']

    games = pd.concat([home[keep], away[keep]], ignore_index=True)
    return games.sort_values('date').reset_index(drop=True)


def rolling_features(games):
    """Recompute all rolling features from a team's prior game history."""
    if len(games) == 0:
        return {}

    pw = games['points_won'].values
    pw_w = games['points_weighted'].values
    g = games['goals'].values
    gs = games['goals_suffered'].values
    gw = games['goals_weighted'].values
    gsw = games['goals_suffered_weighted'].values
    gd = games['goal_diff'].values

    def ma(arr, n):
        return np.mean(arr[-n:]) if len(arr) > 0 else np.nan

    def trend(arr, n):
        a = arr[-n:]
        return np.polyfit(range(len(a)), a, 1)[0] if len(a) >= 2 else np.nan

    return {
        'points_won_ma_5':                 ma(pw, 5),
        'points_won_ma_3':                 ma(pw, 3),
        'points_weighted_ma_5':            ma(pw_w, 5),
        'points_weighted_ma_3':            ma(pw_w, 3),
        'goals_ma_5':                      ma(g, 5),
        'goals_ma_3':                      ma(g, 3),
        'goals_suffered_ma_5':             ma(gs, 5),
        'goals_suffered_ma_3':             ma(gs, 3),
        'goals_weighted_ma_5':             ma(gw, 5),
        'goals_weighted_ma_3':             ma(gw, 3),
        'goals_suffered_weighted_ma_5':    ma(gsw, 5),
        'goals_suffered_weighted_ma_3':    ma(gsw, 3),
        'goal_diff_ma_5':                  ma(gd, 5),
        'goal_diff_ma_3':                  ma(gd, 3),
        'form_trend_5':                    trend(pw, 5),
        'form_trend_3':                    trend(pw, 3),
        'days_since_last_match':           None,   # handled separately
    }


# ── Check helper ──────────────────────────────────────────────────────────────

def check(label, expected, actual, results):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        status = 'SKIP'
    elif isinstance(expected, float) or isinstance(actual, float):
        status = 'PASS' if np.isclose(float(expected), float(actual), atol=TOLERANCE) else 'FAIL'
    else:
        status = 'PASS' if expected == actual else 'FAIL'

    symbol = {'PASS': 'OK  ', 'FAIL': 'FAIL', 'SKIP': 'SKIP'}[status]
    print(f"  {symbol} {label:<45}  expected={expected!r:<20}  actual={actual!r}")
    results.append(status)


# ── Main validation ───────────────────────────────────────────────────────────

def validate_team(team, raw_df, feat_df, elo_df, conf_map, q90_map):
    print(f"\n{'=' * 70}")
    print(f"  Team: {team}")
    print(f"{'=' * 70}")

    # Find a match where the team has ≥ MIN_PRIOR_GAMES prior games
    appearances = feat_df[(feat_df['home_team'] == team) | (feat_df['away_team'] == team)]
    if appearances.empty:
        print(f"  !! {team} not found in features CSV")
        return

    chosen = None
    for _, row in appearances.iterrows():
        prior = get_team_games_before_date(raw_df, team, row['date'], q90_map)
        if len(prior) >= MIN_PRIOR_GAMES:
            chosen = row
            break

    if chosen is None:
        print(f"  !! Could not find a match with ≥{MIN_PRIOR_GAMES} prior games for {team}")
        return

    is_home = chosen['home_team'] == team
    prefix = 'home_' if is_home else 'away_'
    opp_prefix = 'away_' if is_home else 'home_'

    team_score  = chosen['home_score'] if is_home else chosen['away_score']
    opp_score   = chosen['away_score']  if is_home else chosen['home_score']
    opp_points  = chosen['points_away'] if is_home else chosen['points_home']

    print(f"  Match : {chosen['date'].date()}  {chosen['home_team']} {int(chosen['home_score'])}–{int(chosen['away_score'])} {chosen['away_team']}")
    print(f"  Role  : {'HOME' if is_home else 'AWAY'}")
    print(f"  Tournament: {chosen['tournament']}  (neutral={chosen['neutral']})")
    print()

    results = []

    # ── Static features ───────────────────────────────────────────────────────

    # Points won
    exp_pw = calc_points_won(team_score, opp_score)
    check(f'{prefix}points_won', exp_pw, chosen[f'{prefix}points_won'], results)

    # Tournament weight
    exp_tw = get_tournament_weight(chosen['tournament'])
    check('tournament_weight', exp_tw, chosen['tournament_weight'], results)

    # Points weighted — normalised by q90 of points at match date
    q90 = chosen['points_q90']
    exp_pwt = exp_pw * (opp_points / q90)
    check(f'{prefix}points_weighted', exp_pwt, chosen[f'{prefix}points_weighted'], results)

    # Points dif
    exp_pdif = chosen['points_home'] - chosen['points_away']
    check('points_dif', exp_pdif, chosen['points_dif'], results)

    # ELO from elo_ratings.csv
    elo_row = elo_df[
        (elo_df['date'] == chosen['date']) &
        (elo_df['home_team'] == chosen['home_team']) &
        (elo_df['away_team'] == chosen['away_team'])
    ]
    if not elo_row.empty:
        check('home_elo',  elo_row.iloc[0]['home_elo'], chosen['home_elo'],  results)
        check('away_elo',  elo_row.iloc[0]['away_elo'], chosen['away_elo'],  results)
        exp_elo_diff = elo_row.iloc[0]['home_elo'] - elo_row.iloc[0]['away_elo']
        check('elo_diff',  exp_elo_diff,               chosen['elo_diff'],  results)
    else:
        print(f"  ~ elo_ratings: match not found — skipping ELO checks")

    # Confederation (columns are confederation_home / confederation_away)
    exp_conf = conf_map.get(team)
    conf_col = f'confederation_{"home" if is_home else "away"}'
    check(conf_col, exp_conf, chosen[conf_col], results)

    # ── Rolling features ──────────────────────────────────────────────────────
    print()
    prior_games = get_team_games_before_date(raw_df, team, chosen['date'], q90_map)
    exp_rolling = rolling_features(prior_games)

    for metric, exp_val in exp_rolling.items():
        if metric == 'days_since_last_match':
            continue
        check(f'{prefix}{metric}', exp_val, chosen[f'{prefix}{metric}'], results)

    # Days since last match
    exp_days = (chosen['date'] - prior_games['date'].iloc[-1]).days
    check(f'{prefix}days_since_last_match', exp_days, chosen[f'{prefix}days_since_last_match'], results)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = results.count('PASS')
    failed = results.count('FAIL')
    skipped = results.count('SKIP')
    total = passed + failed
    print(f"\n  Result: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")


def main():
    print("Loading data...")
    raw_df  = pd.read_csv("data/ranked_database.csv")
    feat_df = pd.read_csv("data/ranked_database_with_features.csv")
    elo_df  = pd.read_csv("data/elo_ratings.csv")

    raw_df['date']  = pd.to_datetime(raw_df['date'])
    feat_df['date'] = pd.to_datetime(feat_df['date'])
    elo_df['date']  = pd.to_datetime(elo_df['date'])

    conf_df = pd.read_csv("data/resulting_data.csv", low_memory=False)
    conf_df["nation_full_name"] = (
        conf_df["nation_full_name"]
        .str.replace("Czechia", "Czech Republic")
        .str.replace("IR Iran", "Iran")
        .str.replace("Korea Republic", "South Korea")
        .str.replace("USA", "United States")
    )
    conf_map = (
        conf_df.sort_values('rank_date')
        .groupby('nation_full_name')['confederation']
        .last()
        .to_dict()
    )

    for team in TEAMS:
        validate_team(team, raw_df, feat_df, elo_df, conf_map)

    print(f"\n{'=' * 70}\n  Done.\n{'=' * 70}")


if __name__ == "__main__":
    main()
