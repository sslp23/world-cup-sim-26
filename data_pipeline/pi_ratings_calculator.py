"""
Pi-ratings calculator for football match prediction.

Reference: Constantinou & Fenton (2013). Determining the level of ability of
football teams by dynamic ratings based on the relative discrepancies in scores.

Each team maintains two ratings:
  pi_h — performance in home context
  pi_a — performance in away context

After each match (team i at home vs team j away):
  error = actual_goal_diff - expected_goal_diff
        = (home_goals - away_goals) - (pi_h[i] - pi_a[j])

Updates use a squashing function φ(x) = x / (1 + |x|) ∈ (-1, 1) that
dampens the influence of extreme scorelines:
  pi_h[i] += lambda_h × φ(error)   # home team, primary context
  pi_a[j] -= lambda_h × φ(error)   # away team, primary context
  pi_a[i] += lambda_a × φ(error)   # home team, secondary context
  pi_h[j] -= lambda_a × φ(error)   # away team, secondary context

Outputs per match (pre-match, no leakage):
  pi_h_home  — home team's π_h before the match
  pi_a_home  — home team's π_a before the match
  pi_h_away  — away team's π_h before the match
  pi_a_away  — away team's π_a before the match
  pi_diff    — expected goal diff (pi_h_home - pi_a_away); >0 favours home
"""

import pandas as pd
import numpy as np


def phi(x):
    """Squashing function: maps any real to (-1, 1). Dampens outlier scorelines."""
    return x / (1 + abs(x))


def compute_pi_ratings(df, lambda_h=0.035, lambda_a=None):
    """
    Process all matches chronologically, recording pre-match pi-ratings.

    Args:
        df:        DataFrame with columns: date, home_team, away_team,
                   home_score, away_score. Uses full match history.
        lambda_h:  Primary learning rate (home/away context). Default 0.035.
        lambda_a:  Secondary learning rate. Defaults to lambda_h / 3.

    Returns:
        ratings_df: DataFrame with pre-match ratings per game.
        final_pi:   Dict of {team: {'pi_h': float, 'pi_a': float}}.
    """
    if lambda_a is None:
        lambda_a = lambda_h / 3

    pi = {}   # team -> {'pi_h': float, 'pi_a': float}
    results = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Initialise unseen teams at 0
        if home not in pi:
            pi[home] = {'pi_h': 0.0, 'pi_a': 0.0}
        if away not in pi:
            pi[away] = {'pi_h': 0.0, 'pi_a': 0.0}

        pi_h_home = pi[home]['pi_h']
        pi_a_home = pi[home]['pi_a']
        pi_h_away = pi[away]['pi_h']
        pi_a_away = pi[away]['pi_a']

        # Record pre-match ratings (before update)
        results.append({
            'date':      row['date'],
            'home_team': home,
            'away_team': away,
            'pi_h_home': round(pi_h_home, 4),
            'pi_a_home': round(pi_a_home, 4),
            'pi_h_away': round(pi_h_away, 4),
            'pi_a_away': round(pi_a_away, 4),
            'pi_diff':   round(pi_h_home - pi_a_away, 4),
        })

        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            continue

        actual_diff   = row['home_score'] - row['away_score']
        expected_diff = pi_h_home - pi_a_away
        error         = actual_diff - expected_diff

        e = phi(error)

        # Primary updates — context in which each team played
        pi[home]['pi_h'] += lambda_h * e
        pi[away]['pi_a'] -= lambda_h * e

        # Secondary updates — cross-context nudge
        pi[home]['pi_a'] += lambda_a * e
        pi[away]['pi_h'] -= lambda_a * e

    return pd.DataFrame(results), pi


if __name__ == "__main__":
    print("Loading match history...")
    df = pd.read_csv("data/international_results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Processing {len(df)} matches ({df['date'].min().date()} -> {df['date'].max().date()})...")
    ratings_df, final_pi = compute_pi_ratings(df)

    output_path = "data/pi_ratings.csv"
    ratings_df.to_csv(output_path, index=False)
    print(f"Pi-ratings saved to {output_path} ({len(ratings_df)} records)")

    top_teams = sorted(final_pi.items(), key=lambda x: (x[1]['pi_h'] + x[1]['pi_a']) / 2, reverse=True)[:20]
    print("\nTop 20 teams by average pi-rating:")
    for team, r in top_teams:
        avg = (r['pi_h'] + r['pi_a']) / 2
        print(f"  {team:<25} pi_h={r['pi_h']:+.3f}  pi_a={r['pi_a']:+.3f}  avg={avg:+.3f}")
