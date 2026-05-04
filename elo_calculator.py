import pandas as pd
import numpy as np

# ── K-factor tiers (eloratings.net spec) ──────────────────────────────────────
# 60 — World Cup finals
# 50 — Continental championship finals + major intercontinental tournaments
# 40 — World Cup qualifiers, continental qualifiers, major tournaments
# 30 — All other tournaments
# 20 — Friendly matches

# Tournaments that are clearly WC finals (not qualifiers)
K60 = {'FIFA World Cup'}

# Continental finals + major intercontinental
K50 = {
    'UEFA European Championship',
    'UEFA Euro',
    'Copa América',
    'Copa America',
    'Africa Cup of Nations',
    'AFC Asian Cup',
    'Asian Cup',
    'CONCACAF Gold Cup',
    'Gold Cup',
    'FIFA Confederations Cup',
    'Confederations Cup',
    'Olympic Games',
    'CONMEBOL-UEFA Cup of Champions',
    'CONMEBOL–UEFA Cup of Champions',
}

# Qualifiers + other significant tournaments
K40_KEYWORDS = ('qualification', 'qualifier', 'qualifying')

# Friendly keyword
K20_KEYWORDS = ('friendly',)

# Everything else → 30


def get_k_factor(tournament):
    """
    Return base K-factor following eloratings.net tier rules.
    Order matters: check WC finals before qualifier check to avoid
    'FIFA World Cup qualification' matching K60.
    """
    t = str(tournament)
    t_lower = t.lower()

    # Qualifiers first — catches 'FIFA World Cup qualification' before K60 check
    if any(kw in t_lower for kw in K40_KEYWORDS):
        return 40

    # World Cup finals (non-qualifier)
    if t in K60 or t == 'FIFA World Cup':
        return 60

    # Continental finals + major intercontinental
    if any(name in t for name in K50):
        return 50

    # Friendlies
    if any(kw in t_lower for kw in K20_KEYWORDS):
        return 20

    # Everything else (Nations League, regional cups, etc.)
    return 30


def goal_diff_multiplier(goal_diff):
    """
    K multiplier based on winning margin (eloratings.net spec):
      win by 1 → ×1.0
      win by 2 → ×1.5
      win by 3 → ×1.75
      win by N≥4 → ×(1.75 + (N-3)/8)
    Applied only when there is a winner (goal_diff > 0).
    """
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    elif n == 2:
        return 1.5
    elif n == 3:
        return 1.75
    else:
        return 1.75 + (n - 3) / 8


def expected_score(dr):
    """
    We = 1 / (10^(-dr/400) + 1)
    dr = rating_home - rating_away + home_advantage_bonus
    home_advantage_bonus = +100 for home team, -100 for away team (0 on neutral ground).
    """
    return 1 / (10 ** (-dr / 400) + 1)


def compute_elo(df):
    """
    Process all matches chronologically, recording pre-match ELO for each game.
    Implements the full eloratings.net formula:
      - 5-tier K-factor
      - Goal difference multiplier on K
      - Home advantage (+100 to home team's effective rating) on neutral=False matches
    """
    elo = {}  # team -> current ELO
    results = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        home_elo = elo.get(home, 1500.0)
        away_elo = elo.get(away, 1500.0)

        # Record pre-match ELOs (before updating)
        results.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_elo': round(home_elo, 2),
            'away_elo': round(away_elo, 2),
        })

        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            continue

        # Home advantage: +100 to home team's rating unless neutral venue
        is_neutral = bool(row.get('neutral', False))
        home_advantage = 0 if is_neutral else 100

        # Expected scores using rating difference + home advantage
        dr_home = (home_elo - away_elo) + home_advantage
        we_home = expected_score(dr_home)
        we_away = 1 - we_home

        # Actual results
        home_score = row['home_score']
        away_score = row['away_score']
        goal_diff = home_score - away_score

        if goal_diff > 0:
            home_result, away_result = 1.0, 0.0
        elif goal_diff == 0:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0

        # K adjusted for goal difference
        k_base = get_k_factor(row['tournament'])
        k = k_base * goal_diff_multiplier(goal_diff)

        elo[home] = home_elo + k * (home_result - we_home)
        elo[away] = away_elo + k * (away_result - we_away)

    return pd.DataFrame(results), elo


if __name__ == "__main__":
    print("Loading match history...")
    df = pd.read_csv("data/international_results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Processing {len(df)} matches ({df['date'].min().date()} → {df['date'].max().date()})...")
    elo_df, final_elo = compute_elo(df)

    output_path = "data/elo_ratings.csv"
    elo_df.to_csv(output_path, index=False)
    print(f"ELO ratings saved to {output_path} ({len(elo_df)} records)")

    top_teams = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 teams by final ELO:")
    for team, elo_val in top_teams:
        print(f"  {team}: {elo_val:.0f}")
