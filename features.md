# Features Documentation

## Source Dataframe Columns (`ranked_database.csv`)

| Column | Description |
| --- | --- |
| `date` | Match date |
| `home_team` | Home team name |
| `away_team` | Away team name |
| `home_score` | Goals scored by home team |
| `away_score` | Goals scored by away team |
| `tournament` | Competition name (e.g. Friendly, FIFA World Cup) |
| `city` | City where the match was played |
| `country` | Country where the match was played |
| `neutral` | Whether the match was played at a neutral venue (bool) |
| `rank_home` | FIFA ranking of the home team at match time |
| `points_home` | FIFA ranking points of the home team |
| `rank_away` | FIFA ranking of the away team at match time |
| `points_away` | FIFA ranking points of the away team |

---

## Static Features (per match row)

### Points

- `home_points_won` / `away_points_won` — Points earned in the match (3=win, 1=draw, 0=loss)
- `home_points_weighted` / `away_points_weighted` — Points scaled by opponent FIFA points: `points_won × (opponent_points / 1400)`. Results against stronger opponents earn more credit.

### Strength Differential

- `points_dif` — FIFA points difference between home and away team (`points_home - points_away`). Positive means home team is stronger.

### Tournament

- `tournament` — Carried through from source data; used to derive `tournament_weight`.
- `neutral` — Carried through from source data; used in ELO home advantage calculation.
- `tournament_weight` — Competition importance weight:

  | Tournament type | Weight |
  | --- | --- |
  | FIFA World Cup | 1.0 |
  | Qualifiers | 0.8 |
  | Other competitive | 0.9 |
  | Friendly | 0.5 |

### ELO Ratings

ELO ratings are pre-computed by `elo_calculator.py` from the full match history (1872–present) using the [eloratings.net](https://www.eloratings.net) formula.

- `home_elo` / `away_elo` — Each team's ELO rating going into the match (pre-match).
- `elo_diff` — `home_elo - away_elo`. Positive means home team is rated higher.

**ELO formula details:**

- **K-factor tiers**: 60 (WC finals), 50 (continental finals + major intercontinental), 40 (qualifiers), 30 (all other), 20 (friendlies)
- **Goal difference multiplier on K**: ×1.5 for 2-goal wins, ×1.75 for 3, ×(1.75 + (N−3)/8) for N≥4
- **Home advantage**: +100 added to home team's effective rating for expected score, skipped on neutral venues

### Confederation

- `confederation_home` / `confederation_away` — Football confederation of each team (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC). Sourced from `resulting_data.csv`.

---

## Rolling/Moving Average Features (per team, computed from prior games only)

Each feature is computed for both `home_` and `away_` teams, using their last **20**, **10**, **5** and **3** games before the match date.

### Key design choices

- All moving averages are **leak-free** — only games strictly before the match date are used.
- Home and away games are **combined** into a single timeline per team (not separated), so form reflects all recent results regardless of venue.
- Opponent weighting uses FIFA **points** (not rank): `× (opponent_points / 1400)`. Higher opponent points = stronger opponent = more weight.

### Feature table

| Base metric | Description |
| --- | --- |
| `points_won_ma_X` | Avg points earned in last X games |
| `points_weighted_ma_X` | Avg points weighted by opponent FIFA points |
| `goals_ma_X` | Avg goals scored in last X games |
| `goals_suffered_ma_X` | Avg goals conceded in last X games |
| `goals_weighted_ma_X` | Avg goals scored weighted by opponent strength |
| `goals_suffered_weighted_ma_X` | Avg goals conceded weighted by opponent strength |
| `goal_diff_ma_X` | Avg goal difference (scored − conceded) in last X games |
| `form_trend_5` / `_form_trend_3` | Linear slope of `points_won` over last 5/3 games. Positive = improving form, negative = declining. Captures trajectory that identical MAs would mask (e.g. W-W-W after L-L vs L-L after W-W-W). |
| `days_since_last_match` | Days between the current match and the team's previous game. Captures fatigue and match rhythm. |
