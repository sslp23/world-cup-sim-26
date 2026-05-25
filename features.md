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

---

## Model Feature Selection

Selected based on the EDA in [`eda/README.md`](eda/README.md). All features are expressed as **difference features** (home team value − away team value) to be invariant to arbitrary home/away assignment at neutral venues (e.g. World Cup).

### Logical structure

The selected feature set follows a consistent pattern: one **long-window** (last 20 games, captures true level) and one **short-window** (last 5 games, captures recent form) per metric group.

#### Rating features

ELO captures the full head-to-head results history and is the strongest individual predictor. `abs_elo_diff` captures the size of the quality gap independently of direction, allowing the model to learn that large mismatches suppress draws.

| Feature | Description |
| --- | --- |
| `elo_diff` | ELO rating difference (`home_elo − away_elo`) |
| `abs_elo_diff` | Absolute ELO difference — magnitude of quality gap, unchanged by team swap |

#### ELO momentum and history

Multi-horizon ELO features capture how a team's strength has evolved. These complement `elo_diff` (point-in-time) with trajectory and historical pedigree.

| Feature | Description |
| --- | --- |
| `elo_delta_20_diff` | Difference of each team's ELO change over their last 20 games — positive = home team improving faster |
| `elo_ma_8yr_diff` | 8-year rolling mean ELO difference — historical pedigree; most distinct from current ELO (r=0.94 vs 0.97–0.98 for shorter windows) |

`elo_ma_2yr_diff` and `elo_ma_4yr_diff` were excluded after collinearity analysis — Pearson r = 0.97–0.98 with `elo_diff`, near-identical to current ELO because ratings change slowly. Partial r ≈ −0.04 after conditioning on `elo_diff`. See `eda/elo_features_analysis.py`.

#### World Cup experience

| Feature | Description |
| --- | --- |
| `wc_games_diff` | Difference in number of World Cup matches played historically |

#### Weighted points won

Measures which team has been earning more points recently, adjusted for opponent strength (results against stronger sides count more).

| Feature | Description |
| --- | --- |
| `pww_ma20_diff` | Difference of each team's weighted points won average over last 20 games |
| `pww_ma5_diff` | Difference of each team's weighted points won average over last 5 games |

#### Weighted goals scored

Measures attacking output, weighted up by opponent strength — scoring against a top side is worth more than scoring against a weaker one.

| Feature | Description |
| --- | --- |
| `gw_ma20_diff` | Difference of each team's weighted goals scored average over last 20 games |
| `gw_ma5_diff` | Difference of each team's weighted goals scored average over last 5 games |

#### Weighted goals suffered

Measures defensive solidity, weighted down by opponent strength — conceding against a top side is penalised less than conceding against a weaker one.

| Feature | Description |
| --- | --- |
| `gsw_ma20_diff` | Difference of each team's weighted goals conceded average over last 20 games |
| `gsw_ma5_diff` | Difference of each team's weighted goals conceded average over last 5 games |

#### Goal difference

Captures net dominance (attack minus defence combined) for each team. This is distinct from the goals scored and goals suffered features above: while those measure each dimension independently, goal difference reflects overall match control.

| Feature | Description |
| --- | --- |
| `gd_ma20_diff` | Difference of each team's average goal difference over last 20 games |
| `gd_ma5_diff` | Difference of each team's average goal difference over last 5 games |

#### Confederation (CatBoost only)

| Feature | Description |
| --- | --- |
| `confederation_home` / `confederation_away` | Football confederation (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC). Passed as native categorical to CatBoost; not used in XGBoost or ML-Poisson. |

### Excluded features

| Feature group | Reason for exclusion |
| --- | --- |
| `points_dif` / `abs_points_dif` | Pearson r = 0.92 with `elo_diff`; partial r = −0.05 after conditioning on ELO — near-zero independent signal. See [`eda/README.md`](eda/README.md). |
| `pi_diff` / `pi_neutral_diff` | Pearson r = 0.90 with `elo_diff`; partial r = −0.05. Also conflates home/away venue context which is irrelevant at neutral WC venues. |
| Raw (non-weighted) goals / points won | Pearson r ≈ 0.97 with weighted equivalents; weighted version subsumes them |
| `form_trend_5` / `form_trend_3` | Near-zero mutual information; no predictive power |
| `days_since_last_match` | Near-zero mutual information at the dataset level |
| `tournament_weight` | No predictive power on its own |
| `neutral` | Model is designed for neutral-venue prediction; augmentation and symmetrized inference handle venue invariance |
| Windows ma_10 / ma_3 | Highly correlated with ma_20/ma_5 (r > 0.86); keeping two windows is sufficient |
