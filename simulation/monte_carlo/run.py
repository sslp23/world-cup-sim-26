"""
WC 2026 Monte Carlo title probability simulation — CatBoost model.

Pre-computation (runs once):
  - Train CatBoost on pre-WC26 data.
  - Predict probabilities for all 48 group stage matches.
  - Pre-compute P(A beats B) for all 48x47/2 = 1,128 team pairings.

Simulation loop (N=10,000):
  For each run:
    1. Sample group stage outcomes from predicted probabilities.
    2. Compute standings; pick best 8 third-placed teams.
    3. Look up bracket allocation in third_place_combinations.csv
       (all C(12,8)=495 group combinations are covered, so the correct
       slot assignment is always found regardless of which 8 groups
       advance their 3rd-place team).
    4. Simulate R32->R16->QF->SF->Final using pre-computed pairwise probs.
    5. Record the champion.

Run from the project root:
    python -m simulation.monte_carlo.run
    python -m simulation.monte_carlo.run --n 50000
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

from models.catboost.model import CatBoostPredictor
from simulation.dataset    import build as build_dataset
from config import CB_DRAW_WEIGHT

WC26_START = pd.Timestamp('2026-06-11')
WC26_END   = pd.Timestamp('2026-06-27')
OUTPUT_CSV = 'simulation/output/wc_26_mc_title_probs.csv'
N_SIMS     = 10_000

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
ALL_TEAMS   = [t for ts in GROUPS.values() for t in ts]
TEAM_TO_GRP = {t: g for g, ts in GROUPS.items() for t in ts}

# R32 bracket: (match_id, slot_A, slot_B)
R32 = [
    (73,  '2A',    '2B'),
    (74,  '1E',    '3ABCDF'),
    (75,  '1F',    '2C'),
    (76,  '1C',    '2F'),
    (77,  '1I',    '3CDFGH'),
    (78,  '2E',    '2I'),
    (79,  '1A',    '3CEFHI'),
    (80,  '1L',    '3EHIJK'),
    (81,  '1D',    '3BEFIJ'),
    (82,  '1G',    '3AEHIJ'),
    (83,  '2K',    '2L'),
    (84,  '1H',    '2J'),
    (85,  '1B',    '3EFGIJ'),
    (86,  '1J',    '2H'),
    (87,  '1K',    '3DEIJL'),
    (88,  '2D',    '2G'),
]
R16 = [(89,74,77),(90,73,75),(91,76,78),(92,79,80),
       (93,83,84),(94,81,82),(95,86,88),(96,85,87)]
QF  = [(97,89,90),(98,93,94),(99,91,92),(100,95,96)]
SF  = [(101,97,98),(102,99,100)]

# Columns in third_place_combinations.csv that carry slot assignments
SLOT_COLS = {
    '1A vs': 79, '1B vs': 85, '1D vs': 81, '1E vs': 74,
    '1G vs': 82, '1I vs': 77, '1K vs': 87, '1L vs': 80,
}


# ── Pre-computation ────────────────────────────────────────────────────────────

def _precompute_group_probs(wc26_df, model):
    probs = {}
    for _, m in wc26_df.iterrows():
        p = model.predict_proba_row(m)
        probs[(m['home_team'], m['away_team'])] = (
            p['home_win'], p['draw'], p['away_win']
        )
    return probs


def _build_profiles():
    wc_df    = pd.read_csv('data/wc_26_data.csv')
    profiles = {}
    for _, row in wc_df.iterrows():
        for side in ('home', 'away'):
            team = row[f'{side}_team']
            if team not in profiles:
                profiles[team] = {
                    'elo':           row.get(f'{side}_elo', 1500),
                    'pww_ma20':      row.get(f'{side}_points_weighted_ma_20', 0),
                    'pww_ma5':       row.get(f'{side}_points_weighted_ma_5', 0),
                    'gw_ma20':       row.get(f'{side}_goals_weighted_ma_20', 0),
                    'gw_ma5':        row.get(f'{side}_goals_weighted_ma_5', 0),
                    'gsw_ma20':      row.get(f'{side}_goals_suffered_weighted_ma_20', 0),
                    'gsw_ma5':       row.get(f'{side}_goals_suffered_weighted_ma_5', 0),
                    'gd_ma20':       row.get(f'{side}_goal_diff_ma_20', 0),
                    'gd_ma5':        row.get(f'{side}_goal_diff_ma_5', 0),
                    'wc_games':      row.get(f'{side}_wc_games', 0),
                    'wc_best_round': row.get(f'{side}_wc_best_round', 0),
                    'wc_gpg':        row.get(f'{side}_wc_goals_per_game', 0),
                    'wc_gcpg':       row.get(f'{side}_wc_goals_conceded_per_game', 0),
                    'mv_top20_sum':  row.get(f'{side}_mv_top20_sum', np.nan),
                    'confederation': row.get(f'confederation_{side}', 'Unknown'),
                }
    return profiles


def _ko_row(pa, pb):
    """Build the feature Series for a CatBoost knockout prediction."""
    return pd.Series({
        'elo_diff':                           pa.get('elo', 1500)    - pb.get('elo', 1500),
        'home_mv_top20_sum':                  pa.get('mv_top20_sum', np.nan),
        'away_mv_top20_sum':                  pb.get('mv_top20_sum', np.nan),
        'home_wc_best_round':                 pa.get('wc_best_round', 0),
        'away_wc_best_round':                 pb.get('wc_best_round', 0),
        'home_wc_goals_per_game':             pa.get('wc_gpg', 0),
        'away_wc_goals_per_game':             pb.get('wc_gpg', 0),
        'home_wc_games':                      pa.get('wc_games', 0),
        'away_wc_games':                      pb.get('wc_games', 0),
        'home_points_weighted_ma_20':         pa.get('pww_ma20', 0),
        'away_points_weighted_ma_20':         pb.get('pww_ma20', 0),
        'home_points_weighted_ma_5':          pa.get('pww_ma5', 0),
        'away_points_weighted_ma_5':          pb.get('pww_ma5', 0),
        'home_goals_weighted_ma_20':          pa.get('gw_ma20', 0),
        'away_goals_weighted_ma_20':          pb.get('gw_ma20', 0),
        'home_goals_weighted_ma_5':           pa.get('gw_ma5', 0),
        'away_goals_weighted_ma_5':           pb.get('gw_ma5', 0),
        'home_goals_suffered_weighted_ma_20': pa.get('gsw_ma20', 0),
        'away_goals_suffered_weighted_ma_20': pb.get('gsw_ma20', 0),
        'home_goals_suffered_weighted_ma_5':  pa.get('gsw_ma5', 0),
        'away_goals_suffered_weighted_ma_5':  pb.get('gsw_ma5', 0),
        'home_goal_diff_ma_20':               pa.get('gd_ma20', 0),
        'away_goal_diff_ma_20':               pb.get('gd_ma20', 0),
        'home_goal_diff_ma_5':                pa.get('gd_ma5', 0),
        'away_goal_diff_ma_5':                pb.get('gd_ma5', 0),
        'home_wc_goals_conceded_per_game':    pa.get('wc_gcpg', 0),
        'away_wc_goals_conceded_per_game':    pb.get('wc_gcpg', 0),
        'confederation_home':                 pa.get('confederation', 'Unknown'),
        'confederation_away':                 pb.get('confederation', 'Unknown'),
    })


def _precompute_ko_probs(model, profiles):
    """
    P(team_a beats team_b) for all 1,128 pairings.
    Draw probability is split proportionally by relative win strength.
    """
    pairs = list(combinations(ALL_TEAMS, 2))
    print(f'Pre-computing {len(pairs)} pairwise knockout probabilities...')
    ko = {}
    for ta, tb in tqdm(pairs):
        row   = _ko_row(profiles.get(ta, {}), profiles.get(tb, {}))
        p     = model.predict_proba_row(row)
        p_hw, p_d, p_aw = p['home_win'], p['draw'], p['away_win']
        total   = p_hw + p_aw or 1e-9
        ko[(ta, tb)] = float(p_hw + p_d * 0.5)
        ko[(tb, ta)] = float(p_aw + p_d * 0.5)
    return ko


def _precompute_win_prob_sums(group_probs):
    wps = defaultdict(float)
    for (home, away), (p_hw, _, p_aw) in group_probs.items():
        wps[home] += p_hw
        wps[away] += p_aw
    return dict(wps)


def _build_combo_index(combo_df):
    """
    Pre-index the 495-row CSV by frozenset of group letters for O(1) lookup.
    Returns dict {frozenset → row (Series)}.
    """
    grp_cols = [c for c in combo_df.columns if 'groupsvte' in c]
    index = {}
    for _, row in combo_df.iterrows():
        key = frozenset(str(v) for v in row[grp_cols] if pd.notna(v))
        index[key] = row
    return index


# ── One simulation ─────────────────────────────────────────────────────────────

def _sample_group_stage(match_list, probs_arr, rng):
    """
    Vectorised sampling of all 48 group outcomes in one call.
    match_list : list of (home, away) in fixed order
    probs_arr  : np.ndarray shape (48, 3) — columns [p_hw, p_d, p_aw]
    Returns dict {(home, away): 'home_win'|'draw'|'away_win'}.
    """
    outcomes = {}
    labels   = ['home_win', 'draw', 'away_win']
    cumprobs = np.cumsum(probs_arr, axis=1)          # (48, 3) cumulative
    draws    = rng.random(len(match_list))            # (48,)
    indices  = np.argmax(draws[:, None] < cumprobs, axis=1)
    for (home, away), idx in zip(match_list, indices):
        outcomes[(home, away)] = labels[idx]
    return outcomes


def _compute_standings(outcomes, win_prob_sums):
    pts  = defaultdict(int)
    wins = defaultdict(int)
    for (home, away), outcome in outcomes.items():
        if outcome == 'home_win':
            pts[home] += 3; wins[home] += 1
        elif outcome == 'draw':
            pts[home] += 1; pts[away]  += 1
        else:
            pts[away] += 3; wins[away] += 1

    standings = {}
    for grp, teams in GROUPS.items():
        ranked = sorted(teams, key=lambda t: (
            -pts[t], -wins[t], -win_prob_sums.get(t, 0)
        ))
        standings[grp] = [(t, pts[t], wins[t]) for t in ranked]
    return standings


def _pick_best_thirds(standings, win_prob_sums):
    thirds = []
    for grp, rows in standings.items():
        t, p, w = rows[2]
        thirds.append((p, w, win_prob_sums.get(t, 0), t, grp))
    thirds.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return thirds[:8]


def _allocate_thirds(best_thirds, combo_index):
    """
    Look up the bracket allocation for this specific combination of 8 groups.
    Returns {match_id: team} or {} if combination not found (should not happen).
    """
    team_by_grp = {grp: team for _, _, _, team, grp in best_thirds}
    qual_groups = frozenset(team_by_grp.keys())
    row = combo_index.get(qual_groups)
    if row is None:
        return {}
    assignment = {}
    for col, match_id in SLOT_COLS.items():
        val = str(row[col]).strip()
        grp = val.lstrip('3')
        if grp in team_by_grp:
            assignment[match_id] = team_by_grp[grp]
    return assignment


def _simulate_bracket(standings, thirds_assign, ko_probs, rng):
    """Simulate the full knockout bracket. Returns champion name."""
    slot = {}
    for grp, rows in standings.items():
        slot[f'1{grp}'] = rows[0][0]
        slot[f'2{grp}'] = rows[1][0]
    for mid, team in thirds_assign.items():
        for m_id, sA, sB in R32:
            if m_id == mid:
                key = f'r32_{mid}_B' if sB.startswith('3') else f'r32_{mid}_A'
                slot[key] = team

    results = {}

    def resolve(s, mid=None):
        if s[0] in ('1', '2'):
            return slot[s]
        if s[0] == '3':
            return slot.get(f'r32_{mid}_B', slot.get(f'r32_{mid}_A'))
        return results[int(s)]

    def play(ta, tb):
        if ta is None: return tb
        if tb is None: return ta
        return ta if rng.random() < ko_probs.get((ta, tb), 0.5) else tb

    for mid, sA, sB in R32:
        results[mid] = play(resolve(sA, mid), resolve(sB, mid))
    for mid, sA, sB in R16:
        results[mid] = play(results[sA], results[sB])
    for mid, sA, sB in QF:
        results[mid] = play(results[sA], results[sB])

    sf_losers = []
    for mid, sA, sB in SF:
        tA, tB = results[sA], results[sB]
        w = play(tA, tB)
        results[mid] = w
        sf_losers.append(tB if w == tA else tA)

    return play(results[SF[0][0]], results[SF[1][0]])   # Final


# ── Main ───────────────────────────────────────────────────────────────────────

def run(n_sims=N_SIMS):
    # ── Train ────────────────────────────────────────────────────────────────
    print('Building dataset...')
    full_df  = build_dataset()
    train_df = full_df[full_df['date'] < WC26_START].reset_index(drop=True)
    wc26_df  = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC26_START) &
        (full_df['date'] <= WC26_END)
    ].reset_index(drop=True)

    print(f'Training CatBoost (draw_weight={CB_DRAW_WEIGHT})...')
    model = CatBoostPredictor(draw_weight=CB_DRAW_WEIGHT)
    model.fit(train_df)

    # ── Group stage probs ─────────────────────────────────────────────────────
    print('Pre-computing group stage probabilities...')
    group_probs   = _precompute_group_probs(wc26_df, model)
    win_prob_sums = _precompute_win_prob_sums(group_probs)

    # Prepare vectorised sampling structures
    match_list = list(group_probs.keys())
    probs_arr  = np.array([group_probs[k] for k in match_list])   # (48, 3)

    # ── Pairwise KO probs ─────────────────────────────────────────────────────
    profiles = _build_profiles()
    ko_probs = _precompute_ko_probs(model, profiles)

    # ── Bracket allocation table ──────────────────────────────────────────────
    combo_df    = pd.read_csv('simulation/third_place_combinations.csv')
    combo_index = _build_combo_index(combo_df)
    print(f'Combination index built: {len(combo_index)} entries (expect 495)')

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    rng          = np.random.default_rng(seed=42)
    title_counts = defaultdict(int)
    skipped      = 0

    print(f'\nRunning {n_sims:,} Monte Carlo simulations...')
    for _ in tqdm(range(n_sims)):
        outcomes  = _sample_group_stage(match_list, probs_arr, rng)
        standings = _compute_standings(outcomes, win_prob_sums)
        best3     = _pick_best_thirds(standings, win_prob_sums)
        thirds    = _allocate_thirds(best3, combo_index)

        if not thirds:
            skipped += 1
            continue

        champion = _simulate_bracket(standings, thirds, ko_probs, rng)
        if champion:
            title_counts[champion] += 1

    effective = n_sims - skipped
    if skipped:
        print(f'  Skipped {skipped} runs (bracket allocation not found)')

    # ── Results ───────────────────────────────────────────────────────────────
    rows = []
    for team in ALL_TEAMS:
        cnt = title_counts[team]
        rows.append({
            'team':        team,
            'group':       TEAM_TO_GRP[team],
            'title_prob':  round(cnt / effective, 4),
            'title_pct':   f'{cnt / effective:.1%}',
            'title_count': cnt,
        })
    df = (pd.DataFrame(rows)
            .sort_values('title_prob', ascending=False)
            .reset_index(drop=True))
    df.index += 1

    print(f'\n{"="*56}')
    print(f'  WC 2026 Title Probabilities  ({effective:,} simulations)')
    print(f'{"="*56}')
    for i, r in df.iterrows():
        bar = '#' * int(r['title_prob'] * 200)
        print(f"  {i:>2}. {r['team']:<30} {r['title_pct']:>6}  {bar}")

    total = df['title_prob'].sum()
    print(f'\n  Sum of all probabilities: {total:.4f}')

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index_label='rank')
    print(f'  Saved: {OUTPUT_CSV}')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=N_SIMS,
                        help='Number of Monte Carlo simulations')
    args = parser.parse_args()
    run(n_sims=args.n)
