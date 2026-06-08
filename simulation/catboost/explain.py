"""
SHAP explanation for a WC 2026 group stage prediction.

Breaks down the CatBoost prediction for a given match (draw_weight from config.py),
showing which features drove the model's output.

Parameters (edit below):
    HOME_TEAM  : home team name as it appears in the dataset
    AWAY_TEAM  : away team name
    MATCH_DATE : match date string (optional, for disambiguation)

Run from the project root:
    python -m simulation.catboost.explain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from catboost import Pool

from models.catboost.model import CatBoostPredictor, _build_features, FEATURE_COLS as CB_COLS
from simulation.dataset    import build as build_dataset
from simulation.catboost.playoff import build_profiles
from config import CB_DRAW_WEIGHT

# ── Match to explain ──────────────────────────────────────────────────────────
HOME_TEAM  = 'Spain'
AWAY_TEAM  = 'France'
MATCH_DATE = None    # set to a date string to disambiguate group stage matches
# ─────────────────────────────────────────────────────────────────────────────

WC26_START = pd.Timestamp('2026-06-11')
WC26_END   = pd.Timestamp('2026-06-27')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


def _header(title, width=70):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def explain_catboost(model, row):
    _header(f"CatBoost  (draw_weight={CB_DRAW_WEIGHT})")

    from models.catboost.model import CATEGORICAL_COLS, NUMERIC_COLS
    row_df = pd.DataFrame([row])
    X_row  = _build_features(row_df)[CB_COLS]
    for col in NUMERIC_COLS:
        X_row[col] = X_row[col].fillna(0)

    probs = model.predict_proba_row(row)
    pred  = max(probs, key=probs.get)
    print(f"  P(HW)={probs['home_win']:.3f}  P(D)={probs['draw']:.3f}  P(AW)={probs['away_win']:.3f}  => {pred}")

    # CatBoost native SHAP — returns (n_samples, n_classes, n_features+1)
    cat_indices = [CB_COLS.index(c) for c in CATEGORICAL_COLS]
    pool      = Pool(X_row, cat_features=cat_indices)
    shap_vals = model.model.get_feature_importance(pool, type='ShapValues')

    class_names = ['home_win', 'draw', 'away_win']
    for ci, cls in enumerate(class_names):
        shaps = shap_vals[0, ci, :-1]   # (n_features,)
        bias  = shap_vals[0, ci, -1]    # scalar
        order = np.argsort(np.abs(shaps))[::-1]
        print(f"\n  Class: {cls}  (bias={bias:+.4f})")
        print(f"  {'Feature':<24}  {'Value':>12}  {'SHAP':>10}  Direction")
        print(f"  {'-'*62}")
        for i in order:
            feat  = CB_COLS[i]
            val   = str(X_row.iloc[0, i])
            sv_i  = shaps[i]
            arrow = "UP  ^" if sv_i > 0 else "DOWN v"
            print(f"  {feat:<24}  {val:>12}  {sv_i:>+10.4f}  {arrow}")


def run():
    full_df  = build_dataset()
    train_df = full_df[full_df['date'] < WC26_START].reset_index(drop=True)
    wc26_df  = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC26_START) &
        (full_df['date'] <= WC26_END)
    ].copy().reset_index(drop=True)

    # Find match in WC26 group stage schedule
    mask = (wc26_df['home_team'] == HOME_TEAM) & (wc26_df['away_team'] == AWAY_TEAM)
    if MATCH_DATE:
        mask &= wc26_df['date'] == pd.Timestamp(MATCH_DATE)

    if mask.any():
        row = wc26_df[mask].iloc[0]
    else:
        # Match not in group stage — build synthetic row from team profiles
        # (useful for hypothetical knockout matchups)
        print(f"  '{HOME_TEAM} vs {AWAY_TEAM}' not in WC26 schedule — "
              f"building synthetic row from team profiles.")
        profiles = build_profiles()
        pa = profiles.get(HOME_TEAM, {})
        pb = profiles.get(AWAY_TEAM, {})
        if not pa or not pb:
            print(f"  Profile not found for one or both teams. Check team names.")
            return
        row = pd.Series({
            'home_team': HOME_TEAM, 'away_team': AWAY_TEAM,
            'date': pd.Timestamp('2026-07-01'),
            'elo_diff':                           pa.get('elo', 1500) - pb.get('elo', 1500),
            'home_elo':                           pa.get('elo', 1500),
            'away_elo':                           pb.get('elo', 1500),
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
            'confederation_home':                 pa.get('confederation', 'Unknown'),
            'confederation_away':                 pb.get('confederation', 'Unknown'),
        })

    print(f"\n{'#'*70}")
    print(f"  SHAP EXPLANATION: {HOME_TEAM} vs {AWAY_TEAM}  ({row['date'].date()})")
    print(f"{'#'*70}")

    print(f"\nTraining CatBoost (draw_weight={CB_DRAW_WEIGHT})...")
    cb = CatBoostPredictor(draw_weight=CB_DRAW_WEIGHT)
    cb.fit(train_df)

    explain_catboost(cb, row)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    run()
