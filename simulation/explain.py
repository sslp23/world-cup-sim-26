"""
SHAP explanation for a WC 2026 group stage prediction.

Breaks down the CatBoost (draw_weight=0.72) prediction for a given match,
showing which features drove the model's output.

Parameters (edit below):
    HOME_TEAM  : home team name as it appears in the dataset
    AWAY_TEAM  : away team name
    MATCH_DATE : match date string (optional, for disambiguation)

Run from the project root:
    python -m simulation.explain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from catboost import Pool

from models.catboost.model import CatBoostPredictor, _build_features, FEATURE_COLS as CB_COLS
from simulation.dataset    import build as build_dataset

# ── Match to explain ──────────────────────────────────────────────────────────
HOME_TEAM  = 'Colombia'
AWAY_TEAM  = 'Portugal'
MATCH_DATE = '2026-06-27'    # set to None if not needed for disambiguation
# ─────────────────────────────────────────────────────────────────────────────

WC26_START = pd.Timestamp('2026-06-11')
WC26_END   = pd.Timestamp('2026-06-27')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


def _header(title, width=70):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def explain_catboost(model, row):
    _header("CatBoost  (draw_weight=0.65)")

    row_df = pd.DataFrame([row])
    X_row  = _build_features(row_df)[CB_COLS].fillna(0)

    probs = model.predict_proba_row(row)
    pred  = max(probs, key=probs.get)
    print(f"  P(HW)={probs['home_win']:.3f}  P(D)={probs['draw']:.3f}  P(AW)={probs['away_win']:.3f}  => {pred}")

    # CatBoost native SHAP — returns (n_samples, n_classes, n_features+1)
    # Axis 1 = class, axis 2 = features + bias (last column).
    cat_cols  = ['confederation_home', 'confederation_away']
    pool      = Pool(X_row, cat_features=cat_cols)
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

    # Find match
    mask = (wc26_df['home_team'] == HOME_TEAM) & (wc26_df['away_team'] == AWAY_TEAM)
    if MATCH_DATE:
        mask &= wc26_df['date'] == pd.Timestamp(MATCH_DATE)
    if not mask.any():
        print(f"Match not found: {HOME_TEAM} vs {AWAY_TEAM}")
        return
    row = wc26_df[mask].iloc[0]

    print(f"\n{'#'*70}")
    print(f"  SHAP EXPLANATION: {HOME_TEAM} vs {AWAY_TEAM}  ({row['date'].date()})")
    print(f"{'#'*70}")

    print("\nTraining CatBoost (draw_weight=0.65)...")
    cb = CatBoostPredictor(draw_weight=0.65)
    cb.fit(train_df)

    explain_catboost(cb, row)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    run()
