"""
SHAP analysis for ML-Poisson goal regressor.

Explains WHY the model predicts a specific lambda (expected goals) for a given
match, breaking down the contribution of each feature to the prediction.
Run for Qatar vs Ecuador to diagnose the home_win prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import shap

from models.ml_poisson.model import MLPoissonModel
from models.ml_poisson.regressor import _build_attacker_features, FEATURE_COLS

WC_2022_START = pd.Timestamp('2022-11-20')


def explain_match(full_df, train_df, home_team, away_team, match_date=None):
    """
    Train the model and run SHAP on a single match row.
    Shows feature contributions to lambda_home and lambda_away.
    """
    # Train
    model = MLPoissonModel(rho=-0.13)
    model.fit(train_df)
    xgb = model.regressor.model  # underlying XGBRegressor

    # Build SHAP explainer on training data (sample for speed)
    print("Building SHAP explainer on training data...")
    X_home_train = _build_attacker_features(train_df, attacker='home').dropna()
    X_away_train = _build_attacker_features(train_df, attacker='away').dropna()
    X_train_all  = pd.concat([X_home_train, X_away_train], ignore_index=True)
    explainer = shap.TreeExplainer(xgb, data=X_train_all[FEATURE_COLS])

    # Find match row
    mask = (full_df['home_team'] == home_team) & (full_df['away_team'] == away_team)
    if match_date:
        mask &= full_df['date'] == pd.Timestamp(match_date)
    row = full_df[mask].iloc[0]

    lam, mu = model.predict_lambdas(row)
    probs   = model.predict_proba_row(row)
    actual_hw = int(row['home_score'])
    actual_aw = int(row['away_score'])

    print(f"\n{'='*65}")
    print(f"  {home_team} vs {away_team}  ({row['date'].date()})")
    print(f"  Actual: {actual_hw}-{actual_aw}")
    print(f"  lambda_home={lam:.3f}  lambda_away={mu:.3f}")
    print(f"  P(HW)={probs['home_win']:.3f}  P(D)={probs['draw']:.3f}  P(AW)={probs['away_win']:.3f}")
    print(f"{'='*65}")

    # SHAP for each perspective
    for attacker, label in [('home', f'lambda_home  [{home_team} attacks]'),
                             ('away', f'lambda_away  [{away_team} attacks]')]:
        X_row = _build_attacker_features(pd.DataFrame([row]), attacker=attacker)
        X_row_filled = X_row[FEATURE_COLS].fillna(0)

        sv = explainer(X_row_filled)
        base_log = float(sv.base_values[0])
        shaps    = sv.values[0]
        pred_log = base_log + shaps.sum()
        pred_lam = np.exp(pred_log)   # back to goal space

        print(f"\n  {label}  =>  predicted lambda = {pred_lam:.3f} goals")
        print(f"  {'Feature':<22}  {'Raw value':>10}  {'SHAP (log)':>11}  {'Goals impact':>13}  Direction")
        print(f"  {'-'*75}")

        # Sort by absolute SHAP value
        order = np.argsort(np.abs(shaps))[::-1]
        for i in order:
            feat    = FEATURE_COLS[i]
            val     = float(X_row_filled.iloc[0, i])
            sv_i    = shaps[i]
            # Approximate goals impact: exp(base + shap) - exp(base)
            goals_impact = np.exp(base_log + sv_i) - np.exp(base_log)
            arrow   = "UP  ^" if sv_i > 0 else "DOWN v"
            print(f"  {feat:<22}  {val:>10.4f}  {sv_i:>+11.4f}  {goals_impact:>+13.4f}  {arrow}")

        print(f"  {'-'*75}")
        print(f"  Base (avg prediction):  {np.exp(base_log):.3f} goals")
        print(f"  Final predicted lambda: {pred_lam:.3f} goals")


def run():
    print("Loading data...")
    full_df  = pd.read_csv("data/ranked_database_with_features.csv")
    full_df['date'] = pd.to_datetime(full_df['date'])

    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)

    explain_match(full_df, train_df, 'Qatar', 'Ecuador', match_date='2022-11-20')


if __name__ == "__main__":
    run()
