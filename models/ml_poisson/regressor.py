"""
XGBoost regressor for predicting goals scored by a team.

A single model is trained from the "attacker's perspective":
  - Attacker features: their weighted goals scored MAs and weighted points won MA
  - Defender features: opponent's weighted goals suffered MAs and goal diff MA
  - Rating: elo_diff and pi_neutral_diff from the attacker's perspective (positive = attacker is stronger)
            pi_neutral_diff = avg(pi_h, pi_a) of attacker minus avg(pi_h, pi_a) of defender,
            removing home/away context bias that is meaningless at neutral WC venues.

Symmetry: to predict home_goals, use home as attacker.
           to predict away_goals, use away as attacker (negate rating signs).
This single-model approach guarantees consistency and avoids training two
separate models that could diverge.

Selected features from eda/regressor_README.md.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURE_COLS = [
    # Rating from attacker's perspective (positive = attacker stronger)
    # points_dif excluded: Pearson r=0.92 with elo_diff, partial r=-0.05 — see eda/collinearity_points_elo.py
    'elo_diff',
    # Absolute quality gap — same regardless of attacker perspective
    'abs_elo_diff',
    # WC history — attacker advantage in tournament experience
    'wc_games_diff',
    # Attacker's recent offensive output (weighted by opponent strength)
    'att_gw_ma20',
    'att_gw_ma5',
    'att_pww_ma20',
    # Defender's recent defensive record (weighted by opponent strength)
    'def_gsw_ma20',
    'def_gsw_ma5',
    'def_gd_ma20',
]


def _build_attacker_features(df, attacker='home'):
    """
    Build features from the attacker's perspective.

    attacker='home': home team attacks, away team defends.
    attacker='away': away team attacks, home team defends (negate ratings).
    abs_elo_diff is the same in both perspectives — it captures mismatch magnitude.
    """
    feat = pd.DataFrame(index=df.index)

    if attacker == 'home':
        feat['elo_diff']           = df['elo_diff']
        feat['abs_elo_diff']       = df['elo_diff'].abs()
        feat['wc_games_diff']      = df['home_wc_games'] - df['away_wc_games']
        feat['att_gw_ma20']        = df['home_goals_weighted_ma_20']
        feat['att_gw_ma5']         = df['home_goals_weighted_ma_5']
        feat['att_pww_ma20']       = df['home_points_weighted_ma_20']
        feat['def_gsw_ma20']       = df['away_goals_suffered_weighted_ma_20']
        feat['def_gsw_ma5']        = df['away_goals_suffered_weighted_ma_5']
        feat['def_gd_ma20']        = df['away_goal_diff_ma_20']
    else:  # away team attacks
        feat['elo_diff']           = -df['elo_diff']
        feat['abs_elo_diff']       = df['elo_diff'].abs()
        feat['wc_games_diff']      = df['away_wc_games'] - df['home_wc_games']
        feat['att_gw_ma20']        = df['away_goals_weighted_ma_20']
        feat['att_gw_ma5']         = df['away_goals_weighted_ma_5']
        feat['att_pww_ma20']       = df['away_points_weighted_ma_20']
        feat['def_gsw_ma20']       = df['home_goals_suffered_weighted_ma_20']
        feat['def_gsw_ma5']        = df['home_goals_suffered_weighted_ma_5']
        feat['def_gd_ma20']        = df['home_goal_diff_ma_20']

    return feat[FEATURE_COLS]


class GoalRegressor:
    """
    XGBoost regressor trained to predict goals scored by the attacking team.

    Trained on all matches pooling both home and away observations,
    giving 2× the training data and enforcing symmetric learning.

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    max_depth : int
    subsample : float
    colsample_bytree : float
    random_state : int
    """

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='count:poisson',  # Poisson regression — correct for count data
            verbosity=0,
        )
        self._fitted = False

    def fit(self, df):
        """
        Train on all matches, pooling home and away observations.

        Each match contributes two rows:
          - Home team as attacker, target = home_score
          - Away team as attacker, target = away_score

        Parameters
        ----------
        df : DataFrame with raw feature columns, home_score, away_score.
        """
        X_home = _build_attacker_features(df, attacker='home')
        y_home = df['home_score'].values

        X_away = _build_attacker_features(df, attacker='away')
        y_away = df['away_score'].values

        X_home['_y'] = y_home
        X_away['_y'] = y_away
        combined = pd.concat([X_home, X_away], ignore_index=True)
        combined = combined.dropna()

        X_clean = combined[FEATURE_COLS].values
        y_clean = combined['_y'].values

        print(f"Training on {len(X_clean)} rows ({len(df)*2 - len(X_clean)} dropped due to NaN)")
        self.model.fit(X_clean, y_clean)
        self._fitted = True

    def predict_lambda(self, row, attacker='home'):
        """
        Predict expected goals (lambda) for the attacking team.

        Parameters
        ----------
        row : Series with raw feature columns.
        attacker : 'home' or 'away'

        Returns
        -------
        float — predicted expected goals (always >= 0)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = _build_attacker_features(pd.DataFrame([row]), attacker=attacker)
        X = X.fillna(0)
        lam = float(self.model.predict(X.values)[0])
        return max(lam, 0.01)  # clip to avoid degenerate Poisson

    def feature_importance(self):
        """Return feature importances as a sorted DataFrame."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        scores = self.model.feature_importances_
        return (
            pd.DataFrame({'feature': FEATURE_COLS, 'importance': scores})
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )
