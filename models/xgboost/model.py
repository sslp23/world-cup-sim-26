"""
XGBoost classifier for football match outcome prediction.

Predicts 3-class probabilities: home_win / draw / away_win.

Design choices:
  - Augmented training: every match is included twice — once forward (A as home)
    and once flipped (B as home, all diff features negated, outcome swapped).
    This forces the model to learn quality differences symmetrically, making
    it neutral-venue aware without discarding any training data.
  - Class-weighted training: upweights draws to correct for class imbalance.
  - Symmetrized inference: each match is predicted forward and inverted, then
    averaged. Eliminates any residual bias from arbitrary home/away labelling.

All features are difference features (home team value − away team value).
Negating them is equivalent to swapping the two teams.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Ordered for RPS calculation: home_win → draw → away_win
OUTCOME_ORDER  = ['home_win', 'draw', 'away_win']
OUTCOME_TO_INT = {'home_win': 0, 'draw': 1, 'away_win': 2}

# Selected features from EDA (eda/README.md).
# neutral is excluded: augmentation + symmetrized inference make the model
# venue-agnostic without needing an explicit flag.
FEATURE_COLS = [
    # Ratings
    'points_dif',
    'elo_diff',
    'pi_diff',
    # Weighted points won — long and short window
    'pww_ma20_diff',
    'pww_ma5_diff',
    # Weighted goals scored — long and short window
    'gw_ma20_diff',
    'gw_ma5_diff',
    # Weighted goals suffered — long and short window
    'gsw_ma20_diff',
    'gsw_ma5_diff',
    # Goal difference — long and short window
    'gd_ma20_diff',
    'gd_ma5_diff',
]


def _build_diff_features(df):
    """
    Construct all difference features from the raw home/away columns.
    Returns a DataFrame with FEATURE_COLS as columns.
    """
    feat = pd.DataFrame(index=df.index)

    feat['points_dif']    = df['points_dif']
    feat['elo_diff']      = df['elo_diff']
    feat['pi_diff']       = df['pi_diff']

    feat['pww_ma20_diff'] = df['home_points_weighted_ma_20'] - df['away_points_weighted_ma_20']
    feat['pww_ma5_diff']  = df['home_points_weighted_ma_5']  - df['away_points_weighted_ma_5']

    feat['gw_ma20_diff']  = df['home_goals_weighted_ma_20']  - df['away_goals_weighted_ma_20']
    feat['gw_ma5_diff']   = df['home_goals_weighted_ma_5']   - df['away_goals_weighted_ma_5']

    feat['gsw_ma20_diff'] = df['home_goals_suffered_weighted_ma_20'] - df['away_goals_suffered_weighted_ma_20']
    feat['gsw_ma5_diff']  = df['home_goals_suffered_weighted_ma_5']  - df['away_goals_suffered_weighted_ma_5']

    feat['gd_ma20_diff']  = df['home_goal_diff_ma_20'] - df['away_goal_diff_ma_20']
    feat['gd_ma5_diff']   = df['home_goal_diff_ma_5']  - df['away_goal_diff_ma_5']

    return feat[FEATURE_COLS]


def _augment(X, y):
    """
    Double the dataset by adding a flipped version of every match:
      - Negate all diff features (equivalent to swapping home/away teams)
      - Swap home_win (0) ↔ away_win (2); draw (1) stays draw

    Returns (X_aug, y_aug) with 2× the original rows.
    """
    X_flip = -X
    y_flip = np.where(y == 0, 2, np.where(y == 2, 0, 1))
    return np.vstack([X, X_flip]), np.concatenate([y, y_flip])


class XGBoostPredictor:
    """
    XGBoost wrapper for 3-class football outcome prediction.

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    max_depth : int
    subsample : float
    colsample_bytree : float
    random_state : int
    draw_weight : float
        Multiplier applied to draw sample weights on top of the inverse-frequency
        base weight. 1.0 = pure inverse frequency (full compensation for imbalance).
        Values below 1.0 soften the draw boost, reducing over-prediction of draws
        at the cost of slightly lower draw recall.
    """

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        draw_weight=0.75,
    ):
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='mlogloss',
            verbosity=0,
        )
        self.draw_weight = draw_weight
        self._fitted = False

    def fit(self, df):
        """
        Fit using all matches, augmented with flipped versions of each match
        and class weights to correct for draw underrepresentation.

        Parameters
        ----------
        df : DataFrame with all raw feature columns and home_score/away_score.
        """
        X = _build_diff_features(df)
        y = df.apply(
            lambda r: 0 if r['home_score'] > r['away_score']
                      else (1 if r['home_score'] == r['away_score'] else 2),
            axis=1
        ).values

        mask = X.notna().all(axis=1)
        X_clean = X[mask].values
        y_clean = y[mask]

        # Augment: add flipped version of every match
        X_aug, y_aug = _augment(X_clean, y_clean)

        # Class weights: inversely proportional to class frequency, with an
        # additional draw_weight multiplier to control how aggressively draws
        # are upweighted. draw_weight=1.0 = full inverse-frequency compensation.
        class_counts = np.bincount(y_aug, minlength=3)
        class_weights = len(y_aug) / (3 * class_counts)
        class_weights[1] *= self.draw_weight
        sample_weights = class_weights[y_aug]

        print(f"Training rows before augmentation : {mask.sum()} ({(~mask).sum()} dropped due to NaN)")
        print(f"Training rows after augmentation  : {len(X_aug)}")
        print(f"Class distribution — home_win: {class_counts[0]}  draw: {class_counts[1]}  away_win: {class_counts[2]}")
        print(f"Class weights      — home_win: {class_weights[0]:.3f}  draw: {class_weights[1]:.3f}  away_win: {class_weights[2]:.3f}  (draw_weight={self.draw_weight})")

        self.model.fit(X_aug, y_aug, sample_weight=sample_weights)
        self._fitted = True

    def predict_proba_row(self, row):
        """
        Predict outcome probabilities for a single match using symmetrized inference.

        Predicts forward (team A as home) and inverted (team B as home, achieved
        by negating all diff features), then averages. Eliminates residual label bias.

        Parameters
        ----------
        row : Series with the raw feature columns.

        Returns
        -------
        dict with keys 'home_win', 'draw', 'away_win'.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_fwd = _build_diff_features(pd.DataFrame([row])).fillna(0)
        X_inv = -X_fwd  # negate all diffs = swap the two teams

        p_fwd = self.model.predict_proba(X_fwd.values)[0]  # [hw, d, aw] with A as home
        p_inv = self.model.predict_proba(X_inv.values)[0]  # [hw, d, aw] with B as home

        # When teams are swapped: home_win ↔ away_win, draw stays draw
        p_hw = (p_fwd[0] + p_inv[2]) / 2
        p_d  = (p_fwd[1] + p_inv[1]) / 2
        p_aw = (p_fwd[2] + p_inv[0]) / 2

        return {'home_win': float(p_hw), 'draw': float(p_d), 'away_win': float(p_aw)}

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
