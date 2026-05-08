"""
CatBoost classifier for football match outcome prediction.

Same design as XGBoost model (flip augmentation, symmetrized inference,
draw class weighting) with two differences:
  - CatBoost handles categorical features (confederation) natively —
    no one-hot encoding needed.
  - CatBoost is generally more robust to overfitting on tabular data
    and is the top-performing approach in the literature (Razali et al.).

Categorical features: confederation_home, confederation_away.
All other features are difference features (home - away).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

OUTCOME_ORDER  = ['home_win', 'draw', 'away_win']
OUTCOME_TO_INT = {'home_win': 0, 'draw': 1, 'away_win': 2}

NUMERIC_COLS = [
    'points_dif',
    'elo_diff',
    'pi_diff',
    'pww_ma20_diff',
    'pww_ma5_diff',
    'gw_ma20_diff',
    'gw_ma5_diff',
    'gsw_ma20_diff',
    'gsw_ma5_diff',
    'gd_ma20_diff',
    'gd_ma5_diff',
]

CATEGORICAL_COLS = [
    'confederation_home',
    'confederation_away',
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def _build_features(df):
    """
    Build feature DataFrame from raw match columns.
    Numeric features are difference (home - away).
    Categorical features are kept as-is (CatBoost handles encoding).
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

    feat['confederation_home'] = df['confederation_home'].fillna('Unknown')
    feat['confederation_away'] = df['confederation_away'].fillna('Unknown')

    return feat[FEATURE_COLS]


def _flip_features(X_df):
    """
    Return a flipped version of the feature DataFrame:
      - Negate all numeric diff features (equivalent to swapping teams)
      - Swap confederation_home <-> confederation_away
    """
    X_flip = X_df.copy()
    for col in NUMERIC_COLS:
        X_flip[col] = -X_df[col]
    X_flip['confederation_home'] = X_df['confederation_away']
    X_flip['confederation_away'] = X_df['confederation_home']
    return X_flip


def _augment(X_df, y):
    """
    Double the dataset with flipped matches.
    home_win (0) <-> away_win (2), draw (1) stays draw.
    """
    X_flip = _flip_features(X_df)
    y_flip = np.where(y == 0, 2, np.where(y == 2, 0, 1))
    X_aug = pd.concat([X_df, X_flip], ignore_index=True)
    y_aug = np.concatenate([y, y_flip])
    return X_aug, y_aug


class CatBoostPredictor:
    """
    CatBoost wrapper for 3-class football outcome prediction.

    Parameters
    ----------
    iterations : int
        Number of boosting rounds.
    learning_rate : float
    depth : int
        Tree depth.
    draw_weight : float
        Multiplier on the draw inverse-frequency class weight.
        draw_weight=0.75 is the default, tuned to balance draw recall vs accuracy.
    random_seed : int
    """

    def __init__(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        draw_weight=0.7,
        random_seed=42,
    ):
        self.iterations   = iterations
        self.learning_rate = learning_rate
        self.depth        = depth
        self.draw_weight  = draw_weight
        self.random_seed  = random_seed
        self.model        = None
        self._fitted      = False

    def fit(self, df):
        """
        Fit using all matches, augmented with flipped versions.
        Class weights correct for draw underrepresentation.

        Parameters
        ----------
        df : DataFrame with raw feature columns and home_score/away_score.
        """
        X = _build_features(df)
        y = df.apply(
            lambda r: 0 if r['home_score'] > r['away_score']
                      else (1 if r['home_score'] == r['away_score'] else 2),
            axis=1
        ).values

        # Drop rows missing any numeric feature
        numeric_mask = X[NUMERIC_COLS].notna().all(axis=1)
        X_clean = X[numeric_mask].reset_index(drop=True)
        y_clean = y[numeric_mask]

        X_aug, y_aug = _augment(X_clean, y_clean)

        # Class weights
        class_counts = np.bincount(y_aug, minlength=3)
        class_weights = len(y_aug) / (3 * class_counts)
        class_weights[1] *= self.draw_weight
        sample_weights = class_weights[y_aug]

        print(f"Training rows before augmentation : {numeric_mask.sum()} ({(~numeric_mask).sum()} dropped)")
        print(f"Training rows after augmentation  : {len(X_aug)}")
        print(f"Class distribution — home_win: {class_counts[0]}  draw: {class_counts[1]}  away_win: {class_counts[2]}")
        print(f"Class weights      — home_win: {class_weights[0]:.3f}  draw: {class_weights[1]:.3f}  away_win: {class_weights[2]:.3f}  (draw_weight={self.draw_weight})")

        cat_indices = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function='MultiClass',
            classes_count=3,
            random_seed=self.random_seed,
            cat_features=cat_indices,
            verbose=0,
            train_dir=None,
        )
        self.model.fit(X_aug, y_aug, sample_weight=sample_weights)
        self._fitted = True

    def predict_proba_row(self, row):
        """
        Symmetrized prediction: average forward and flipped inference.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_fwd = _build_features(pd.DataFrame([row])).reset_index(drop=True)

        # Fill numeric NaNs with 0 for sparse teams
        for col in NUMERIC_COLS:
            X_fwd[col] = X_fwd[col].fillna(0)

        X_inv = _flip_features(X_fwd)

        p_fwd = self.model.predict_proba(X_fwd)[0]
        p_inv = self.model.predict_proba(X_inv)[0]

        # Swap home_win <-> away_win for inverted prediction, then average
        p_hw = (p_fwd[0] + p_inv[2]) / 2
        p_d  = (p_fwd[1] + p_inv[1]) / 2
        p_aw = (p_fwd[2] + p_inv[0]) / 2

        return {'home_win': float(p_hw), 'draw': float(p_d), 'away_win': float(p_aw)}

    def feature_importance(self):
        """Return feature importances as a sorted DataFrame."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        scores = self.model.get_feature_importance()
        return (
            pd.DataFrame({'feature': FEATURE_COLS, 'importance': scores})
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )
