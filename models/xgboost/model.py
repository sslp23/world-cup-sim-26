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

# Signed difference features — negated when teams are swapped during augmentation.
MV_COLS = ['mv_sum_diff', 'mv_log_ratio']   # both NaN together — excluded from required mask

DIFF_COLS = [
    'elo_diff',
    'mv_sum_diff',         # Transfermarkt top-20 squad MV differential, home − away (€)
    'mv_log_ratio',        # log((home_mv+1)/(away_mv+1)) — relative squad strength ratio
    'wc_best_round_diff',  # best WC round ever reached (0=DNQ … 6=champion)
    'wc_gpg_diff',         # goals scored per game in WC history (0 if never qualified)
    'wc_games_diff',
    'pww_ma20_diff',
    'pww_ma5_diff',
    'gw_ma20_diff',
    'gw_ma5_diff',
    'gsw_ma20_diff',
    'gsw_ma5_diff',
    'gd_ma20_diff',
    'gd_ma5_diff',
]

# Absolute-value features — magnitude of quality gap, unchanged by team swap.
# points_dif excluded: Pearson r=0.92 with elo_diff, partial r=-0.05 after conditioning
# on elo_diff — near-zero independent signal. See eda/collinearity_points_elo.py.
ABS_COLS = ['abs_elo_diff']

FEATURE_COLS = DIFF_COLS + ABS_COLS

# mv_sum_diff is NaN for ~29% of training rows; XGBoost handles NaN natively via learned
# default directions, so those rows are kept. Only non-MV features trigger the NaN mask.
REQUIRED_DIFF_COLS = [c for c in DIFF_COLS if c not in MV_COLS]


def _build_features(df):
    """
    Construct all features from the raw home/away columns.
    Diff features are (home − away). Abs features capture mismatch magnitude.
    """
    feat = pd.DataFrame(index=df.index)

    feat['elo_diff']      = df['elo_diff']
    h_mv = df['home_mv_top20_sum']
    a_mv = df['away_mv_top20_sum']
    feat['mv_sum_diff']  = h_mv - a_mv                              # NaN kept intentionally
    feat['mv_log_ratio'] = np.log((h_mv + 1) / (a_mv + 1))         # NaN kept intentionally

    feat['wc_best_round_diff'] = df['home_wc_best_round']     - df['away_wc_best_round']
    feat['wc_gpg_diff']        = df['home_wc_goals_per_game'] - df['away_wc_goals_per_game']

    feat['wc_games_diff'] = df['home_wc_games'] - df['away_wc_games']

    feat['pww_ma20_diff'] = df['home_points_weighted_ma_20'] - df['away_points_weighted_ma_20']
    feat['pww_ma5_diff']  = df['home_points_weighted_ma_5']  - df['away_points_weighted_ma_5']

    feat['gw_ma20_diff']  = df['home_goals_weighted_ma_20']  - df['away_goals_weighted_ma_20']
    feat['gw_ma5_diff']   = df['home_goals_weighted_ma_5']   - df['away_goals_weighted_ma_5']

    feat['gsw_ma20_diff'] = df['home_goals_suffered_weighted_ma_20'] - df['away_goals_suffered_weighted_ma_20']
    feat['gsw_ma5_diff']  = df['home_goals_suffered_weighted_ma_5']  - df['away_goals_suffered_weighted_ma_5']

    feat['gd_ma20_diff']  = df['home_goal_diff_ma_20'] - df['away_goal_diff_ma_20']
    feat['gd_ma5_diff']   = df['home_goal_diff_ma_5']  - df['away_goal_diff_ma_5']

    feat['abs_elo_diff']   = df['elo_diff'].abs()

    return feat[FEATURE_COLS]


def _augment(X_df, y):
    """
    Double the dataset by adding a flipped version of every match:
      - Negate DIFF_COLS (equivalent to swapping home/away teams)
      - ABS_COLS are unchanged (mismatch magnitude is the same regardless of labelling)
      - Swap home_win (0) ↔ away_win (2); draw (1) stays draw

    Returns (X_aug, y_aug) as (DataFrame, ndarray) with 2× the original rows.
    """
    X_flip = X_df.copy()
    for col in DIFF_COLS:
        X_flip[col] = -X_df[col]
    y_flip = np.where(y == 0, 2, np.where(y == 2, 0, 1))
    X_aug  = pd.concat([X_df, X_flip], ignore_index=True)
    y_aug  = np.concatenate([y, y_flip])
    return X_aug, y_aug


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
        self._fitted     = False

    def fit(self, df):
        """
        Fit using all matches, augmented with flipped versions of each match
        and class weights to correct for draw underrepresentation.

        Parameters
        ----------
        df : DataFrame with all raw feature columns and home_score/away_score.
        """
        X = _build_features(df)
        y = df.apply(
            lambda r: 0 if r['home_score'] > r['away_score']
                      else (1 if r['home_score'] == r['away_score'] else 2),
            axis=1
        ).values

        mask = X[REQUIRED_DIFF_COLS].notna().all(axis=1)
        X_clean = X[mask].reset_index(drop=True)
        y_clean = y[mask]

        X_aug, y_aug = _augment(X_clean, y_clean)

        class_counts  = np.bincount(y_aug, minlength=3)
        class_weights = len(y_aug) / (3 * class_counts)
        class_weights[1] *= self.draw_weight
        sample_weights = class_weights[y_aug]

        print(f"Training rows before augmentation : {mask.sum()} ({(~mask).sum()} dropped due to NaN)")
        print(f"Training rows after augmentation  : {len(X_aug)}")
        print(f"Class distribution — home_win: {class_counts[0]}  draw: {class_counts[1]}  away_win: {class_counts[2]}")
        print(f"Class weights      — home_win: {class_weights[0]:.3f}  draw: {class_weights[1]:.3f}  away_win: {class_weights[2]:.3f}  (draw_weight={self.draw_weight})")

        self.model.fit(X_aug[FEATURE_COLS].values, y_aug, sample_weight=sample_weights)
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

        X_fwd = _build_features(pd.DataFrame([row]))[FEATURE_COLS]
        for col in REQUIRED_DIFF_COLS + ABS_COLS:
            X_fwd[col] = X_fwd[col].fillna(0)
        # mv_sum_diff left as NaN — XGBoost uses its trained default direction

        X_inv = X_fwd.copy()
        for col in DIFF_COLS:
            X_inv[col] = -X_fwd[col]
        # ABS_COLS stay the same

        p_fwd = self.model.predict_proba(X_fwd.values)[0]
        p_inv = self.model.predict_proba(X_inv.values)[0]

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
