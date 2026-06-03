"""
Ordered Logit model for football match outcome prediction.

Models the three outcomes (away_win < draw < home_win) as thresholds
on a latent continuous variable: the "strength gap" between teams.

    z = b1*elo_diff + b2*pi_diff + b3*points_dif

    P(away_win) = sigmoid(t1 - z)
    P(draw)     = sigmoid(t2 - z) - sigmoid(t1 - z)
    P(home_win) = 1 - sigmoid(t2 - z)

Only rating features are used — no form MAs, no confederation.
This is a clean statistical baseline measuring how much pure team
quality explains without any additional feature engineering.

Symmetrized inference is applied at prediction time (same as XGBoost
and CatBoost): predict forward and with teams swapped, then average.
"""

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

OUTCOME_ORDER  = ['home_win', 'draw', 'away_win']

# Rating-only features — the "fences" model
# mv_sum_diff: NaN rows are dropped (statsmodels has no native NaN handling);
# at prediction time fillna(0) is applied (neutral: equal market values).
FEATURE_COLS = ['elo_diff', 'points_dif', 'mv_sum_diff']


def _build_features(df):
    feat = pd.DataFrame(index=df.index)
    feat['elo_diff']    = df['elo_diff']
    feat['points_dif']  = df['points_dif']
    feat['mv_sum_diff'] = (df['home_mv_top20_sum'] - df['away_mv_top20_sum']) / 1e6  # expressed in €M
    return feat[FEATURE_COLS]


class OrderedLogitPredictor:
    """
    Ordered Logit wrapper for 3-class football outcome prediction.

    Outcome encoding for the ordered model:
        0 = away_win  (lowest on the latent scale)
        1 = draw
        2 = home_win  (highest on the latent scale)
    """

    def __init__(self):
        self.result = None
        self._fitted = False

    def fit(self, df):
        """
        Fit on all matches with complete rating features.

        Parameters
        ----------
        df : DataFrame with raw feature columns and home_score/away_score.
        """
        X = _build_features(df)
        y = df.apply(
            lambda r: 2 if r['home_score'] > r['away_score']
                      else (1 if r['home_score'] == r['away_score'] else 0),
            axis=1
        )

        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask].reset_index(drop=True)
        y_clean = pd.Categorical(y[mask].values, categories=[0, 1, 2], ordered=True)

        print(f"Training on {mask.sum()} rows ({(~mask).sum()} dropped due to NaN)")
        print(f"Outcome distribution — away_win: {(y_clean==0).sum()}  draw: {(y_clean==1).sum()}  home_win: {(y_clean==2).sum()}")

        model = OrderedModel(y_clean, X_clean, distr='logit')
        self.result = model.fit(method='bfgs', disp=False)
        self._fitted = True

        # Print fitted coefficients and thresholds
        print("\nFitted coefficients:")
        for name, coef in zip(FEATURE_COLS, self.result.params[:len(FEATURE_COLS)]):
            print(f"  {name:<14}  coef = {coef:+.4f}")
        thresholds = self.result.params[len(FEATURE_COLS):].values
        print(f"  threshold t1   = {thresholds[0]:+.4f}  (away_win | draw boundary)")
        print(f"  threshold t2   = {thresholds[1]:+.4f}  (draw | home_win boundary)")

    def predict_proba_row(self, row):
        """
        Symmetrized prediction: average forward and flipped inference.
        Flipping = negate all features (swap the two teams).

        Returns dict with keys 'home_win', 'draw', 'away_win'.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_fwd = _build_features(pd.DataFrame([row])).fillna(0)
        X_inv = -X_fwd

        # OrderedModel predicts [P(cat=0), P(cat=1), P(cat=2)]
        # = [P(away_win), P(draw), P(home_win)]
        p_fwd = self.result.model.predict(self.result.params, exog=X_fwd.values)[0]
        p_inv = self.result.model.predict(self.result.params, exog=X_inv.values)[0]

        # p_fwd: [aw, d, hw] with A as home
        # p_inv: [aw, d, hw] with B as home → swap aw ↔ hw
        p_hw = (p_fwd[2] + p_inv[0]) / 2
        p_d  = (p_fwd[1] + p_inv[1]) / 2
        p_aw = (p_fwd[0] + p_inv[2]) / 2

        return {'home_win': float(p_hw), 'draw': float(p_d), 'away_win': float(p_aw)}
