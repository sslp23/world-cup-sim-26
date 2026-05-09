"""
ML-Poisson hybrid model for football match outcome prediction.

Stage 1 — GoalRegressor predicts expected goals per team:
  lambda_home = regressor(home attack features, away defense features)
  lambda_away = regressor(away attack features, home defense features)

Stage 2 — Dixon-Coles score matrix converts lambdas to outcome probabilities:
  P(home_goals=i, away_goals=j) = tau(i,j) * Poisson(i|lambda_home) * Poisson(j|lambda_away)
  where tau is the low-score correction that fixes Poisson's underestimation
  of 0-0 and 1-0/0-1 scorelines.

Symmetrized inference: each match is predicted forward and with teams swapped,
then averaged — removes residual label bias at neutral venues.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from models.ml_poisson.regressor import GoalRegressor

OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


def _tau_scalar(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction (scalar version)."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _effective_rho(lam, mu, rho_max):
    """
    Match-conditional rho: scales draw correction by how balanced the match is.

    When lam == mu (perfectly even), balance = 1.0 -> rho_eff = rho_max.
    When one team dominates (lam >> mu), balance -> 0 -> rho_eff -> 0.

    balance = 1 - |lam - mu| / (lam + mu)
    """
    balance = 1.0 - abs(lam - mu) / (lam + mu)
    return rho_max * balance


def _score_matrix(lam, mu, rho_max, max_goals=10):
    """
    Build (max_goals+1 x max_goals+1) score probability matrix.
    M[i, j] = P(home_goals=i, away_goals=j)

    rho_max is the maximum Dixon-Coles correction, applied at full strength
    only when both teams have equal expected goals (balance = 1).
    """
    rho = _effective_rho(lam, mu, rho_max)
    M = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = _tau_scalar(i, j, lam, mu, rho)
            M[i, j] = tau * poisson.pmf(i, lam) * poisson.pmf(j, mu)
    M = np.clip(M, 0, None)
    M /= M.sum()
    return M


def _matrix_to_probs(M):
    home_win = float(np.sum(np.tril(M, -1)))
    draw     = float(np.sum(np.diag(M)))
    away_win = float(np.sum(np.triu(M, 1)))
    return {'home_win': home_win, 'draw': draw, 'away_win': away_win}


class MLPoissonModel:
    """
    ML-Poisson hybrid: XGBoost goal regressors + Dixon-Coles score matrix.

    Parameters
    ----------
    rho : float
        Maximum Dixon-Coles low-score correction (applied when both teams
        have equal expected goals). Scales down to 0 as the match becomes
        more lopsided. Negative values boost 0-0 and 1-1 draw probabilities.
    regressor_kwargs : dict
        Passed to GoalRegressor constructor.
    """

    def __init__(self, rho=-0.13, **regressor_kwargs):
        self.rho       = rho
        self.regressor = GoalRegressor(**regressor_kwargs)
        self._fitted   = False

    def fit(self, df):
        """
        Fit the goal regressor on all training matches.

        Parameters
        ----------
        df : DataFrame with raw feature columns, home_score, away_score.
        """
        self.regressor.fit(df)
        self._fitted = True

    def predict_proba_row(self, row, max_goals=10):
        """
        Predict outcome probabilities for a single match using symmetrized inference.

        Forward:  lambda_home from home-attacker perspective,
                  lambda_away from away-attacker perspective.
        Inverted: swap teams (negate attacker perspective).
        Average both score matrices, then sum to outcome probabilities.

        Parameters
        ----------
        row : Series with raw feature columns.
        max_goals : int

        Returns
        -------
        dict with keys 'home_win', 'draw', 'away_win'.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Forward prediction
        lam_fwd = self.regressor.predict_lambda(row, attacker='home')
        mu_fwd  = self.regressor.predict_lambda(row, attacker='away')
        M_fwd   = _score_matrix(lam_fwd, mu_fwd, self.rho, max_goals)

        # Inverted prediction: swap home/away by negating attacker perspective
        lam_inv = self.regressor.predict_lambda(row, attacker='away')
        mu_inv  = self.regressor.predict_lambda(row, attacker='home')
        # Inverted matrix has axes flipped: M_inv[i,j] = P(away_goals=i, home_goals=j)
        # Transpose to restore M[i,j] = P(home_goals=i, away_goals=j)
        M_inv   = _score_matrix(lam_inv, mu_inv, self.rho, max_goals).T

        M_avg = (M_fwd + M_inv) / 2
        return _matrix_to_probs(M_avg)

    def predict_lambdas(self, row):
        """Return (lambda_home, lambda_away) without symmetrization — for inspection."""
        lam = self.regressor.predict_lambda(row, attacker='home')
        mu  = self.regressor.predict_lambda(row, attacker='away')
        return lam, mu

    def feature_importance(self):
        return self.regressor.feature_importance()
