"""
Dixon-Coles model for football score prediction.

Reference: Dixon, M. & Coles, S. (1997). Modelling Association Football Scores
and Inefficiencies in the Football Betting Market.

Key components:
  - Per-team attack (α) and defense (β) parameters, estimated via MLE.
  - Home advantage parameter (γ), skipped for neutral venues.
  - Low-score correction (ρ) that fixes Poisson's underestimation of 0-0 and 1-1.
  - Time decay: recent matches weighted more heavily (xi controls decay rate).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class DixonColes:
    def __init__(self, xi=0.0065):
        """
        Args:
            xi: Time decay constant. Higher = faster decay.
                0.0065 gives ~half-life of 107 days (Dixon & Coles original).
                Set to 0 to disable decay.
        """
        self.xi = xi
        self.teams_ = None
        self.attack_ = None
        self.defense_ = None
        self.home_adv_ = None
        self.rho_ = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _time_weights(self, dates, ref_date):
        days = (ref_date - pd.to_datetime(dates)).dt.days.values
        return np.exp(-self.xi * days)

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        """
        Vectorized low-score correction for arrays x, y, lam, mu.
        Adjusts Poisson independence assumption for low scores.
        """
        tau = np.ones(len(x))
        m00 = (x == 0) & (y == 0)
        m10 = (x == 1) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m11 = (x == 1) & (y == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m11] = 1 - rho
        return tau

    @staticmethod
    def _tau_scalar(x, y, lam, mu, rho):
        """Scalar tau for score matrix prediction."""
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _neg_log_likelihood(self, params, home_idx, away_idx,
                             home_goals, away_goals, weights, neutrals, n):
        # Unpack — attack[0] fixed at 1 (log=0) for identifiability
        log_att = np.zeros(n)
        log_att[1:] = params[:n - 1]
        log_def = params[n - 1:2 * n - 1]
        log_home = params[2 * n - 1]
        rho = params[2 * n]

        att = np.exp(log_att)
        dfn = np.exp(log_def)
        home_adv = np.exp(log_home)

        # Home advantage is 1.0 on neutral grounds
        ha = np.where(neutrals, 1.0, home_adv)

        lam = att[home_idx] * dfn[away_idx] * ha
        mu = att[away_idx] * dfn[home_idx]

        tau = self._tau(home_goals, away_goals, lam, mu, rho)

        # Guard against invalid tau (rho out of range)
        if np.any(tau <= 0):
            return 1e10

        log_p = (np.log(tau)
                 + poisson.logpmf(home_goals, lam)
                 + poisson.logpmf(away_goals, mu))

        return -np.dot(weights, log_p)

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df, ref_date=None):
        """
        Fit the Dixon-Coles model.

        Args:
            df: DataFrame with columns:
                date, home_team, away_team, home_score, away_score, neutral
            ref_date: Reference date for time decay. Defaults to df['date'].max().

        Returns:
            self
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)

        if ref_date is None:
            ref_date = df['date'].max()

        self.teams_ = sorted(set(df['home_team']) | set(df['away_team']))
        n = len(self.teams_)
        idx = {t: i for i, t in enumerate(self.teams_)}

        home_idx = df['home_team'].map(idx).values
        away_idx = df['away_team'].map(idx).values
        home_goals = df['home_score'].values
        away_goals = df['away_score'].values
        neutrals = df['neutral'].astype(bool).values
        weights = self._time_weights(df['date'], ref_date)

        # Initial params: zeros in log-space (all params = 1 except rho = 0)
        n_params = (n - 1) + n + 1 + 1
        x0 = np.zeros(n_params)

        # rho bounded so tau stays valid: tau(0,0) = 1 - λμρ > 0 requires ρ < 1/(λμ)
        # Simple safe bound: rho in (-1, 1)
        bounds = [(None, None)] * (n_params - 1) + [(-0.99, 0.99)]

        print(f"Fitting Dixon-Coles on {len(df)} matches, {n} teams...")
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights, neutrals, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50000, 'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': 500000},
        )

        if not result.success:
            print(f"  Warning: {result.message}")

        log_att = np.zeros(n)
        log_att[1:] = result.x[:n - 1]
        log_def = result.x[n - 1:2 * n - 1]

        self.attack_ = dict(zip(self.teams_, np.exp(log_att)))
        self.defense_ = dict(zip(self.teams_, np.exp(log_def)))
        self.home_adv_ = float(np.exp(result.x[2 * n - 1]))
        self.rho_ = float(result.x[2 * n])

        print(f"  Done. home_adv={self.home_adv_:.3f}, rho={self.rho_:.3f}")
        return self

    def predict_score_matrix(self, home_team, away_team, neutral=False, max_goals=10):
        """
        Returns matrix M where M[i, j] = P(home_goals=i, away_goals=j).
        """
        att_h = self.attack_.get(home_team, 1.0)
        def_h = self.defense_.get(home_team, 1.0)
        att_a = self.attack_.get(away_team, 1.0)
        def_a = self.defense_.get(away_team, 1.0)

        ha = 1.0 if neutral else self.home_adv_
        lam = att_h * def_a * ha
        mu = att_a * def_h

        M = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = self._tau_scalar(i, j, lam, mu, self.rho_)
                M[i, j] = tau * poisson.pmf(i, lam) * poisson.pmf(j, mu)

        return M

    def predict_outcome_probs(self, home_team, away_team, neutral=False, max_goals=10):
        """
        Returns dict with 'home_win', 'draw', 'away_win' probabilities.
        """
        M = self.predict_score_matrix(home_team, away_team, neutral, max_goals)

        home_win = float(np.sum(np.tril(M, -1)))   # i > j (home scores more)
        draw = float(np.sum(np.diag(M)))            # i == j
        away_win = float(np.sum(np.triu(M, 1)))     # j > i (away scores more)

        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total,
        }

    def top_teams(self, n=20):
        """Return top n teams ranked by attack / defense ratio (overall strength)."""
        if self.attack_ is None:
            raise RuntimeError("Model not fitted yet.")
        strength = {t: self.attack_[t] / self.defense_[t] for t in self.teams_}
        return sorted(strength.items(), key=lambda x: x[1], reverse=True)[:n]
