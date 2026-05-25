"""
Collinearity analysis: pi ratings vs ELO.

Tests both the raw pi_diff and a neutral-venue version (avg of home/away
pi components) against elo_diff, since WC matches are at neutral venues.

Run from the project root:
    python -m eda.collinearity_pi_elo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

DATA_PATH    = 'data/ranked_database_with_features.csv'
TRAIN_CUTOFF = '2022-11-20'

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['result'] = df.apply(
    lambda r: 0 if r['home_score'] > r['away_score']
              else (1 if r['home_score'] == r['away_score'] else 2),
    axis=1,
)
train = df[df['date'] < TRAIN_CUTOFF].dropna(
    subset=['elo_diff', 'pi_h_home', 'pi_a_home', 'pi_h_away', 'pi_a_away']
).copy()

# Neutral-venue pi: average of home-attack and away-attack rating per team.
# Removes home/away bias that is meaningless at WC neutral venues.
train['pi_neutral_home'] = (train['pi_h_home'] + train['pi_a_home']) / 2
train['pi_neutral_away'] = (train['pi_h_away'] + train['pi_a_away']) / 2
train['pi_neutral_diff'] = train['pi_neutral_home'] - train['pi_neutral_away']

elo = train['elo_diff'].values
pid = train['pi_diff'].values
pin = train['pi_neutral_diff'].values
y   = train['result'].values

print(f'Training rows: {len(train)}')

# ── 1. Correlation with ELO ───────────────────────────────────────────────────
print()
print('=' * 60)
print('1. CORRELATION WITH ELO')
print('=' * 60)
for name, x in [('pi_diff', pid), ('pi_neutral_diff', pin)]:
    pr, _ = stats.pearsonr(x, elo)
    sr, _ = stats.spearmanr(x, elo)
    print(f'  {name:<22}  Pearson r={pr:+.4f}  R²={pr**2:.4f}  Spearman r={sr:+.4f}')

# ── 2. Partial correlation with result ───────────────────────────────────────
print()
print('=' * 60)
print('2. PARTIAL CORRELATION WITH RESULT (controlling for elo_diff)')
print('   Recall: partial r(points_dif | elo_diff) = -0.050 → removed')
print('=' * 60)

def partial_corr(x, y, z):
    bz  = np.cov(x, z)[0, 1] / np.var(z);  rx = x - bz  * z
    bz2 = np.cov(y, z)[0, 1] / np.var(z);  ry = y - bz2 * z
    return stats.pearsonr(rx, ry)

for name, x in [('pi_diff', pid), ('pi_neutral_diff', pin)]:
    r, p = partial_corr(x, y.astype(float), elo)
    print(f'  partial r({name:<20} | elo_diff) = {r:+.4f}   p={p:.2e}')

# ── 3. Predictive value ───────────────────────────────────────────────────────
print()
print('=' * 60)
print('3. PREDICTIVE VALUE  (logistic regression, in-sample log-loss)')
print('   Lower = better.  Baseline (uniform) = 1.0986')
print('=' * 60)

configs = {
    'elo_diff only':          train[['elo_diff']].values,
    'pi_diff only':           train[['pi_diff']].values,
    'pi_neutral_diff only':   train[['pi_neutral_diff']].values,
    'elo + pi_diff':          train[['elo_diff', 'pi_diff']].values,
    'elo + pi_neutral_diff':  train[['elo_diff', 'pi_neutral_diff']].values,
}
for label, X in configs.items():
    Xs  = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xs, y)
    ll  = log_loss(y, clf.predict_proba(Xs))
    acc = (clf.predict(Xs) == y).mean()
    print(f'  {label:<30}  log-loss={ll:.4f}  accuracy={acc:.3f}')
