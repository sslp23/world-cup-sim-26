"""
Collinearity analysis: FIFA points vs ELO rating.

Questions:
  1. How correlated are points_dif and elo_diff? (Pearson, Spearman)
  2. Do they have high VIF in the full model feature set?
  3. Does each add independent predictive value over the other?
     - Train a logistic regression with only points_dif → log-loss
     - Train with only elo_diff → log-loss
     - Train with both → log-loss
     - Check the improvement

Run from the project root:
    python -m eda.collinearity_points_elo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_PATH    = 'data/ranked_database_with_features.csv'
TRAIN_CUTOFF = '2022-11-20'

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['result'] = df.apply(
    lambda r: 0 if r['home_score'] > r['away_score']
              else (1 if r['home_score'] == r['away_score'] else 2),
    axis=1,
)
train = df[df['date'] < TRAIN_CUTOFF].dropna(subset=['elo_diff', 'points_dif']).copy()
print(f'Training rows: {len(train)}')

pts = train['points_dif'].values
elo = train['elo_diff'].values
y   = train['result'].values

# ── 1. Correlation ────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('1. CORRELATION  (points_dif  vs  elo_diff)')
print('=' * 60)

pearson_r,  pearson_p  = stats.pearsonr(pts, elo)
spearman_r, spearman_p = stats.spearmanr(pts, elo)

print(f'  Pearson  r = {pearson_r:+.4f}   p = {pearson_p:.2e}')
print(f'  Spearman r = {spearman_r:+.4f}   p = {spearman_p:.2e}')
print(f'  R²         = {pearson_r**2:.4f}  (shared variance)')

# ── 2. VIF ────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('2. VARIANCE INFLATION FACTOR  (full model feature set)')
print('   VIF > 5 = moderate collinearity, VIF > 10 = severe')
print('=' * 60)

MODEL_FEATURES = {
    'points_dif':    train['points_dif'],
    'elo_diff':      train['elo_diff'],
    'abs_elo_diff':  train['elo_diff'].abs(),
    'abs_points_dif': train['points_dif'].abs(),
    'pww_ma20_diff': train['home_points_weighted_ma_20'] - train['away_points_weighted_ma_20'],
    'pww_ma5_diff':  train['home_points_weighted_ma_5']  - train['away_points_weighted_ma_5'],
    'gw_ma20_diff':  train['home_goals_weighted_ma_20']  - train['away_goals_weighted_ma_20'],
    'gw_ma5_diff':   train['home_goals_weighted_ma_5']   - train['away_goals_weighted_ma_5'],
    'gsw_ma20_diff': train['home_goals_suffered_weighted_ma_20'] - train['away_goals_suffered_weighted_ma_20'],
    'gsw_ma5_diff':  train['home_goals_suffered_weighted_ma_5']  - train['away_goals_suffered_weighted_ma_5'],
    'gd_ma20_diff':  train['home_goal_diff_ma_20'] - train['away_goal_diff_ma_20'],
    'gd_ma5_diff':   train['home_goal_diff_ma_5']  - train['away_goal_diff_ma_5'],
}

vif_df = pd.DataFrame(MODEL_FEATURES).dropna()
vif_vals = []
for i, col in enumerate(vif_df.columns):
    vif = variance_inflation_factor(vif_df.values, i)
    vif_vals.append({'feature': col, 'VIF': vif})

vif_result = pd.DataFrame(vif_vals).sort_values('VIF', ascending=False)
for _, row in vif_result.iterrows():
    flag = '  <<< HIGH' if row['VIF'] > 10 else ('  < moderate' if row['VIF'] > 5 else '')
    print(f"  {row['feature']:<22}  VIF = {row['VIF']:6.2f}{flag}")

# ── 3. Predictive value: alone vs together ────────────────────────────────────
print()
print('=' * 60)
print('3. INDEPENDENT PREDICTIVE VALUE  (logistic regression, log-loss)')
print('   Lower log-loss = better.  Baseline (uniform) = 1.0986')
print('=' * 60)

clean = pd.DataFrame({'pts': pts, 'elo': elo, 'y': y}).dropna()
y_clean = clean['y'].values

configs = {
    'points_dif only':   clean[['pts']].values,
    'elo_diff only':     clean[['elo']].values,
    'pts + elo':         clean[['pts', 'elo']].values,
    'pts + elo + abs':   clean[['pts', 'elo',
                                 np.abs(clean['pts']),
                                 np.abs(clean['elo'])]].values
                         if False else None,  # placeholder, built below
}

# Build correctly
X_sets = {
    'points_dif only':        clean[['pts']].values,
    'elo_diff only':          clean[['elo']].values,
    'pts + elo':              clean[['pts', 'elo']].values,
    'pts + elo + abs(both)':  np.column_stack([clean['pts'], clean['elo'],
                                               clean['pts'].abs(), clean['elo'].abs()]),
}

for label, X in X_sets.items():
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    clf    = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    clf.fit(X_sc, y_clean)
    probs  = clf.predict_proba(X_sc)
    ll     = log_loss(y_clean, probs)
    acc    = (clf.predict(X_sc) == y_clean).mean()
    print(f"  {label:<28}  log-loss = {ll:.4f}   accuracy = {acc:.3f}")

# ── 4. Partial correlation ─────────────────────────────────────────────────────
print()
print('=' * 60)
print('4. PARTIAL CORRELATION WITH RESULT')
print('   points_dif controlling for elo_diff (and vice versa)')
print('=' * 60)

from scipy.stats import pearsonr

def partial_corr(x, y, z):
    """Partial correlation of x with y controlling for z."""
    # Regress x on z, get residuals
    bz_x = np.cov(x, z)[0, 1] / np.var(z)
    resid_x = x - bz_x * z
    # Regress y on z, get residuals
    bz_y = np.cov(y, z)[0, 1] / np.var(z)
    resid_y = y - bz_y * z
    r, p = pearsonr(resid_x, resid_y)
    return r, p

pc_pts_given_elo_r, pc_pts_given_elo_p = partial_corr(clean['pts'].values, y_clean.astype(float), clean['elo'].values)
pc_elo_given_pts_r, pc_elo_given_pts_p = partial_corr(clean['elo'].values, y_clean.astype(float), clean['pts'].values)

print(f'  partial r(points_dif | elo_diff)   = {pc_pts_given_elo_r:+.4f}   p = {pc_pts_given_elo_p:.2e}')
print(f'  partial r(elo_diff   | points_dif) = {pc_elo_given_pts_r:+.4f}   p = {pc_elo_given_pts_p:.2e}')
print()
print('  If partial r is still significant: the feature adds signal beyond the other.')
print('  If near-zero: the feature is redundant once the other is known.')
