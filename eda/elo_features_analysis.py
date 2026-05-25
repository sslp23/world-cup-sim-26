"""
Predictive power analysis for ELO variant features.

Evaluates the following features relative to the baseline elo_diff:
  - elo_delta_20_diff  : ELO momentum (change over last 20 games)
  - elo_ma_2yr_diff    : 2-year rolling mean ELO
  - elo_ma_4yr_diff    : 4-year rolling mean ELO (WC cycle)
  - elo_ma_8yr_diff    : 8-year rolling mean ELO (historical pedigree)
  - abs_elo_diff       : absolute ELO gap (suppresses draws for large mismatches)
  - wc_games_diff      : World Cup experience

Sections:
  1. Spearman correlation and mutual information with result
  2. Collinearity with elo_diff (Pearson r)
  3. Partial correlation with result, controlling for elo_diff
  4. Log-loss: baseline (elo_diff only) vs adding each new feature
  5. VIF for the full ELO feature group
  6. Feature means by outcome (sanity check)

Run from the project root:
    python -m eda.elo_features_analysis
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_PATH    = 'data/ranked_database_with_features.csv'
TRAIN_CUTOFF = '2022-11-20'

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['result'] = df.apply(
    lambda r: 0 if r['home_score'] > r['away_score']
              else (1 if r['home_score'] == r['away_score'] else 2),
    axis=1,
)
train = df[df['date'] < TRAIN_CUTOFF].copy()

# ── Build difference features ─────────────────────────────────────────────────
train = train.copy()
train['elo_delta_20_diff'] = train['home_elo_delta_20'] - train['away_elo_delta_20']
train['elo_ma_2yr_diff']   = train['home_elo_ma_2yr']   - train['away_elo_ma_2yr']
train['elo_ma_4yr_diff']   = train['home_elo_ma_4yr']   - train['away_elo_ma_4yr']
train['elo_ma_8yr_diff']   = train['home_elo_ma_8yr']   - train['away_elo_ma_8yr']
train['abs_elo_diff']      = train['elo_diff'].abs()
train['wc_games_diff']     = train['home_wc_games']     - train['away_wc_games']

NEW_FEATURES = [
    'elo_delta_20_diff',
    'elo_ma_2yr_diff',
    'elo_ma_4yr_diff',
    'elo_ma_8yr_diff',
    'abs_elo_diff',
    'wc_games_diff',
]
ALL_ELO_FEATURES = ['elo_diff'] + NEW_FEATURES

required = ALL_ELO_FEATURES
clean = train.dropna(subset=required).copy()
y = clean['result'].values
print(f'Training rows (all ELO features present): {len(clean)}')

RESULT_LABELS = {0: 'home_win', 1: 'draw', 2: 'away_win'}


# ── helpers ───────────────────────────────────────────────────────────────────

def partial_corr(x, y, z):
    """Partial correlation of x with y controlling for z."""
    bz_x   = np.cov(x, z)[0, 1] / np.var(z)
    resid_x = x - bz_x * z
    bz_y   = np.cov(y, z)[0, 1] / np.var(z)
    resid_y = y - bz_y * z
    r, p   = pearsonr(resid_x, resid_y)
    return r, p


def logit_ll(X, y_):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    clf    = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
    clf.fit(X_sc, y_)
    probs  = clf.predict_proba(X_sc)
    return log_loss(y_, probs), (clf.predict(X_sc) == y_).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPEARMAN CORRELATION + MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('1. SPEARMAN CORRELATION AND MUTUAL INFORMATION WITH RESULT')
print('   (negative Spearman = predicts home win, positive = away win)')
print('=' * 70)

X_mi = clean[ALL_ELO_FEATURES].values
mi_scores = mutual_info_classif(X_mi, y, random_state=42)
mi_map = dict(zip(ALL_ELO_FEATURES, mi_scores))

rows = []
for feat in ALL_ELO_FEATURES:
    r, p = stats.spearmanr(clean[feat], y)
    rows.append({'feature': feat, 'spearman_r': r, 'p': p, 'MI': mi_map[feat]})

result_df = pd.DataFrame(rows).sort_values('MI', ascending=False)
print(f"\n  {'Feature':<24}  {'Spearman r':>10}  {'p-value':>10}  {'MI':>8}")
print(f"  {'-'*60}")
for _, row in result_df.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else '   '))
    marker = ' <-- baseline' if row['feature'] == 'elo_diff' else ''
    print(f"  {row['feature']:<24}  {row['spearman_r']:>+10.4f}  {row['p']:>10.2e}  {row['MI']:>8.4f}  {sig}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COLLINEARITY WITH elo_diff (Pearson r)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('2. COLLINEARITY WITH elo_diff  (Pearson r)')
print('   r > 0.85 = high collinearity; r > 0.92 = severe (like points_dif)')
print('=' * 70)

elo = clean['elo_diff'].values
print(f"\n  {'Feature':<24}  {'Pearson r':>10}  {'R²':>8}  Note")
print(f"  {'-'*60}")
for feat in NEW_FEATURES:
    r, _ = pearsonr(clean[feat], elo)
    note = '  <<< high'  if abs(r) > 0.85 else ('  < moderate' if abs(r) > 0.60 else '  low')
    print(f"  {feat:<24}  {r:>+10.4f}  {r**2:>8.4f}  {note}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARTIAL CORRELATION WITH RESULT, CONTROLLING FOR elo_diff
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('3. PARTIAL CORRELATION WITH RESULT  (controlling for elo_diff)')
print('   Near-zero partial r = feature adds no signal beyond elo_diff')
print('=' * 70)

y_float = y.astype(float)
elo_v   = clean['elo_diff'].values

print(f"\n  {'Feature':<24}  {'partial r':>10}  {'p-value':>10}  Assessment")
print(f"  {'-'*66}")
for feat in NEW_FEATURES:
    r, p = partial_corr(clean[feat].values, y_float, elo_v)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '   '))
    if abs(r) < 0.05:
        assessment = 'near-zero, likely redundant'
    elif abs(r) < 0.10:
        assessment = 'weak independent signal'
    else:
        assessment = 'meaningful independent signal'
    print(f"  {feat:<24}  {r:>+10.4f}  {p:>10.2e}  {sig}  {assessment}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOG-LOSS: BASELINE vs ADDING EACH FEATURE
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('4. LOG-LOSS  (logistic regression, in-sample)')
print('   Baseline = elo_diff only.  Lower = better.  Uniform = 1.0986')
print('=' * 70)

X_base = clean[['elo_diff']].values
ll_base, acc_base = logit_ll(X_base, y)
print(f"\n  {'Model':<40}  {'log-loss':>9}  {'vs base':>10}  {'accuracy':>9}")
print(f"  {'-'*72}")
print(f"  {'elo_diff (baseline)':<40}  {ll_base:>9.4f}  {'—':>10}  {acc_base:>9.3f}")

for feat in NEW_FEATURES:
    X = clean[['elo_diff', feat]].values
    ll, acc = logit_ll(X, y)
    delta = ll - ll_base
    print(f"  {'elo_diff + ' + feat:<40}  {ll:>9.4f}  {delta:>+10.4f}  {acc:>9.3f}")

# All new features together
X_all = clean[ALL_ELO_FEATURES].values
ll_all, acc_all = logit_ll(X_all, y)
print(f"  {'elo_diff + all new features':<40}  {ll_all:>9.4f}  {ll_all - ll_base:>+10.4f}  {acc_all:>9.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VIF FOR THE FULL ELO FEATURE GROUP
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('5. VARIANCE INFLATION FACTOR  (ELO feature group)')
print('   VIF > 5 = moderate collinearity, VIF > 10 = severe')
print('=' * 70)

vif_df   = clean[ALL_ELO_FEATURES].copy()
vif_vals = []
for i, col in enumerate(vif_df.columns):
    vif = variance_inflation_factor(vif_df.values, i)
    vif_vals.append({'feature': col, 'VIF': round(vif, 2)})

vif_result = pd.DataFrame(vif_vals).sort_values('VIF', ascending=False)
print(f"\n  {'Feature':<24}  {'VIF':>8}  Flag")
print(f"  {'-'*44}")
for _, row in vif_result.iterrows():
    flag = '  <<< HIGH' if row['VIF'] > 10 else ('  < moderate' if row['VIF'] > 5 else '')
    print(f"  {row['feature']:<24}  {row['VIF']:>8.2f}  {flag}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE MEANS BY OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('6. FEATURE MEANS BY OUTCOME  (sanity check)')
print('   home_win should have positive values, away_win negative (for diff features)')
print('=' * 70)

check_df = clean[ALL_ELO_FEATURES].copy()
check_df['result'] = y
grouped = check_df.groupby('result')[ALL_ELO_FEATURES].mean()
grouped.index = [RESULT_LABELS[i] for i in grouped.index]
print('\n' + grouped.round(1).to_string())

print()
