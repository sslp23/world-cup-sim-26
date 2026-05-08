"""
Feature analysis for WC 2026 prediction models.

Evaluates feature predictiveness and redundancy using difference-based
features (home - away), the natural format for neutral venue predictions.

Sections:
  1. Outcome distribution
  2. Spearman correlation with result
  3. Mutual information (captures non-linear relationships)
  4. Multicollinearity (Pearson |r| > 0.85 pairs)
  5. Feature means by outcome (sanity check)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/ranked_database_with_features.csv"
TRAIN_CUTOFF = "2022-11-20"   # everything before WC 2022

RESULT_LABELS = {0: 'home_win', 1: 'draw', 2: 'away_win'}

# ── Load & prepare ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['result'] = df.apply(
    lambda r: 0 if r['home_score'] > r['away_score']
              else (1 if r['home_score'] == r['away_score'] else 2),
    axis=1
)

train = df[df['date'] < TRAIN_CUTOFF].copy()
train = train.dropna(subset=['elo_diff', 'pi_diff'])  # require rating features
print(f"Training rows after dropping rating NaNs: {len(train)}")

# ── Build difference features ─────────────────────────────────────────────────
def diff(col_home, col_away):
    return train[col_home] - train[col_away]

feature_defs = {
    # Rating-based
    'points_dif':           train['points_dif'],
    'elo_diff':             train['elo_diff'],
    'pi_diff':              train['pi_diff'],
    # Points won MAs
    'pw_ma20_diff':         diff('home_points_won_ma_20', 'away_points_won_ma_20'),
    'pw_ma10_diff':         diff('home_points_won_ma_10', 'away_points_won_ma_10'),
    'pw_ma5_diff':          diff('home_points_won_ma_5',  'away_points_won_ma_5'),
    'pw_ma3_diff':          diff('home_points_won_ma_3',  'away_points_won_ma_3'),
    # Weighted points won MAs
    'pww_ma20_diff':        diff('home_points_weighted_ma_20', 'away_points_weighted_ma_20'),
    'pww_ma10_diff':        diff('home_points_weighted_ma_10', 'away_points_weighted_ma_10'),
    'pww_ma5_diff':         diff('home_points_weighted_ma_5',  'away_points_weighted_ma_5'),
    'pww_ma3_diff':         diff('home_points_weighted_ma_3',  'away_points_weighted_ma_3'),
    # Goals scored MAs
    'goals_ma20_diff':      diff('home_goals_ma_20', 'away_goals_ma_20'),
    'goals_ma10_diff':      diff('home_goals_ma_10', 'away_goals_ma_10'),
    'goals_ma5_diff':       diff('home_goals_ma_5',  'away_goals_ma_5'),
    'goals_ma3_diff':       diff('home_goals_ma_3',  'away_goals_ma_3'),
    # Weighted goals scored MAs
    'gw_ma20_diff':         diff('home_goals_weighted_ma_20', 'away_goals_weighted_ma_20'),
    'gw_ma10_diff':         diff('home_goals_weighted_ma_10', 'away_goals_weighted_ma_10'),
    'gw_ma5_diff':          diff('home_goals_weighted_ma_5',  'away_goals_weighted_ma_5'),
    'gw_ma3_diff':          diff('home_goals_weighted_ma_3',  'away_goals_weighted_ma_3'),
    # Goals suffered MAs
    'gs_ma20_diff':         diff('home_goals_suffered_ma_20', 'away_goals_suffered_ma_20'),
    'gs_ma10_diff':         diff('home_goals_suffered_ma_10', 'away_goals_suffered_ma_10'),
    'gs_ma5_diff':          diff('home_goals_suffered_ma_5',  'away_goals_suffered_ma_5'),
    'gs_ma3_diff':          diff('home_goals_suffered_ma_3',  'away_goals_suffered_ma_3'),
    # Weighted goals suffered MAs
    'gsw_ma20_diff':        diff('home_goals_suffered_weighted_ma_20', 'away_goals_suffered_weighted_ma_20'),
    'gsw_ma10_diff':        diff('home_goals_suffered_weighted_ma_10', 'away_goals_suffered_weighted_ma_10'),
    'gsw_ma5_diff':         diff('home_goals_suffered_weighted_ma_5',  'away_goals_suffered_weighted_ma_5'),
    'gsw_ma3_diff':         diff('home_goals_suffered_weighted_ma_3',  'away_goals_suffered_weighted_ma_3'),
    # Goal difference MAs
    'gd_ma20_diff':         diff('home_goal_diff_ma_20', 'away_goal_diff_ma_20'),
    'gd_ma10_diff':         diff('home_goal_diff_ma_10', 'away_goal_diff_ma_10'),
    'gd_ma5_diff':          diff('home_goal_diff_ma_5',  'away_goal_diff_ma_5'),
    'gd_ma3_diff':          diff('home_goal_diff_ma_3',  'away_goal_diff_ma_3'),
    # Form trends
    'trend5_diff':          diff('home_form_trend_5', 'away_form_trend_5'),
    'trend3_diff':          diff('home_form_trend_3', 'away_form_trend_3'),
    # Context
    'days_diff':            diff('home_days_since_last_match', 'away_days_since_last_match'),
    'neutral':              train['neutral'].astype(int),
    'tournament_weight':    train['tournament_weight'],
}

feat_df = pd.DataFrame(feature_defs)
result   = train['result'].values

# ═══════════════════════════════════════════════════════════════════════════════
# 1. OUTCOME DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("1. OUTCOME DISTRIBUTION")
print("=" * 60)
counts = train['result'].value_counts().sort_index()
for code, label in RESULT_LABELS.items():
    n = counts.get(code, 0)
    print(f"  {label:<10} {n:>5}  ({n/len(train)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPEARMAN CORRELATION WITH RESULT
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("2. SPEARMAN CORRELATION WITH RESULT")
print("   (negative = predicts home win, positive = predicts away win)")
print("=" * 60)

corrs = []
for name, feat in feature_defs.items():
    mask = feat.notna()
    if mask.sum() < 100:
        continue
    r, p = stats.spearmanr(feat[mask], result[mask])
    corrs.append({'feature': name, 'spearman_r': r, 'p_value': p, 'abs_r': abs(r)})

corr_df = pd.DataFrame(corrs).sort_values('abs_r', ascending=False)
for _, row in corr_df.iterrows():
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else '   '))
    print(f"  {row['feature']:<26}  r={row['spearman_r']:+.4f}  {sig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("3. MUTUAL INFORMATION WITH RESULT")
print("   (higher = more predictive, captures non-linear effects)")
print("=" * 60)

mi_df = feat_df.copy()
mi_df['result'] = result
mi_df = mi_df.dropna()
X_mi = mi_df.drop('result', axis=1).values
y_mi = mi_df['result'].values

mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_results = sorted(zip(feat_df.columns, mi_scores), key=lambda x: x[1], reverse=True)

for name, score in mi_results:
    bar = '#' * int(score * 200)
    print(f"  {name:<26}  MI={score:.4f}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MULTICOLLINEARITY — pairwise Pearson among features
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. MULTICOLLINEARITY (Pearson |r| > 0.85 pairs)")
print("=" * 60)

clean = feat_df.dropna()
corr_matrix = clean.corr(method='pearson')

high_corr = []
cols = corr_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            high_corr.append((cols[i], cols[j], r))

high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
if high_corr:
    for a, b, r in high_corr:
        print(f"  {a:<26} <-> {b:<26}  r={r:+.3f}")
else:
    print("  No pairs above 0.85")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE MEANS BY OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("5. FEATURE MEANS BY OUTCOME (rating & form features)")
print("=" * 60)

check_features = ['points_dif', 'elo_diff', 'pi_diff',
                  'pw_ma5_diff', 'pww_ma5_diff', 'gd_ma5_diff',
                  'gd_ma10_diff', 'gd_ma20_diff', 'trend5_diff']

mi_df_check = feat_df[check_features].copy()
mi_df_check['result'] = result
mi_df_check = mi_df_check.dropna()

grouped = mi_df_check.groupby('result')[check_features].mean()
grouped.index = [RESULT_LABELS[i] for i in grouped.index]
print(grouped.round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONFEDERATION ANALYSIS (categorical)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("6. CONFEDERATION ANALYSIS")
print("=" * 60)

conf_train = train[['confederation_home', 'confederation_away', 'result']].dropna()
conf_result = conf_train['result'].values

# 6a. Win rate by home confederation
print("\n  a) Outcome rates by home confederation")
print(f"  {'Confederation':<12}  {'N':>5}  {'home_win%':>9}  {'draw%':>6}  {'away_win%':>9}")
for conf in sorted(conf_train['confederation_home'].unique()):
    mask = conf_train['confederation_home'] == conf
    sub = conf_train[mask]['result']
    n = len(sub)
    hw = (sub == 0).mean() * 100
    dr = (sub == 1).mean() * 100
    aw = (sub == 2).mean() * 100
    print(f"  {conf:<12}  {n:>5}  {hw:>8.1f}%  {dr:>5.1f}%  {aw:>8.1f}%")

# 6b. Chi-squared test: confederation_home vs result
ct_home = pd.crosstab(conf_train['confederation_home'], conf_train['result'])
chi2_home, p_home, _, _ = stats.chi2_contingency(ct_home)
n_home = ct_home.values.sum()
cramers_v_home = np.sqrt(chi2_home / (n_home * (min(ct_home.shape) - 1)))

ct_away = pd.crosstab(conf_train['confederation_away'], conf_train['result'])
chi2_away, p_away, _, _ = stats.chi2_contingency(ct_away)
n_away = ct_away.values.sum()
cramers_v_away = np.sqrt(chi2_away / (n_away * (min(ct_away.shape) - 1)))

print(f"\n  b) Chi-squared test vs result")
print(f"  confederation_home:  chi2={chi2_home:.1f}  p={p_home:.4f}  Cramer's V={cramers_v_home:.3f}")
print(f"  confederation_away:  chi2={chi2_away:.1f}  p={p_away:.4f}  Cramer's V={cramers_v_away:.3f}")

# 6c. Mutual information — one-hot encoded confederations
conf_encoded = pd.get_dummies(conf_train[['confederation_home', 'confederation_away']])
mi_conf = mutual_info_classif(conf_encoded.values, conf_result, random_state=42)
mi_conf_results = sorted(zip(conf_encoded.columns, mi_conf), key=lambda x: x[1], reverse=True)

print(f"\n  c) Mutual information — one-hot encoded confederation columns")
for name, score in mi_conf_results:
    if score > 0.001:
        print(f"  {name:<40}  MI={score:.4f}")

# 6d. Same-confederation matchup
conf_train = conf_train.copy()
conf_train['same_conf'] = (conf_train['confederation_home'] == conf_train['confederation_away']).astype(int)
print(f"\n  d) Outcome rates: same confederation vs cross-confederation")
print(f"  {'matchup':<22}  {'N':>5}  {'home_win%':>9}  {'draw%':>6}  {'away_win%':>9}")
for label, mask in [('same conf', conf_train['same_conf'] == 1), ('cross conf', conf_train['same_conf'] == 0)]:
    sub = conf_train[mask]['result']
    n = len(sub)
    hw = (sub == 0).mean() * 100
    dr = (sub == 1).mean() * 100
    aw = (sub == 2).mean() * 100
    print(f"  {label:<22}  {n:>5}  {hw:>8.1f}%  {dr:>5.1f}%  {aw:>8.1f}%")
