"""
Feature analysis for goal-scoring regressors (ML-Poisson model).

Unlike the classifier EDA (feature_analysis.py) which used difference features
and a categorical target, here we predict goals scored by each team separately:

  home_goals ~ f(home attack features, away defense features, match context)
  away_goals ~ f(away attack features, home defense features, match context)

Two separate analyses are run — one per target — to identify the best
predictors for each regression task.

Sections:
  1. Target distribution (home_goals, away_goals)
  2. Spearman correlation with goals scored
  3. Mutual information (regression, captures non-linear effects)
  4. Multicollinearity among candidate features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

DATA_PATH    = "data/ranked_database_with_features.csv"
TRAIN_CUTOFF = "2022-11-20"

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
train = df[df['date'] < TRAIN_CUTOFF].dropna(subset=['elo_diff', 'pi_diff']).copy()
print(f"Training rows: {len(train)}")

# ── Candidate feature sets ────────────────────────────────────────────────────
# For home_goals: home team attacks, away team defends
home_goal_features = {
    # Home attack
    'h_gw_ma20':          train['home_goals_weighted_ma_20'],
    'h_gw_ma10':          train['home_goals_weighted_ma_10'],
    'h_gw_ma5':           train['home_goals_weighted_ma_5'],
    'h_gd_ma20':          train['home_goal_diff_ma_20'],
    'h_gd_ma10':          train['home_goal_diff_ma_10'],
    'h_gd_ma5':           train['home_goal_diff_ma_5'],
    'h_pww_ma20':         train['home_points_weighted_ma_20'],
    'h_pww_ma10':         train['home_points_weighted_ma_10'],
    'h_pww_ma5':          train['home_points_weighted_ma_5'],
    # Away defense (how many goals they concede)
    'a_gsw_ma20':         train['away_goals_suffered_weighted_ma_20'],
    'a_gsw_ma10':         train['away_goals_suffered_weighted_ma_10'],
    'a_gsw_ma5':          train['away_goals_suffered_weighted_ma_5'],
    'a_gd_ma20':          train['away_goal_diff_ma_20'],
    'a_gd_ma10':          train['away_goal_diff_ma_10'],
    'a_gd_ma5':           train['away_goal_diff_ma_5'],
    # Match-level ratings
    'elo_diff':           train['elo_diff'],
    'pi_diff':            train['pi_diff'],
    'points_dif':         train['points_dif'],
    # Context
    'neutral':            train['neutral'].astype(int),
    'tournament_weight':  train['tournament_weight'],
}

# For away_goals: away team attacks, home team defends
away_goal_features = {
    # Away attack
    'a_gw_ma20':          train['away_goals_weighted_ma_20'],
    'a_gw_ma10':          train['away_goals_weighted_ma_10'],
    'a_gw_ma5':           train['away_goals_weighted_ma_5'],
    'a_gd_ma20':          train['away_goal_diff_ma_20'],
    'a_gd_ma10':          train['away_goal_diff_ma_10'],
    'a_gd_ma5':           train['away_goal_diff_ma_5'],
    'a_pww_ma20':         train['away_points_weighted_ma_20'],
    'a_pww_ma10':         train['away_points_weighted_ma_10'],
    'a_pww_ma5':          train['away_points_weighted_ma_5'],
    # Home defense (how many goals they concede)
    'h_gsw_ma20':         train['home_goals_suffered_weighted_ma_20'],
    'h_gsw_ma10':         train['home_goals_suffered_weighted_ma_10'],
    'h_gsw_ma5':          train['home_goals_suffered_weighted_ma_5'],
    'h_gd_ma20':          train['home_goal_diff_ma_20'],
    'h_gd_ma10':          train['home_goal_diff_ma_10'],
    'h_gd_ma5':           train['home_goal_diff_ma_5'],
    # Match-level ratings (negated — now from away perspective)
    'elo_diff':           -train['elo_diff'],
    'pi_diff':            -train['pi_diff'],
    'points_dif':         -train['points_dif'],
    # Context
    'neutral':            train['neutral'].astype(int),
    'tournament_weight':  train['tournament_weight'],
}

targets = {
    'home_goals': (train['home_score'], home_goal_features),
    'away_goals': (train['away_score'], away_goal_features),
}


def run_analysis(target_name, y, feature_defs):
    print()
    print("=" * 65)
    print(f"TARGET: {target_name}")
    print("=" * 65)

    feat_df = pd.DataFrame(feature_defs)

    # 1. Target distribution
    print(f"\n  Mean={y.mean():.3f}  Std={y.std():.3f}  "
          f"0-goals={( y==0).mean()*100:.1f}%  "
          f"1-goal={(y==1).mean()*100:.1f}%  "
          f"2-goals={(y==2).mean()*100:.1f}%  "
          f"3+={(y>=3).mean()*100:.1f}%")

    # 2. Spearman correlation
    print(f"\n  {'Feature':<22}  {'Spearman r':>10}  sig")
    corrs = []
    for name, feat in feature_defs.items():
        mask = feat.notna() & y.notna()
        r, p = stats.spearmanr(feat[mask], y[mask])
        corrs.append((name, r, p))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, p in corrs:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {name:<22}  {r:>+10.4f}  {sig}")

    # 3. Mutual information
    print(f"\n  {'Feature':<22}  {'MI':>6}")
    mi_df = feat_df.copy()
    mi_df['y'] = y.values
    mi_df = mi_df.dropna()
    X_mi = mi_df.drop('y', axis=1).values
    y_mi = mi_df['y'].values
    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
    mi_results = sorted(zip(feat_df.columns, mi_scores), key=lambda x: x[1], reverse=True)
    for name, score in mi_results:
        bar = '#' * int(score * 100)
        print(f"  {name:<22}  {score:>6.4f}  {bar}")

    # 4. Multicollinearity (|r| > 0.85)
    clean = feat_df.dropna()
    corr_matrix = clean.corr(method='pearson')
    high_corr = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr.append((cols[i], cols[j], r))
    if high_corr:
        print(f"\n  Multicollinear pairs (|r| > 0.85):")
        for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {a:<22} <-> {b:<22}  r={r:+.3f}")
    else:
        print("\n  No multicollinear pairs above 0.85")


for target_name, (y, feature_defs) in targets.items():
    run_analysis(target_name, y, feature_defs)
