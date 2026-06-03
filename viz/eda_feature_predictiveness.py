"""
EDA: Feature predictiveness — Spearman |r| with match outcome.

Ranks all model candidate features by absolute Spearman correlation
with match result (0=home win, 1=draw, 2=away win), highlighting
which features were included in the final models vs excluded.

Data source: concatenated past_wc files (2005-2022, no overlaps).

Run from the project root:
    python -m viz.eda_feature_predictiveness
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

OUTPUT_PATH = 'viz/output/eda_feature_predictiveness.png'
TRAIN_CUTOFF = '2022-11-20'

C_INCLUDED = '#1F4E79'   # dark blue — features in final models
C_EXCLUDED = '#B0BEC5'   # light grey — excluded (e.g. points_dif)
C_ANNOT    = '#546E7A'
C_SPINE    = '#CFD8DC'

# Features to evaluate.  True = included in at least one final model.
FEATURES = [
    ('elo_diff',                          True,  'ELO difference'),
    ('mv_sum_diff',                       True,  'Market value sum diff  (top-20 squad)'),
    ('pi_diff',                           False, 'π-rating diff  (excluded — r=0.90 with ELO)'),
    ('points_dif',                        False, 'FIFA points diff  (excluded — r=0.92 with ELO)'),
    ('mv_median_diff',                    False, 'Market value median diff  (excluded — r=0.91 with MV sum)'),
    ('home_wc_best_round',                True,  'Home WC best round'),
    ('away_wc_best_round',                True,  'Away WC best round'),
    ('home_wc_goals_per_game',            True,  'Home WC goals/game'),
    ('away_wc_goals_per_game',            True,  'Away WC goals/game'),
    ('home_wc_games',                     True,  'Home WC games played'),
    ('away_wc_games',                     True,  'Away WC games played'),
    ('home_points_weighted_ma_20',        True,  'Home points wtd-MA 20'),
    ('home_points_weighted_ma_5',         True,  'Home points wtd-MA 5'),
    ('home_goals_weighted_ma_20',         True,  'Home goals wtd-MA 20'),
    ('home_goals_weighted_ma_5',          True,  'Home goals wtd-MA 5'),
    ('home_goals_suffered_weighted_ma_20',True,  'Home goals suffered wtd-MA 20'),
    ('home_goals_suffered_weighted_ma_5', True,  'Home goals suffered wtd-MA 5'),
    ('home_goal_diff_ma_20',              True,  'Home goal diff MA 20'),
    ('home_goal_diff_ma_5',               True,  'Home goal diff MA 5'),
]


def run():
    # Past-WC files cover 2005-2022 with no overlap — use them as training source
    # so MV columns (only in past_wc files) are available alongside all other features.
    files = sorted(glob.glob('data/past_wc/wc*/ranked_database_with_features.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['mv_sum_diff']    = df['home_mv_top20_sum']    - df['away_mv_top20_sum']
    df['mv_median_diff'] = df['home_mv_top20_median'] - df['away_mv_top20_median']
    df['date'] = pd.to_datetime(df['date'])
    train = (
        df[df['date'] < TRAIN_CUTOFF]
        .dropna(subset=['home_score', 'away_score'])
        .copy()
    )
    train['result'] = train.apply(
        lambda r: 0 if r['home_score'] > r['away_score']
        else (1 if r['home_score'] == r['away_score'] else 2),
        axis=1,
    )

    rows = []
    for col, included, label in FEATURES:
        if col not in train.columns:
            continue
        sub = train[[col, 'result']].dropna()
        r, _ = spearmanr(sub[col], sub['result'])
        rows.append({'label': label, 'abs_r': abs(r), 'included': included})

    results = pd.DataFrame(rows).sort_values('abs_r', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = [C_INCLUDED if inc else C_EXCLUDED for inc in results['included']]
    bars = ax.barh(results['label'], results['abs_r'],
                   color=colors, height=0.65, zorder=2)

    # Value labels at bar ends
    for bar, val in zip(bars, results['abs_r']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=7.5, color=C_ANNOT)

    ax.set_xlabel('|Spearman r|  with match outcome', fontsize=10, color=C_ANNOT)
    ax.set_title(
        'Feature predictiveness  (Spearman |r| with outcome, training data pre-WC2022)',
        fontsize=13, fontweight='bold', pad=12,
    )

    ax.set_xlim(0, results['abs_r'].max() * 1.18)
    ax.tick_params(axis='both', labelsize=8, colors=C_ANNOT)
    ax.xaxis.grid(True, color=C_SPINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_INCLUDED, label='Included in final models'),
        mpatches.Patch(color=C_EXCLUDED, label='Excluded'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right',
              framealpha=0.9, edgecolor=C_SPINE)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
