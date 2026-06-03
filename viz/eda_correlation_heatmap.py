"""
EDA: Pearson correlation heatmap for model candidate features.

Lower-triangular heatmap. Key findings:
  elo_diff / points_dif     r=0.92 -> points_dif excluded
  mv_sum_diff / mv_med_diff r=0.91 -> mv_median_diff excluded

Data source: concatenated past_wc files (2005-2022, no overlaps).

Run from the project root:
    python -m viz.eda_correlation_heatmap
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = 'viz/output/eda_correlation_heatmap.png'
TRAIN_CUTOFF = '2022-11-20'

C_ANNOT  = '#546E7A'
C_SPINE  = '#CFD8DC'

# Columns to include and their short display labels
COLS_AND_LABELS = [
    ('elo_diff',                           'elo_diff'),
    ('mv_sum_diff',                        'mv_sum_diff'),
    ('pi_diff',                            'pi_diff'),
    ('points_dif',                         'points_dif'),
    ('mv_median_diff',                     'mv_median_diff'),
    ('home_wc_best_round',                 'wc_best_home'),
    ('away_wc_best_round',                 'wc_best_away'),
    ('home_wc_goals_per_game',             'wc_gpg_home'),
    ('away_wc_goals_per_game',             'wc_gpg_away'),
    ('home_points_weighted_ma_20',         'pts_ma20_H'),
    ('home_points_weighted_ma_5',          'pts_ma5_H'),
    ('home_goals_weighted_ma_20',          'gls_ma20_H'),
    ('home_goals_suffered_weighted_ma_20', 'gls_suf_ma20_H'),
    ('home_goal_diff_ma_20',               'gd_ma20_H'),
]


def run():
    # Past-WC files cover 2005-2022 with no overlap and include MV columns.
    files = sorted(glob.glob('data/past_wc/wc*/ranked_database_with_features.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['mv_sum_diff']    = df['home_mv_top20_sum']    - df['away_mv_top20_sum']
    df['mv_median_diff'] = df['home_mv_top20_median'] - df['away_mv_top20_median']
    df['date'] = pd.to_datetime(df['date'])
    train = (
        df[df['date'] < TRAIN_CUTOFF]
        .dropna(subset=['elo_diff', 'home_score', 'away_score'])
        .copy()
    )

    cols   = [c for c, _ in COLS_AND_LABELS if c in train.columns]
    labels = [l for c, l in COLS_AND_LABELS if c in train.columns]

    # Pairwise Pearson (pandas handles NaN pairs automatically)
    corr = train[cols].corr(method='pearson')

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, True)

    n = len(cols)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'rb_div', ['#C62828', '#FFFFFF', '#1F4E79'], N=256
    )

    masked_corr = corr.copy()
    masked_corr.values[mask] = np.nan

    im = ax.imshow(masked_corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            val = corr.values[i, j]
            text_col = 'white' if abs(val) > 0.65 else C_ANNOT
            weight = 'bold' if abs(val) > 0.85 else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=text_col, fontweight=weight)

    def highlight_cell(row_label, col_label, color):
        """Highlight a lower-triangle cell and annotate with its actual Pearson r."""
        try:
            ri = labels.index(row_label)
            ci = labels.index(col_label)
            if ri <= ci:
                ri, ci = ci, ri
            val = corr.values[ri, ci]
            rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                  linewidth=2, edgecolor=color,
                                  facecolor='none', zorder=5)
            ax.add_patch(rect)
            ax.text(ci + 1.35, ri, f'r = {val:.2f}', ha='left', va='center',
                    fontsize=7.5, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.85))
        except ValueError:
            pass

    highlight_cell('points_dif',    'elo_diff',    '#E53935')
    highlight_cell('mv_median_diff', 'mv_sum_diff', '#E53935')

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='both', colors=C_ANNOT, length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=C_ANNOT)
    cbar.outline.set_edgecolor(C_SPINE)

    ax.set_title(
        'Pearson correlation — model candidate features  (training data, pre-WC2022)',
        fontsize=13, fontweight='bold', pad=14,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
