"""
EDA: Pearson correlation heatmap for model candidate features.

Lower-triangular heatmap. Key finding: elo_diff and points_dif
have r=0.92 → points_dif was excluded from all final models.

Run from the project root:
    python -m viz.eda_correlation_heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_PATH = 'viz/output/eda_correlation_heatmap.png'
TRAIN_CUTOFF = '2022-11-20'

C_ANNOT  = '#546E7A'
C_SPINE  = '#CFD8DC'

# Columns to include and their short display labels
COLS_AND_LABELS = [
    ('elo_diff',                           'elo_diff'),
    ('pi_diff',                            'pi_diff'),
    ('points_dif',                         'points_dif'),
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
    df = pd.read_csv('data/ranked_database_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    train = (
        df[df['date'] < TRAIN_CUTOFF]
        .dropna(subset=['elo_diff', 'home_score', 'away_score'])
        .copy()
    )

    cols   = [c for c, _ in COLS_AND_LABELS if c in train.columns]
    labels = [l for c, l in COLS_AND_LABELS if c in train.columns]

    corr = train[cols].corr(method='pearson')

    # Mask upper triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, True)

    n = len(cols)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Custom diverging colormap: red (−1) → white (0) → dark blue (+1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'rb_div', ['#C62828', '#FFFFFF', '#1F4E79'], N=256
    )

    masked_corr = corr.copy()
    masked_corr.values[mask] = np.nan

    im = ax.imshow(masked_corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Cell annotations
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            val = corr.values[i, j]
            text_col = 'white' if abs(val) > 0.65 else C_ANNOT
            weight = 'bold' if abs(val) > 0.85 else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=text_col, fontweight=weight)

    # Highlight the elo_diff / points_dif cell border
    # In a lower-triangular layout the cell sits at (row=points_dif, col=elo_diff)
    try:
        ri = labels.index('points_dif')   # row — the larger index
        ci = labels.index('elo_diff')     # col — the smaller index
        rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                              linewidth=2, edgecolor='#E53935',
                              facecolor='none', zorder=5)
        ax.add_patch(rect)
        ax.text(ci + 1.3, ri, 'r = 0.92', ha='left', va='center',
                fontsize=7.5, color='#E53935', fontweight='bold')
    except ValueError:
        pass

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='both', colors=C_ANNOT, length=0)

    # Colorbar
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
