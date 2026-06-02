"""
EDA: ELO rating difference distribution by match outcome.

Shows how elo_diff separates home wins, draws, and away wins
in the training data (pre-WC2022).

Run from the project root:
    python -m viz.eda_elo_by_outcome
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_PATH = 'viz/output/eda_elo_by_outcome.png'
TRAIN_CUTOFF = '2022-11-20'

# Colour palette (consistent with dataset_overview.py)
C_HW     = '#1F4E79'   # dark blue — home win
C_DRAW   = '#90A4AE'   # blue-grey — draw
C_AW     = '#B0BEC5'   # lighter grey — away win
C_ANNOT  = '#546E7A'
C_SPINE  = '#CFD8DC'
C_MEDIAN = '#FFFFFF'

OUTCOME_LABELS = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
OUTCOME_COLORS = {0: C_HW, 1: C_DRAW, 2: C_AW}


def run():
    df = pd.read_csv('data/ranked_database_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    train = (
        df[df['date'] < TRAIN_CUTOFF]
        .dropna(subset=['elo_diff', 'home_score', 'away_score'])
        .copy()
    )
    train['result'] = train.apply(
        lambda r: 0 if r['home_score'] > r['away_score']
        else (1 if r['home_score'] == r['away_score'] else 2),
        axis=1,
    )

    groups = [train.loc[train['result'] == k, 'elo_diff'].values for k in [0, 1, 2]]
    labels = [OUTCOME_LABELS[k] for k in [0, 1, 2]]
    colors = [OUTCOME_COLORS[k] for k in [0, 1, 2]]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    positions = [1, 2, 3]

    # Violin bodies
    parts = ax.violinplot(groups, positions=positions, widths=0.7,
                          showmedians=False, showextrema=False)
    for body, col in zip(parts['bodies'], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor(col)
        body.set_linewidth(1.0)

    # Box-plot overlay
    bp = ax.boxplot(groups, positions=positions, widths=0.22,
                    patch_artist=True, notch=False,
                    manage_ticks=False,
                    medianprops=dict(color=C_MEDIAN, linewidth=2),
                    whiskerprops=dict(color=C_ANNOT, linewidth=0.8, linestyle='--'),
                    capprops=dict(color=C_ANNOT, linewidth=0.8),
                    flierprops=dict(marker='o', markersize=2,
                                    markerfacecolor=C_ANNOT, alpha=0.3,
                                    linestyle='none'),
                    boxprops=dict(linewidth=0))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

    # Median annotation
    for pos, grp, col in zip(positions, groups, colors):
        med = np.median(grp)
        ax.text(pos, med, f'{med:+.0f}', ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=6)

    # Zero reference line
    ax.axhline(0, color=C_SPINE, linewidth=1.0, linestyle='-', zorder=0)

    # Sample size annotations
    for pos, grp, col in zip(positions, groups, colors):
        ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] < -600 else -620,
                f'n={len(grp):,}', ha='center', va='top',
                fontsize=8, color=C_ANNOT)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('ELO difference  (home − away)', fontsize=10, color=C_ANNOT)
    ax.set_title(
        'ELO rating difference by match outcome  (training data, pre-WC2022)',
        fontsize=13, fontweight='bold', pad=12,
    )

    ax.tick_params(axis='both', labelsize=8, colors=C_ANNOT)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.grid(True, color=C_SPINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor=C_SPINE)

    # Training set note
    n_total = sum(len(g) for g in groups)
    ax.text(0.01, 0.97, f'{n_total:,} matches  ·  training set only',
            transform=ax.transAxes, fontsize=7.5, color=C_ANNOT,
            ha='left', va='top', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
