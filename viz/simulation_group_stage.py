"""
WC 2026 predicted group stage standings — 3×4 grid of group cards.

Each card shows the four teams in predicted finishing order with
W/D/L/Pts, color-coded by qualification status.

Data: simulation/output/wc_26_catboost.xlsx

Run from the project root:
    python -m viz.simulation_group_stage
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = 'viz/output/simulation_group_stage.png'
SIM_FILE    = 'simulation/output/wc_26_catboost.xlsx'

GROUPS = list('ABCDEFGHIJKL')

# Green palette — matches the playoff bracket colour scheme
C_1ST      = '#1B5E20'   # very dark green  — 1st (advance)
C_2ND      = '#388E3C'   # dark green       — 2nd (advance)
C_3RD_ADV  = '#81C784'   # medium green     — 3rd best (advance)
C_3RD_OUT  = '#E8F5E9'   # very light green — 3rd (eliminated)
C_4TH      = '#F5F5F5'   # light grey       — 4th (eliminated)
C_HEADER   = '#2E7D32'   # group header

C_LIGHT  = '#FFFFFF'
C_DARK   = '#1A1A1A'
C_ANNOT  = '#546E7A'
C_SPINE  = '#CFD8DC'

# Abbreviate team names that would overflow the card
ABBREV = {
    'Bosnia and Herzegovina': 'Bosnia & Herz.',
    'United States':          'USA',
    'Czech Republic':         'Czech Rep.',
}


def shorten(name):
    if name in ABBREV:
        return ABBREV[name]
    return name if len(name) <= 14 else name[:13] + '.'


def row_colors(pos, team, adv_3rds):
    if pos == 1:
        return C_1ST, C_LIGHT
    if pos == 2:
        return C_2ND, C_LIGHT
    if pos == 3:
        return (C_3RD_ADV, C_DARK) if team in adv_3rds else (C_3RD_OUT, C_DARK)
    return C_4TH, C_DARK


def load_data():
    xls = pd.read_excel(SIM_FILE, sheet_name=None)

    standings = {}
    for g in GROUPS:
        raw = xls[f'Group {g}']
        tbl = raw.iloc[11:15, :7].copy()
        tbl.columns = ['pos', 'team', 'mp', 'w', 'd', 'l', 'pts']
        tbl = tbl.reset_index(drop=True)
        for c in ['pos', 'w', 'd', 'l', 'pts']:
            tbl[c] = tbl[c].astype(int)
        standings[g] = tbl

    raw_b3 = xls['BEST 3rds']
    col_team = raw_b3.columns[2]
    col_qual = raw_b3.columns[11]
    adv_mask = raw_b3[col_qual] == 'Advance to knockout stage'
    adv_3rds = set(raw_b3.loc[adv_mask, col_team].values)

    return standings, adv_3rds


def draw_group(ax, letter, tbl, adv_3rds):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    # Header bar
    ax.add_patch(FancyBboxPatch(
        (0, 5.2), 10, 0.9,
        boxstyle='round,pad=0.04',
        facecolor=C_HEADER, edgecolor='none', zorder=2,
    ))
    ax.text(5, 5.65, f'GROUP  {letter}', ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='white', zorder=3)

    # Column label row
    for x, lbl in [(0.5, '#'), (7.3, 'W'), (8.2, 'D'), (9.0, 'L'), (9.8, 'Pts')]:
        ax.text(x, 5.0, lbl, ha='center', va='center',
                fontsize=6.5, color=C_ANNOT, fontstyle='italic')
    ax.text(1.1, 5.0, 'Team', ha='left', va='center',
            fontsize=6.5, color=C_ANNOT, fontstyle='italic')

    # Team rows
    for i, row in tbl.iterrows():
        y_mid = 4.1 - i * 1.0    # centres at 4.1, 3.1, 2.1, 1.1
        fc, tc = row_colors(int(row['pos']), row['team'], adv_3rds)

        ax.add_patch(FancyBboxPatch(
            (0, y_mid - 0.44), 10, 0.88,
            boxstyle='round,pad=0.02',
            facecolor=fc, edgecolor='white', linewidth=0.5, zorder=2,
        ))

        ax.text(0.5,  y_mid, str(int(row['pos'])), ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=tc, zorder=3)
        ax.text(1.1,  y_mid, shorten(row['team']),   ha='left',   va='center',
                fontsize=8,   color=tc, zorder=3)
        ax.text(7.3,  y_mid, str(int(row['w'])),     ha='center', va='center',
                fontsize=8,   color=tc, zorder=3)
        ax.text(8.2,  y_mid, str(int(row['d'])),     ha='center', va='center',
                fontsize=8,   color=tc, zorder=3)
        ax.text(9.0,  y_mid, str(int(row['l'])),     ha='center', va='center',
                fontsize=8,   color=tc, zorder=3)
        ax.text(9.8,  y_mid, str(int(row['pts'])),   ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=tc, zorder=3)

        # Star marker on best-3rd teams that advance
        if int(row['pos']) == 3 and row['team'] in adv_3rds:
            ax.text(9.8, y_mid + 0.36, '★', ha='center', va='center',
                    fontsize=5.5, color='#2E7D32', zorder=4)


def run():
    standings, adv_3rds = load_data()

    fig, axes = plt.subplots(3, 4, figsize=(17, 9.5))
    fig.patch.set_facecolor('white')

    for i, g in enumerate(GROUPS):
        draw_group(axes[i // 4, i % 4], g, standings[g], adv_3rds)

    # Spacing between cards
    plt.subplots_adjust(wspace=0.08, hspace=0.12)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_1ST,     label='1st — advances'),
        mpatches.Patch(color=C_2ND,     label='2nd — advances'),
        mpatches.Patch(color=C_3RD_ADV, label='3rd ★ — advances (best 3rd)'),
        mpatches.Patch(color=C_3RD_OUT, edgecolor=C_SPINE, linewidth=0.5,
                       label='3rd — eliminated'),
        mpatches.Patch(color=C_4TH,     edgecolor=C_SPINE, linewidth=0.5,
                       label='4th — eliminated'),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, 0.005),
               framealpha=0.95, edgecolor=C_SPINE)

    fig.suptitle(
        'WC 2026 — Predicted Group Stage Standings  (CatBoost model)',
        fontsize=14, fontweight='bold', color=C_ANNOT, y=0.995,
    )
    plt.tight_layout(rect=[0, 0.055, 1, 0.993])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
