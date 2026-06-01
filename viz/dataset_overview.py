"""
Dataset overview: international match volume per year (1872–2026).

Highlights:
  - FIFA World Cup editions (dark bar overlay)
  - Post-WC2022 simulation window (right-hand shaded region)
  - COVID-19 dip (2020)
  - WWII gap (1940–1945)

Run from the project root:
    python -m viz.dataset_overview
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_PATH = 'viz/output/dataset_overview.png'

WC_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966,
    1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998,
    2002, 2006, 2010, 2014, 2018, 2022, 2026,
]

# Colour palette
C_BAR      = '#90A4AE'   # blue-grey — all matches
C_WC       = '#1F4E79'   # dark blue — WC editions
C_WINDOW   = '#E3F2FD'   # very light blue — simulation window
C_ANNOT    = '#546E7A'   # annotation text
C_SPINE    = '#CFD8DC'


def run():
    df = pd.read_csv('data/international_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    per_year   = df.groupby('year').size().reset_index(name='matches')
    wc_per_year = (
        df[df['tournament'] == 'FIFA World Cup']
        .groupby('year').size()
        .reset_index(name='wc_matches')
    )
    per_year = per_year.merge(wc_per_year, on='year', how='left').fillna(0)
    per_year['wc_matches'] = per_year['wc_matches'].astype(int)

    years   = per_year['year'].values
    matches = per_year['matches'].values
    wc_m    = per_year['wc_matches'].values
    is_wc   = np.isin(years, WC_YEARS)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Simulation window shade (post-WC2022 training window)
    ax.axvspan(2022.4, years.max() + 0.6, color=C_WINDOW, zorder=0, linewidth=0)

    # All-matches bars
    ax.bar(years, matches, color=C_BAR, width=0.8, zorder=2, label='All matches')

    # WC-edition overlay (stacked on top in dark blue)
    ax.bar(years[is_wc], wc_m[is_wc], color=C_WC, width=0.8, zorder=3, label='FIFA World Cup matches')

    # ── Annotations ───────────────────────────────────────────────────────────
    # WWII gap
    ax.annotate(
        'WWII\n(1940–45)',
        xy=(1942.5, 15), xytext=(1935, 260),
        fontsize=8, color=C_ANNOT, ha='center',
        arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=0.8),
    )

    # COVID dip
    ax.annotate(
        'COVID-19\n(2020)',
        xy=(2020, per_year.loc[per_year['year'] == 2020, 'matches'].values[0] + 10),
        xytext=(2016.5, 680),
        fontsize=8, color=C_ANNOT, ha='center',
        arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=0.8),
    )

    # Simulation window label — placed mid-height inside shaded area
    y_top = matches.max() * 1.18
    ax.text(
        2024.3, y_top * 0.60,
        'Simulation\nwindow',
        fontsize=8, color=C_WC, ha='center', va='center', fontstyle='italic',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2),
    )

    # WC edition tick marks on x-axis
    for yr in WC_YEARS:
        ax.axvline(yr, color=C_WC, linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)

    # ── Style ──────────────────────────────────────────────────────────────────
    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(0, matches.max() * 1.18)

    ax.set_xlabel('Year', fontsize=10, color=C_ANNOT)
    ax.set_ylabel('Matches', fontsize=10, color=C_ANNOT)
    ax.set_title(
        'International football matches in the dataset  (1872 – 2026)',
        fontsize=13, fontweight='bold', pad=12,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.tick_params(axis='both', labelsize=8, colors=C_ANNOT)

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, color=C_SPINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend_handles = [
        mpatches.Patch(color=C_BAR, label='All international matches'),
        mpatches.Patch(color=C_WC,  label='FIFA World Cup matches'),
        mpatches.Patch(facecolor=C_WINDOW, edgecolor=C_WC, linewidth=0.8,
                       label='Simulation window (post-WC2022)'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor=C_SPINE)

    # Total match count — bottom right
    total = len(df)
    ax.text(0.99, 0.03, f'{total:,} matches total',
            transform=ax.transAxes, fontsize=8, color=C_ANNOT,
            ha='right', va='bottom', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
