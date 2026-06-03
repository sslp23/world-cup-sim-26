"""
Backtest results: grouped bar chart comparing model performance
across 5 World Cup editions (WC 2006 – 2022).

Three vertically stacked subplots: Accuracy (↑), RPS (↓), Log-Loss (↓).
X-axis: WC editions (chronological). Bars: one per model.

Run from the project root:
    python -m viz.backtest_grouped_bars
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_PATH = 'viz/output/backtest_grouped_bars.png'

MODEL_ORDER = ['Ord_Logit', 'XGBoost', 'Ensemble', 'ML-Poisson', 'CatBoost']
MODEL_NAMES = {
    'Ord_Logit':  'Ord. Logit',
    'XGBoost':    'XGBoost',
    'Ensemble':   'Ensemble',
    'ML-Poisson': 'ML-Poisson',
    'CatBoost':   'CatBoost',
}
MODEL_COLORS = {
    'Ord_Logit':  '#78909C',
    'XGBoost':    '#6A1B9A',
    'Ensemble':   '#00897B',
    'ML-Poisson': '#1976D2',
    'CatBoost':   '#1F4E79',
}

# Chronological order for x-axis; columns in avg_performance match these names
WC_CHRONO  = ['WC06', 'WC10', 'WC14', 'WC18', 'WC22']
WC_LABELS  = ['WC 2006', 'WC 2010', 'WC 2014', 'WC 2018', 'WC 2022']

C_ANNOT = '#546E7A'
C_SPINE = '#CFD8DC'
C_BG    = '#F5F9FF'


def load_data():
    # File has no real header row — row 0 is the section label ("ACCURACY" etc.)
    raw = pd.read_excel('backtest/avg_performance.xlsx', header=None)
    col_names = ['model', 'WC22', 'WC18', 'WC14', 'WC10', 'WC06', 'avg']

    # Row layout (0-indexed with header=None):
    #  0: "ACCURACY" label     1: embedded header (Model, WC22 …)
    #  2-6: accuracy data
    #  8: "LOG-LOSS" label     9: header    10-14: data
    # 16: "RPS" label         17: header   18-22: data
    def block(start):
        df = raw.iloc[start:start + 5, :7].copy()
        df.columns = col_names
        return df.set_index('model')

    acc = block(2)
    for c in ['WC22', 'WC18', 'WC14', 'WC10', 'WC06', 'avg']:
        acc[c] = acc[c].astype(str).str.rstrip('%').astype(float)

    ll  = block(10)
    rps = block(18)
    for df in [ll, rps]:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return acc, ll, rps


def draw_bars(ax, df, higher_better, ylabel, ylim):
    x = np.arange(len(WC_CHRONO))
    n = len(MODEL_ORDER)
    bar_w = 0.14
    offsets = np.linspace(-(n - 1) / 2 * bar_w, (n - 1) / 2 * bar_w, n)

    for mi, model in enumerate(MODEL_ORDER):
        if model not in df.index:
            continue
        vals = [df.loc[model, wc] for wc in WC_CHRONO]
        ax.bar(x + offsets[mi], vals, width=bar_w,
               color=MODEL_COLORS[model], alpha=0.88, zorder=3,
               label=MODEL_NAMES[model])

    # Best-per-edition marker (small tick above/below the winning bar)
    for xi, wc in enumerate(WC_CHRONO):
        col_vals = {m: df.loc[m, wc] for m in MODEL_ORDER if m in df.index}
        best_model = max(col_vals, key=col_vals.get) if higher_better \
                     else min(col_vals, key=col_vals.get)
        mi = MODEL_ORDER.index(best_model)
        best_val = col_vals[best_model]
        offset = bar_w * 0.6 if higher_better else -bar_w * 0.6
        ax.plot(xi + offsets[mi], best_val + offset, marker='*', markersize=5,
                color='white', markeredgecolor=MODEL_COLORS[best_model],
                markeredgewidth=0.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(WC_LABELS, fontsize=9, color=C_ANNOT)
    ax.set_ylabel(ylabel, fontsize=9, color=C_ANNOT)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelsize=8, colors=C_ANNOT)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid(True, color=C_SPINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')

    direction = '↑ higher is better' if higher_better else '↓ lower is better'
    ax.text(0.99, 0.97, direction, transform=ax.transAxes,
            fontsize=7.5, color=C_ANNOT, ha='right', va='top', fontstyle='italic')

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def run():
    acc_df, ll_df, rps_df = load_data()

    fig, (ax_acc, ax_rps, ax_ll) = plt.subplots(3, 1, figsize=(13, 11),
                                                 facecolor='white')
    fig.patch.set_facecolor('white')

    draw_bars(ax_acc, acc_df, higher_better=True,
              ylabel='Accuracy (%)', ylim=(42, 70))
    ax_acc.set_title('Accuracy', fontsize=11, fontweight='bold',
                     color=C_ANNOT, pad=5, loc='left')

    draw_bars(ax_rps, rps_df, higher_better=False,
              ylabel='RPS', ylim=(0.14, 0.24))
    ax_rps.set_title('RPS', fontsize=11, fontweight='bold',
                     color=C_ANNOT, pad=5, loc='left')

    draw_bars(ax_ll, ll_df, higher_better=False,
              ylabel='Log-Loss', ylim=(0.82, 1.22))
    ax_ll.set_title('Log-Loss', fontsize=11, fontweight='bold',
                    color=C_ANNOT, pad=5, loc='left')

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_NAMES[m])
               for m in MODEL_ORDER]
    fig.legend(handles=handles, fontsize=9, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), framealpha=0.9, edgecolor=C_SPINE)

    fig.suptitle(
        'Model performance across 5 World Cup editions  (2006 – 2022)  ·  ★ = best per edition',
        fontsize=13, fontweight='bold', color=C_ANNOT, y=0.998,
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.996])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
