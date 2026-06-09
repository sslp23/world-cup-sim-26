"""
Cumulative accuracy line chart: game by game from WC 2006 to WC 2022.

Each line shows the running mean accuracy of a model, starting from the
first group-stage match in WC 2006 and ending at the WC 2022 final.
Lines converge to each model's overall average accuracy (higher is better).

Run from the project root:
    python -m viz.backtest_cumulative_accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_PATH = 'viz/output/backtest_cumulative_accuracy.png'

MODELS = {
    'Ord_Logit':  'backtest/output/Ord_Logit.xlsx',
    'XGBoost':    'backtest/output/XGBoost.xlsx',
    'Ensemble':   'backtest/output/Ensemble.xlsx',
    'ML-Poisson': 'backtest/output/ML-Poisson.xlsx',
    'CatBoost':   'backtest/output/CatBoost.xlsx',
}

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

# Draw lowest-accuracy first (underneath), highest-accuracy last (on top)
MODEL_ORDER = ['Ensemble', 'XGBoost', 'CatBoost', 'ML-Poisson', 'Ord_Logit']

WC_SHEETS  = ['WC 06', 'WC 10', 'WC 14', 'WC 18', 'WC 22']
WC_DISPLAY = ['WC 2006', 'WC 2010', 'WC 2014', 'WC 2018', 'WC 2022']

C_ANNOT = '#546E7A'
C_SPINE = '#CFD8DC'
C_SHADE = '#EEF5FD'


def load_series(path):
    xls = pd.read_excel(path, sheet_name=None)
    frames, boundaries, n = [], [], 0
    for sheet in WC_SHEETS:
        if sheet not in xls:
            continue
        df = xls[sheet][['Result']].copy()
        df = df.dropna(subset=['Result']).reset_index(drop=True)
        n += len(df)
        frames.append(df)
        boundaries.append(n)

    all_matches = pd.concat(frames, ignore_index=True)
    correct = (all_matches['Result'] == 'OK').astype(int)
    idx = np.arange(1, len(correct) + 1)
    cum_acc = correct.cumsum().values / idx
    return cum_acc, boundaries


def run():
    series = {}
    boundaries = None
    for model in MODEL_ORDER:
        cum_acc, bnd = load_series(MODELS[model])
        series[model] = cum_acc
        if boundaries is None:
            boundaries = bnd

    total = boundaries[-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    data_min = min(s.min() for s in series.values())
    data_max = max(s.max() for s in series.values())
    Y_MIN = max(0.0, round(data_min * 0.94, 2))
    Y_MAX = min(1.0, round(data_max * 1.06, 2))
    ax.set_xlim(0, total + 55)
    ax.set_ylim(Y_MIN, Y_MAX)

    # Alternating WC era shading + labels
    prev = 0
    for i, (bnd, lbl) in enumerate(zip(boundaries, WC_DISPLAY)):
        if i % 2 == 0:
            ax.axvspan(prev, bnd, facecolor=C_SHADE, alpha=0.55,
                       linewidth=0, zorder=0)
        mid = (prev + bnd) / 2
        ax.text(mid, Y_MAX - 0.005, lbl, ha='center', va='top',
                fontsize=8.5, fontweight='bold', color=C_ANNOT)
        if i < len(boundaries) - 1:
            ax.axvline(bnd, color=C_SPINE, linewidth=1.0,
                       linestyle='--', alpha=0.8, zorder=1)
        prev = bnd

    # Spread right-side labels with a minimum vertical gap
    final_vals = {m: series[m][-1] for m in MODEL_ORDER}
    sorted_models = sorted(MODEL_ORDER, key=lambda m: final_vals[m])  # ascending

    MIN_GAP = 0.008
    ys = sorted([final_vals[m] for m in MODEL_ORDER])
    adjusted = list(ys)
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < MIN_GAP:
            adjusted[i] = adjusted[i - 1] + MIN_GAP
    label_y = {model: adj for model, adj in zip(sorted_models, adjusted)}

    # Draw lines
    for model in MODEL_ORDER:
        cum_acc = series[model]
        x = np.arange(1, len(cum_acc) + 1)
        ax.plot(x, cum_acc, color=MODEL_COLORS[model], linewidth=1.7,
                zorder=3, alpha=0.92)
        ax.scatter([x[-1]], [cum_acc[-1]], color=MODEL_COLORS[model],
                   s=45, zorder=5)
        ly = label_y[model]
        if abs(ly - cum_acc[-1]) > 0.001:
            ax.plot([x[-1], x[-1] + 3], [cum_acc[-1], ly],
                    color=MODEL_COLORS[model], lw=0.7, zorder=4)
        ax.text(x[-1] + 5, ly,
                f'{MODEL_NAMES[model]}  {cum_acc[-1]:.1%}',
                va='center', fontsize=8.5, color=MODEL_COLORS[model],
                fontweight='bold')

    ax.text(0.01, 0.03, '↑ higher is better', transform=ax.transAxes,
            fontsize=8, color=C_ANNOT, fontstyle='italic', va='bottom')

    ax.set_xlabel('Match  (chronological: WC 2006 → WC 2022)',
                  fontsize=10, color=C_ANNOT)
    ax.set_ylabel('Cumulative accuracy (running mean)', fontsize=10, color=C_ANNOT)
    ax.set_title(
        'Model accuracy — cumulative game by game across 5 World Cups',
        fontsize=13, fontweight='bold', pad=12,
    )

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.yaxis.grid(True, color=C_SPINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8, colors=C_ANNOT)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(32))

    for spine in ax.spines.values():
        spine.set_edgecolor(C_SPINE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
