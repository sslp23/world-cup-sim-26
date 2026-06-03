"""
Calibration curves: predicted probability vs actual frequency.

One-vs-all approach across all 5 WC editions (320 matches per model):
  For each outcome class, predicted probabilities are binned and the
  fraction of times the event actually occurred is computed per bin.
  Perfect calibration = the diagonal (predicted == actual).

3 subplots (Home Win | Draw | Away Win), 5 model lines each.
Error bars: 95% Wilson CI. Bins with fewer than MIN_OBS are hidden.

Run from the project root:
    python -m viz.backtest_calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

OUTPUT_PATH = 'viz/output/backtest_calibration.png'

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
MODEL_ORDER = ['Ord_Logit', 'XGBoost', 'Ensemble', 'ML-Poisson', 'CatBoost']

WC_SHEETS  = ['WC 06', 'WC 10', 'WC 14', 'WC 18', 'WC 22']
CLASSES    = ['home_win', 'draw', 'away_win']
CLASS_LABELS = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
PROB_COLS    = {'home_win': 'P(HW)', 'draw': 'P(D)', 'away_win': 'P(AW)'}

N_BINS  = 10
MIN_OBS = 10

C_ANNOT = '#546E7A'
C_SPINE = '#CFD8DC'
C_DIAG  = '#B0BEC5'


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def load_model_data(path):
    xls = pd.read_excel(path, sheet_name=None)
    frames = []
    for sheet in WC_SHEETS:
        if sheet not in xls:
            continue
        df = xls[sheet][['Actual', 'P(HW)', 'P(D)', 'P(AW)']].copy()
        df = df.dropna(subset=['Actual', 'P(HW)']).reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def calibration_curve(probs, actuals, n_bins=N_BINS):
    """
    Returns (mean_predicted, fraction_positive, err_lo, err_hi, counts)
    using mean predicted probability per bin as the x-value.
    Bins with fewer than MIN_OBS observations are excluded.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        mask = (probs >= lo_e) & (probs <= hi_e if i == n_bins - 1 else probs < hi_e)
        n = int(mask.sum())
        if n < MIN_OBS:
            continue
        k = int(actuals[mask].sum())
        p_pred = float(probs[mask].mean())
        p_act  = k / n
        ci_lo, ci_hi = wilson_ci(k, n)
        out.append((p_pred, p_act, max(0.0, p_act - ci_lo), max(0.0, ci_hi - p_act), n))
    if not out:
        return [np.array([]) for _ in range(5)]
    arr = np.array(out)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4].astype(int)


def run():
    model_data = {m: load_model_data(p) for m, p in MODELS.items()}
    n_matches = len(model_data[MODEL_ORDER[0]])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for ax, cls in zip(axes, CLASSES):
        ax.set_facecolor('white')
        prob_col = PROB_COLS[cls]

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], color=C_DIAG, linewidth=1.4,
                linestyle='--', zorder=1, label='Perfect calibration')

        for model in MODEL_ORDER:
            df      = model_data[model]
            probs   = df[prob_col].values.astype(float)
            actuals = (df['Actual'] == cls).astype(int).values

            x, y, err_lo, err_hi, _ = calibration_curve(probs, actuals)
            if len(x) == 0:
                continue

            ax.plot(x, y, 'o-', color=MODEL_COLORS[model],
                    linewidth=1.5, markersize=5, alpha=0.90,
                    label=MODEL_NAMES[model], zorder=3)
            ax.errorbar(x, y, yerr=[err_lo, err_hi],
                        fmt='none', color=MODEL_COLORS[model],
                        capsize=3, linewidth=0.9, alpha=0.55, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted probability', fontsize=9, color=C_ANNOT)
        ax.set_ylabel('Actual frequency', fontsize=9, color=C_ANNOT)
        ax.set_title(CLASS_LABELS[cls], fontsize=11, fontweight='bold',
                     color=C_ANNOT, pad=6)
        ax.tick_params(labelsize=8, colors=C_ANNOT)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(True, which='major', color=C_SPINE, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_SPINE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend on first subplot only
    handles = [mpatches.Patch(color='none', label='')]   # spacer
    handles = (
        [plt.Line2D([0], [0], color=C_DIAG, linestyle='--', linewidth=1.2,
                    label='Perfect calibration')] +
        [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_NAMES[m])
         for m in MODEL_ORDER]
    )
    axes[0].legend(handles=handles, fontsize=7.5, loc='upper left',
                   framealpha=0.9, edgecolor=C_SPINE)

    fig.text(
        0.5, 0.01,
        f'{n_matches} matches  ·  WC 2006–2022  ·  one-vs-all per outcome  '
        f'·  {N_BINS} equal-width bins (min {MIN_OBS} obs)  ·  error bars: 95% Wilson CI',
        ha='center', fontsize=7.5, color=C_ANNOT, fontstyle='italic',
    )
    fig.suptitle(
        'Calibration — predicted probability vs actual frequency',
        fontsize=13, fontweight='bold', color=C_ANNOT, y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
