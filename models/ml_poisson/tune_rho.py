"""
Tune Dixon-Coles rho parameter for ML-Poisson model against WC 2022.

rho controls the low-score correction in the score matrix:
  - More negative rho -> higher P(0-0) and P(1-1) -> more draws predicted
  - rho = 0 -> standard independent Poisson (no correction)

Grid search over rho values, reporting RPS, log-loss, accuracy and
predicted draw count for each value.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from models.ml_poisson.model import MLPoissonModel

WC_2022_START = pd.Timestamp('2022-11-20')
WC_2022_END   = pd.Timestamp('2022-12-18')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


def actual_outcome(home_score, away_score):
    if home_score > away_score: return 'home_win'
    elif home_score == away_score: return 'draw'
    return 'away_win'


def rps(probs, outcome):
    actual_vec = [1.0 if o == outcome else 0.0 for o in OUTCOME_ORDER]
    pred_cum   = np.cumsum([probs[o] for o in OUTCOME_ORDER])
    actual_cum = np.cumsum(actual_vec)
    return float(np.sum((pred_cum - actual_cum) ** 2) / (len(OUTCOME_ORDER) - 1))


def log_loss_single(probs, outcome):
    return -np.log(probs[outcome] + 1e-15)


def evaluate(model, wc22):
    rows = []
    for _, match in wc22.iterrows():
        probs   = model.predict_proba_row(match)
        outcome = actual_outcome(match['home_score'], match['away_score'])
        rows.append({
            'actual':    outcome,
            'predicted': max(probs, key=probs.get),
            'rps':       rps(probs, outcome),
            'log_loss':  log_loss_single(probs, outcome),
        })
    df = pd.DataFrame(rows)
    return {
        'accuracy':        df['correct'].mean() if 'correct' in df else (df['actual'] == df['predicted']).mean(),
        'rps':             df['rps'].mean(),
        'log_loss':        df['log_loss'].mean(),
        'pred_draws':      (df['predicted'] == 'draw').sum(),
        'correct_draws':   ((df['predicted'] == 'draw') & (df['actual'] == 'draw')).sum(),
        'actual_draws':    (df['actual'] == 'draw').sum(),
    }


def run():
    print("Loading data...")
    full_df  = pd.read_csv("data/past_wc/wc2022/ranked_database_with_features.csv")
    full_df['date'] = pd.to_datetime(full_df['date'])

    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)
    wc22     = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC_2022_START) &
        (full_df['date'] <= WC_2022_END)
    ].copy().reset_index(drop=True)

    print(f"Training on {len(train_df)} matches, evaluating on {len(wc22)} WC 2022 matches")
    print(f"Actual draws in WC 2022: {(wc22.apply(lambda r: actual_outcome(r['home_score'], r['away_score']), axis=1) == 'draw').sum()}")

    # Train regressor once — only rho changes between evaluations
    base_model = MLPoissonModel(rho=0.0)
    base_model.fit(train_df)
    print("Regressor trained.\n")

    rho_values = [-0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50]

    print(f"  {'rho':>6}  {'Accuracy':>9}  {'Log-Loss':>10}  {'RPS':>8}  {'Pred draws':>11}  {'Correct draws':>14}")
    print(f"  {'-'*70}")

    results = []
    for rho_val in rho_values:
        base_model.rho = rho_val
        m = evaluate(base_model, wc22)
        results.append({'rho': rho_val, **m})
        print(f"  {rho_val:>6.2f}  {m['accuracy']:>9.3f}  {m['log_loss']:>10.4f}  {m['rps']:>8.4f}"
              f"  {m['pred_draws']:>8}/{len(wc22)}  {m['correct_draws']:>8}/{m['actual_draws']}")

    results_df = pd.DataFrame(results)
    best_rps = results_df.loc[results_df['rps'].idxmin()]
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]

    print(f"\n  Best RPS:      rho = {best_rps['rho']:.2f}  (RPS={best_rps['rps']:.4f})")
    print(f"  Best Accuracy: rho = {best_acc['rho']:.2f}  (acc={best_acc['accuracy']:.3f})")


if __name__ == "__main__":
    run()
