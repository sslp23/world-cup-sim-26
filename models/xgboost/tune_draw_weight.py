"""
Tunes the draw_weight parameter of XGBoostPredictor against WC 2022.

draw_weight multiplies the inverse-frequency draw class weight:
  - 0.0 = no draw upweighting (model free to under-predict draws)
  - 1.0 = full inverse-frequency compensation
  - >1.0 = extra draw boost beyond what frequency alone suggests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from models.xgboost.model import XGBoostPredictor

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
            'correct':  max(probs, key=probs.get) == outcome,
            'log_loss': log_loss_single(probs, outcome),
            'rps':      rps(probs, outcome),
        })
    df = pd.DataFrame(rows)
    return df['correct'].mean(), df['log_loss'].mean(), df['rps'].mean()


if __name__ == "__main__":
    print("Loading data...")
    full_df = pd.read_csv("data/ranked_database_with_features.csv")
    full_df['date'] = pd.to_datetime(full_df['date'])

    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)
    wc22 = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC_2022_START) &
        (full_df['date'] <= WC_2022_END)
    ].copy().reset_index(drop=True)

    draw_weights = np.arange(0.6, 0.8, 0.01)
    # draw_weights = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    results = []

    for dw in draw_weights:
        model = XGBoostPredictor(draw_weight=dw)
        model.fit(train_df)
        acc, ll, rps_val = evaluate(model, wc22)
        results.append({'draw_weight': dw, 'accuracy': acc, 'log_loss': ll, 'rps': rps_val})
        print(f"  draw_weight={dw:.2f}  acc={acc:.3f}  log_loss={ll:.4f}  rps={rps_val:.4f}")

    print()
    df_results = pd.DataFrame(results)
    print(df_results.sort_values('rps').to_string(index=False))
