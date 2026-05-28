"""
Tuning Dixon-Coles time decay (xi) against FIFA World Cup 2022.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from models.dixon_coles.model import DixonColes

WC_2022_START = pd.Timestamp('2022-11-20')
WC_2022_END   = pd.Timestamp('2022-12-18')

def actual_outcome(home_score, away_score):
    if home_score > away_score: return 'home_win'
    elif home_score == away_score: return 'draw'
    return 'away_win'

def log_loss_single(probs, outcome):
    return -np.log(probs[outcome] + 1e-15)

def run_tuning():
    print("Loading data...")
    full_df = pd.read_csv("data/past_wc/wc2022/ranked_database_with_features.csv")
    full_df['date'] = pd.to_datetime(full_df['date'])

    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)
    wc22 = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC_2022_START) &
        (full_df['date'] <= WC_2022_END)
    ].copy().reset_index(drop=True)

    xi_values = [0.001, 0.0005, 0.0001, 0.00005]
    tuning_results = []

    for xi in xi_values:
        print(f"\nTesting xi = {xi}...")
        model = DixonColes(xi=xi)
        model.fit(train_df, ref_date=WC_2022_START)

        losses = []
        correct = 0

        for _, match in wc22.iterrows():
            probs = model.predict_outcome_probs(match['home_team'], match['away_team'], neutral=True)
            outcome = actual_outcome(match['home_score'], match['away_score'])
            
            losses.append(log_loss_single(probs, outcome))
            if max(probs, key=probs.get) == outcome:
                correct += 1
        
        avg_ll = np.mean(losses)
        acc = correct / len(wc22)
        print(f"  Accuracy: {acc:.3f}, Log-Loss: {avg_ll:.4f}")
        
        tuning_results.append({
            'xi': xi,
            'accuracy': acc,
            'log_loss': avg_ll
        })

    tuning_df = pd.DataFrame(tuning_results)
    print("\n" + "="*30)
    print("      TUNING RESULTS")
    print("="*30)
    print(tuning_df.sort_values('log_loss'))

if __name__ == "__main__":
    run_tuning()
