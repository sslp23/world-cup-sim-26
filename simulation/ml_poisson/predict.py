"""
Trains ML-Poisson on pre-WC26 data and predicts WC26 group stage probabilities.

Uses MLPoissonModel (XGBoost goal regressors + Dixon-Coles score matrix).
rho set via config.py (MLP_RHO).

Run from the project root:
    python -m simulation.ml_poisson.run
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from tqdm import tqdm

from models.ml_poisson.model import MLPoissonModel
from config import MLP_RHO

WC26_START    = pd.Timestamp('2026-06-11')
WC26_END      = pd.Timestamp('2026-06-27')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


def run(full_df: pd.DataFrame) -> list[dict]:
    """
    Train on pre-WC26 rows, predict on WC26 group stage matches.
    Returns list of prediction dicts, one per match.
    """
    train_df = full_df[full_df['date'] < WC26_START].reset_index(drop=True)
    wc26_df  = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC26_START) &
        (full_df['date'] <= WC26_END)
    ].copy().reset_index(drop=True)

    print(f'Training rows: {len(train_df)}  |  WC26 matches: {len(wc26_df)}')

    wc26_df.to_csv('data/wc_26_data.csv', index=False)
    print(f'WC26 dataset saved to data/wc_26_data.csv')

    print(f'Training ML-Poisson (rho={MLP_RHO})...')
    model = MLPoissonModel(rho=MLP_RHO)
    model.fit(train_df)

    print('Predicting...')
    predictions = []
    for _, match in tqdm(wc26_df.iterrows(), total=len(wc26_df)):
        probs    = model.predict_proba_row(match)
        pred_raw = max(probs, key=probs.get)
        pred_lbl = {
            'home_win': match['home_team'],
            'draw':     'Draw',
            'away_win': match['away_team'],
        }[pred_raw]

        predictions.append({
            'date':       str(match['date'].date()),
            'home':       match['home_team'],
            'away':       match['away_team'],
            'p_home_win': round(probs['home_win'], 3),
            'p_draw':     round(probs['draw'], 3),
            'p_away_win': round(probs['away_win'], 3),
            'predicted':  pred_lbl,
            'confidence': f"{probs[pred_raw]:.1%}",
            '_pred_raw':  pred_raw,
        })

    hw = sum(1 for p in predictions if p['_pred_raw'] == 'home_win')
    dr = sum(1 for p in predictions if p['_pred_raw'] == 'draw')
    aw = sum(1 for p in predictions if p['_pred_raw'] == 'away_win')
    print(f'Results: {hw} home wins  {dr} draws  {aw} away wins')

    return predictions
