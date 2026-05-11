"""
Experiment: effect of training window size on model performance.

Tests three training windows, all evaluated on the same 64 WC 2022 matches:
  - Baseline : after WC 2018 (2018-08-01 → 2022-11-19)
  - Window B : after WC 2010 (2010-07-12 → 2022-11-19)
  - Window A : after WC 2006 (2006-07-10 → 2022-11-19)

ELO and pi-ratings are always computed from full match history (1872–present)
and reused from the existing CSVs — only the ranked database (FIFA-ranked
matches with form features) changes per window.

Models compared: XGBoost, CatBoost, ML-Poisson, Ensemble (XGB+CB+MLP).

Run from the project root:
    python -m experiments.training_window_test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from data_pipeline.features_creator import FeaturesCreator
from models.xgboost.model       import XGBoostPredictor
from models.catboost.model      import CatBoostPredictor
from models.ml_poisson.model    import MLPoissonModel

WC_2022_START = pd.Timestamp('2022-11-20')
WC_2022_END   = pd.Timestamp('2022-12-18')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']

WINDOWS = [
    ('Baseline (post-WC18)', pd.Timestamp('2018-08-01')),
    ('post-WC10',            pd.Timestamp('2010-07-12')),
    ('post-WC06',            pd.Timestamp('2006-07-10')),
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def actual_outcome(home_score, away_score):
    if home_score > away_score:    return 'home_win'
    elif home_score == away_score: return 'draw'
    return 'away_win'


def rps(probs, outcome):
    actual_vec = [1.0 if o == outcome else 0.0 for o in OUTCOME_ORDER]
    pred_cum   = np.cumsum([probs[o] for o in OUTCOME_ORDER])
    actual_cum = np.cumsum(actual_vec)
    return float(np.sum((pred_cum - actual_cum) ** 2) / (len(OUTCOME_ORDER) - 1))


def log_loss_single(probs, outcome):
    return -np.log(probs[outcome] + 1e-15)


def avg_probs(*prob_dicts):
    result = {o: 0.0 for o in OUTCOME_ORDER}
    for p in prob_dicts:
        for o in OUTCOME_ORDER:
            result[o] += p[o] / len(prob_dicts)
    return result


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build ranked_database_with_features for matches from start_date onwards.
    Reuses existing elo_ratings.csv and pi_ratings.csv (full-history ratings).
    """
    # --- Match results filtered to window ---
    results_full = pd.read_csv("data/international_results.csv")
    results_full['date'] = pd.to_datetime(results_full['date'])
    df = results_full[results_full['date'] >= start_date].reset_index(drop=True)

    # --- FIFA rankings filtered to window ---
    rank = pd.read_csv("data/resulting_data.csv", low_memory=False)
    rank = rank[['rank', 'nation_full_name', 'points', 'rank_date']]
    rank['rank_date'] = pd.to_datetime(rank['rank_date'])
    rank['points'] = pd.to_numeric(rank['points'], errors='coerce')
    rank = rank[rank['rank_date'] >= start_date].reset_index(drop=True)
    rank['nation_full_name'] = (
        rank['nation_full_name']
        .str.replace('Czechia', 'Czech Republic')
        .str.replace('IR Iran', 'Iran')
        .str.replace('Korea Republic', 'South Korea')
        .str.replace('USA', 'United States')
        .str.replace('Bosnia-Herzegovina', 'Bosnia and Herzegovina')
        .str.replace('Turkiye', 'Turkey')
        .str.replace('China', 'China PR', regex=False)
    )
    rank = (rank.set_index('rank_date')
                .groupby('nation_full_name', group_keys=False)
                .resample('D').first().ffill().reset_index())

    # --- Merge matches with rankings ---
    ranked = df.merge(rank, left_on=['date', 'home_team'],
                      right_on=['rank_date', 'nation_full_name']).drop(['rank_date', 'nation_full_name'], axis=1)
    ranked = ranked.merge(rank, left_on=['date', 'away_team'],
                          right_on=['rank_date', 'nation_full_name'],
                          suffixes=('_home', '_away')).drop(['rank_date', 'nation_full_name'], axis=1)

    # --- Features (form MAs, ELO, pi-ratings) ---
    # Save to a temp CSV so FeaturesCreator can read it (it requires a csv_path)
    tmp_path = f"data/_tmp_ranked_{start_date.date()}.csv"
    ranked.to_csv(tmp_path, index=False)

    elo_df = pd.read_csv("data/elo_ratings.csv")
    pi_df  = pd.read_csv("data/pi_ratings.csv")

    creator = FeaturesCreator(csv_path=tmp_path, conf_path="data/resulting_data.csv")
    df_feat = creator.create_all_features(elo_df=elo_df, pi_df=pi_df)

    os.remove(tmp_path)
    return df_feat


# ── Model evaluator ────────────────────────────────────────────────────────────

def evaluate_models(full_df: pd.DataFrame, label: str) -> dict:
    """Train all models and evaluate on WC 2022. Returns metrics dict."""
    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)
    wc22 = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC_2022_START) &
        (full_df['date'] <= WC_2022_END)
    ].copy().reset_index(drop=True)

    print(f"  Training rows: {len(train_df)}  |  WC 2022 matches: {len(wc22)}")

    # Train
    xgb = XGBoostPredictor(draw_weight=0.75)
    xgb.fit(train_df)

    cb = CatBoostPredictor(draw_weight=0.7)
    cb.fit(train_df)

    mlp = MLPoissonModel(rho=-0.30)
    mlp.fit(train_df)

    # Evaluate each model + ensemble
    def score_model(get_probs_fn):
        accs, lls, rpss = [], [], []
        for _, match in wc22.iterrows():
            probs   = get_probs_fn(match)
            outcome = actual_outcome(match['home_score'], match['away_score'])
            accs.append(int(max(probs, key=probs.get) == outcome))
            lls.append(log_loss_single(probs, outcome))
            rpss.append(rps(probs, outcome))
        return {
            'accuracy': np.mean(accs),
            'log_loss': np.mean(lls),
            'rps':      np.mean(rpss),
            'correct':  int(np.sum(accs)),
            'n':        len(accs),
        }

    results = {
        'xgboost':   score_model(lambda m: xgb.predict_proba_row(m)),
        'catboost':  score_model(lambda m: cb.predict_proba_row(m)),
        'ml_poisson':score_model(lambda m: mlp.predict_proba_row(m)),
        'ensemble':  score_model(lambda m: avg_probs(
                         xgb.predict_proba_row(m),
                         cb.predict_proba_row(m),
                         mlp.predict_proba_row(m))),
    }
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    all_results = {}

    for label, start_date in WINDOWS:
        print(f"\n{'='*60}")
        print(f"  Window: {label}  (from {start_date.date()})")
        print(f"{'='*60}")
        print("  Building dataset...")
        df = build_dataset(start_date)
        metrics = evaluate_models(df, label)
        all_results[label] = metrics
        print(f"  Done.")

    # ── Summary table ──────────────────────────────────────────────────────────
    models = ['xgboost', 'catboost', 'ml_poisson', 'ensemble']
    model_labels = {'xgboost': 'XGBoost', 'catboost': 'CatBoost',
                    'ml_poisson': 'ML-Poisson', 'ensemble': 'Ensemble'}
    metrics_order = [('accuracy', 'Accuracy', '{:.1%}'), ('log_loss', 'Log-Loss', '{:.4f}'), ('rps', 'RPS', '{:.4f}')]

    window_labels = [w[0] for w in WINDOWS]

    for metric_key, metric_name, fmt in metrics_order:
        print(f"\n  {metric_name}")
        header = f"  {'Model':<14}" + "".join(f"  {w:<24}" for w in window_labels)
        print(header)
        print("  " + "-" * (14 + 26 * len(window_labels)))
        for m in models:
            row = f"  {model_labels[m]:<14}"
            for w in window_labels:
                val = all_results[w][m][metric_key]
                correct = all_results[w][m]['correct']
                n = all_results[w][m]['n']
                if metric_key == 'accuracy':
                    row += f"  {fmt.format(val)} ({correct}/{n})          "
                else:
                    row += f"  {fmt.format(val):<24}"
            print(row)

    print()


if __name__ == '__main__':
    run()
