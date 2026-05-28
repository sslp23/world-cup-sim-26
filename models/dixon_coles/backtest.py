"""
Backtest Dixon-Coles against FIFA World Cup 2022.

Training data : all matches in international_results.csv before 2022-11-20
                (full history gives better team parameter estimates;
                 time decay means matches older than ~3 years have negligible weight)
Test set      : all 64 WC 2022 matches (treated as neutral venue)

Metrics reported per match and in aggregate:
  - Accuracy   : % of matches where argmax(probs) == actual outcome
  - Log-Loss   : -mean(log p_correct) — measures calibration
  - RPS        : Ranked Probability Score — ordered W/D/L, penalises distance from truth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from models.dixon_coles.model import DixonColes

WC_2022_START = pd.Timestamp('2022-11-20')
WC_2022_END   = pd.Timestamp('2022-12-18')
OUTCOME_ORDER = ['home_win', 'draw', 'away_win']


# ── Metric functions ──────────────────────────────────────────────────────────

def actual_outcome(home_score, away_score):
    if home_score > away_score:
        return 'home_win'
    elif home_score == away_score:
        return 'draw'
    return 'away_win'


def rps(probs, outcome):
    """
    Ranked Probability Score for a single match.
    Outcomes ordered: home_win → draw → away_win.
    RPS = (1/(K-1)) * sum_k (cumulative_pred_k - cumulative_actual_k)^2
    Range [0, 1]. Lower is better.
    """
    actual_vec = [1.0 if o == outcome else 0.0 for o in OUTCOME_ORDER]
    pred_cum    = np.cumsum([probs[o] for o in OUTCOME_ORDER])
    actual_cum  = np.cumsum(actual_vec)
    return float(np.sum((pred_cum - actual_cum) ** 2) / (len(OUTCOME_ORDER) - 1))


def log_loss_single(probs, outcome):
    return -np.log(probs[outcome] + 1e-15)


def accuracy_single(probs, outcome):
    predicted = max(probs, key=probs.get)
    return int(predicted == outcome)


# ── Main backtest ─────────────────────────────────────────────────────────────

def run():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    full_df = pd.read_csv("data/past_wc/wc2022/ranked_database_with_features.csv")
    full_df['date'] = pd.to_datetime(full_df['date'])

    train_df = full_df[full_df['date'] < WC_2022_START].reset_index(drop=True)
    wc22 = full_df[
        (full_df['tournament'] == 'FIFA World Cup') &
        (full_df['date'] >= WC_2022_START) &
        (full_df['date'] <= WC_2022_END)
    ].copy().reset_index(drop=True)

    print(f"Training matches : {len(train_df)}")
    print(f"WC 2022 matches  : {len(wc22)}")

    # ── Fit model ─────────────────────────────────────────────────────────────
    model = DixonColes(xi=0.00005)
    model.fit(train_df, ref_date=WC_2022_START)

    print(f"\nTop 10 teams by attack/defense ratio:")
    for team, strength in model.top_teams(10):
        print(f"  {team:<25} {strength:.3f}")

    # ── Predict each WC 2022 match ────────────────────────────────────────────
    print("\nPredicting WC 2022 matches...")
    rows = []
    for _, match in wc22.iterrows():
        home = match['home_team']
        away = match['away_team']
        # All WC 2022 matches are at neutral venues (Qatar)
        probs = model.predict_outcome_probs(home, away, neutral=True)
        outcome = actual_outcome(match['home_score'], match['away_score'])
        predicted = max(probs, key=probs.get)

        rows.append({
            'date':       match['date'].date(),
            'home_team':  home,
            'away_team':  away,
            'home_score': int(match['home_score']),
            'away_score': int(match['away_score']),
            'actual':     outcome,
            'predicted':  predicted,
            'correct':    predicted == outcome,
            'p_home_win': round(probs['home_win'], 3),
            'p_draw':     round(probs['draw'], 3),
            'p_away_win': round(probs['away_win'], 3),
            'p_correct':  round(probs[outcome], 3),
            'log_loss':   round(log_loss_single(probs, outcome), 4),
            'rps':        round(rps(probs, outcome), 4),
        })

    results = pd.DataFrame(rows)

    # ── Print per-match results ───────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Date':<12} {'Home':<22} {'Away':<22} {'Score':<6} {'Actual':<10} {'Pred':<10} {'P(HW)':<7} {'P(D)':<7} {'P(AW)':<7} {'RPS':<7}")
    print("=" * 95)
    for _, r in results.iterrows():
        mark = "OK  " if r['correct'] else "FAIL"
        score = f"{r['home_score']}-{r['away_score']}"
        print(
            f"{str(r['date']):<12} {r['home_team']:<22} {r['away_team']:<22} "
            f"{score:<6} {r['actual']:<10} {r['predicted']:<10} "
            f"{r['p_home_win']:<7} {r['p_draw']:<7} {r['p_away_win']:<7} "
            f"{r['rps']:<7}  {mark}"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    accuracy  = results['correct'].mean()
    mean_ll   = results['log_loss'].mean()
    mean_rps  = results['rps'].mean()

    # Baseline: uniform 1/3 probabilities (naive benchmark)
    uniform_probs = {'home_win': 1/3, 'draw': 1/3, 'away_win': 1/3}
    baseline_rps = np.mean([
        rps(uniform_probs, actual_outcome(r['home_score'], r['away_score']))
        for _, r in results.iterrows()
    ])
    baseline_ll = -np.log(1/3)

    print("\n" + "=" * 50)
    print("  AGGREGATE METRICS — WC 2022 (64 matches)")
    print("=" * 50)
    print(f"  Accuracy   : {accuracy:.3f}  ({int(results['correct'].sum())}/{len(results)} correct)")
    print(f"  Log-Loss   : {mean_ll:.4f}  (baseline uniform: {baseline_ll:.4f})")
    print(f"  RPS        : {mean_rps:.4f}  (baseline uniform: {baseline_rps:.4f})")
    print()

    # Breakdown by match stage
    print("  Accuracy by stage:")
    stage_map = {
        'Group stage':     lambda d: d <= pd.Timestamp('2022-12-02'),
        'Round of 16':     lambda d: pd.Timestamp('2022-12-03') <= d <= pd.Timestamp('2022-12-06'),
        'Quarter-finals':  lambda d: pd.Timestamp('2022-12-09') <= d <= pd.Timestamp('2022-12-10'),
        'Semi-finals':     lambda d: pd.Timestamp('2022-12-13') <= d <= pd.Timestamp('2022-12-14'),
        'Final / 3rd':     lambda d: d >= pd.Timestamp('2022-12-17'),
    }
    results['date_ts'] = pd.to_datetime(results['date'])
    for stage, mask_fn in stage_map.items():
        stage_rows = results[results['date_ts'].apply(mask_fn)]
        if len(stage_rows):
            acc = stage_rows['correct'].mean()
            print(f"    {stage:<16}: {acc:.3f}  ({int(stage_rows['correct'].sum())}/{len(stage_rows)})")

    print("=" * 50)

    # Save results
    out_path = "models/dixon_coles/wc2022_backtest_results.csv"
    results.drop(columns=['date_ts']).to_csv(out_path, index=False)
    print(f"\n  Full results saved to {out_path}")

    return results


if __name__ == "__main__":
    run()
