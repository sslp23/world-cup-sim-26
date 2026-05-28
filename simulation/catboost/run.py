"""
WC 2026 Group Stage Simulation — CatBoost entry point.

Run from the project root:
    python -m simulation.catboost.run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simulation.dataset               import build
from simulation.catboost.predict      import run as predict
from simulation.catboost.export_excel import save


def main():
    df          = build()
    predictions = predict(df)
    save(predictions, 'simulation/output/wc_26_catboost.xlsx')


if __name__ == '__main__':
    main()
