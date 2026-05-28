"""
WC 2026 Group Stage Simulation — ML-Poisson entry point.

Run from the project root:
    python -m simulation.ml_poisson.run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simulation.dataset                 import build
from simulation.ml_poisson.predict      import run as predict
from simulation.ml_poisson.export_excel import save


def main():
    df          = build()
    predictions = predict(df)
    save(predictions, 'simulation/output/wc_26_ml_poisson.xlsx')


if __name__ == '__main__':
    main()
