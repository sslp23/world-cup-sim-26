"""
WC 2026 Group Stage Simulation — entry point.

Run from the project root:
    python -m simulation.run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.dataset      import build
from simulation.predict      import run as predict
from simulation.export_excel import save


def main():
    df          = build()
    predictions = predict(df)
    save(predictions, 'simulation/output/wc_26_sim.xlsx')


if __name__ == '__main__':
    main()
