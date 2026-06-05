"""
WC 2026 Group Stage Simulation — Ordered Logit entry point.

Run from the project root:
    python -m simulation.ordered_logit.run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from simulation.dataset                  import build
from simulation.ordered_logit.predict    import run as predict
from simulation.ordered_logit.export_excel import save


def main():
    df          = build()
    predictions = predict(df)
    save(predictions, 'simulation/output/wc_26_ordered_logit.xlsx')


if __name__ == '__main__':
    main()
