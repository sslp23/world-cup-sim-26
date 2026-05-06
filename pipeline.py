"""
Master pipeline — runs all sub-pipelines in order.

Usage:
    python pipeline.py
"""

from data_pipeline.pipeline import run as run_data

# from models.pipeline import run as run_models      # future
# from backtest.pipeline import run as run_backtest  # future


if __name__ == "__main__":
    run_data()
    # run_models()
    # run_backtest()
