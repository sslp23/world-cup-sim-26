"""
Central model hyperparameters.

Edit here to apply changes across all simulation and backtest scripts.
"""

# Draw class weight multiplier for XGBoost and CatBoost classifiers.
# Multiplies the inverse-frequency draw weight; < 1.0 softens the draw boost.
XGB_DRAW_WEIGHT = 0.67
CB_DRAW_WEIGHT  = 0.66

# Maximum Dixon-Coles low-score correction for ML-Poisson.
# Applied at full strength when teams are evenly matched; scaled toward 0
# for lopsided matches (dynamic rho logic in ml_poisson/model.py).
MLP_RHO = -0.35
