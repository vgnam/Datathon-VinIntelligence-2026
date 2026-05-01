"""
stacking_v1.py
============================================================
Two-Level Stacking (Stacked Generalization) for Revenue & COGS
using Selected V1 features (8 features).

Architecture:
  Layer 0 (Base Models - OOF via TimeSeriesSplit):
    - Baseline: hist_monthday_mean * (1 + yoy_growth)
    - RidgeCV (linear, robust)
    - ElasticNetCV (sparse linear)
    - RandomForest (tree-based)
    - LightGBM (gradient boosting)

  Layer 1 (Meta-Learner):
    - RidgeCV trained on OOF predictions from all base models
    - Separate meta-learner for Revenue and COGS

Why this works better than simple averaging:
  1. Base models capture different patterns (linear vs non-linear)
  2. Meta-learner learns OPTIMAL WEIGHTS per sample region
  3. TimeSeriesSplit avoids lookahead leakage in OOF generation
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASE_DIR = Path(r'D:\Datathon-2026')
OUTPUT_DIR = BASE_DIR / 'output'
FORECAST_DIR = BASE_DIR / 'data_cleaned' / 'forecast'

FEATURE_COLS = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
]

N_SPLITS = 5
RANDOM_STATE = 42

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
train = pd.read_csv(OUTPUT_DIR / 'train_features_selected.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(OUTPUT_DIR / 'test_features_selected.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print("="*70)
print("STACKING MODEL - V1 FEATURES")
print("="*70)
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Features: {len(FEATURE_COLS)}")

X_train = train[FEATURE_COLS].copy()
X_test = test[FEATURE_COLS].copy()
y_rev = train['Revenue'].values.copy()
y_cogs = train['COGS'].values.copy()

# Fill NA just in case
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# ------------------------------------------------------------------
# BASELINE MODEL (rule-based, no training needed)
# ------------------------------------------------------------------
def baseline_predict(df):
    """Baseline: historical month-day mean scaled by YoY growth."""
    rev = df['hist_monthday_revenue_mean'].values * (1 + df['hist_yoy_revenue_growth'].fillna(0).values)
    cogs = df['hist_monthday_cogs_mean'].values * (1 + df['hist_yoy_cogs_growth'].fillna(0).values)
    return rev, cogs

# ------------------------------------------------------------------
# BASE MODELS
# ------------------------------------------------------------------
base_models = {
    'ridge': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]),
    'elastic': ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
        cv=3,
        max_iter=3000,
        n_jobs=-1,
    ),
    'rf': RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

# Add LightGBM if available
try:
    from lightgbm import LGBMRegressor
    base_models['lgbm'] = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    print("LightGBM: enabled")
except Exception as e:
    print(f"LightGBM: disabled ({e})")

model_names = list(base_models.keys()) + ['baseline']
print(f"Base models: {model_names}")

# ------------------------------------------------------------------
# OOF GENERATION via TimeSeriesSplit
# ------------------------------------------------------------------
print("\n" + "="*70)
print("GENERATING OOF PREDICTIONS (TimeSeriesSplit)")
print("="*70)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
n_train = len(train)
n_test = len(test)
n_models = len(model_names)

oof_rev = np.zeros((n_train, n_models))
oof_cogs = np.zeros((n_train, n_models))
test_rev = np.zeros((n_test, n_models))
test_cogs = np.zeros((n_test, n_models))

# Fill baseline column (no CV needed)
base_tr_rev, base_tr_cogs = baseline_predict(train)
base_te_rev, base_te_cogs = baseline_predict(test)
oof_rev[:, -1] = base_tr_rev
oof_cogs[:, -1] = base_tr_cogs
test_rev[:, -1] = base_te_rev
test_cogs[:, -1] = base_te_cogs

# Train ML base models with CV
for idx, (name, model_template) in enumerate(base_models.items()):
    print(f"\n[{idx+1}/{len(base_models)}] {name} ...")
    oof_r = np.zeros(n_train)
    oof_c = np.zeros(n_train)
    test_r_folds = np.zeros((n_test, N_SPLITS))
    test_c_folds = np.zeros((n_test, N_SPLITS))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_r, y_val_r = y_rev[tr_idx], y_rev[val_idx]
        y_tr_c, y_val_c = y_cogs[tr_idx], y_cogs[val_idx]

        # Revenue
        m_r = clone(model_template)
        m_r.fit(X_tr, y_tr_r)
        oof_r[val_idx] = m_r.predict(X_val)
        test_r_folds[:, fold] = m_r.predict(X_test)

        # COGS
        m_c = clone(model_template)
        m_c.fit(X_tr, y_tr_c)
        oof_c[val_idx] = m_c.predict(X_val)
        test_c_folds[:, fold] = m_c.predict(X_test)

        # Quick MAPE on this fold
        def mape(a, p):
            return np.mean(np.abs(a - p) / (np.abs(a) + 1e-9))
        print(f"  Fold {fold+1} | Val MAPE Rev: {mape(y_val_r, oof_r[val_idx]):.4f} | Cogs: {mape(y_val_c, oof_c[val_idx]):.4f}")

    oof_rev[:, idx] = oof_r
    oof_cogs[:, idx] = oof_c
    test_rev[:, idx] = test_r_folds.mean(axis=1)
    test_cogs[:, idx] = test_c_folds.mean(axis=1)

# ------------------------------------------------------------------
# META-LEARNER (Level 1)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TRAINING META-LEARNER")
print("="*70)

meta_rev = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
meta_cogs = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

meta_rev.fit(oof_rev, y_rev)
meta_cogs.fit(oof_cogs, y_cogs)

print(f"Meta Revenue  alpha: {meta_rev.alpha_}")
print(f"Meta COGS     alpha: {meta_cogs.alpha_}")
print(f"Meta Revenue  coefs: {dict(zip(model_names, np.round(meta_rev.coef_, 4)))}")
print(f"Meta COGS     coefs: {dict(zip(model_names, np.round(meta_cogs.coef_, 4)))}")

# OOF performance of meta-learner
meta_oof_rev = meta_rev.predict(oof_rev)
meta_oof_cogs = meta_cogs.predict(oof_cogs)

def mape(a, p):
    return float(np.mean(np.abs(a - p) / (np.abs(a) + 1e-9)))

def r2_score(a, p):
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))

print(f"\nOOF Meta MAPE Revenue : {mape(y_rev, meta_oof_rev):.4f}")
print(f"OOF Meta MAPE COGS    : {mape(y_cogs, meta_oof_cogs):.4f}")
print(f"OOF Meta R2   Revenue : {r2_score(y_rev, meta_oof_rev):.4f}")
print(f"OOF Meta R2   COGS    : {r2_score(y_cogs, meta_oof_cogs):.4f}")

# Compare with simple average ensemble
simple_rev = np.mean(oof_rev[:, :-1], axis=1)  # exclude baseline
simple_cogs = np.mean(oof_cogs[:, :-1], axis=1)
print(f"\nSimple Avg MAPE Revenue: {mape(y_rev, simple_rev):.4f}")
print(f"Simple Avg MAPE COGS   : {mape(y_cogs, simple_cogs):.4f}")

# ------------------------------------------------------------------
# FINAL TEST PREDICTION
# ------------------------------------------------------------------
final_rev = meta_rev.predict(test_rev)
final_cogs = meta_cogs.predict(test_cogs)

# Constraints
final_rev = np.maximum(final_rev, 0.0)
final_cogs = np.maximum(final_cogs, 0.0)
final_cogs = np.minimum(final_cogs, final_rev * 0.995)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
sub = pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(final_rev, 2),
    'COGS': np.round(final_cogs, 2),
})
sub.to_csv(FORECAST_DIR / 'stacking_v1_submission.csv', index=False)
print(f"\n[SAVED] {FORECAST_DIR / 'stacking_v1_submission.csv'}")

# Also save OOF and test predictions for potential second-stage stacking
stacking_oof = pd.DataFrame(oof_rev, columns=[f'{n}_rev' for n in model_names])
stacking_oof['Date'] = train['Date']
stacking_oof['Actual_Revenue'] = y_rev
stacking_oof['Actual_COGS'] = y_cogs
stacking_oof['Meta_Revenue'] = meta_oof_rev
stacking_oof['Meta_COGS'] = meta_oof_cogs
stacking_oof.to_csv(OUTPUT_DIR / 'stacking_v1_oof.csv', index=False)
print(f"[SAVED] {OUTPUT_DIR / 'stacking_v1_oof.csv'}")

print("\n" + "="*70)
print("DONE")
print("="*70)
