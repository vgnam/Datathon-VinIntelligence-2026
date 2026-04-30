import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BASELINE++: SEASONAL PROFILE + SELECTED FEATURES + RESIDUAL CORRECTION")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Date range: {train['Date'].min().date()} to {train['Date'].max().date()}")

# ============================================
# 1. BASE PREDICTION: hist_monthday_revenue_mean
# ============================================
print("\n" + "="*70)
print("1. BASE PREDICTION")
print("="*70)

def compute_base_prediction(df):
    """Base = hist_monthday mean * (1 + YoY growth for that month)"""
    base_rev = df['hist_monthday_revenue_mean'].copy()
    base_cogs = df['hist_monthday_cogs_mean'].copy()
    
    # Apply monthly growth trend
    growth_rev = 1 + df['hist_yoy_revenue_growth'].fillna(0)
    growth_cogs = 1 + df['hist_yoy_cogs_growth'].fillna(0)
    
    base_rev = base_rev * growth_rev
    base_cogs = base_cogs * growth_cogs
    
    return base_rev, base_cogs

train['base_revenue'], train['base_cogs'] = compute_base_prediction(train)
test['base_revenue'], test['base_cogs'] = compute_base_prediction(test)

print(f"Base Revenue: mean={train['base_revenue'].mean():,.0f}, std={train['base_revenue'].std():,.0f}")
print(f"Actual Revenue: mean={train['Revenue'].mean():,.0f}, std={train['Revenue'].std():,.0f}")

# ============================================
# 2. TRAFFIC ADJUSTMENT
# ============================================
print("\n" + "="*70)
print("2. TRAFFIC ADJUSTMENT")
print("="*70)

# Compute historical expected_sessions mean by month
train['month_day'] = train['Date'].dt.strftime('%m-%d')
session_profile = train.groupby('month_day')['expected_sessions'].mean().to_dict()

train['session_profile'] = train['month_day'].map(session_profile)
test['month_day'] = test['Date'].dt.strftime('%m-%d')
test['session_profile'] = test['month_day'].map(session_profile).fillna(test['expected_sessions'].mean())

# Traffic multiplier: current / profile
train['traffic_mult'] = train['expected_sessions'] / (train['session_profile'] + 1)
test['traffic_mult'] = test['expected_sessions'] / (test['session_profile'] + 1)

# Clip multiplier to reasonable range
train['traffic_mult'] = train['traffic_mult'].clip(0.5, 2.0)
test['traffic_mult'] = test['traffic_mult'].clip(0.5, 2.0)

print(f"Traffic multiplier: mean={train['traffic_mult'].mean():.3f}, range=[{train['traffic_mult'].min():.3f}, {train['traffic_mult'].max():.3f}]")

# Apply traffic adjustment
train['adj_revenue'] = train['base_revenue'] * train['traffic_mult']
train['adj_cogs'] = train['base_cogs'] * train['traffic_mult']
test['adj_revenue'] = test['base_revenue'] * test['traffic_mult']
test['adj_cogs'] = test['base_cogs'] * test['traffic_mult']

# ============================================
# 3. PROMO ADJUSTMENT
# ============================================
print("\n" + "="*70)
print("3. PROMO ADJUSTMENT")
print("="*70)

# Promo factor: if seasonal prob is high, boost prediction
# Compute promo impact from train
train['promo_flag'] = (train['promo_seasonal_prob'] > train['promo_seasonal_prob'].median()).astype(int)
promo_impact_rev = train.groupby('promo_flag')['Revenue'].mean()
promo_impact_cogs = train.groupby('promo_flag')['COGS'].mean()

if len(promo_impact_rev) > 1:
    promo_factor_rev = promo_impact_rev[1] / (promo_impact_rev[0] + 1)
    promo_factor_cogs = promo_impact_cogs[1] / (promo_impact_cogs[0] + 1)
else:
    promo_factor_rev = 1.0
    promo_factor_cogs = 1.0

print(f"Promo impact: Revenue factor={promo_factor_rev:.3f}, COGS factor={promo_factor_cogs:.3f}")

# Apply smooth promo factor based on promo_seasonal_prob
def apply_promo_factor(df, base_col, factor):
    # Scale factor by promo_prob: max boost when prob=1, no boost when prob=0
    prob_scale = df['promo_seasonal_prob'].clip(0, 1)
    adjusted_factor = 1 + (factor - 1) * prob_scale
    return df[base_col] * adjusted_factor

train['promo_revenue'] = apply_promo_factor(train, 'adj_revenue', promo_factor_rev)
train['promo_cogs'] = apply_promo_factor(train, 'adj_cogs', promo_factor_cogs)
test['promo_revenue'] = apply_promo_factor(test, 'adj_revenue', promo_factor_rev)
test['promo_cogs'] = apply_promo_factor(test, 'adj_cogs', promo_factor_cogs)

# ============================================
# 4. TET ADJUSTMENT
# ============================================
print("\n" + "="*70)
print("4. TET ADJUSTMENT")
print("="*70)

# Tet impact from train
tet_impact_rev = train.groupby('is_tet_period')['Revenue'].mean()
tet_impact_cogs = train.groupby('is_tet_period')['COGS'].mean()

if len(tet_impact_rev) > 1:
    tet_factor_rev = tet_impact_rev[1] / (tet_impact_rev[0] + 1)
    tet_factor_cogs = tet_impact_cogs[1] / (tet_impact_cogs[0] + 1)
else:
    tet_factor_rev = 1.0
    tet_factor_cogs = 1.0

print(f"Tet impact: Revenue factor={tet_factor_rev:.3f}, COGS factor={tet_factor_cogs:.3f}")

# Apply tet factor
train['tet_revenue'] = train['promo_revenue'] * (1 + (tet_factor_rev - 1) * train['is_tet_period'])
train['tet_cogs'] = train['promo_cogs'] * (1 + (tet_factor_cogs - 1) * train['is_tet_period'])
test['tet_revenue'] = test['promo_revenue'] * (1 + (tet_factor_rev - 1) * test['is_tet_period'])
test['tet_cogs'] = test['promo_cogs'] * (1 + (tet_factor_cogs - 1) * test['is_tet_period'])

# ============================================
# 5. RESIDUAL MODEL
# ============================================
print("\n" + "="*70)
print("5. RESIDUAL MODEL (Ridge)")
print("="*70)

# Compute residual
train['residual_rev'] = train['Revenue'] - train['tet_revenue']
train['residual_cogs'] = train['COGS'] - train['tet_cogs']

# Features for residual model
residual_features = [
    'hist_monthday_revenue_mean_recent',
    'hist_monthday_cogs_mean_recent',
    'day_cos', 'day_sin',
    'expected_sessions',
    'traffic_uncertainty',
    'days_to_tet',
    'promo_monthly_prob',
]

# Train/val split
val_days = 180
split_date = train['Date'].max() - pd.Timedelta(days=val_days)
train_mask = train['Date'] <= split_date
val_mask = train['Date'] > split_date

X_train = train.loc[train_mask, residual_features].fillna(0)
y_train_rev = train.loc[train_mask, 'residual_rev']
y_train_cogs = train.loc[train_mask, 'residual_cogs']

X_val = train.loc[val_mask, residual_features].fillna(0)
y_val_rev = train.loc[val_mask, 'Revenue']
y_val_cogs = train.loc[val_mask, 'COGS']

# Fit Ridge
ridge_rev = Ridge(alpha=1000.0)
ridge_cogs = Ridge(alpha=1000.0)
ridge_rev.fit(X_train, y_train_rev)
ridge_cogs.fit(X_train, y_train_cogs)

# Predict residual on full train and test
X_full = train[residual_features].fillna(0)
X_test = test[residual_features].fillna(0)

train['pred_residual_rev'] = ridge_rev.predict(X_full)
train['pred_residual_cogs'] = ridge_cogs.predict(X_full)
test['pred_residual_rev'] = ridge_rev.predict(X_test)
test['pred_residual_cogs'] = ridge_cogs.predict(X_test)

# Final prediction
train['final_revenue'] = train['tet_revenue'] + train['pred_residual_rev']
train['final_cogs'] = train['tet_cogs'] + train['pred_residual_cogs']

# Validation prediction
val_pred_rev = train.loc[val_mask, 'final_revenue']
val_pred_cogs = train.loc[val_mask, 'final_cogs']

# ============================================
# 6. METRICS
# ============================================
print("\n" + "="*70)
print("6. VALIDATION METRICS")
print("="*70)

def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs(actual - pred) / (np.abs(actual) + eps)))

def r2(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-9))

print(f"MAPE Revenue: {mape(y_val_rev, val_pred_rev):.4f}")
print(f"MAPE COGS:    {mape(y_val_cogs, val_pred_cogs):.4f}")
print(f"R2 Revenue:   {r2(y_val_rev, val_pred_rev):.4f}")
print(f"R2 COGS:      {r2(y_val_cogs, val_pred_cogs):.4f}")

# Base-only metrics for comparison
base_val_rev = train.loc[val_mask, 'base_revenue']
base_val_cogs = train.loc[val_mask, 'base_cogs']
print(f"\nBase-only MAPE Revenue: {mape(y_val_rev, base_val_rev):.4f}")
print(f"Base-only R2 Revenue:   {r2(y_val_rev, base_val_rev):.4f}")

# ============================================
# 7. TEST PREDICTION
# ============================================
print("\n" + "="*70)
print("7. TEST PREDICTION")
print("="*70)

# Full retrain on all data
ridge_rev_full = Ridge(alpha=1000.0)
ridge_cogs_full = Ridge(alpha=1000.0)
ridge_rev_full.fit(X_full, train['residual_rev'])
ridge_cogs_full.fit(X_full, train['residual_cogs'])

test['final_revenue'] = test['tet_revenue'] + ridge_rev_full.predict(X_test)
test['final_cogs'] = test['tet_cogs'] + ridge_cogs_full.predict(X_test)

# Ensure COGS <= Revenue * 0.995
test['final_cogs'] = np.minimum(test['final_cogs'], test['final_revenue'] * 0.995)
test['final_revenue'] = np.maximum(test['final_revenue'], 0)
test['final_cogs'] = np.maximum(test['final_cogs'], 0)

# ============================================
# 8. SAVE
# ============================================
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)

submission = pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['final_revenue'], 2),
    'COGS': np.round(test['final_cogs'], 2),
})
submission.to_csv(out_dir / 'baseline_plus_plus_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'baseline_plus_plus_submission.csv'}")

print("\n" + "="*70)
print("DONE")
print("="*70)
