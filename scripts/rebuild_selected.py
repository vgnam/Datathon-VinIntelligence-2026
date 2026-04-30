import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("REBUILD SELECTED FEATURES V1 & V2 FROM FIXED BALANCED DATA")
print("="*70)

# Load fixed balanced features
train_balanced = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test_balanced = pd.read_csv(r'D:\Datathon-2026\output\test_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Balanced train: {train_balanced.shape}")
print(f"Balanced test:  {test_balanced.shape}")

# ============================================
# V1: 12 features (original selected)
# ============================================
print("\n" + "="*70)
print("BUILDING SELECTED V1 (12 features)")
print("="*70)

v1_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
    'inventory_sell_through',  # In test this is now NaN (fixed)
    'inventory_days_since_snapshot',
    'traffic_uncertainty',
    'promo_seasonal_prob',     # FIXED: was promo_intensity=0 in test
]

# Check availability
missing_train = [c for c in v1_features if c not in train_balanced.columns]
missing_test = [c for c in v1_features if c not in test_balanced.columns]
print(f"Missing in train: {missing_train}")
print(f"Missing in test:  {missing_test}")

train_v1 = train_balanced[['Date'] + v1_features].copy()
test_v1 = test_balanced[['Date'] + v1_features].copy()

# Add targets to train
train_target = pd.read_csv(r'D:\Datathon-2026\output\train_target.csv', parse_dates=['Date'])
if 'net_revenue' in train_target.columns and 'Revenue' not in train_target.columns:
    train_target = train_target.rename(columns={'net_revenue': 'Revenue'})
train_v1 = train_v1.merge(train_target[['Date','Revenue','COGS']], on='Date', how='left')

# Fill NaN in inventory_sell_through with median from train
median_sell_through = train_v1['inventory_sell_through'].median()
train_v1['inventory_sell_through'] = train_v1['inventory_sell_through'].fillna(median_sell_through)
test_v1['inventory_sell_through'] = test_v1['inventory_sell_through'].fillna(median_sell_through)

# Save
train_v1.to_csv(r'D:\Datathon-2026\output\train_features_selected.csv', index=False)
test_v1.to_csv(r'D:\Datathon-2026\output\test_features_selected.csv', index=False)

print(f"Saved V1 train: {train_v1.shape}")
print(f"Saved V1 test:  {test_v1.shape}")
for i, f in enumerate(v1_features, 1):
    print(f"  {i:2d}. {f}")

# ============================================
# V2: 15 features (with recency, calendar, promo fix)
# ============================================
print("\n" + "="*70)
print("BUILDING SELECTED V2 (15 features)")
print("="*70)

# Need to compute hist_monthday_revenue_mean_recent from train
train_with_target = train_balanced.merge(train_target[['Date','Revenue','COGS']], on='Date', how='left')
train_with_target['year'] = train_with_target['Date'].dt.year
train_with_target['month_day'] = train_with_target['Date'].dt.strftime('%m-%d')

# Recent stats (2019-2022)
recent_train = train_with_target[train_with_target['year'] >= 2019].copy()
hist_recent_rev = recent_train.groupby('month_day')['Revenue'].mean().reset_index().rename(columns={'Revenue': 'hist_monthday_revenue_mean_recent'})
hist_recent_cogs = recent_train.groupby('month_day')['COGS'].mean().reset_index().rename(columns={'COGS': 'hist_monthday_cogs_mean_recent'})

# Merge into balanced
train_balanced = train_balanced.merge(hist_recent_rev, on='month_day', how='left')
train_balanced = train_balanced.merge(hist_recent_cogs, on='month_day', how='left')
test_balanced = test_balanced.merge(hist_recent_rev, on='month_day', how='left')
test_balanced = test_balanced.merge(hist_recent_cogs, on='month_day', how='left')

v2_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_revenue_mean_recent',  # NEW
    'hist_monthday_cogs_mean',
    'hist_monthday_cogs_mean_recent',     # NEW
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
    'is_tet_period',                      # NEW
    'days_to_tet',                        # NEW
    'traffic_uncertainty',
    'promo_seasonal_prob',                # FIXED
    'promo_monthly_prob',                 # NEW
]

missing_train_v2 = [c for c in v2_features if c not in train_balanced.columns]
missing_test_v2 = [c for c in v2_features if c not in test_balanced.columns]
print(f"Missing in train: {missing_train_v2}")
print(f"Missing in test:  {missing_test_v2}")

# Fill any NaN in recent features with regular hist mean
train_balanced['hist_monthday_revenue_mean_recent'] = train_balanced['hist_monthday_revenue_mean_recent'].fillna(train_balanced['hist_monthday_revenue_mean'])
train_balanced['hist_monthday_cogs_mean_recent'] = train_balanced['hist_monthday_cogs_mean_recent'].fillna(train_balanced['hist_monthday_cogs_mean'])
test_balanced['hist_monthday_revenue_mean_recent'] = test_balanced['hist_monthday_revenue_mean_recent'].fillna(test_balanced['hist_monthday_revenue_mean'])
test_balanced['hist_monthday_cogs_mean_recent'] = test_balanced['hist_monthday_cogs_mean_recent'].fillna(test_balanced['hist_monthday_cogs_mean'])

train_v2 = train_balanced[['Date'] + v2_features].copy()
test_v2 = test_balanced[['Date'] + v2_features].copy()
train_v2 = train_v2.merge(train_target[['Date','Revenue','COGS']], on='Date', how='left')

# Save
train_v2.to_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', index=False)
test_v2.to_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', index=False)

print(f"Saved V2 train: {train_v2.shape}")
print(f"Saved V2 test:  {test_v2.shape}")
for i, f in enumerate(v2_features, 1):
    print(f"  {i:2d}. {f}")

print("\n" + "="*70)
print("DONE - Selected V1 & V2 rebuilt from fixed balanced data")
print("="*70)
