import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FIX SELECTED FEATURES: khac phuc 4 nhuoc diem")
print("="*70)

# Load balanced features (day du nhat)
train_balanced = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test_balanced = pd.read_csv(r'D:\Datathon-2026\output\test_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
train_target = pd.read_csv(r'D:\Datathon-2026\output\train_target.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

if 'net_revenue' in train_target.columns and 'Revenue' not in train_target.columns:
    train_target = train_target.rename(columns={'net_revenue': 'Revenue'})

train = train_balanced.merge(train_target[['Date','Revenue','COGS']], on='Date', how='left')

# ============================================
# 1. TAO PROMO_SEASONAL_PROB thay cho promo_intensity=0
# ============================================
print("\n1. TAO PROMO_SEASONAL_PROB")
print("-" * 70)

# Lay lich su promo tu train (2012-2022)
train['month'] = train['Date'].dt.month
train['day'] = train['Date'].dt.day

# Xac suat co promo theo (month, day)
promo_by_md = train.groupby(['month','day'])['promo_active'].mean().reset_index()
promo_by_md = promo_by_md.rename(columns={'promo_active': 'promo_seasonal_prob'})

# Xac suat co promo theo month
promo_by_month = train.groupby('month')['promo_active'].mean().reset_index()
promo_by_month = promo_by_month.rename(columns={'promo_active': 'promo_monthly_prob'})

# Merge vao train va test
train = train.merge(promo_by_md, on=['month','day'], how='left')
train = train.merge(promo_by_month, on='month', how='left')

test_balanced['month'] = test_balanced['Date'].dt.month
test_balanced['day'] = test_balanced['Date'].dt.day
test_balanced = test_balanced.merge(promo_by_md, on=['month','day'], how='left')
test_balanced = test_balanced.merge(promo_by_month, on='month', how='left')

print(f"   promo_seasonal_prob train: mean={train['promo_seasonal_prob'].mean():.3f}, max={train['promo_seasonal_prob'].max():.3f}")
print(f"   promo_seasonal_prob test:  mean={test_balanced['promo_seasonal_prob'].mean():.3f}, max={test_balanced['promo_seasonal_prob'].max():.3f}")
print(f"   promo_monthly_prob  train: mean={train['promo_monthly_prob'].mean():.3f}")
print(f"   promo_monthly_prob  test:  mean={test_balanced['promo_monthly_prob'].mean():.3f}")

# ============================================
# 2. TAO hist_monthday_revenue_mean_RECENT (2019-2022 only)
# ============================================
print("\n2. TAO hist_monthday_revenue_mean_RECENT (2019-2022)")
print("-" * 70)

train['year'] = train['Date'].dt.year
train['month_day'] = train['Date'].dt.strftime('%m-%d')

recent_train = train[train['year'] >= 2019].copy()
hist_recent = recent_train.groupby('month_day')['Revenue'].mean().reset_index()
hist_recent = hist_recent.rename(columns={'Revenue': 'hist_monthday_revenue_mean_recent'})

# COGS recent
hist_recent_cogs = recent_train.groupby('month_day')['COGS'].mean().reset_index()
hist_recent_cogs = hist_recent_cogs.rename(columns={'COGS': 'hist_monthday_cogs_mean_recent'})

# Merge
train = train.merge(hist_recent, on='month_day', how='left')
train = train.merge(hist_recent_cogs, on='month_day', how='left')

test_balanced['month_day'] = test_balanced['Date'].dt.strftime('%m-%d')
test_balanced = test_balanced.merge(hist_recent, on='month_day', how='left')
test_balanced = test_balanced.merge(hist_recent_cogs, on='month_day', how='left')

print(f"   hist_monthday_revenue_mean_recent: train mean={train['hist_monthday_revenue_mean_recent'].mean():.0f}, test mean={test_balanced['hist_monthday_revenue_mean_recent'].mean():.0f}")
print(f"   hist_monthday_cogs_mean_recent:    train mean={train['hist_monthday_cogs_mean_recent'].mean():.0f}, test mean={test_balanced['hist_monthday_cogs_mean_recent'].mean():.0f}")

# ============================================
# 3. INVENTORY: Thay inventory_sell_through bang inventory_is_stale_rate (co y nghia hon trong test)
# ============================================
print("\n3. KIEM TRA INVENTORY FEATURES")
print("-" * 70)

print(f"   inventory_is_stale_rate test: unique={test_balanced['inventory_is_stale_rate'].nunique()}")
print(f"      first 10: {test_balanced['inventory_is_stale_rate'].head(10).tolist()}")
print(f"      last 10:  {test_balanced['inventory_is_stale_rate'].tail(10).tolist()}")

# ============================================
# 4. CHON FEATURES MOI (15 features)
# ============================================
print("\n4. CHON FEATURES MOI (15 features)")
print("-" * 70)

SELECTED_FEATURES_V2 = [
    'hist_monthday_revenue_mean',           # giu lai
    'hist_monthday_revenue_mean_recent',    # MOI: recency weighted
    'hist_monthday_cogs_mean',              # giu lai
    'hist_monthday_cogs_mean_recent',       # MOI: recency weighted
    'day_cos',
    'day_sin',
    'hist_yoy_revenue_growth',
    'hist_yoy_cogs_growth',
    'expected_sessions',
    'month',
    'is_tet_period',                        # THEM: calendar quan trong
    'days_to_tet',                          # THEM: calendar quan trong
    'traffic_uncertainty',
    'promo_seasonal_prob',                  # THAY: thay cho promo_intensity=0
    'promo_monthly_prob',                   # THEM: bo sung
]

# Kiem tra cac cot ton tai
missing_train = [c for c in SELECTED_FEATURES_V2 if c not in train.columns]
missing_test = [c for c in SELECTED_FEATURES_V2 if c not in test_balanced.columns]
print(f"   Missing in train: {missing_train}")
print(f"   Missing in test:  {missing_test}")

# Fill missing
train = train.fillna(0.0)
test_balanced = test_balanced.fillna(0.0)

# Tao output
output_cols = ['Date'] + SELECTED_FEATURES_V2 + ['Revenue', 'COGS']
train_selected_v2 = train[output_cols].copy()
test_selected_v2 = test_balanced[['Date'] + SELECTED_FEATURES_V2].copy()

# Luu file
train_selected_v2.to_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', index=False)
test_selected_v2.to_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', index=False)

print(f"\n   Da luu:")
print(f"   - output/train_features_selected_v2.csv ({train_selected_v2.shape})")
print(f"   - output/test_features_selected_v2.csv ({test_selected_v2.shape})")

for i, f in enumerate(SELECTED_FEATURES_V2, 1):
    print(f"   {i:2d}. {f}")

# ============================================
# 5. SO SANH TEST V2 vs V1
# ============================================
print("\n5. SO SANH TEST V2 vs V1")
print("-" * 70)

test_old = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected.csv')

print("   Feature V1 (old):")
for c in test_old.columns:
    if c == 'Date': continue
    u = test_old[c].nunique()
    print(f"      {c}: unique={u}, min={test_old[c].min():.4f}, max={test_old[c].max():.4f}")

print("\n   Feature V2 (new):")
for c in SELECTED_FEATURES_V2:
    u = test_selected_v2[c].nunique()
    print(f"      {c}: unique={u}, min={test_selected_v2[c].min():.4f}, max={test_selected_v2[c].max():.4f}")

print("\n" + "="*70)
print("DONE — Da tao selected features v2")
print("="*70)
