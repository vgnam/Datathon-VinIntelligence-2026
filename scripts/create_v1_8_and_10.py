"""
create_v1_8_and_10.py
============================================================
Tạo 2 bộ selected features từ V1:
  • V1_8  : 8 features (tinh gọn, bỏ các cột yếu nhất)
  • V1_10 : 10 features (giữ thêm 2 cột inventory + traffic)

Input : train/test_features_balanced.csv
Output: train/test_features_selected_v1_8.csv
        train/test_features_selected_v1_10.csv
============================================================
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r'D:\Datathon-2026')
OUTPUT_DIR = BASE_DIR / 'output'

# Load balanced data + target
train_balanced = pd.read_csv(OUTPUT_DIR / 'train_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test_balanced = pd.read_csv(OUTPUT_DIR / 'test_features_balanced.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
train_target = pd.read_csv(OUTPUT_DIR / 'train_target.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

if 'net_revenue' in train_target.columns and 'Revenue' not in train_target.columns:
    train_target = train_target.rename(columns={'net_revenue': 'Revenue'})

print("="*70)
print("CREATE V1_8 AND V1_10 FEATURE SETS")
print("="*70)
print(f"Balanced train: {train_balanced.shape}")
print(f"Balanced test : {test_balanced.shape}")

# ======================================================================
# V1_8 – 8 features (tinh gọn nhất)
# ======================================================================
v1_8_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
]

# ======================================================================
# V1_10 – 10 features (thêm 2 cột từ bản cũ)
# ======================================================================
v1_10_features = v1_8_features + [
    'inventory_days_since_snapshot',   # Composite ~0.118
    'traffic_uncertainty',              # Composite ~0.082
]

# --- Kiểm tra tồn tại -------------------------------------------------
def check_features(name, feat_list, train_df, test_df):
    missing_train = [c for c in feat_list if c not in train_df.columns]
    missing_test = [c for c in feat_list if c not in test_df.columns]
    if missing_train:
        print(f"[WARN] {name} missing in train: {missing_train}")
    if missing_test:
        print(f"[WARN] {name} missing in test : {missing_test}")
    return len(missing_train) == 0 and len(missing_test) == 0

check_features('V1_8',  v1_8_features,  train_balanced, test_balanced)
check_features('V1_10', v1_10_features, train_balanced, test_balanced)

# --- Build V1_8 -------------------------------------------------------
train_v1_8 = train_balanced[['Date'] + v1_8_features].copy()
test_v1_8  = test_balanced[['Date'] + v1_8_features].copy()
train_v1_8 = train_v1_8.merge(train_target[['Date', 'Revenue', 'COGS']], on='Date', how='left')

train_v1_8.to_csv(OUTPUT_DIR / 'train_features_selected_v1_8.csv', index=False)
test_v1_8.to_csv(OUTPUT_DIR / 'test_features_selected_v1_8.csv', index=False)

print(f"\n[SAVED] V1_8  – train: {train_v1_8.shape}, test: {test_v1_8.shape}")
for i, f in enumerate(v1_8_features, 1):
    print(f"   {i}. {f}")

# --- Build V1_10 ------------------------------------------------------
train_v1_10 = train_balanced[['Date'] + v1_10_features].copy()
test_v1_10  = test_balanced[['Date'] + v1_10_features].copy()
train_v1_10 = train_v1_10.merge(train_target[['Date', 'Revenue', 'COGS']], on='Date', how='left')

train_v1_10.to_csv(OUTPUT_DIR / 'train_features_selected_v1_10.csv', index=False)
test_v1_10.to_csv(OUTPUT_DIR / 'test_features_selected_v1_10.csv', index=False)

print(f"\n[SAVED] V1_10 – train: {train_v1_10.shape}, test: {test_v1_10.shape}")
for i, f in enumerate(v1_10_features, 1):
    print(f"   {i}. {f}")

# --- Also update the canonical 'selected' to point to V1_8 ------------
train_v1_8.to_csv(OUTPUT_DIR / 'train_features_selected.csv', index=False)
test_v1_8.to_csv(OUTPUT_DIR / 'test_features_selected.csv', index=False)
print(f"\n[UPDATED] train_features_selected.csv & test_features_selected.csv -> V1_8")

print("\n" + "="*70)
print("DONE")
print("="*70)
