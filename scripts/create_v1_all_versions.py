"""
create_v1_all_versions.py
============================================================
Tạo lại đầy đủ 3 phiên bản V1 từ lịch sử biến đổi:
  • V1_12 : 12 features (nguyên bản ban đầu)
  • V1_10 : 10 features (bỏ inventory_sell_through + promo_seasonal_prob)
  • V1_8  : 8 features  (bỏ thêm inventory_days_since_snapshot + traffic_uncertainty)

Input : train/test_features_balanced.csv
Output: train/test_features_selected_v1_{8,10,12}.csv
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
print("CREATE V1_8, V1_10, V1_12 FEATURE SETS")
print("="*70)
print(f"Balanced train: {train_balanced.shape}")
print(f"Balanced test : {test_balanced.shape}")

# ======================================================================
# V1_12 – 12 features (nguyên bản đầu tiên)
# ======================================================================
v1_12_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
    'inventory_sell_through',
    'inventory_days_since_snapshot',
    'traffic_uncertainty',
    'promo_seasonal_prob',
]

# ======================================================================
# V1_10 – 10 features (bỏ inventory_sell_through + promo_seasonal_prob)
# ======================================================================
v1_10_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
    'inventory_days_since_snapshot',
    'traffic_uncertainty',
]

# ======================================================================
# V1_8 – 8 features (bỏ thêm inventory_days_since_snapshot + traffic_uncertainty)
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

# --- Kiểm tra tồn tại -------------------------------------------------
def check_features(name, feat_list, train_df, test_df):
    missing_train = [c for c in feat_list if c not in train_df.columns]
    missing_test = [c for c in feat_list if c not in test_df.columns]
    if missing_train:
        print(f"[WARN] {name} missing in train: {missing_train}")
    if missing_test:
        print(f"[WARN] {name} missing in test : {missing_test}")
    return len(missing_train) == 0 and len(missing_test) == 0

# ======================================================================
# BUILD & SAVE
# ======================================================================
def build_and_save(name, feat_list, train_df, test_df, target_df):
    train_out = train_df[['Date'] + feat_list].copy()
    test_out = test_df[['Date'] + feat_list].copy()
    train_out = train_out.merge(target_df[['Date', 'Revenue', 'COGS']], on='Date', how='left')
    
    train_path = OUTPUT_DIR / f'train_features_selected_{name}.csv'
    test_path = OUTPUT_DIR / f'test_features_selected_{name}.csv'
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    
    print(f"\n[SAVED] {name} – train: {train_out.shape}, test: {test_out.shape}")
    for i, f in enumerate(feat_list, 1):
        print(f"   {i:2d}. {f}")

# V1_12
check_features('V1_12', v1_12_features, train_balanced, test_balanced)
build_and_save('v1_12', v1_12_features, train_balanced, test_balanced, train_target)

# V1_10
check_features('V1_10', v1_10_features, train_balanced, test_balanced)
build_and_save('v1_10', v1_10_features, train_balanced, test_balanced, train_target)

# V1_8
check_features('V1_8', v1_8_features, train_balanced, test_balanced)
build_and_save('v1_8', v1_8_features, train_balanced, test_balanced, train_target)

# Update canonical selected to V1_8
train_v1_8 = pd.read_csv(OUTPUT_DIR / 'train_features_selected_v1_8.csv')
test_v1_8 = pd.read_csv(OUTPUT_DIR / 'test_features_selected_v1_8.csv')
train_v1_8.to_csv(OUTPUT_DIR / 'train_features_selected.csv', index=False)
test_v1_8.to_csv(OUTPUT_DIR / 'test_features_selected.csv', index=False)
print("\n[UPDATED] train_features_selected.csv & test_features_selected.csv -> V1_8")

print("\n" + "="*70)
print("DONE")
print("="*70)
