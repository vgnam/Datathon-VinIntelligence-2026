import pandas as pd
import numpy as np

print("="*70)
print("KIEM TRA TEST FEATURES — TIM NHUOC DIEM")
print("="*70)

# Load test features
test_balanced = pd.read_csv(r'D:\Datathon-2026\output\test_features_balanced.csv')
test_clean = pd.read_csv(r'D:\Datathon-2026\output\test_features_clean.csv')
test_selected = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected.csv')
train_balanced = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv')

print("\n1. SO LUONG FEATURES")
print(f"   test_balanced:  {test_balanced.shape[1]} cols, {test_balanced.shape[0]} rows")
print(f"   test_clean:     {test_clean.shape[1]} cols, {test_clean.shape[0]} rows")
print(f"   test_selected:  {test_selected.shape[1]} cols, {test_selected.shape[0]} rows")

print("\n2. CHECK NaN TRONG TEST SET")
for name, df in [("balanced", test_balanced), ("clean", test_clean), ("selected", test_selected)]:
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n   {name}: {len(nan_cols)} columns have NaN:")
        for col, cnt in nan_cols.items():
            print(f"      {col}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")
    else:
        print(f"\n   {name}: No NaN")

print("\n3. CHECK CONSTANT VALUES TRONG TEST SET")
for name, df in [("balanced", test_balanced), ("clean", test_clean), ("selected", test_selected)]:
    const_cols = []
    for col in df.columns:
        if col == 'Date':
            continue
        unique_vals = df[col].nunique()
        if unique_vals == 1:
            const_cols.append((col, df[col].iloc[0]))
    if const_cols:
        print(f"\n   {name}: {len(const_cols)} constant columns:")
        for col, val in const_cols:
            print(f"      {col} = {val}")
    else:
        print(f"\n   {name}: No fully constant columns")

print("\n4. CHECK NEAR-CONSTANT COLUMNS (std < 0.001)")
for name, df in [("balanced", test_balanced), ("clean", test_clean), ("selected", test_selected)]:
    near_const = []
    for col in df.columns:
        if col == 'Date':
            continue
        std = df[col].std()
        if std < 0.001:
            near_const.append((col, std, df[col].mean()))
    if near_const:
        print(f"\n   {name}: {len(near_const)} near-constant columns:")
        for col, std, mean in near_const:
            print(f"      {col}: std={std:.6f}, mean={mean:.6f}")

print("\n5. INVENTORY FEATURES — CHECK STALE BEHAVIOR")
inv_cols = ['inventory_fill_rate','inventory_sell_through','inventory_days_since_snapshot','inventory_is_stale_rate']
for col in inv_cols:
    if col in test_balanced.columns:
        vals = test_balanced[col]
        print(f"\n   {col}:")
        print(f"      range: {vals.min():.4f} to {vals.max():.4f}")
        print(f"      unique: {vals.nunique()}")
        print(f"      first 10: {vals.head(10).tolist()}")
        print(f"      last 10:  {vals.tail(10).tolist()}")

print("\n6. PROMO FEATURES — CHECK IF ALWAYS ZERO")
promo_cols = ['promo_active','promo_intensity','promo_stackable','promo_carryover_5d']
for col in promo_cols:
    if col in test_balanced.columns:
        vals = test_balanced[col]
        print(f"\n   {col}: min={vals.min()}, max={vals.max()}, mean={vals.mean():.4f}")

print("\n7. HIST_* FEATURES — CHECK VARIANCE ACROSS TEST PERIOD")
hist_cols = [c for c in test_balanced.columns if c.startswith('hist_')]
for col in hist_cols:
    vals = test_balanced[col]
    print(f"\n   {col}: min={vals.min():.2f}, max={vals.max():.2f}, std={vals.std():.2f}, unique={vals.nunique()}")

print("\n8. CONVERSION RATE — CHECK IF ZERO")
if 'conversion_rate_overall' in test_balanced.columns:
    vals = test_balanced['conversion_rate_overall']
    print(f"   conversion_rate_overall: min={vals.min()}, max={vals.max()}, mean={vals.mean():.4f}")
    print(f"      unique values: {sorted(vals.unique())[:20]}")

print("\n9. COMPARE TRAIN vs TEST DISTRIBUTION (hist_monthday_revenue_mean)")
if 'hist_monthday_revenue_mean' in train_balanced.columns and 'hist_monthday_revenue_mean' in test_balanced.columns:
    tr = train_balanced['hist_monthday_revenue_mean']
    te = test_balanced['hist_monthday_revenue_mean']
    print(f"   Train: mean={tr.mean():.0f}, std={tr.std():.0f}, min={tr.min():.0f}, max={tr.max():.0f}")
    print(f"   Test:  mean={te.mean():.0f}, std={te.std():.0f}, min={te.min():.0f}, max={te.max():.0f}")
    print(f"   Ratio test/train mean: {te.mean()/tr.mean():.3f}")

print("\n" + "="*70)
print("DONE")
print("="*70)
