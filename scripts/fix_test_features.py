import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FIX TEST FEATURES DIRECTLY ON EXISTING CSV FILES")
print("="*70)

# Load existing features (from previous pipeline run)
train = pd.read_csv(r'D:\Datathon-2026\output\train_features.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================
# 1. FIX conversion_rate_overall in TEST
# ============================================
print("\n1. FIX conversion_rate_overall")
train['month'] = train['Date'].dt.month
conv_by_month = train.groupby('month')['conversion_rate_overall'].mean().to_dict()
print(f"   Conversion rate by month (from train): {conv_by_month}")

test['month'] = test['Date'].dt.month
old_conv_mean = test['conversion_rate_overall'].mean()
test['conversion_rate_overall'] = test['month'].map(conv_by_month).fillna(0)
print(f"   Old test mean: {old_conv_mean:.6f} -> New test mean: {test['conversion_rate_overall'].mean():.6f}")

# ============================================
# 2. ADD promo_seasonal_prob and promo_monthly_prob
# ============================================
print("\n2. ADD promo_seasonal_prob & promo_monthly_prob")

train['day_of_week'] = train['Date'].dt.dayofweek
train['month_day'] = train['Date'].dt.strftime('%m-%d')

# By month-day
promo_by_md = train.groupby('month_day')['promo_active'].mean().to_dict()
# By month
promo_by_month = train.groupby('month')['promo_active'].mean().to_dict()
# By day_of_week
promo_by_dow = train.groupby('day_of_week')['promo_active'].mean().to_dict()

print(f"   Promo by month: {promo_by_month}")

test['month_day'] = test['Date'].dt.strftime('%m-%d')
test['promo_seasonal_prob'] = test['month_day'].map(promo_by_md).fillna(promo_by_month.get(1, 0))
test['promo_monthly_prob'] = test['month'].map(promo_by_month).fillna(0)
test['promo_dow_prob'] = test['Date'].dt.dayofweek.map(promo_by_dow).fillna(0)

print(f"   Test promo_seasonal_prob: mean={test['promo_seasonal_prob'].mean():.3f}, max={test['promo_seasonal_prob'].max():.3f}")
print(f"   Test promo_monthly_prob:  mean={test['promo_monthly_prob'].mean():.3f}, max={test['promo_monthly_prob'].max():.3f}")

# ============================================
# 3. FIX INVENTORY in TEST: mark all as stale/projected, keep only is_stale_rate
# ============================================
print("\n3. FIX inventory features in TEST")
inv_cols = ['inventory_fill_rate','inventory_stockout_freq','inventory_sell_through','inventory_reorder_freq','inventory_days_since_snapshot']
for col in inv_cols:
    if col in test.columns:
        old_unique = test[col].nunique()
        if col == 'inventory_days_since_snapshot':
            # Keep as is (it's technically correct) but it's still frozen
            print(f"   {col}: kept (unique={old_unique})")
        else:
            # Replace with NaN to indicate unknown in test period
            test[col] = np.nan
            print(f"   {col}: set to NaN (was unique={old_unique})")

if 'inventory_is_stale_rate' in test.columns:
    test['inventory_is_stale_rate'] = 1.0
    print(f"   inventory_is_stale_rate: set to 1.0")

# ============================================
# 4. SAVE FIXED FILES
# ============================================
print("\n4. SAVE")
out_dir = Path(r'D:\Datathon-2026\output')
train.to_csv(out_dir / 'train_features_fixed.csv', index=False)
test.to_csv(out_dir / 'test_features_fixed.csv', index=False)
print(f"   Saved: {out_dir / 'train_features_fixed.csv'}")
print(f"   Saved: {out_dir / 'test_features_fixed.csv'}")

# ============================================
# 5. RUN reduce_features.py ON FIXED FILES
# ============================================
print("\n5. RUN reduce_features.py")
import subprocess
result = subprocess.run(['python', 'scripts/reduce_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr[-500:])

# ============================================
# 6. RUN add_historical_features.py ON FIXED FILES
# ============================================
print("\n6. RUN add_historical_features.py")
result = subprocess.run(['python', 'scripts/add_historical_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr[-500:])

print("\n" + "="*70)
print("DONE")
print("="*70)
