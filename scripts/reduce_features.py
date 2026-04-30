import pandas as pd
import numpy as np
from pathlib import Path

# Load data
print("Loading features...")
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_fixed.csv', parse_dates=['Date'])
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_fixed.csv', parse_dates=['Date'])

print(f"Original train shape: {train.shape}")
print(f"Original test shape: {test.shape}")

# Features to REMOVE
features_to_drop = []

# 1. STATIC features (zero variance - no predictive power)
static_features = [
    'inventory_reorder_freq',
    'is_imputed_conversion_rate', 
    'return_rate_overall',
    'promo_efficiency_overall',
    'customer_count_total',
    'avg_signup_tenure',
    'gender_pct_female',
    'gender_pct_male',
    'gender_pct_unknown',
    'acquisition_pct_direct',
    'acquisition_pct_email_campaign',
    'acquisition_pct_organic_search',
    'acquisition_pct_paid_search',
    'acquisition_pct_referral',
    'acquisition_pct_social_media',
    'age_group_pct_18-24',
    'age_group_pct_25-34',
    'age_group_pct_35-44',
    'age_group_pct_45-54',
    'age_group_pct_55+',
]
features_to_drop.extend(static_features)
print(f"\n1. Static features to drop: {len(static_features)}")

# 2. HIGHLY CORRELATED - remove redundant
# day_of_year (0.997 with month) → keep month
features_to_drop.append('day_of_year')

# week_of_year (0.970 with month) → keep month  
features_to_drop.append('week_of_year')

# week_sin/week_cos (0.997 with day_sin/day_cos) → keep day_sin/day_cos
features_to_drop.extend(['week_sin', 'week_cos'])

# promo_carryover 3d, 7d, recency_weighted (0.96-0.99 with 5d) → keep 5d
features_to_drop.extend(['promo_carryover_3d', 'promo_carryover_7d', 'promo_recency_weighted'])

# traffic_carryover_2d, traffic_recency_weighted (0.95-0.99 with expected_sessions) → keep expected_sessions
features_to_drop.extend(['traffic_carryover_2d', 'traffic_recency_weighted'])

# is_profile_based (0.983 with conversion_rate_overall) → keep conversion_rate_overall
features_to_drop.append('is_profile_based')

# inventory_stockout_freq (0.950 with inventory_fill_rate) → keep fill_rate
features_to_drop.append('inventory_stockout_freq')

print(f"2. Correlated features to drop: 10")
print(f"   - day_of_year, week_of_year, week_sin, week_cos")
print(f"   - promo_carryover_3d/7d/recency_weighted")
print(f"   - traffic_carryover_2d/traffic_recency_weighted")
print(f"   - is_profile_based, inventory_stockout_freq")

# 3. REDUNDANT calendar features
# is_month_start/end and is_qtr_end - very sparse and captured by day_of_week/month
features_to_drop.extend(['is_month_start', 'is_month_end', 'is_qtr_end'])
print(f"3. Sparse calendar features: 3")

# 4. year - just a trend counter, not useful for prediction
features_to_drop.append('year')
print(f"4. Year feature: 1")

# Total
print(f"\n=== TOTAL FEATURES TO DROP: {len(features_to_drop)} ===")

# Apply dropping
train_clean = train.drop(columns=features_to_drop)
test_clean = test.drop(columns=features_to_drop)

print(f"\nCleaned train shape: {train_clean.shape}")
print(f"Cleaned test shape: {test_clean.shape}")

# Verify no zero-variance features remain
print(f"\n=== VERIFY NO ZERO-VARIANCE FEATURES ===")
zero_var = []
for col in train_clean.columns:
    if col == 'Date':
        continue
    if train_clean[col].std() == 0:
        zero_var.append(col)
if zero_var:
    print(f"WARNING: Still have zero-variance features: {zero_var}")
else:
    print("OK - No zero-variance features remain")

# Show remaining features
print(f"\n=== REMAINING FEATURES ({len(train_clean.columns)}) ===")
for i, col in enumerate(train_clean.columns, 1):
    print(f"{i:2d}. {col}")

# Save
out_dir = Path(r'D:\Datathon-2026\output')
train_clean.to_csv(out_dir / 'train_features_clean.csv', index=False)
test_clean.to_csv(out_dir / 'test_features_clean.csv', index=False)

print(f"\nSaved:")
print(f"  {out_dir / 'train_features_clean.csv'}")
print(f"  {out_dir / 'test_features_clean.csv'}")
