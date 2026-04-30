import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("REDUCING HISTORICAL FEATURES (53 to 35 columns)")
print("="*60)

# Load enhanced features
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_enhanced.csv', parse_dates=['Date'])
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_enhanced.csv', parse_dates=['Date'])

print(f"Original enhanced: {train.shape[1]} features")

# ============================================
# REMOVE REDUNDANT HISTORICAL FEATURES
# ============================================

# Bỏ median và std — mean đã đủ đại diện
# Bỏ monthly stats — month-day stats cụ thể hơn và chứa cùng thông tin seasonal
# Bỏ cogs mirror nếu revenue đã có (hoặc giữ cả nhưng chỉ giữ mean)

cols_to_drop = [
    # Medians (mean đã đủ)
    'hist_monthly_revenue_median', 'hist_monthly_cogs_median',
    'hist_weekday_revenue_median', 'hist_weekday_cogs_median',
    'hist_monthday_revenue_median', 'hist_monthday_cogs_median',
    'hist_weekend_revenue_median', 'hist_weekend_cogs_median',
    'hist_promo_revenue_median', 'hist_promo_cogs_median',
    'hist_tet_revenue_median', 'hist_tet_cogs_median',
    
    # Std (quá noisy với ít data)
    'hist_monthly_revenue_std', 'hist_monthly_cogs_std',
    'hist_weekday_revenue_std', 'hist_weekday_cogs_std',
    
    # Monthly mean (month-day mean cụ thể hơn, chứa cùng seasonal info)
    'hist_monthly_revenue_mean', 'hist_monthly_cogs_mean',
    
    # Cogs mirrors — giữ lại hist_monthday_cogs_mean vì quan trọng, 
    # nhưng bỏ các cogs phụ khác để giảm feature count
    'hist_weekday_cogs_mean',
    'hist_weekend_cogs_mean', 
    'hist_promo_cogs_mean',
    'hist_tet_cogs_mean',
]

train_reduced = train.drop(columns=cols_to_drop)
test_reduced = test.drop(columns=cols_to_drop)

# Check correlation của remaining historical features
print("\n=== CORRELATION CHECK (remaining hist features) ===")
hist_cols = [c for c in train_reduced.columns if c.startswith('hist_')]
print(f"Remaining historical features: {len(hist_cols)}")

corr = train_reduced[hist_cols].corr()
high_corr = []
for i in range(len(hist_cols)):
    for j in range(i+1, len(hist_cols)):
        if abs(corr.iloc[i,j]) > 0.85:
            high_corr.append((hist_cols[i], hist_cols[j], corr.iloc[i,j]))
            print(f"{hist_cols[i]} <-> {hist_cols[j]}: {corr.iloc[i,j]:.3f}")

if not high_corr:
    print("No high correlations (>0.85) found!")

# ============================================
# FINAL STATS
# ============================================
print(f"\n{'='*60}")
print(f"Final feature count: {train_reduced.shape[1]}")
print(f"Samples: {len(train_reduced)}")
print(f"Ratio: {len(train_reduced)/train_reduced.shape[1]:.1f} samples/feature")
print("="*60)

print(f"\nRemaining features ({train_reduced.shape[1]}):")
for i, col in enumerate(train_reduced.columns, 1):
    print(f"{i:2d}. {col}")

# Save
out_dir = Path(r'D:\Datathon-2026\output')
train_reduced.to_csv(out_dir / 'train_features_balanced.csv', index=False)
test_reduced.to_csv(out_dir / 'test_features_balanced.csv', index=False)

print(f"\nSaved:")
print(f"  {out_dir / 'train_features_balanced.csv'}")
print(f"  {out_dir / 'test_features_balanced.csv'}")
