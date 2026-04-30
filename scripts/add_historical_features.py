import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("ADDING HISTORICAL STATISTICS AS FEATURES")
print("="*60)

# Load data
TRAIN_FEATURES = Path(r'D:\Datathon-2026\output\train_features_fixed.csv')
TEST_FEATURES = Path(r'D:\Datathon-2026\output\test_features_fixed.csv')
TRAIN_TARGET = Path(r'D:\Datathon-2026\output\train_target.csv')

train_feat = pd.read_csv(TRAIN_FEATURES, parse_dates=["Date"])
test_feat = pd.read_csv(TEST_FEATURES, parse_dates=["Date"])
train_target = pd.read_csv(TRAIN_TARGET, parse_dates=["Date"])

if "net_revenue" in train_target.columns and "Revenue" not in train_target.columns:
    train_target = train_target.rename(columns={"net_revenue": "Revenue"})

# Merge features with targets for historical stats
train_full = train_feat.merge(train_target[["Date", "Revenue", "COGS"]], on="Date", how="left")

print(f"Train: {len(train_full)} days")
print(f"Test: {len(test_feat)} days")
print(f"Date range: {train_full['Date'].min().date()} to {train_full['Date'].max().date()}")

# ============================================
# COMPUTE HISTORICAL STATISTICS
# ============================================

print("\n" + "="*60)
print("COMPUTING HISTORICAL STATISTICS")
print("="*60)

# 1. Monthly historical averages
def compute_monthly_stats(df):
    """Monthly Revenue/COGS statistics"""
    monthly = df.groupby('month').agg({
        'Revenue': ['mean', 'median', 'std'],
        'COGS': ['mean', 'median', 'std']
    }).reset_index()
    monthly.columns = ['month', 
                       'hist_monthly_revenue_mean', 'hist_monthly_revenue_median', 'hist_monthly_revenue_std',
                       'hist_monthly_cogs_mean', 'hist_monthly_cogs_median', 'hist_monthly_cogs_std']
    return monthly

monthly_stats = compute_monthly_stats(train_full)
print(f"1. Monthly stats: {len(monthly_stats)} months")

# 2. Weekday historical averages
def compute_weekday_stats(df):
    """Weekday Revenue/COGS statistics"""
    weekday = df.groupby('day_of_week').agg({
        'Revenue': ['mean', 'median', 'std'],
        'COGS': ['mean', 'median', 'std']
    }).reset_index()
    weekday.columns = ['day_of_week',
                       'hist_weekday_revenue_mean', 'hist_weekday_revenue_median', 'hist_weekday_revenue_std',
                       'hist_weekday_cogs_mean', 'hist_weekday_cogs_median', 'hist_weekday_cogs_std']
    return weekday

weekday_stats = compute_weekday_stats(train_full)
print(f"2. Weekday stats: {len(weekday_stats)} weekdays")

# 3. Month-Day historical averages (seasonal profile)
def compute_monthday_stats(df):
    """Month-Day Revenue/COGS statistics (captures yearly seasonality)"""
    df = df.copy()
    df['month_day'] = df['Date'].dt.strftime('%m-%d')
    monthday = df.groupby('month_day').agg({
        'Revenue': ['mean', 'median', 'std', 'count'],
        'COGS': ['mean', 'median', 'std', 'count']
    }).reset_index()
    monthday.columns = ['month_day',
                        'hist_monthday_revenue_mean', 'hist_monthday_revenue_median', 'hist_monthday_revenue_std', 'hist_monthday_count',
                        'hist_monthday_cogs_mean', 'hist_monthday_cogs_median', 'hist_monthday_cogs_std', 'hist_monthday_cogs_count']
    # Only keep month-days with at least 3 years of data
    monthday = monthday[monthday['hist_monthday_count'] >= 3]
    return monthday

monthday_stats = compute_monthday_stats(train_full)
print(f"3. Month-Day stats: {len(monthday_stats)} unique month-days (min 3 years)")

# 4. Weekend vs Weekday historical averages
def compute_weekend_stats(df):
    """Weekend vs Weekday Revenue/COGS statistics"""
    weekend = df.groupby('is_weekend').agg({
        'Revenue': ['mean', 'median'],
        'COGS': ['mean', 'median']
    }).reset_index()
    weekend.columns = ['is_weekend',
                       'hist_weekend_revenue_mean', 'hist_weekend_revenue_median',
                       'hist_weekend_cogs_mean', 'hist_weekend_cogs_median']
    return weekend

weekend_stats = compute_weekend_stats(train_full)
print(f"4. Weekend stats: {len(weekend_stats)} groups")

# 5. Promo vs Non-Promo historical averages
def compute_promo_stats(df):
    """Promo vs Non-Promo Revenue/COGS statistics"""
    promo = df.groupby('promo_active').agg({
        'Revenue': ['mean', 'median'],
        'COGS': ['mean', 'median']
    }).reset_index()
    promo.columns = ['promo_active',
                     'hist_promo_revenue_mean', 'hist_promo_revenue_median',
                     'hist_promo_cogs_mean', 'hist_promo_cogs_median']
    return promo

promo_stats = compute_promo_stats(train_full)
print(f"5. Promo stats: {len(promo_stats)} groups")

# 6. TET period historical averages
def compute_tet_stats(df):
    """TET vs Non-TET Revenue/COGS statistics"""
    tet = df.groupby('is_tet_period').agg({
        'Revenue': ['mean', 'median'],
        'COGS': ['mean', 'median']
    }).reset_index()
    tet.columns = ['is_tet_period',
                   'hist_tet_revenue_mean', 'hist_tet_revenue_median',
                   'hist_tet_cogs_mean', 'hist_tet_cogs_median']
    return tet

tet_stats = compute_tet_stats(train_full)
print(f"6. TET stats: {len(tet_stats)} groups")

# 7. Monthly growth rate (YoY)
def compute_yoy_growth(df):
    """Year-over-year growth rate by month"""
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    yearly_monthly = df.groupby(['year', 'month'])[['Revenue', 'COGS']].sum().reset_index()
    
    growth_rates = []
    for month in range(1, 13):
        month_data = yearly_monthly[yearly_monthly['month'] == month].sort_values('year')
        if len(month_data) >= 2:
            for i in range(1, len(month_data)):
                rev_growth = (month_data.iloc[i]['Revenue'] - month_data.iloc[i-1]['Revenue']) / (month_data.iloc[i-1]['Revenue'] + 1e-9)
                cogs_growth = (month_data.iloc[i]['COGS'] - month_data.iloc[i-1]['COGS']) / (month_data.iloc[i-1]['COGS'] + 1e-9)
                growth_rates.append({
                    'year': month_data.iloc[i]['year'],
                    'month': month,
                    'yoy_revenue_growth': rev_growth,
                    'yoy_cogs_growth': cogs_growth
                })
    
    growth_df = pd.DataFrame(growth_rates)
    # Average YoY growth by month (across all years)
    avg_growth = growth_df.groupby('month')[['yoy_revenue_growth', 'yoy_cogs_growth']].mean().reset_index()
    avg_growth.columns = ['month', 'hist_yoy_revenue_growth', 'hist_yoy_cogs_growth']
    return avg_growth

yoy_growth = compute_yoy_growth(train_full)
print(f"7. YoY growth: {len(yoy_growth)} months")

# ============================================
# MERGE HISTORICAL STATS INTO FEATURES
# ============================================

def add_historical_features(df, is_train=False):
    """Add historical statistics as features"""
    df = df.copy()
    
    # 1. Monthly stats
    df = df.merge(monthly_stats, on='month', how='left')
    
    # 2. Weekday stats
    df = df.merge(weekday_stats, on='day_of_week', how='left')
    
    # 3. Month-Day stats
    df['month_day'] = df['Date'].dt.strftime('%m-%d')
    df = df.merge(monthday_stats[['month_day', 
                                   'hist_monthday_revenue_mean', 'hist_monthday_revenue_median',
                                   'hist_monthday_cogs_mean', 'hist_monthday_cogs_median']], 
                  on='month_day', how='left')
    df = df.drop('month_day', axis=1)
    
    # 4. Weekend stats
    df = df.merge(weekend_stats, on='is_weekend', how='left')
    
    # 5. Promo stats (skip for test since promo_active is always 0 in test)
    if is_train:
        df = df.merge(promo_stats, on='promo_active', how='left')
    else:
        # For test: assign non-promo historical means directly
        promo_mean_row = promo_stats[promo_stats['promo_active'] == 0]
        if not promo_mean_row.empty:
            for col in promo_mean_row.columns:
                if col != 'promo_active' and col not in df.columns:
                    df[col] = promo_mean_row.iloc[0][col]
        else:
            df['hist_promo_revenue_mean'] = df['hist_monthday_revenue_mean']
            df['hist_promo_revenue_median'] = df['hist_monthday_revenue_median']
            df['hist_promo_cogs_mean'] = df['hist_monthday_cogs_mean']
            df['hist_promo_cogs_median'] = df['hist_monthday_cogs_median']
    
    # 6. TET stats
    df = df.merge(tet_stats, on='is_tet_period', how='left')
    
    # 7. YoY growth
    df = df.merge(yoy_growth, on='month', how='left')
    
    return df

print("\n" + "="*60)
print("ADDING FEATURES TO TRAIN AND TEST")
print("="*60)

train_enhanced = add_historical_features(train_full, is_train=True)
test_enhanced = add_historical_features(test_feat, is_train=False)

# Drop original targets from train (they were only needed for stats computation)
target_cols = ['Revenue', 'COGS']
if all(col in train_enhanced.columns for col in target_cols):
    train_enhanced = train_enhanced.drop(columns=target_cols)

print(f"Enhanced train: {train_enhanced.shape}")
print(f"Enhanced test: {test_enhanced.shape}")

# ============================================
# FILL MISSING VALUES
# ============================================

# For month-day stats that might be missing (e.g., leap day 02-29, or sparse data)
# Fall back to monthly mean
hist_cols = [c for c in train_enhanced.columns if c.startswith('hist_')]
print(f"\nHistorical feature columns ({len(hist_cols)}):")
for col in hist_cols:
    print(f"  - {col}")

# Check missing values
print("\n=== MISSING VALUES ===")
for col in hist_cols:
    train_missing = train_enhanced[col].isna().sum()
    test_missing = test_enhanced[col].isna().sum()
    if train_missing > 0 or test_missing > 0:
        print(f"{col}: train={train_missing}, test={test_missing}")
        # Fill with median from training data
        median_val = train_enhanced[col].median()
        train_enhanced[col] = train_enhanced[col].fillna(median_val)
        test_enhanced[col] = test_enhanced[col].fillna(median_val)

# ============================================
# SAVE
# ============================================

out_dir = Path(r'D:\Datathon-2026\output')
train_enhanced.to_csv(out_dir / 'train_features_enhanced.csv', index=False)
test_enhanced.to_csv(out_dir / 'test_features_enhanced.csv', index=False)

print(f"\n{'='*60}")
print("SAVED:")
print(f"  {out_dir / 'train_features_enhanced.csv'}")
print(f"  {out_dir / 'test_features_enhanced.csv'}")
print(f"\nFeature count: {len(train_enhanced.columns)} (was 23, added {len(hist_cols)} hist features)")
print("="*60)

# Show sample
print("\n=== SAMPLE ENHANCED FEATURES ===")
print(train_enhanced[['Date', 'hist_monthly_revenue_mean', 'hist_weekday_revenue_mean', 
                       'hist_monthday_revenue_mean', 'hist_yoy_revenue_growth']].head())
