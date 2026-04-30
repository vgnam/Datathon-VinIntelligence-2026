import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("SYNC promo_seasonal_prob INTO TRAIN (was missing)")
print("="*70)

# Load test (has promo_seasonal_prob)
test_balanced = pd.read_csv(r'D:\Datathon-2026\output\test_features_balanced.csv', parse_dates=['Date'])
test_enhanced = pd.read_csv(r'D:\Datathon-2026\output\test_features_enhanced.csv', parse_dates=['Date'])

# Load train (missing promo_seasonal_prob)
train_balanced = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv', parse_dates=['Date'])
train_enhanced = pd.read_csv(r'D:\Datathon-2026\output\train_features_enhanced.csv', parse_dates=['Date'])

# Compute from train itself (same logic as test)
train_balanced['month'] = train_balanced['Date'].dt.month
train_balanced['month_day'] = train_balanced['Date'].dt.strftime('%m-%d')
train_balanced['day_of_week'] = train_balanced['Date'].dt.dayofweek

promo_by_md = train_balanced.groupby('month_day')['promo_active'].mean().to_dict()
promo_by_month = train_balanced.groupby('month')['promo_active'].mean().to_dict()
promo_by_dow = train_balanced.groupby('day_of_week')['promo_active'].mean().to_dict()

train_balanced['promo_seasonal_prob'] = train_balanced['month_day'].map(promo_by_md).fillna(0)
train_balanced['promo_monthly_prob'] = train_balanced['month'].map(promo_by_month).fillna(0)
train_balanced['promo_dow_prob'] = train_balanced['day_of_week'].map(promo_by_dow).fillna(0)

# Same for enhanced
train_enhanced['month'] = train_enhanced['Date'].dt.month
train_enhanced['month_day'] = train_enhanced['Date'].dt.strftime('%m-%d')
train_enhanced['day_of_week'] = train_enhanced['Date'].dt.dayofweek

train_enhanced['promo_seasonal_prob'] = train_enhanced['month_day'].map(promo_by_md).fillna(0)
train_enhanced['promo_monthly_prob'] = train_enhanced['month'].map(promo_by_month).fillna(0)
train_enhanced['promo_dow_prob'] = train_enhanced['day_of_week'].map(promo_by_dow).fillna(0)

# Drop temp columns
train_balanced = train_balanced.drop(columns=['month','month_day','day_of_week'])
train_enhanced = train_enhanced.drop(columns=['month','month_day','day_of_week'])

# Save
train_balanced.to_csv(r'D:\Datathon-2026\output\train_features_balanced.csv', index=False)
train_enhanced.to_csv(r'D:\Datathon-2026\output\train_features_enhanced.csv', index=False)

print(f"Train balanced now: {train_balanced.shape}")
print(f"Train enhanced now: {train_enhanced.shape}")
print("Done")
