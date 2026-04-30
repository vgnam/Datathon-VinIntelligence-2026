import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("RECALCULATE conversion_rate_overall FROM ORDERS + WEB_TRAFFIC")
print("="*70)

# Load orders
orders = pd.read_csv(r'D:\Datathon-2026\data\transaction\orders.csv', parse_dates=['order_date'])
orders['year_month'] = orders['order_date'].dt.to_period('M')
orders_by_month = orders.groupby('year_month').size().reset_index(name='order_count')

# Load web traffic
traffic = pd.read_csv(r'D:\Datathon-2026\data\operational\web_traffic.csv', parse_dates=['date'])
traffic['year_month'] = traffic['date'].dt.to_period('M')
sessions_by_month = traffic.groupby('year_month')['sessions'].sum().reset_index(name='session_count')

# Merge
conv = orders_by_month.merge(sessions_by_month, on='year_month', how='outer').fillna(0)
conv['conversion_rate'] = conv['order_count'] / (conv['session_count'] + 1e-9)

# Monthly average (for test period)
conv['month'] = conv['year_month'].dt.month
monthly_avg = conv.groupby('month')['conversion_rate'].mean().to_dict()
print(f"Monthly avg conversion rate: {monthly_avg}")

# Load fixed features
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_fixed.csv', parse_dates=['Date'])
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_fixed.csv', parse_dates=['Date'])

# Map year-month to train
train['year_month'] = train['Date'].dt.to_period('M')
train = train.merge(conv[['year_month','conversion_rate']], on='year_month', how='left')
train['conversion_rate_overall'] = train['conversion_rate'].fillna(train['conversion_rate_overall'])
train = train.drop(columns=['year_month','conversion_rate'])

# Map month-only to test
test['month'] = test['Date'].dt.month
test['conversion_rate_overall'] = test['month'].map(monthly_avg).fillna(0)

print(f"\nTrain conversion_rate: min={train['conversion_rate_overall'].min():.6f}, max={train['conversion_rate_overall'].max():.6f}, mean={train['conversion_rate_overall'].mean():.6f}")
print(f"Test conversion_rate:  min={test['conversion_rate_overall'].min():.6f}, max={test['conversion_rate_overall'].max():.6f}, mean={test['conversion_rate_overall'].mean():.6f}")

# Save back
train.to_csv(r'D:\Datathon-2026\output\train_features_fixed.csv', index=False)
test.to_csv(r'D:\Datathon-2026\output\test_features_fixed.csv', index=False)

print("\nSaved fixed files with corrected conversion_rate")

# Now rerun reduce_features and add_historical_features
import subprocess
print("\nRerunning reduce_features.py...")
result = subprocess.run(['python', 'scripts/reduce_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)

print("\nRerunning add_historical_features.py...")
result = subprocess.run(['python', 'scripts/add_historical_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)

print("\nDONE")
