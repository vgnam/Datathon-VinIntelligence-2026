import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

print("="*70)
print("DROP BROKEN conversion_rate_overall, KEEP ONLY GOOD FEATURES")
print("="*70)

# Load fixed features
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_fixed.csv', parse_dates=['Date'])
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_fixed.csv', parse_dates=['Date'])

# Drop broken column
if 'conversion_rate_overall' in train.columns:
    print(f"Dropping conversion_rate_overall from train (was broken: max={train['conversion_rate_overall'].max():.0f})")
    train = train.drop(columns=['conversion_rate_overall'])
if 'conversion_rate_overall' in test.columns:
    test = test.drop(columns=['conversion_rate_overall'])

# Also drop is_imputed_conversion_rate
for col in ['is_imputed_conversion_rate']:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

# Save
train.to_csv(r'D:\Datathon-2026\output\train_features_fixed.csv', index=False)
test.to_csv(r'D:\Datathon-2026\output\test_features_fixed.csv', index=False)

print(f"Train shape after drop: {train.shape}")
print(f"Test shape after drop: {test.shape}")

# Rerun pipeline steps
print("\nRerunning reduce_features.py...")
result = subprocess.run(['python', 'scripts/reduce_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)

print("\nRerunning add_historical_features.py...")
result = subprocess.run(['python', 'scripts/add_historical_features.py'], capture_output=True, text=True, cwd=r'D:\Datathon-2026')
print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)

print("\nDONE")
