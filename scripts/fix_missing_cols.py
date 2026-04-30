import pandas as pd

# Fix missing month/day_of_week in enhanced/balanced train
for fname in ['train_features_enhanced.csv', 'train_features_balanced.csv']:
    df = pd.read_csv(f'D:/Datathon-2026/output/{fname}', parse_dates=['Date'])
    if 'month' not in df.columns:
        df['month'] = df['Date'].dt.month
        print(f"Added 'month' to {fname}")
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['Date'].dt.dayofweek
        print(f"Added 'day_of_week' to {fname}")
    df.to_csv(f'D:/Datathon-2026/output/{fname}', index=False)
    print(f"Saved {fname}: {df.shape}")

print("Done")
