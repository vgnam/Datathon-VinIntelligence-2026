import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAIN: SELECTED FEATURES V2 (FIXED) vs V1 (OLD)")
print("="*70)

# Load V1
train_v1 = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test_v1 = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# Load V2
train_v2 = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test_v2 = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# Log transform
EPS = 1.0
for df in [train_v1, train_v2]:
    df['Revenue_log'] = np.log1p(df['Revenue'] + EPS)
    df['COGS_log'] = np.log1p(df['COGS'] + EPS)

# Metrics
def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs(actual - pred) / (np.abs(actual) + eps)))

def r2(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-9))

def inverse_transform(log_pred):
    return np.expm1(log_pred) - EPS

def train_and_eval(train_df, test_df, features, label):
    X = train_df[features].copy()
    y_log = train_df[['Revenue_log', 'COGS_log']].copy()
    y_raw = train_df[['Revenue', 'COGS']].copy()

    val_days = 180
    split_date = train_df['Date'].max() - pd.Timedelta(days=val_days)
    train_mask = train_df['Date'] <= split_date
    val_mask = train_df['Date'] > split_date

    X_train = X.loc[train_mask].copy()
    X_val = X.loc[val_mask].copy()
    y_train_log = y_log.loc[train_mask].copy()
    y_val_raw = y_raw.loc[val_mask].copy()

    median_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(median_values).fillna(0.0)
    X_val = X_val.fillna(median_values).fillna(0.0)
    X_full = X.fillna(median_values).fillna(0.0)
    X_test = test_df[features].copy()
    X_test = X_test.fillna(median_values).fillna(0.0)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train_log)
    rf_val = pd.DataFrame(inverse_transform(rf.predict(X_val)), columns=['Revenue', 'COGS'])

    # ExtraTrees
    et = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
    et.fit(X_train, y_train_log)
    et_val = pd.DataFrame(inverse_transform(et.predict(X_val)), columns=['Revenue', 'COGS'])

    print(f"\n{label} — {len(features)} features")
    print(f"  RF  MAPE Rev: {mape(y_val_raw['Revenue'], rf_val['Revenue']):.4f} | COGS: {mape(y_val_raw['COGS'], rf_val['COGS']):.4f} | R2 Rev: {r2(y_val_raw['Revenue'], rf_val['Revenue']):.4f}")
    print(f"  ET  MAPE Rev: {mape(y_val_raw['Revenue'], et_val['Revenue']):.4f} | COGS: {mape(y_val_raw['COGS'], et_val['COGS']):.4f} | R2 Rev: {r2(y_val_raw['Revenue'], et_val['Revenue']):.4f}")

    # Feature importance
    importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
    print(f"  Top 5 RF importance:")
    for _, row in importance.head(5).iterrows():
        print(f"    {row['Feature']:40s} {row['Importance']:.4f}")

    # Full retrain RF
    rf_full = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=2, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
    rf_full.fit(X_full, y_log)
    test_pred = pd.DataFrame(inverse_transform(rf_full.predict(X_test)), columns=['Revenue', 'COGS'])
    return test_pred

# V1 features
features_v1 = [
    'hist_monthday_revenue_mean', 'hist_monthday_cogs_mean', 'day_cos',
    'hist_yoy_cogs_growth', 'day_sin', 'hist_yoy_revenue_growth',
    'expected_sessions', 'month', 'inventory_sell_through',
    'inventory_days_since_snapshot', 'traffic_uncertainty', 'promo_intensity'
]

# V2 features
features_v2 = [
    'hist_monthday_revenue_mean', 'hist_monthday_revenue_mean_recent',
    'hist_monthday_cogs_mean', 'hist_monthday_cogs_mean_recent',
    'day_cos', 'day_sin', 'hist_yoy_revenue_growth', 'hist_yoy_cogs_growth',
    'expected_sessions', 'month', 'is_tet_period', 'days_to_tet',
    'traffic_uncertainty', 'promo_seasonal_prob', 'promo_monthly_prob'
]

print("\n" + "="*70)
print("TRAIN V1 (OLD)")
print("="*70)
pred_v1 = train_and_eval(train_v1, test_v1, features_v1, "V1 (OLD)")

print("\n" + "="*70)
print("TRAIN V2 (FIXED)")
print("="*70)
pred_v2 = train_and_eval(train_v2, test_v2, features_v2, "V2 (FIXED)")

# Save V2 submission
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)

sub_v2 = pd.DataFrame({
    'Date': test_v2['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.maximum(pred_v2['Revenue'], 0.0).round(2),
    'COGS': np.maximum(pred_v2['COGS'], 0.0).round(2),
})
sub_v2['COGS'] = np.minimum(sub_v2['COGS'], sub_v2['Revenue'] * 0.995)
sub_v2.to_csv(out_dir / 'rf_selected_v2_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'rf_selected_v2_submission.csv'}")

print("\n" + "="*70)
print("DONE")
print("="*70)
