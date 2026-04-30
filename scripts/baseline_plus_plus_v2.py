import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BASELINE++ V2: SEASONAL BASE + TRAFFIC + RESIDUAL (SIMPLIFIED)")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================
# 1. BASE = hist_monthday_revenue_mean_recent (2019-2022, more relevant)
# ============================================
train['base_rev'] = train['hist_monthday_revenue_mean_recent'].fillna(train['hist_monthday_revenue_mean'])
train['base_cogs'] = train['hist_monthday_cogs_mean_recent'].fillna(train['hist_monthday_cogs_mean'])
test['base_rev'] = test['hist_monthday_revenue_mean_recent'].fillna(test['hist_monthday_revenue_mean'])
test['base_cogs'] = test['hist_monthday_cogs_mean_recent'].fillna(test['hist_monthday_cogs_mean'])

# ============================================
# 2. TRAFFIC ADJUSTMENT: expected_sessions vs historical profile
# ============================================
train['month_day'] = train['Date'].dt.strftime('%m-%d')
session_md = train.groupby('month_day')['expected_sessions'].mean().to_dict()
train['sess_md'] = train['month_day'].map(session_md)
test['month_day'] = test['Date'].dt.strftime('%m-%d')
test['sess_md'] = test['month_day'].map(session_md).fillna(test['expected_sessions'].median())

# Smooth multiplier
train['traffic_mult'] = (train['expected_sessions'] / (train['sess_md'] + 1)).clip(0.7, 1.3)
test['traffic_mult'] = (test['expected_sessions'] / (test['sess_md'] + 1)).clip(0.7, 1.3)

# Apply
train['adj_rev'] = train['base_rev'] * train['traffic_mult']
train['adj_cogs'] = train['base_cogs'] * train['traffic_mult']

# ============================================
# 3. RESIDUAL = Ridge on remaining features
# ============================================
train['resid_rev'] = train['Revenue'] - train['adj_rev']
train['resid_cogs'] = train['COGS'] - train['adj_cogs']

feat_cols = ['day_cos','day_sin','hist_yoy_revenue_growth','hist_yoy_cogs_growth',
             'month','is_tet_period','days_to_tet','traffic_uncertainty',
             'promo_seasonal_prob','promo_monthly_prob']

# Train/val split
val_days = 180
split_date = train['Date'].max() - pd.Timedelta(days=val_days)
train_mask = train['Date'] <= split_date
val_mask = train['Date'] > split_date

X_tr = train.loc[train_mask, feat_cols].fillna(0)
y_tr_rev = train.loc[train_mask, 'resid_rev']
y_tr_cogs = train.loc[train_mask, 'resid_cogs']

ridge_rev = Ridge(alpha=5000)
ridge_cogs = Ridge(alpha=5000)
ridge_rev.fit(X_tr, y_tr_rev)
ridge_cogs.fit(X_tr, y_tr_cogs)

# Predict
train['pred_resid_rev'] = ridge_rev.predict(train[feat_cols].fillna(0))
train['pred_resid_cogs'] = ridge_cogs.predict(train[feat_cols].fillna(0))

# Final
train['final_rev'] = train['adj_rev'] + train['pred_resid_rev']
train['final_cogs'] = train['adj_cogs'] + train['pred_resid_cogs']

# ============================================
# 4. VALIDATION METRICS
# ============================================
def mape(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    return float(np.mean(np.abs(a-p)/(np.abs(a)+eps)))
def r2(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    ss_res=np.sum((a-p)**2); ss_tot=np.sum((a-np.mean(a))**2)
    return float(1-ss_res/(ss_tot+eps))

val_rev = train.loc[val_mask, 'Revenue']
val_cogs = train.loc[val_mask, 'COGS']
val_pred_rev = train.loc[val_mask, 'final_rev']
val_pred_cogs = train.loc[val_mask, 'final_cogs']

print(f"\nBase-only MAPE Rev: {mape(val_rev, train.loc[val_mask,'base_rev']):.4f}")
print(f"Base+Traffic MAPE:  {mape(val_rev, train.loc[val_mask,'adj_rev']):.4f}")
print(f"Final++ MAPE Rev:   {mape(val_rev, val_pred_rev):.4f}")
print(f"Final++ MAPE COGS:  {mape(val_cogs, val_pred_cogs):.4f}")
print(f"Final++ R2 Rev:     {r2(val_rev, val_pred_rev):.4f}")
print(f"Final++ R2 COGS:    {r2(val_cogs, val_pred_cogs):.4f}")

# ============================================
# 5. TEST PREDICTION
# ============================================
# Full retrain
ridge_rev_f = Ridge(alpha=5000)
ridge_cogs_f = Ridge(alpha=5000)
ridge_rev_f.fit(train[feat_cols].fillna(0), train['resid_rev'])
ridge_cogs_f.fit(train[feat_cols].fillna(0), train['resid_cogs'])

test['adj_rev'] = test['base_rev'] * test['traffic_mult']
test['adj_cogs'] = test['base_cogs'] * test['traffic_mult']
test['final_rev'] = test['adj_rev'] + ridge_rev_f.predict(test[feat_cols].fillna(0))
test['final_cogs'] = test['adj_cogs'] + ridge_cogs_f.predict(test[feat_cols].fillna(0))

# Constraints
test['final_rev'] = np.maximum(test['final_rev'], 0)
test['final_cogs'] = np.maximum(test['final_cogs'], 0)
test['final_cogs'] = np.minimum(test['final_cogs'], test['final_rev'] * 0.995)

# Save
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['final_rev'], 2),
    'COGS': np.round(test['final_cogs'], 2),
}).to_csv(out_dir / 'baseline_plus_plus_v2_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'baseline_plus_plus_v2_submission.csv'}")
print("DONE")
