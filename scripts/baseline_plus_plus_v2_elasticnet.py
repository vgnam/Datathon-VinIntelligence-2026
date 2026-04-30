import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BASELINE++ V2 + ELASTIC NET RESIDUAL")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================
# 1. BASE = Recent seasonal mean + traffic adjustment (from Baseline++ v2)
# ============================================
train['base_rev'] = train['hist_monthday_revenue_mean_recent'].fillna(train['hist_monthday_revenue_mean'])
train['base_cogs'] = train['hist_monthday_cogs_mean_recent'].fillna(train['hist_monthday_cogs_mean'])
test['base_rev'] = test['hist_monthday_revenue_mean_recent'].fillna(test['hist_monthday_revenue_mean'])
test['base_cogs'] = test['hist_monthday_cogs_mean_recent'].fillna(test['hist_monthday_cogs_mean'])

# Traffic adjustment
train['month_day'] = train['Date'].dt.strftime('%m-%d')
session_md = train.groupby('month_day')['expected_sessions'].mean().to_dict()
train['sess_md'] = train['month_day'].map(session_md)
test['month_day'] = test['Date'].dt.strftime('%m-%d')
test['sess_md'] = test['month_day'].map(session_md).fillna(test['expected_sessions'].median())

train['traffic_mult'] = (train['expected_sessions'] / (train['sess_md'] + 1)).clip(0.7, 1.3)
test['traffic_mult'] = (test['expected_sessions'] / (test['sess_md'] + 1)).clip(0.7, 1.3)

train['base_rev'] = train['base_rev'] * train['traffic_mult']
train['base_cogs'] = train['base_cogs'] * train['traffic_mult']
test['base_rev'] = test['base_rev'] * test['traffic_mult']
test['base_cogs'] = test['base_cogs'] * test['traffic_mult']

# ============================================
# 2. RESIDUAL = Actual - Base
# ============================================
train['resid_rev'] = train['Revenue'] - train['base_rev']
train['resid_cogs'] = train['COGS'] - train['base_cogs']

# ============================================
# 3. ELASTIC NET on residual
# ============================================
resid_features = [
    'day_cos','day_sin',
    'hist_yoy_revenue_growth','hist_yoy_cogs_growth',
    'month','is_tet_period','days_to_tet',
    'promo_seasonal_prob','promo_monthly_prob',
    'traffic_uncertainty',
]

for c in resid_features:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())

# Train/val split
val_days = 180
split_date = train['Date'].max() - pd.Timedelta(days=val_days)
train_mask = train['Date'] <= split_date
val_mask = train['Date'] > split_date

X_tr = train.loc[train_mask, resid_features]
y_tr_rev = train.loc[train_mask, 'resid_rev']
y_tr_cogs = train.loc[train_mask, 'resid_cogs']
X_val = train.loc[val_mask, resid_features]

# Grid search ElasticNet
best_mae_rev = np.inf
best_en_rev = None
best_mae_cogs = np.inf
best_en_cogs = None

print("\nGrid search ElasticNet residual...")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        en_rev = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
        en_cogs = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
        en_rev.fit(X_tr, y_tr_rev)
        en_cogs.fit(X_tr, y_tr_cogs)
        
        pred_rev = train.loc[val_mask, 'base_rev'] + en_rev.predict(X_val)
        pred_cogs = train.loc[val_mask, 'base_cogs'] + en_cogs.predict(X_val)
        
        mae_rev = np.mean(np.abs(train.loc[val_mask, 'Revenue'] - pred_rev))
        mae_cogs = np.mean(np.abs(train.loc[val_mask, 'COGS'] - pred_cogs))
        
        if mae_rev < best_mae_rev:
            best_mae_rev = mae_rev
            best_en_rev = en_rev
        if mae_cogs < best_mae_cogs:
            best_mae_cogs = mae_cogs
            best_en_cogs = en_cogs

print(f"Best Rev: alpha={best_en_rev.alpha}, l1={best_en_rev.l1_ratio}")
print(f"Best COGS: alpha={best_en_cogs.alpha}, l1={best_en_cogs.l1_ratio}")

# ============================================
# 4. VALIDATION METRICS
# ============================================
val_pred_rev = train.loc[val_mask, 'base_rev'] + best_en_rev.predict(X_val)
val_pred_cogs = train.loc[val_mask, 'base_cogs'] + best_en_cogs.predict(X_val)

def mape(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    return float(np.mean(np.abs(a-p)/(np.abs(a)+eps)))

def r2(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    ss_res=np.sum((a-p)**2); ss_tot=np.sum((a-np.mean(a))**2)
    return float(1-ss_res/(ss_tot+eps))

print(f"\nValidation ({val_mask.sum()} days):")
print(f"MAE  Revenue: {np.mean(np.abs(train.loc[val_mask,'Revenue'] - val_pred_rev)):,.0f}")
print(f"MAE  COGS:    {np.mean(np.abs(train.loc[val_mask,'COGS'] - val_pred_cogs)):,.0f}")
print(f"MAPE Revenue: {mape(train.loc[val_mask,'Revenue'], val_pred_rev):.4f}")
print(f"MAPE COGS:    {mape(train.loc[val_mask,'COGS'], val_pred_cogs):.4f}")
print(f"R2   Revenue: {r2(train.loc[val_mask,'Revenue'], val_pred_rev):.4f}")
print(f"R2   COGS:    {r2(train.loc[val_mask,'COGS'], val_pred_cogs):.4f}")

print(f"\nBase-only MAE Revenue: {np.mean(np.abs(train.loc[val_mask,'Revenue'] - train.loc[val_mask,'base_rev'])):,.0f}")

# ============================================
# 5. FULL RETRAIN & TEST
# ============================================
en_rev_f = ElasticNet(alpha=best_en_rev.alpha, l1_ratio=best_en_rev.l1_ratio, max_iter=10000, random_state=42)
en_cogs_f = ElasticNet(alpha=best_en_cogs.alpha, l1_ratio=best_en_cogs.l1_ratio, max_iter=10000, random_state=42)
en_rev_f.fit(train[resid_features], train['resid_rev'])
en_cogs_f.fit(train[resid_features], train['resid_cogs'])

test['pred_rev'] = test['base_rev'] + en_rev_f.predict(test[resid_features])
test['pred_cogs'] = test['base_cogs'] + en_cogs_f.predict(test[resid_features])

test['pred_rev'] = np.maximum(test['pred_rev'], 0)
test['pred_cogs'] = np.maximum(test['pred_cogs'], 0)
test['pred_cogs'] = np.minimum(test['pred_cogs'], test['pred_rev'] * 0.995)

# ============================================
# 6. SAVE
# ============================================
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['pred_rev'], 2),
    'COGS': np.round(test['pred_cogs'], 2),
}).to_csv(out_dir / 'baseline_plus_plus_v2_elasticnet_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'baseline_plus_plus_v2_elasticnet_submission.csv'}")

# Coefficients
coef_rev = pd.DataFrame({'Feature': resid_features, 'Coef': en_rev_f.coef_}).sort_values('Coef', key=abs, ascending=False)
print("\nTop residual coefficients (Revenue):")
print(coef_rev.to_string(index=False))

print("\nDONE")
