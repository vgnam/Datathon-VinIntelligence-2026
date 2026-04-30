import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RESIDUAL ELASTIC NET on SELECTED V2")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# ============================================
# 1. BASE PREDICTION (same as baseline+)
# ============================================
train['base_rev'] = train['hist_monthday_revenue_mean'].fillna(train['hist_monthday_revenue_mean_recent'])
train['base_cogs'] = train['hist_monthday_cogs_mean'].fillna(train['hist_monthday_cogs_mean_recent'])
test['base_rev'] = test['hist_monthday_revenue_mean'].fillna(test['hist_monthday_revenue_mean_recent'])
test['base_cogs'] = test['hist_monthday_cogs_mean'].fillna(test['hist_monthday_cogs_mean_recent'])

# ============================================
# 2. RESIDUAL
# ============================================
train['resid_rev'] = train['Revenue'] - train['base_rev']
train['resid_cogs'] = train['COGS'] - train['base_cogs']

# ============================================
# 3. FEATURES for residual model
# ============================================
feat_cols = [
    'day_cos','day_sin',
    'hist_yoy_revenue_growth','hist_yoy_cogs_growth',
    'expected_sessions','traffic_uncertainty',
    'month','is_tet_period','days_to_tet',
    'promo_seasonal_prob','promo_monthly_prob',
    'hist_monthday_revenue_mean_recent',
    'hist_monthday_cogs_mean_recent',
]

# Fill NaN
for c in feat_cols:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())

X = train[feat_cols]
y_rev = train['resid_rev']
y_cogs = train['resid_cogs']

# ============================================
# 4. TRAIN/VAL SPLIT
# ============================================
val_days = 180
split_date = train['Date'].max() - pd.Timedelta(days=val_days)
train_mask = train['Date'] <= split_date
val_mask = train['Date'] > split_date

X_train = X.loc[train_mask]
y_train_rev = y_rev.loc[train_mask]
y_train_cogs = y_cogs.loc[train_mask]
X_val = X.loc[val_mask]
y_val_rev = train.loc[val_mask, 'Revenue']
y_val_cogs = train.loc[val_mask, 'COGS']

# ============================================
# 5. ELASTIC NET GRID SEARCH
# ============================================
print("\nGrid search ElasticNet...")

best_mae_rev = np.inf
best_en_rev = None
best_mae_cogs = np.inf
best_en_cogs = None

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]:
    for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        en_rev = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        en_cogs = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        en_rev.fit(X_train, y_train_rev)
        en_cogs.fit(X_train, y_train_cogs)
        
        pred_rev = train.loc[val_mask, 'base_rev'] + en_rev.predict(X_val)
        pred_cogs = train.loc[val_mask, 'base_cogs'] + en_cogs.predict(X_val)
        
        mae_rev = np.mean(np.abs(y_val_rev - pred_rev))
        mae_cogs = np.mean(np.abs(y_val_cogs - pred_cogs))
        
        if mae_rev < best_mae_rev:
            best_mae_rev = mae_rev
            best_en_rev = en_rev
        if mae_cogs < best_mae_cogs:
            best_mae_cogs = mae_cogs
            best_en_cogs = en_cogs

print(f"Best EN Rev: alpha={best_en_rev.alpha}, l1={best_en_rev.l1_ratio}")
print(f"Best EN COGS: alpha={best_en_cogs.alpha}, l1={best_en_cogs.l1_ratio}")

# ============================================
# 6. VALIDATION METRICS
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
print(f"MAE  Revenue: {np.mean(np.abs(y_val_rev - val_pred_rev)):,.0f}")
print(f"MAE  COGS:    {np.mean(np.abs(y_val_cogs - val_pred_cogs)):,.0f}")
print(f"MAPE Revenue: {mape(y_val_rev, val_pred_rev):.4f}")
print(f"MAPE COGS:    {mape(y_val_cogs, val_pred_cogs):.4f}")
print(f"R2   Revenue: {r2(y_val_rev, val_pred_rev):.4f}")
print(f"R2   COGS:    {r2(y_val_cogs, val_pred_cogs):.4f}")

# Base-only comparison
print(f"\nBase-only MAE Revenue: {np.mean(np.abs(y_val_rev - train.loc[val_mask,'base_rev'])):,.0f}")
print(f"Base-only MAPE Revenue: {mape(y_val_rev, train.loc[val_mask,'base_rev']):.4f}")

# ============================================
# 7. FULL RETRAIN & TEST
# ============================================
en_rev_full = ElasticNet(alpha=best_en_rev.alpha, l1_ratio=best_en_rev.l1_ratio, max_iter=5000, random_state=42)
en_cogs_full = ElasticNet(alpha=best_en_cogs.alpha, l1_ratio=best_en_cogs.l1_ratio, max_iter=5000, random_state=42)
en_rev_full.fit(X, y_rev)
en_cogs_full.fit(X, y_cogs)

test['pred_rev'] = test['base_rev'] + en_rev_full.predict(test[feat_cols])
test['pred_cogs'] = test['base_cogs'] + en_cogs_full.predict(test[feat_cols])

# Constraints
test['pred_rev'] = np.maximum(test['pred_rev'], 0)
test['pred_cogs'] = np.maximum(test['pred_cogs'], 0)
test['pred_cogs'] = np.minimum(test['pred_cogs'], test['pred_rev'] * 0.995)

# ============================================
# 8. SAVE
# ============================================
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['pred_rev'], 2),
    'COGS': np.round(test['pred_cogs'], 2),
}).to_csv(out_dir / 'residual_elasticnet_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'residual_elasticnet_submission.csv'}")

# Feature importance (coefficients)
coef_rev = pd.DataFrame({'Feature': feat_cols, 'Coef': en_rev_full.coef_}).sort_values('Coef', key=abs, ascending=False)
print("\nTop ElasticNet coefficients (Revenue residual):")
print(coef_rev.head(10).to_string(index=False))

print("\nDONE")
