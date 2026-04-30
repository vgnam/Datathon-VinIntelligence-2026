import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("BASELINE++ GROUP-MEAN: BASE + MEAN RESIDUAL BY TET/PROMO/DOW")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# ============================================
# 1. BASE = Recent seasonal mean
# ============================================
train['base_rev'] = train['hist_monthday_revenue_mean_recent'].fillna(train['hist_monthday_revenue_mean'])
train['base_cogs'] = train['hist_monthday_cogs_mean_recent'].fillna(train['hist_monthday_cogs_mean'])
test['base_rev'] = test['hist_monthday_revenue_mean_recent'].fillna(test['hist_monthday_revenue_mean'])
test['base_cogs'] = test['hist_monthday_cogs_mean_recent'].fillna(test['hist_monthday_cogs_mean'])

# ============================================
# 2. RESIDUAL
# ============================================
train['resid_rev'] = train['Revenue'] - train['base_rev']
train['resid_cogs'] = train['COGS'] - train['base_cogs']

# ============================================
# 3. GROUP-MEAN RESIDUAL (by is_tet_period + promo_flag + weekend)
# ============================================
train['promo_flag'] = (train['promo_seasonal_prob'] > train['promo_seasonal_prob'].median()).astype(int)
train['weekend_flag'] = (train['day_sin'].abs() > 0.5).astype(int)  # proxy for weekend

# Group mean residual
resid_group_rev = train.groupby(['is_tet_period','promo_flag','weekend_flag'])['resid_rev'].mean().reset_index()
resid_group_cogs = train.groupby(['is_tet_period','promo_flag','weekend_flag'])['resid_cogs'].mean().reset_index()

print("Group-mean residuals (Revenue):")
print(resid_group_rev.to_string(index=False))

# Apply to train and test
test['promo_flag'] = (test['promo_seasonal_prob'] > train['promo_seasonal_prob'].median()).astype(int)
test['weekend_flag'] = (test['day_sin'].abs() > 0.5).astype(int)

resid_group_rev = resid_group_rev.rename(columns={'resid_rev': 'resid_rev_group'})
resid_group_cogs = resid_group_cogs.rename(columns={'resid_cogs': 'resid_cogs_group'})

train = train.merge(resid_group_rev, on=['is_tet_period','promo_flag','weekend_flag'], how='left')
train = train.merge(resid_group_cogs, on=['is_tet_period','promo_flag','weekend_flag'], how='left')
test = test.merge(resid_group_rev, on=['is_tet_period','promo_flag','weekend_flag'], how='left')
test = test.merge(resid_group_cogs, on=['is_tet_period','promo_flag','weekend_flag'], how='left')

train['final_rev'] = train['base_rev'] + train['resid_rev_group'].fillna(0)
train['final_cogs'] = train['base_cogs'] + train['resid_cogs_group'].fillna(0)
test['final_rev'] = test['base_rev'] + test['resid_rev_group'].fillna(0)
test['final_cogs'] = test['base_cogs'] + test['resid_cogs_group'].fillna(0)

# ============================================
# 4. VALIDATION
# ============================================
val_days = 180
split_date = train['Date'].max() - pd.Timedelta(days=val_days)
val_mask = train['Date'] > split_date

def mape(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    return float(np.mean(np.abs(a-p)/(np.abs(a)+eps)))

def r2(a,p):
    a=np.asarray(a,dtype=float); p=np.asarray(p,dtype=float); eps=1e-9
    ss_res=np.sum((a-p)**2); ss_tot=np.sum((a-np.mean(a))**2)
    return float(1-ss_res/(ss_tot+eps))

print(f"\nValidation ({val_mask.sum()} days):")
print(f"MAE  Revenue: {np.mean(np.abs(train.loc[val_mask,'Revenue'] - train.loc[val_mask,'final_rev'])):,.0f}")
print(f"MAE  COGS:    {np.mean(np.abs(train.loc[val_mask,'COGS'] - train.loc[val_mask,'final_cogs'])):,.0f}")
print(f"MAPE Revenue: {mape(train.loc[val_mask,'Revenue'], train.loc[val_mask,'final_rev']):.4f}")
print(f"MAPE COGS:    {mape(train.loc[val_mask,'COGS'], train.loc[val_mask,'final_cogs']):.4f}")
print(f"R2   Revenue: {r2(train.loc[val_mask,'Revenue'], train.loc[val_mask,'final_rev']):.4f}")
print(f"R2   COGS:    {r2(train.loc[val_mask,'COGS'], train.loc[val_mask,'final_cogs']):.4f}")

print(f"\nBase-only MAE Revenue: {np.mean(np.abs(train.loc[val_mask,'Revenue'] - train.loc[val_mask,'base_rev'])):,.0f}")

# ============================================
# 5. TEST & SAVE
# ============================================
test['final_rev'] = np.maximum(test['final_rev'], 0)
test['final_cogs'] = np.maximum(test['final_cogs'], 0)
test['final_cogs'] = np.minimum(test['final_cogs'], test['final_rev'] * 0.995)

out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['final_rev'], 2),
    'COGS': np.round(test['final_cogs'], 2),
}).to_csv(out_dir / 'baseline_plus_plus_groupmean_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'baseline_plus_plus_groupmean_submission.csv'}")
print("DONE")
