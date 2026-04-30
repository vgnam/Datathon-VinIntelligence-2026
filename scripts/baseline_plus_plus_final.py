import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("BASELINE++ FINAL: RECENT SEASONAL MEAN + YOY GROWTH")
print("="*70)

# Load selected V2
train = pd.read_csv(r'D:\Datathon-2026\output\train_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
test = pd.read_csv(r'D:\Datathon-2026\output\test_features_selected_v2.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

# ============================================
# PREDICTION
# ============================================
# Revenue = hist_monthday_revenue_mean_recent * (1 + hist_yoy_revenue_growth)
# COGS    = hist_monthday_cogs_mean_recent    * (1 + hist_yoy_cogs_growth)
# Fallback to full-period mean if recent missing

train['pred_rev'] = train['hist_monthday_revenue_mean_recent'].fillna(train['hist_monthday_revenue_mean']) * (1 + train['hist_yoy_revenue_growth'].fillna(0))
train['pred_cogs'] = train['hist_monthday_cogs_mean_recent'].fillna(train['hist_monthday_cogs_mean']) * (1 + train['hist_yoy_cogs_growth'].fillna(0))

test['pred_rev'] = test['hist_monthday_revenue_mean_recent'].fillna(test['hist_monthday_revenue_mean']) * (1 + test['hist_yoy_revenue_growth'].fillna(0))
test['pred_cogs'] = test['hist_monthday_cogs_mean_recent'].fillna(test['hist_monthday_cogs_mean']) * (1 + test['hist_yoy_cogs_growth'].fillna(0))

# ============================================
# VALIDATION
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
print(f"MAPE Revenue: {mape(train.loc[val_mask,'Revenue'], train.loc[val_mask,'pred_rev']):.4f}")
print(f"MAPE COGS:    {mape(train.loc[val_mask,'COGS'], train.loc[val_mask,'pred_cogs']):.4f}")
print(f"R2 Revenue:   {r2(train.loc[val_mask,'Revenue'], train.loc[val_mask,'pred_rev']):.4f}")
print(f"R2 COGS:      {r2(train.loc[val_mask,'COGS'], train.loc[val_mask,'pred_cogs']):.4f}")

# ============================================
# TEST SUBMISSION
# ============================================
test['pred_rev'] = np.maximum(test['pred_rev'], 0)
test['pred_cogs'] = np.maximum(test['pred_cogs'], 0)
test['pred_cogs'] = np.minimum(test['pred_cogs'], test['pred_rev'] * 0.995)

out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    'Date': test['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(test['pred_rev'], 2),
    'COGS': np.round(test['pred_cogs'], 2),
}).to_csv(out_dir / 'baseline_plus_plus_final_submission.csv', index=False)
print(f"\nSaved: {out_dir / 'baseline_plus_plus_final_submission.csv'}")
print("DONE")
