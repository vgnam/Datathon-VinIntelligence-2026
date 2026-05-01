import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

FEAT_DIR = Path('../output')
train = pd.read_csv(FEAT_DIR/'train_features.csv', parse_dates=['Date']).sort_values('Date')
target = pd.read_csv(FEAT_DIR/'train_target.csv', parse_dates=['Date']).sort_values('Date')

sales = pd.read_csv('analytical/sales.csv', parse_dates=['Date'])

month_stats = sales.groupby(sales['Date'].dt.month).agg(mrev=('Revenue','mean'), mcogs=('COGS','mean')).reset_index()
month_stats.columns = ['month','hist_month_revenue_mean','hist_month_cogs_mean']
weekday_stats = sales.groupby(sales['Date'].dt.weekday).agg(wrev=('Revenue','mean'), wcogs=('COGS','mean')).reset_index()
weekday_stats.columns = ['weekday','hist_weekday_revenue_mean','hist_weekday_cogs_mean']

train = train.merge(month_stats, on='month', how='left').merge(weekday_stats, on='weekday', how='left')

X = train.drop(columns=['Date'])
y = target[['Revenue','COGS']]
y_log = y.copy()
y_log['Revenue'] = np.log1p(y_log['Revenue'])
y_log['COGS'] = np.log1p(y_log['COGS'])

med = X.median(numeric_only=True)
X = X.fillna(med).fillna(0.0)

def mape(a,p):
    a=np.array(a,dtype=float); p=np.array(p,dtype=float)
    m=np.isfinite(a)&np.isfinite(p)
    a=a[m]; p=p[m]
    return float(np.mean(np.abs(a-p)/(np.abs(a)+1e-9)))

def r2(a,p):
    a=np.array(a,dtype=float); p=np.array(p,dtype=float)
    m=np.isfinite(a)&np.isfinite(p)
    a=a[m]; p=p[m]
    ss_res=np.sum((a-p)**2); ss_tot=np.sum((a-np.mean(a))**2)
    return 1.0-ss_res/ss_tot if ss_tot>1e-12 else (1.0 if ss_res<1e-12 else 0.0)

def safe_expm1(arr):
    arr=np.clip(np.array(arr,dtype=float), -20, 20)
    return np.expm1(arr)

tscv = TimeSeriesSplit(n_splits=5)
oof_rev={'rf':np.zeros(len(X)),'lgb':np.zeros(len(X)),'xgb':np.zeros(len(X))}
oof_cogs={'rf':np.zeros(len(X)),'lgb':np.zeros(len(X)),'xgb':np.zeros(len(X))}

for fold,(tr_idx,va_idx) in enumerate(tscv.split(X),1):
    X_tr,X_va = X.iloc[tr_idx], X.iloc[va_idx]
    yl_tr,yl_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_tr, yl_tr)
    p = pd.DataFrame(rf.predict(X_va), columns=['Revenue','COGS'])
    oof_rev['rf'][va_idx] = np.maximum(safe_expm1(p['Revenue']),0)
    oof_cogs['rf'][va_idx] = np.maximum(safe_expm1(p['COGS']),0)
    
    dtrain = lgb.Dataset(X_tr, label=yl_tr['Revenue'])
    dval = lgb.Dataset(X_va, label=yl_va['Revenue'])
    lgbm = lgb.train({'objective':'regression','verbosity':-1,'learning_rate':0.03,'num_leaves':20,'max_depth':6,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5}, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(50,verbose=False)])
    oof_rev['lgb'][va_idx] = np.maximum(safe_expm1(lgbm.predict(X_va)),0)
    
    dtrain = lgb.Dataset(X_tr, label=yl_tr['COGS'])
    dval = lgb.Dataset(X_va, label=yl_va['COGS'])
    lgbm = lgb.train({'objective':'regression','verbosity':-1,'learning_rate':0.03,'num_leaves':20,'max_depth':6,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5}, dtrain, num_boost_round=2000, valid_sets=[dval], callbacks=[lgb.early_stopping(50,verbose=False)])
    oof_cogs['lgb'][va_idx] = np.maximum(safe_expm1(lgbm.predict(X_va)),0)
    
    dtrain = xgb.DMatrix(X_tr, label=yl_tr['Revenue'])
    dval = xgb.DMatrix(X_va, label=yl_va['Revenue'])
    xgbr = xgb.train({'objective':'reg:squarederror','seed':42,'learning_rate':0.03,'max_depth':4,'subsample':0.8,'colsample_bytree':0.8}, dtrain, num_boost_round=800, evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=False)
    oof_rev['xgb'][va_idx] = np.maximum(safe_expm1(xgbr.predict(dval)),0)
    
    dtrain = xgb.DMatrix(X_tr, label=yl_tr['COGS'])
    dval = xgb.DMatrix(X_va, label=yl_va['COGS'])
    xgbr = xgb.train({'objective':'reg:squarederror','seed':42,'learning_rate':0.03,'max_depth':4,'subsample':0.8,'colsample_bytree':0.8}, dtrain, num_boost_round=800, evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=False)
    oof_cogs['xgb'][va_idx] = np.maximum(safe_expm1(xgbr.predict(dval)),0)

X_stack_rev = np.column_stack([oof_rev['rf'], oof_rev['lgb'], oof_rev['xgb']])
X_stack_cogs = np.column_stack([oof_cogs['rf'], oof_cogs['lgb'], oof_cogs['xgb']])
gbm_rev = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbm_cogs = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbm_rev.fit(X_stack_rev, y['Revenue'])
gbm_cogs.fit(X_stack_cogs, y['COGS'])
srev = gbm_rev.predict(X_stack_rev)
scogs = gbm_cogs.predict(X_stack_cogs)

# ResEN
en_rev = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
en_cogs = ElasticNet(alpha=0.001, l1_ratio=0.8, max_iter=5000)
resid_rev = np.log1p(y['Revenue']) - np.log1p(np.maximum(srev,0))
resid_cogs = np.log1p(y['COGS']) - np.log1p(np.maximum(scogs,0))
en_rev.fit(X, resid_rev)
en_cogs.fit(X, resid_cogs)
final_rev = np.maximum(np.expm1(np.log1p(np.maximum(srev,0)) + en_rev.predict(X)),0)
final_cogs = np.maximum(np.expm1(np.log1p(np.maximum(scogs,0)) + en_cogs.predict(X)),0)
final_cogs = np.minimum(final_cogs, final_rev*0.995)

print('WITH month/weekday means:')
print('  GBM  Rev MAPE ' + str(round(mape(y['Revenue'], srev),4)) + ' | COGS MAPE ' + str(round(mape(y['COGS'], scogs),4)))
print('  GBM+ResEN Rev MAPE ' + str(round(mape(y['Revenue'], final_rev),4)) + ' | COGS MAPE ' + str(round(mape(y['COGS'], final_cogs),4)) + ' | R2 Rev ' + str(round(r2(y['Revenue'], final_rev),4)) + ' COGS ' + str(round(r2(y['COGS'], final_cogs),4)))
