import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAIN ML MODELS WITH BALANCED FEATURES + LOG TRANSFORM")
print("="*70)

# ============================================
# 1. LOAD DATA
# ============================================
TRAIN_FEATURES = Path(r'D:\Datathon-2026\output\train_features_balanced.csv')
TEST_FEATURES = Path(r'D:\Datathon-2026\output\test_features_balanced.csv')
TRAIN_TARGET = Path(r'D:\Datathon-2026\output\train_target.csv')

if not TRAIN_FEATURES.exists():
    raise FileNotFoundError(f"Missing {TRAIN_FEATURES}")

train_feat = pd.read_csv(TRAIN_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
test_feat = pd.read_csv(TEST_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
train_target = pd.read_csv(TRAIN_TARGET, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

if "net_revenue" in train_target.columns and "Revenue" not in train_target.columns:
    train_target = train_target.rename(columns={"net_revenue": "Revenue"})

train = train_feat.merge(train_target[["Date", "Revenue", "COGS"]], on="Date", how="left")

print(f"Train features: {train_feat.shape}")
print(f"Test features:  {test_feat.shape}")
print(f"Date range:     {train['Date'].min().date()} to {train['Date'].max().date()}")

# ============================================
# 2. LOG TRANSFORM TARGET
# ============================================
print("\n" + "="*70)
print("2. LOG TRANSFORM TARGET")
print("="*70)

# Add small epsilon to handle zeros
EPS = 1.0

train['Revenue_log'] = np.log1p(train['Revenue'] + EPS)
train['COGS_log'] = np.log1p(train['COGS'] + EPS)

print(f"Revenue range:      {train['Revenue'].min():,.0f} to {train['Revenue'].max():,.0f}")
print(f"Revenue_log range:  {train['Revenue_log'].min():.4f} to {train['Revenue_log'].max():.4f}")
print(f"COGS range:         {train['COGS'].min():,.0f} to {train['COGS'].max():,.0f}")
print(f"COGS_log range:     {train['COGS_log'].min():.4f} to {train['COGS_log'].max():.4f}")

# ============================================
# 3. TRAIN/VAL SPLIT
# ============================================
X = train.drop(columns=["Date", "Revenue", "COGS", "Revenue_log", "COGS_log"]).copy()
y_raw = train[["Revenue", "COGS"]].copy()
y_log = train[["Revenue_log", "COGS_log"]].copy()

val_days = 180
split_date = train["Date"].max() - pd.Timedelta(days=val_days)
train_mask = train["Date"] <= split_date
val_mask = train["Date"] > split_date

X_train = X.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
y_train_log = y_log.loc[train_mask].copy()
y_val_log = y_log.loc[val_mask].copy()
y_val_raw = y_raw.loc[val_mask].copy()

# Fill NA
median_values = X_train.median(numeric_only=True)
X_train = X_train.fillna(median_values).fillna(0.0)
X_val = X_val.fillna(median_values).fillna(0.0)
X_full = X.fillna(median_values).fillna(0.0)
X_test = test_feat.drop(columns=["Date"]).copy()
X_test = X_test.fillna(median_values).fillna(0.0)
# Ensure same column order as training
X_test = X_test[X_train.columns]

print(f"\nTrain: {len(X_train)} days, Val: {len(X_val)} days")
print(f"Features: {X.shape[1]}")

# ============================================
# 4. METRICS
# ============================================
def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs(actual - pred) / (np.abs(actual) + eps)))

def smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-9
    return float(np.mean(2.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + eps)))

def r2(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-9))

def sse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sum((actual - pred) ** 2))

def inverse_transform(log_pred):
    """Convert log predictions back to original scale"""
    return np.expm1(log_pred) - EPS

results = []

# ============================================
# 5. RANDOM FOREST
# ============================================
print("\n" + "="*70)
print("5. RANDOM FOREST (Log Target)")
print("="*70)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import ParameterGrid
except ImportError:
    raise ImportError("scikit-learn required. Install: pip install scikit-learn")

rf_grid = {
    "n_estimators": [200, 400],
    "max_depth": [12, 16],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

best_score = np.inf
best_model = None
best_val_pred_log = None

for params in ParameterGrid(rf_grid):
    model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_val)
    pred_raw = inverse_transform(pred_log)
    score = (mape(y_val_raw["Revenue"], pred_raw[:,0]) + mape(y_val_raw["COGS"], pred_raw[:,1])) / 2
    if score < best_score:
        best_score = score
        best_model = model
        best_val_pred_log = pred_log

rf_val_pred_raw = inverse_transform(best_val_pred_log)
rf_val_pred = pd.DataFrame(rf_val_pred_raw, columns=["Revenue", "COGS"])

print(f"MAPE Revenue: {mape(y_val_raw['Revenue'], rf_val_pred['Revenue']):.4f}")
print(f"MAPE COGS   : {mape(y_val_raw['COGS'], rf_val_pred['COGS']):.4f}")
print(f"R2 Revenue  : {r2(y_val_raw['Revenue'], rf_val_pred['Revenue']):.4f}")
print(f"R2 COGS     : {r2(y_val_raw['COGS'], rf_val_pred['COGS']):.4f}")

results.append({"target": "Revenue", "model": "RF_log",
    "MAPE": mape(y_val_raw['Revenue'], rf_val_pred['Revenue']),
    "SMAPE": smape(y_val_raw['Revenue'], rf_val_pred['Revenue']),
    "R2": r2(y_val_raw['Revenue'], rf_val_pred['Revenue']),
    "SSE": sse(y_val_raw['Revenue'], rf_val_pred['Revenue'])})
results.append({"target": "COGS", "model": "RF_log",
    "MAPE": mape(y_val_raw['COGS'], rf_val_pred['COGS']),
    "SMAPE": smape(y_val_raw['COGS'], rf_val_pred['COGS']),
    "R2": r2(y_val_raw['COGS'], rf_val_pred['COGS']),
    "SSE": sse(y_val_raw['COGS'], rf_val_pred['COGS'])})

# Full retrain
rf_params = {k: v for k, v in best_model.get_params().items() if k not in ("n_jobs", "random_state")}
rf_full = RandomForestRegressor(n_jobs=-1, random_state=42, **rf_params)
rf_full.fit(X_full, y_log)
rf_test_pred_log = rf_full.predict(X_test)
rf_test_pred = pd.DataFrame(inverse_transform(rf_test_pred_log), columns=["Revenue", "COGS"])

# ============================================
# 6. LIGHTGBM
# ============================================
print("\n" + "="*70)
print("6. LIGHTGBM (Log Target)")
print("="*70)

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed, skipping...")
    lgb_val_pred = None
    lgb_test_pred = None
else:
    lgb_base_params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    lgb_grid = {
        "learning_rate": [0.005, 0.05],
        "num_leaves": [31, 63],
        "max_depth": [8, 12],
        "feature_fraction": [0.8, 1.0],
        "bagging_fraction": [0.8],
        "bagging_freq": [5],
    }

    def lgb_grid_search(X_tr, y_tr, X_va, y_va, grid):
        lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns), free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr, free_raw_data=False)
        best_p = None
        best_s = np.inf
        best_m = None
        for params in ParameterGrid(grid):
            p = {**lgb_base_params, **params}
            model = lgb.train(p, lgb_tr, num_boost_round=2000, valid_sets=[lgb_va],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = model.predict(X_va)
            score = np.mean(np.abs(y_va - pred))  # MAE on log scale
            if score < best_s:
                best_s = score
                best_p = params
                best_m = model
        return best_m, best_p, best_s

    lgb_rev, best_p_rev, _ = lgb_grid_search(X_train, y_train_log["Revenue_log"], X_val, y_val_log["Revenue_log"], lgb_grid)
    lgb_cogs, best_p_cogs, _ = lgb_grid_search(X_train, y_train_log["COGS_log"], X_val, y_val_log["COGS_log"], lgb_grid)

    val_pred_rev_log = lgb_rev.predict(X_val)
    val_pred_cogs_log = lgb_cogs.predict(X_val)
    val_pred_rev = inverse_transform(val_pred_rev_log)
    val_pred_cogs = inverse_transform(val_pred_cogs_log)

    print(f"MAPE Revenue: {mape(y_val_raw['Revenue'], val_pred_rev):.4f}")
    print(f"MAPE COGS   : {mape(y_val_raw['COGS'], val_pred_cogs):.4f}")
    print(f"R2 Revenue  : {r2(y_val_raw['Revenue'], val_pred_rev):.4f}")
    print(f"R2 COGS     : {r2(y_val_raw['COGS'], val_pred_cogs):.4f}")

    lgb_val_pred = pd.DataFrame({"Revenue": val_pred_rev, "COGS": val_pred_cogs})
    results.append({"target": "Revenue", "model": "LGB_log",
        "MAPE": mape(y_val_raw['Revenue'], val_pred_rev), "SMAPE": smape(y_val_raw['Revenue'], val_pred_rev),
        "R2": r2(y_val_raw['Revenue'], val_pred_rev), "SSE": sse(y_val_raw['Revenue'], val_pred_rev)})
    results.append({"target": "COGS", "model": "LGB_log",
        "MAPE": mape(y_val_raw['COGS'], val_pred_cogs), "SMAPE": smape(y_val_raw['COGS'], val_pred_cogs),
        "R2": r2(y_val_raw['COGS'], val_pred_cogs), "SSE": sse(y_val_raw['COGS'], val_pred_cogs)})

    # Full retrain
    best_iter_rev = getattr(lgb_rev, "best_iteration", None) or 2000
    best_iter_cogs = getattr(lgb_cogs, "best_iteration", None) or 1000
    lgb_rev_full = lgb.train({**lgb_base_params, **best_p_rev},
                              lgb.Dataset(X_full, label=y_log["Revenue_log"], feature_name=list(X_full.columns), free_raw_data=False),
                              num_boost_round=best_iter_rev)
    lgb_cogs_full = lgb.train({**lgb_base_params, **best_p_cogs},
                               lgb.Dataset(X_full, label=y_log["COGS_log"], feature_name=list(X_full.columns), free_raw_data=False),
                               num_boost_round=best_iter_cogs)
    lgb_test_pred = pd.DataFrame({
        "Revenue": inverse_transform(lgb_rev_full.predict(X_test)),
        "COGS": inverse_transform(lgb_cogs_full.predict(X_test)),
    })

# ============================================
# 7. XGBOOST
# ============================================
print("\n" + "="*70)
print("7. XGBOOST (Log Target)")
print("="*70)

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed, skipping...")
    xgb_val_pred = None
    xgb_test_pred = None
else:
    xgb_base_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 42, "nthread": -1}
    xgb_grid = {"max_depth": [4, 6], "learning_rate": [0.05, 0.1], "subsample": [0.8], "colsample_bytree": [0.8]}

    def xgb_grid_search(X_tr, y_tr, X_va, y_va, grid):
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        best_p = None
        best_s = np.inf
        best_m = None
        for params in ParameterGrid(grid):
            p = {**xgb_base_params, **params}
            model = xgb.train(params=p, dtrain=dtr, num_boost_round=800,
                              evals=[(dtr, "train"), (dva, "val")],
                              early_stopping_rounds=50, verbose_eval=False)
            preds = model.predict(dva)
            score = np.mean(np.abs(y_va - preds))
            if score < best_s:
                best_s = score
                best_p = params
                best_m = model
        return best_m, best_p, best_s

    xgb_rev, best_p_rev, _ = xgb_grid_search(X_train, y_train_log["Revenue_log"], X_val, y_val_log["Revenue_log"], xgb_grid)
    xgb_cogs, best_p_cogs, _ = xgb_grid_search(X_train, y_train_log["COGS_log"], X_val, y_val_log["COGS_log"], xgb_grid)

    def predict_best(model, X):
        dtest = xgb.DMatrix(X)
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            return model.predict(dtest)
        return model.predict(dtest, iteration_range=(0, best_iter + 1))

    val_pred_rev = inverse_transform(predict_best(xgb_rev, X_val))
    val_pred_cogs = inverse_transform(predict_best(xgb_cogs, X_val))

    print(f"MAPE Revenue: {mape(y_val_raw['Revenue'], val_pred_rev):.4f}")
    print(f"MAPE COGS   : {mape(y_val_raw['COGS'], val_pred_cogs):.4f}")
    print(f"R2 Revenue  : {r2(y_val_raw['Revenue'], val_pred_rev):.4f}")
    print(f"R2 COGS     : {r2(y_val_raw['COGS'], val_pred_cogs):.4f}")

    xgb_val_pred = pd.DataFrame({"Revenue": val_pred_rev, "COGS": val_pred_cogs})
    results.append({"target": "Revenue", "model": "XGB_log",
        "MAPE": mape(y_val_raw['Revenue'], val_pred_rev), "SMAPE": smape(y_val_raw['Revenue'], val_pred_rev),
        "R2": r2(y_val_raw['Revenue'], val_pred_rev), "SSE": sse(y_val_raw['Revenue'], val_pred_rev)})
    results.append({"target": "COGS", "model": "XGB_log",
        "MAPE": mape(y_val_raw['COGS'], val_pred_cogs), "SMAPE": smape(y_val_raw['COGS'], val_pred_cogs),
        "R2": r2(y_val_raw['COGS'], val_pred_cogs), "SSE": sse(y_val_raw['COGS'], val_pred_cogs)})

    xgb_rev_full = xgb.train({**xgb_base_params, **best_p_rev},
                              xgb.DMatrix(X_full, label=y_log["Revenue_log"]),
                              num_boost_round=getattr(xgb_rev, "best_iteration", 800),
                              verbose_eval=False)
    xgb_cogs_full = xgb.train({**xgb_base_params, **best_p_cogs},
                               xgb.DMatrix(X_full, label=y_log["COGS_log"]),
                               num_boost_round=getattr(xgb_cogs, "best_iteration", 800),
                               verbose_eval=False)
    xgb_test_pred = pd.DataFrame({
        "Revenue": inverse_transform(predict_best(xgb_rev_full, X_test)),
        "COGS": inverse_transform(predict_best(xgb_cogs_full, X_test)),
    })

# ============================================
# 8. EXTRATREES
# ============================================
print("\n" + "="*70)
print("8. EXTRATREES (Log Target)")
print("="*70)

try:
    from sklearn.ensemble import ExtraTreesRegressor
except ImportError:
    raise ImportError("scikit-learn required")

et_grid = {
    "n_estimators": [300, 500],
    "max_depth": [16, 20],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

best_score = np.inf
best_model = None
best_val_pred_log = None

for params in ParameterGrid(et_grid):
    model = ExtraTreesRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_val)
    pred_raw = inverse_transform(pred_log)
    score = (mape(y_val_raw["Revenue"], pred_raw[:,0]) + mape(y_val_raw["COGS"], pred_raw[:,1])) / 2
    if score < best_score:
        best_score = score
        best_model = model
        best_val_pred_log = pred_log

et_val_pred_raw = inverse_transform(best_val_pred_log)
et_val_pred = pd.DataFrame(et_val_pred_raw, columns=["Revenue", "COGS"])

print(f"MAPE Revenue: {mape(y_val_raw['Revenue'], et_val_pred['Revenue']):.4f}")
print(f"MAPE COGS   : {mape(y_val_raw['COGS'], et_val_pred['COGS']):.4f}")
print(f"R2 Revenue  : {r2(y_val_raw['Revenue'], et_val_pred['Revenue']):.4f}")
print(f"R2 COGS     : {r2(y_val_raw['COGS'], et_val_pred['COGS']):.4f}")

results.append({"target": "Revenue", "model": "ET_log",
    "MAPE": mape(y_val_raw['Revenue'], et_val_pred['Revenue']),
    "SMAPE": smape(y_val_raw['Revenue'], et_val_pred['Revenue']),
    "R2": r2(y_val_raw['Revenue'], et_val_pred['Revenue']),
    "SSE": sse(y_val_raw['Revenue'], et_val_pred['Revenue'])})
results.append({"target": "COGS", "model": "ET_log",
    "MAPE": mape(y_val_raw['COGS'], et_val_pred['COGS']),
    "SMAPE": smape(y_val_raw['COGS'], et_val_pred['COGS']),
    "R2": r2(y_val_raw['COGS'], et_val_pred['COGS']),
    "SSE": sse(y_val_raw['COGS'], et_val_pred['COGS'])})

et_params = {k: v for k, v in best_model.get_params().items() if k not in ("n_jobs", "random_state")}
et_full = ExtraTreesRegressor(n_jobs=-1, random_state=42, **et_params)
et_full.fit(X_full, y_log)
et_test_pred_log = et_full.predict(X_test)
et_test_pred = pd.DataFrame(inverse_transform(et_test_pred_log), columns=["Revenue", "COGS"])

# ============================================
# 9. ENSEMBLE (Inverse MAPE weights)
# ============================================
print("\n" + "="*70)
print("9. ENSEMBLE (RF + XGB + LGB + ET)")
print("="*70)

def mape_np(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs(actual - pred) / (np.abs(actual) + eps)))

def inv_weights(mape_dict, floor=1e-4):
    inv = {k: 1.0 / max(v, floor) for k, v in mape_dict.items()}
    total = sum(inv.values())
    return {k: inv[k] / total for k in inv}

val_pred_map = {
    "rf": rf_val_pred,
    "et": et_val_pred,
    "lgb": lgb_val_pred,
}
val_pred_map = {k: v for k, v in val_pred_map.items() if v is not None}

mape_rev = {k: mape_np(y_val_raw["Revenue"], v["Revenue"]) for k, v in val_pred_map.items()}
mape_cogs = {k: mape_np(y_val_raw["COGS"], v["COGS"]) for k, v in val_pred_map.items()}

weights_rev = inv_weights(mape_rev)
weights_cogs = inv_weights(mape_cogs)

print("\nModel weights (Revenue):", {k: f"{v:.3f}" for k, v in weights_rev.items()})
print("Model weights (COGS):", {k: f"{v:.3f}" for k, v in weights_cogs.items()})

def weighted_sum(pred_map, weights, col):
    out = np.zeros(len(next(iter(pred_map.values()))), dtype=float)
    for name, df in pred_map.items():
        out += weights[name] * df[col].to_numpy(dtype=float)
    return out

ens_val_rev = weighted_sum(val_pred_map, weights_rev, "Revenue")
ens_val_cogs = weighted_sum(val_pred_map, weights_cogs, "COGS")
ens_val_cogs = np.minimum(ens_val_cogs, ens_val_rev * 0.995)

print(f"\nEnsemble MAPE Revenue: {mape_np(y_val_raw['Revenue'], ens_val_rev):.4f}")
print(f"Ensemble MAPE COGS   : {mape_np(y_val_raw['COGS'], ens_val_cogs):.4f}")
print(f"Ensemble R2 Revenue  : {r2(y_val_raw['Revenue'], ens_val_rev):.4f}")
print(f"Ensemble R2 COGS     : {r2(y_val_raw['COGS'], ens_val_cogs):.4f}")

results.append({"target": "Revenue", "model": "Ensemble_log",
    "MAPE": mape_np(y_val_raw['Revenue'], ens_val_rev),
    "SMAPE": smape(y_val_raw['Revenue'], ens_val_rev),
    "R2": r2(y_val_raw['Revenue'], ens_val_rev),
    "SSE": sse(y_val_raw['Revenue'], ens_val_rev)})
results.append({"target": "COGS", "model": "Ensemble_log",
    "MAPE": mape_np(y_val_raw['COGS'], ens_val_cogs),
    "SMAPE": smape(y_val_raw['COGS'], ens_val_cogs),
    "R2": r2(y_val_raw['COGS'], ens_val_cogs),
    "SSE": sse(y_val_raw['COGS'], ens_val_cogs)})

# ============================================
# 10. SUMMARY TABLE
# ============================================
print("\n" + "="*70)
print("10. SUMMARY: LOG TRANSFORM + BALANCED FEATURES")
print("="*70)

summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(["target", "R2"], ascending=[True, False]).reset_index(drop=True)
print(summary_df.to_string(index=False))

# ============================================
# 11. SAVE SUBMISSIONS
# ============================================
print("\n" + "="*70)
print("11. SAVE SUBMISSIONS")
print("="*70)

out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)

for name, pred_df in [("rf_log", rf_test_pred), ("lgb_log", lgb_test_pred),
                       ("xgb_log", xgb_test_pred), ("et_log", et_test_pred)]:
    if pred_df is not None:
        sub = pd.DataFrame({
            "Date": test_feat["Date"].dt.strftime("%Y-%m-%d"),
            "Revenue": np.maximum(pred_df["Revenue"], 0.0).round(2),
            "COGS": np.maximum(pred_df["COGS"], 0.0).round(2),
        })
        sub["COGS"] = np.minimum(sub["COGS"], sub["Revenue"] * 0.995)
        sub.to_csv(out_dir / f"{name}_submission.csv", index=False)
        print(f"Saved: {out_dir / f'{name}_submission.csv'}")

# Ensemble submission
test_pred_map = {
    "rf": rf_test_pred,
    "et": et_test_pred,
    "lgb": lgb_test_pred,
    "xgb": xgb_test_pred,
}
ens_test_rev = weighted_sum(test_pred_map, weights_rev, "Revenue")
ens_test_cogs = weighted_sum(test_pred_map, weights_cogs, "COGS")
ens_test_rev = np.maximum(ens_test_rev, 0.0)
ens_test_cogs = np.maximum(ens_test_cogs, 0.0)
ens_test_cogs = np.minimum(ens_test_cogs, ens_test_rev * 0.995)

ensemble_sub = pd.DataFrame({
    "Date": test_feat["Date"].dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(ens_test_rev, 2),
    "COGS": np.round(ens_test_cogs, 2),
})
ensemble_sub.to_csv(out_dir / "ensemble_log_submission.csv", index=False)
print(f"Saved: {out_dir / 'ensemble_log_submission.csv'}")

print("\n" + "="*70)
print("DONE")
print("="*70)
