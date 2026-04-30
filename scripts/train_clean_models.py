import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING ML MODELS WITH CLEANED FEATURES")
print("="*60)

# Load data
TRAIN_FEATURES = Path(r'D:\Datathon-2026\output\train_features_clean.csv')
TEST_FEATURES = Path(r'D:\Datathon-2026\output\test_features_clean.csv')
TRAIN_TARGET = Path(r'D:\Datathon-2026\output\train_target.csv')

train_feat = pd.read_csv(TRAIN_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
test_feat = pd.read_csv(TEST_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
train_target = pd.read_csv(TRAIN_TARGET, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

if "net_revenue" in train_target.columns and "Revenue" not in train_target.columns:
    train_target = train_target.rename(columns={"net_revenue": "Revenue"})

train = train_feat.merge(train_target, on="Date", how="left")
X = train.drop(columns=["Date", "Revenue", "COGS"]).copy()
y = train[["Revenue", "COGS"]].copy()

val_days = 180
split_date = train["Date"].max() - pd.Timedelta(days=val_days)
train_mask = train["Date"] <= split_date
val_mask = train["Date"] > split_date

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()
X_val = X.loc[val_mask].copy()
y_val = y.loc[val_mask].copy()

median_values = X_train.median(numeric_only=True)
X_train = X_train.fillna(median_values).fillna(0.0)
X_val = X_val.fillna(median_values).fillna(0.0)
X_full = X.fillna(median_values).fillna(0.0)
X_test = test_feat.drop(columns=["Date"]).copy()
X_test = X_test.fillna(median_values).fillna(0.0)

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

print(f"\nFeatures: {X.shape[1]}")
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

results = []

# ============================================
# 1. RANDOM FOREST
# ============================================
print("\n" + "="*60)
print("1. RANDOM FOREST")
print("="*60)

from sklearn.ensemble import RandomForestRegressor

rf_grid = {
    "n_estimators": [200, 400],
    "max_depth": [12, 16],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

best_params = None
best_score = np.inf
best_model = None
best_val_pred = None

for params in ParameterGrid(rf_grid):
    model = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    pred = pd.DataFrame(pred, columns=["Revenue", "COGS"])
    score = (mape(y_val["Revenue"], pred["Revenue"]) + mape(y_val["COGS"], pred["COGS"])) / 2
    if score < best_score:
        best_score = score
        best_params = params
        best_model = model
        best_val_pred = pred

print(f"Best params: {best_params}")
print(f"MAPE Revenue: {mape(y_val['Revenue'], best_val_pred['Revenue']):.4f}")
print(f"MAPE COGS   : {mape(y_val['COGS'], best_val_pred['COGS']):.4f}")
print(f"R2 Revenue  : {r2(y_val['Revenue'], best_val_pred['Revenue']):.4f}")
print(f"R2 COGS     : {r2(y_val['COGS'], best_val_pred['COGS']):.4f}")

results.append({
    "target": "Revenue", "model": "RandomForest",
    "MAPE": mape(y_val['Revenue'], best_val_pred['Revenue']),
    "SMAPE": smape(y_val['Revenue'], best_val_pred['Revenue']),
    "R2": r2(y_val['Revenue'], best_val_pred['Revenue']),
    "SSE": sse(y_val['Revenue'], best_val_pred['Revenue']),
})
results.append({
    "target": "COGS", "model": "RandomForest",
    "MAPE": mape(y_val['COGS'], best_val_pred['COGS']),
    "SMAPE": smape(y_val['COGS'], best_val_pred['COGS']),
    "R2": r2(y_val['COGS'], best_val_pred['COGS']),
    "SSE": sse(y_val['COGS'], best_val_pred['COGS']),
})

rf_val_pred = best_val_pred.copy()
rf_full = RandomForestRegressor(n_jobs=-1, random_state=42, **best_params)
rf_full.fit(X_full, y)
rf_test_pred = pd.DataFrame(rf_full.predict(X_test), columns=["Revenue", "COGS"])

# ============================================
# 2. LIGHTGBM
# ============================================
print("\n" + "="*60)
print("2. LIGHTGBM")
print("="*60)

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed, skipping...")
    lgb_val_pred = None
    lgb_test_pred = None
else:
    lgb_base_params = {"objective": "regression", "metric": "mape", "verbosity": -1}
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
            score = mape(y_va, pred)
            if score < best_s:
                best_s = score
                best_p = params
                best_m = model
        return best_m, best_p, best_s

    lgb_rev, best_p_rev, _ = lgb_grid_search(X_train, y_train["Revenue"], X_val, y_val["Revenue"], lgb_grid)
    lgb_cogs, best_p_cogs, _ = lgb_grid_search(X_train, y_train["COGS"], X_val, y_val["COGS"], lgb_grid)

    val_pred_rev = lgb_rev.predict(X_val)
    val_pred_cogs = lgb_cogs.predict(X_val)

    print(f"Best Rev params: {best_p_rev}")
    print(f"Best Cogs params: {best_p_cogs}")
    print(f"MAPE Revenue: {mape(y_val['Revenue'], val_pred_rev):.4f}")
    print(f"MAPE COGS   : {mape(y_val['COGS'], val_pred_cogs):.4f}")
    print(f"R2 Revenue  : {r2(y_val['Revenue'], val_pred_rev):.4f}")
    print(f"R2 COGS     : {r2(y_val['COGS'], val_pred_cogs):.4f}")

    lgb_val_pred = pd.DataFrame({"Revenue": val_pred_rev, "COGS": val_pred_cogs})
    results.append({"target": "Revenue", "model": "LightGBM",
        "MAPE": mape(y_val['Revenue'], val_pred_rev), "SMAPE": smape(y_val['Revenue'], val_pred_rev),
        "R2": r2(y_val['Revenue'], val_pred_rev), "SSE": sse(y_val['Revenue'], val_pred_rev)})
    results.append({"target": "COGS", "model": "LightGBM",
        "MAPE": mape(y_val['COGS'], val_pred_cogs), "SMAPE": smape(y_val['COGS'], val_pred_cogs),
        "R2": r2(y_val['COGS'], val_pred_cogs), "SSE": sse(y_val['COGS'], val_pred_cogs)})

    # Full retrain
    best_iter_rev = getattr(lgb_rev, "best_iteration", None) or 2000
    best_iter_cogs = getattr(lgb_cogs, "best_iteration", None) or 1000
    lgb_rev_full = lgb.train({**lgb_base_params, **best_p_rev},
                              lgb.Dataset(X_full, label=y["Revenue"], feature_name=list(X_full.columns), free_raw_data=False),
                              num_boost_round=best_iter_rev)
    lgb_cogs_full = lgb.train({**lgb_base_params, **best_p_cogs},
                               lgb.Dataset(X_full, label=y["COGS"], feature_name=list(X_full.columns), free_raw_data=False),
                               num_boost_round=best_iter_cogs)
    lgb_test_pred = pd.DataFrame({
        "Revenue": lgb_rev_full.predict(X_test),
        "COGS": lgb_cogs_full.predict(X_test),
    })

# ============================================
# 3. XGBOOST
# ============================================
print("\n" + "="*60)
print("3. XGBOOST")
print("="*60)

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed, skipping...")
    xgb_val_pred = None
    xgb_test_pred = None
else:
    xgb_base_params = {"objective": "reg:squarederror", "eval_metric": "mape", "seed": 42, "nthread": -1}
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
            score = mape(y_va, preds)
            if score < best_s:
                best_s = score
                best_p = params
                best_m = model
        return best_m, best_p, best_s

    xgb_rev, best_p_rev, _ = xgb_grid_search(X_train, y_train["Revenue"], X_val, y_val["Revenue"], xgb_grid)
    xgb_cogs, best_p_cogs, _ = xgb_grid_search(X_train, y_train["COGS"], X_val, y_val["COGS"], xgb_grid)

    def predict_best(model, X):
        dtest = xgb.DMatrix(X)
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            return model.predict(dtest)
        return model.predict(dtest, iteration_range=(0, best_iter + 1))

    val_pred_rev = predict_best(xgb_rev, X_val)
    val_pred_cogs = predict_best(xgb_cogs, X_val)

    print(f"MAPE Revenue: {mape(y_val['Revenue'], val_pred_rev):.4f}")
    print(f"MAPE COGS   : {mape(y_val['COGS'], val_pred_cogs):.4f}")
    print(f"R2 Revenue  : {r2(y_val['Revenue'], val_pred_rev):.4f}")
    print(f"R2 COGS     : {r2(y_val['COGS'], val_pred_cogs):.4f}")

    xgb_val_pred = pd.DataFrame({"Revenue": val_pred_rev, "COGS": val_pred_cogs})
    results.append({"target": "Revenue", "model": "XGBoost",
        "MAPE": mape(y_val['Revenue'], val_pred_rev), "SMAPE": smape(y_val['Revenue'], val_pred_rev),
        "R2": r2(y_val['Revenue'], val_pred_rev), "SSE": sse(y_val['Revenue'], val_pred_rev)})
    results.append({"target": "COGS", "model": "XGBoost",
        "MAPE": mape(y_val['COGS'], val_pred_cogs), "SMAPE": smape(y_val['COGS'], val_pred_cogs),
        "R2": r2(y_val['COGS'], val_pred_cogs), "SSE": sse(y_val['COGS'], val_pred_cogs)})

    xgb_rev_full = xgb.train({**xgb_base_params, **best_p_rev},
                              xgb.DMatrix(X_full, label=y["Revenue"]),
                              num_boost_round=getattr(xgb_rev, "best_iteration", 800),
                              verbose_eval=False)
    xgb_cogs_full = xgb.train({**xgb_base_params, **best_p_cogs},
                               xgb.DMatrix(X_full, label=y["COGS"]),
                               num_boost_round=getattr(xgb_cogs, "best_iteration", 800),
                               verbose_eval=False)
    xgb_test_pred = pd.DataFrame({
        "Revenue": predict_best(xgb_rev_full, X_test),
        "COGS": predict_best(xgb_cogs_full, X_test),
    })

# ============================================
# 4. EXTRATREES
# ============================================
print("\n" + "="*60)
print("4. EXTRATREES")
print("="*60)

from sklearn.ensemble import ExtraTreesRegressor

et_grid = {
    "n_estimators": [300, 500],
    "max_depth": [16, 20],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

best_params = None
best_score = np.inf
best_model = None
best_val_pred = None

for params in ParameterGrid(et_grid):
    model = ExtraTreesRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    pred = pd.DataFrame(pred, columns=["Revenue", "COGS"])
    score = (mape(y_val["Revenue"], pred["Revenue"]) + mape(y_val["COGS"], pred["COGS"])) / 2
    if score < best_score:
        best_score = score
        best_params = params
        best_model = model
        best_val_pred = pred

print(f"Best params: {best_params}")
print(f"MAPE Revenue: {mape(y_val['Revenue'], best_val_pred['Revenue']):.4f}")
print(f"MAPE COGS   : {mape(y_val['COGS'], best_val_pred['COGS']):.4f}")
print(f"R2 Revenue  : {r2(y_val['Revenue'], best_val_pred['Revenue']):.4f}")
print(f"R2 COGS     : {r2(y_val['COGS'], best_val_pred['COGS']):.4f}")

et_val_pred = best_val_pred.copy()
results.append({
    "target": "Revenue", "model": "ExtraTrees",
    "MAPE": mape(y_val['Revenue'], best_val_pred['Revenue']),
    "SMAPE": smape(y_val['Revenue'], best_val_pred['Revenue']),
    "R2": r2(y_val['Revenue'], best_val_pred['Revenue']),
    "SSE": sse(y_val['Revenue'], best_val_pred['Revenue']),
})
results.append({
    "target": "COGS", "model": "ExtraTrees",
    "MAPE": mape(y_val['COGS'], best_val_pred['COGS']),
    "SMAPE": smape(y_val['COGS'], best_val_pred['COGS']),
    "R2": r2(y_val['COGS'], best_val_pred['COGS']),
    "SSE": sse(y_val['COGS'], best_val_pred['COGS']),
})

et_full = ExtraTreesRegressor(n_jobs=-1, random_state=42, **best_params)
et_full.fit(X_full, y)
et_test_pred = pd.DataFrame(et_full.predict(X_test), columns=["Revenue", "COGS"])

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*60)
print("SUMMARY: CLEANED FEATURES (22 features)")
print("="*60)

summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values(["target", "MAPE"]).reset_index(drop=True)
print(summary_df.to_string(index=False))

# ============================================
# SAVE SUBMISSIONS
# ============================================
out_dir = Path(r'D:\Datathon-2026\data_cleaned\forecast')
out_dir.mkdir(parents=True, exist_ok=True)

for name, pred_df in [("rf_clean", rf_test_pred), ("lgb_clean", lgb_test_pred),
                       ("xgb_clean", xgb_test_pred), ("et_clean", et_test_pred)]:
    if pred_df is not None:
        sub = pd.DataFrame({
            "Date": test_feat["Date"].dt.strftime("%Y-%m-%d"),
            "Revenue": np.maximum(pred_df["Revenue"], 0.0).round(2),
            "COGS": np.maximum(pred_df["COGS"], 0.0).round(2),
        })
        sub["COGS"] = np.minimum(sub["COGS"], sub["Revenue"] * 0.995)
        sub.to_csv(out_dir / f"{name}_submission.csv", index=False)
        print(f"\nSaved: {out_dir / f'{name}_submission.csv'}")

print("\n" + "="*60)
print("DONE")
print("="*60)
