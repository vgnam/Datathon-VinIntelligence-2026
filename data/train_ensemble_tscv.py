"""
train_ensemble_tscv.py
Train tree-based models (RF, LGB, XGB) with TimeSeriesCV,
then stack by multiple meta-learners (Huber, Ridge, Lasso, ElasticNet, BayesianRidge, GBM).
Optional: add ResEN residual layer on top of stacking.
Export submission CSV + diagnostic plots.

Assumes working directory contains the script or is project-root,
with processed features at ../output/ and raw data at ./analytical/.
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FEAT_DIR = ROOT.parent / "output"
RAW_DIR = ROOT / "analytical"
OUT_DIR = ROOT / "forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FEATURES = FEAT_DIR / "train_features.csv"
TEST_FEATURES = FEAT_DIR / "test_features.csv"
TRAIN_TARGET = FEAT_DIR / "train_target.csv"

TRAIN_FILE = RAW_DIR / "sales.csv"
TEST_FILE = RAW_DIR / "sample_submission.csv"
PROMO_FILE = ROOT / "master" / "promotions.csv"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return np.inf
    actual = actual[mask]
    pred = pred[mask]
    eps = 1e-9
    return float(np.mean(np.abs(actual - pred) / (np.abs(actual) + eps)))


def r2_score_fn(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return 0.0
    actual = actual[mask]
    pred = pred[mask]
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def safe_expm1(arr, clip_max=20.0):
    """Clip before expm1 to avoid overflow."""
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, -clip_max, clip_max)
    return np.expm1(arr)


# ---------------------------------------------------------------------------
# 1. Load processed features
# ---------------------------------------------------------------------------
def load_processed_features():
    train_feat = pd.read_csv(TRAIN_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test_feat = pd.read_csv(TEST_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    train_target = pd.read_csv(TRAIN_TARGET, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    lower_cols = {c.lower(): c for c in train_target.columns}
    renames = {}
    if "Revenue" not in train_target.columns and "revenue" in lower_cols:
        renames[lower_cols["revenue"]] = "Revenue"
    if "COGS" not in train_target.columns and "cogs" in lower_cols:
        renames[lower_cols["cogs"]] = "COGS"
    if "Revenue" not in train_target.columns and "net_revenue" in train_target.columns:
        renames["net_revenue"] = "Revenue"
    if renames:
        train_target = train_target.rename(columns=renames)

    overlap = {"Revenue", "COGS"}.intersection(train_feat.columns)
    if overlap:
        train_feat = train_feat.drop(columns=sorted(overlap))

    train = train_feat.merge(train_target, on="Date", how="left")
    if train["Revenue"].isna().any() or train["COGS"].isna().any():
        raise ValueError("Target merge failed.")

    return train, test_feat


# ---------------------------------------------------------------------------
# 3. Time-Series Cross Validation (RF + LGB + XGB only)
# ---------------------------------------------------------------------------
def run_tscv(
    train,
    test_feat,
    n_splits=5,
):
    dates = train["Date"].reset_index(drop=True)
    X = train.drop(columns=["Date", "Revenue", "COGS"]).copy()
    y = train[["Revenue", "COGS"]].reset_index(drop=True)
    y_log = y.copy()
    y_log["Revenue"] = np.log1p(y_log["Revenue"])
    y_log["COGS"] = np.log1p(y_log["COGS"])

    median_vals = X.median(numeric_only=True)
    X = X.fillna(median_vals).fillna(0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(median_vals).fillna(0.0)

    feature_names = list(X.columns)
    n_train = len(train)
    n_test = len(test_feat)

    oof = {
        "rf_rev": np.zeros(n_train), "rf_cogs": np.zeros(n_train),
        "lgb_rev": np.zeros(n_train), "lgb_cogs": np.zeros(n_train),
        "xgb_rev": np.zeros(n_train), "xgb_cogs": np.zeros(n_train),
    }
    test_fold = {
        "rf_rev": [], "rf_cogs": [],
        "lgb_rev": [], "lgb_cogs": [],
        "xgb_rev": [], "xgb_cogs": [],
    }

    fold_records = []

    test_X = test_feat.drop(columns=["Date"], errors="ignore").copy()
    test_X = test_X.reindex(columns=feature_names)
    test_X = test_X.fillna(median_vals).fillna(0.0)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
        yl_tr, yl_va = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        # ---- Random Forest ----
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42,
        )
        rf.fit(X_tr, yl_tr)
        p = rf.predict(X_va)
        pr = pd.DataFrame(p, columns=["Revenue", "COGS"])
        pr["Revenue"] = np.maximum(safe_expm1(pr["Revenue"]), 0.0)
        pr["COGS"] = np.maximum(safe_expm1(pr["COGS"]), 0.0)
        oof["rf_rev"][val_idx] = pr["Revenue"].values
        oof["rf_cogs"][val_idx] = pr["COGS"].values
        pt = rf.predict(test_X)
        pt = pd.DataFrame(pt, columns=["Revenue", "COGS"])
        pt["Revenue"] = np.maximum(safe_expm1(pt["Revenue"]), 0.0)
        pt["COGS"] = np.maximum(safe_expm1(pt["COGS"]), 0.0)
        test_fold["rf_rev"].append(pt["Revenue"].values)
        test_fold["rf_cogs"].append(pt["COGS"].values)
        fold_records.append({"fold": fold, "model": "rf", "target": "Revenue",
                             "mape": mape(y_va["Revenue"], pr["Revenue"]), "r2": r2_score_fn(y_va["Revenue"], pr["Revenue"])})
        fold_records.append({"fold": fold, "model": "rf", "target": "COGS",
                             "mape": mape(y_va["COGS"], pr["COGS"]), "r2": r2_score_fn(y_va["COGS"], pr["COGS"])})
        print(f"  RF   MAPE Rev: {mape(y_va['Revenue'], pr['Revenue']):.4f}  COGS: {mape(y_va['COGS'], pr['COGS']):.4f} | R2 Rev: {r2_score_fn(y_va['Revenue'], pr['Revenue']):.4f}  COGS: {r2_score_fn(y_va['COGS'], pr['COGS']):.4f}")

        # ---- LightGBM Revenue ----
        dtrain = lgb.Dataset(X_tr, label=yl_tr["Revenue"], feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=yl_va["Revenue"], reference=dtrain)
        lgb_rev = lgb.train(
            {"objective": "regression", "metric": "mape", "verbosity": -1,
             "learning_rate": 0.03, "num_leaves": 20, "max_depth": 6,
             "min_child_samples": 20,
             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5},
            dtrain, num_boost_round=2000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pr_rev = np.maximum(safe_expm1(lgb_rev.predict(X_va)), 0.0)
        oof["lgb_rev"][val_idx] = pr_rev
        pt_rev = np.maximum(safe_expm1(lgb_rev.predict(test_X)), 0.0)
        test_fold["lgb_rev"].append(pt_rev)

        # ---- LightGBM COGS ----
        dtrain = lgb.Dataset(X_tr, label=yl_tr["COGS"], feature_name=feature_names)
        dval = lgb.Dataset(X_va, label=yl_va["COGS"], reference=dtrain)
        lgb_cogs = lgb.train(
            {"objective": "regression", "metric": "mape", "verbosity": -1,
             "learning_rate": 0.03, "num_leaves": 20, "max_depth": 6,
             "min_child_samples": 20,
             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5},
            dtrain, num_boost_round=2000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pr_cogs = np.maximum(safe_expm1(lgb_cogs.predict(X_va)), 0.0)
        oof["lgb_cogs"][val_idx] = pr_cogs
        pt_cogs = np.maximum(safe_expm1(lgb_cogs.predict(test_X)), 0.0)
        test_fold["lgb_cogs"].append(pt_cogs)
        fold_records.append({"fold": fold, "model": "lgb", "target": "Revenue",
                             "mape": mape(y_va["Revenue"], pr_rev), "r2": r2_score_fn(y_va["Revenue"], pr_rev)})
        fold_records.append({"fold": fold, "model": "lgb", "target": "COGS",
                             "mape": mape(y_va["COGS"], pr_cogs), "r2": r2_score_fn(y_va["COGS"], pr_cogs)})
        print(f"  LGB  MAPE Rev: {mape(y_va['Revenue'], pr_rev):.4f}  COGS: {mape(y_va['COGS'], pr_cogs):.4f} | R2 Rev: {r2_score_fn(y_va['Revenue'], pr_rev):.4f}  COGS: {r2_score_fn(y_va['COGS'], pr_cogs):.4f}")

        # ---- XGBoost Revenue ----
        dtrain = xgb.DMatrix(X_tr, label=yl_tr["Revenue"])
        dval = xgb.DMatrix(X_va, label=yl_va["Revenue"])
        xgb_rev = xgb.train(
            {"objective": "reg:squarederror", "eval_metric": "mape", "seed": 42,
             "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0,
             "nthread": -1},
            dtrain, num_boost_round=800, evals=[(dval, "val")],
            early_stopping_rounds=50, verbose_eval=False,
        )
        pr_rev = np.maximum(safe_expm1(xgb_rev.predict(dval)), 0.0)
        oof["xgb_rev"][val_idx] = pr_rev
        pt_rev = np.maximum(safe_expm1(xgb_rev.predict(xgb.DMatrix(test_X))), 0.0)
        test_fold["xgb_rev"].append(pt_rev)

        # ---- XGBoost COGS ----
        dtrain = xgb.DMatrix(X_tr, label=yl_tr["COGS"])
        dval = xgb.DMatrix(X_va, label=yl_va["COGS"])
        xgb_cogs = xgb.train(
            {"objective": "reg:squarederror", "eval_metric": "mape", "seed": 42,
             "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0,
             "nthread": -1},
            dtrain, num_boost_round=800, evals=[(dval, "val")],
            early_stopping_rounds=50, verbose_eval=False,
        )
        pr_cogs = np.maximum(safe_expm1(xgb_cogs.predict(dval)), 0.0)
        oof["xgb_cogs"][val_idx] = pr_cogs
        pt_cogs = np.maximum(safe_expm1(xgb_cogs.predict(xgb.DMatrix(test_X))), 0.0)
        test_fold["xgb_cogs"].append(pt_cogs)
        fold_records.append({"fold": fold, "model": "xgb", "target": "Revenue",
                             "mape": mape(y_va["Revenue"], pr_rev), "r2": r2_score_fn(y_va["Revenue"], pr_rev)})
        fold_records.append({"fold": fold, "model": "xgb", "target": "COGS",
                             "mape": mape(y_va["COGS"], pr_cogs), "r2": r2_score_fn(y_va["COGS"], pr_cogs)})
        print(f"  XGB  MAPE Rev: {mape(y_va['Revenue'], pr_rev):.4f}  COGS: {mape(y_va['COGS'], pr_cogs):.4f} | R2 Rev: {r2_score_fn(y_va['Revenue'], pr_rev):.4f}  COGS: {r2_score_fn(y_va['COGS'], pr_cogs):.4f}")

    # Average test predictions across folds
    test_avg = {}
    for k, v in test_fold.items():
        test_avg[k] = np.mean(np.vstack(v), axis=0)

    fold_df = pd.DataFrame(fold_records)
    
    # Store fold boundaries for plotting
    fold_boundaries = []
    tscv_for_boundaries = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, val_idx) in enumerate(tscv_for_boundaries.split(X), 1):
        fold_boundaries.append({
            "fold": fold,
            "train_start": dates.iloc[tr_idx[0]],
            "train_end": dates.iloc[tr_idx[-1]],
            "val_start": dates.iloc[val_idx[0]],
            "val_end": dates.iloc[val_idx[-1]],
        })
    
    return oof, test_avg, y, fold_df, dates, fold_boundaries


# ---------------------------------------------------------------------------
# 4. Stacking with multiple meta-learners
# ---------------------------------------------------------------------------
META_LEARNERS = {
    "huber": HuberRegressor(epsilon=1.35, max_iter=2000),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1.0, max_iter=5000),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "bayesian_ridge": BayesianRidge(),
    "gbm": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}


def build_stack_features(oof, test_avg):
    """Build feature matrices for stacking from 3 tree models."""
    X_stack_rev = np.column_stack([oof["rf_rev"], oof["lgb_rev"], oof["xgb_rev"]])
    X_stack_cogs = np.column_stack([oof["rf_cogs"], oof["lgb_cogs"], oof["xgb_cogs"]])
    X_test_rev = np.column_stack([test_avg["rf_rev"], test_avg["lgb_rev"], test_avg["xgb_rev"]])
    X_test_cogs = np.column_stack([test_avg["rf_cogs"], test_avg["lgb_cogs"], test_avg["xgb_cogs"]])
    return X_stack_rev, X_stack_cogs, X_test_rev, X_test_cogs


def stack_models(oof, test_avg, y_train, test_dates, meta_name="huber"):
    """Fit chosen meta-learner on 4 tree model OOF predictions."""
    meta = META_LEARNERS.get(meta_name)
    if meta is None:
        raise ValueError(f"Unknown meta-learner: {meta_name}. Choose from {list(META_LEARNERS.keys())}")

    X_stack_rev, X_stack_cogs, X_test_rev, X_test_cogs = build_stack_features(oof, test_avg)

    meta_rev = meta.__class__(**meta.get_params())
    meta_cogs = meta.__class__(**meta.get_params())
    meta_rev.fit(X_stack_rev, y_train["Revenue"].to_numpy())
    meta_cogs.fit(X_stack_cogs, y_train["COGS"].to_numpy())
    pred_rev = meta_rev.predict(X_test_rev)
    pred_cogs = meta_cogs.predict(X_test_cogs)

    pred_rev = np.maximum(pred_rev, 0.0)
    pred_cogs = np.maximum(pred_cogs, 0.0)
    pred_cogs = np.minimum(pred_cogs, pred_rev * 0.995)

    stack_rev_train = meta_rev.predict(X_stack_rev)
    stack_cogs_train = meta_cogs.predict(X_stack_cogs)

    print(f"\n=== Stacking 3 models: {meta_name.upper()} ===")
    model_names = ['rf', 'lgb', 'xgb']
    if hasattr(meta_rev, "coef_"):
        print(f"Weights (Revenue): {dict(zip(model_names, np.round(meta_rev.coef_, 4)))}")
        print(f"Intercept (Revenue): {round(meta_rev.intercept_, 2) if hasattr(meta_rev, 'intercept_') else 'N/A'}")
        print(f"Weights (COGS):    {dict(zip(model_names, np.round(meta_cogs.coef_, 4)))}")
        print(f"Intercept (COGS): {round(meta_cogs.intercept_, 2) if hasattr(meta_cogs, 'intercept_') else 'N/A'}")
    elif hasattr(meta_rev, "feature_importances_"):
        print(f"Feature importances (Revenue): {dict(zip(model_names, np.round(meta_rev.feature_importances_, 4)))}")
        print(f"Feature importances (COGS):    {dict(zip(model_names, np.round(meta_cogs.feature_importances_, 4)))}")

    sub = pd.DataFrame({
        "Date": pd.to_datetime(test_dates).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(pred_rev, 2),
        "COGS": np.round(pred_cogs, 2),
    })
    return sub, stack_rev_train, stack_cogs_train


def stack_with_residual(oof, test_avg, y_train, test_dates, X_train_raw, X_test_raw, meta_name="huber"):
    """
    Stack 3 models, then fit ElasticNet on residual of stacking output.
    Returns final prediction = stacking_pred + ResEN_residual.
    """
    # Step 1: Stack 3 models
    meta = META_LEARNERS.get(meta_name)
    X_stack_rev, X_stack_cogs, X_test_rev, X_test_cogs = build_stack_features(oof, test_avg)

    meta_rev = meta.__class__(**meta.get_params())
    meta_cogs = meta.__class__(**meta.get_params())
    meta_rev.fit(X_stack_rev, y_train["Revenue"].to_numpy())
    meta_cogs.fit(X_stack_cogs, y_train["COGS"].to_numpy())

    stack_rev_pred = meta_rev.predict(X_stack_rev)
    stack_cogs_pred = meta_cogs.predict(X_stack_cogs)
    stack_test_rev = meta_rev.predict(X_test_rev)
    stack_test_cogs = meta_cogs.predict(X_test_cogs)

    # Step 2: Residual = actual_log - log1p(stacking_pred)
    resid_rev = np.log1p(y_train["Revenue"].to_numpy()) - np.log1p(np.maximum(stack_rev_pred, 0.0))
    resid_cogs = np.log1p(y_train["COGS"].to_numpy()) - np.log1p(np.maximum(stack_cogs_pred, 0.0))

    # Step 3: Fit ElasticNet on residual
    en_rev = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ])
    en_cogs = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(alpha=0.001, l1_ratio=0.8, max_iter=5000)),
    ])
    en_rev.fit(X_train_raw, resid_rev)
    en_cogs.fit(X_train_raw, resid_cogs)

    # Step 4: Final prediction = expm1(log1p(stacking) + EN_residual)
    resid_rev_test = en_rev.predict(X_test_raw)
    resid_cogs_test = en_cogs.predict(X_test_raw)
    final_rev = np.maximum(safe_expm1(np.log1p(np.maximum(stack_test_rev, 0.0)) + resid_rev_test), 0.0)
    final_cogs = np.maximum(safe_expm1(np.log1p(np.maximum(stack_test_cogs, 0.0)) + resid_cogs_test), 0.0)
    final_cogs = np.minimum(final_cogs, final_rev * 0.995)

    # OOF final predictions
    resid_rev_oof = en_rev.predict(X_train_raw)
    resid_cogs_oof = en_cogs.predict(X_train_raw)
    final_rev_oof = np.maximum(safe_expm1(np.log1p(np.maximum(stack_rev_pred, 0.0)) + resid_rev_oof), 0.0)
    final_cogs_oof = np.maximum(safe_expm1(np.log1p(np.maximum(stack_cogs_pred, 0.0)) + resid_cogs_oof), 0.0)

    print(f"\n=== Stacking 3 models + ResEN residual: {meta_name.upper()} ===")
    print(f"Stacking-only MAPE Rev: {mape(y_train['Revenue'], stack_rev_pred):.4f}  COGS: {mape(y_train['COGS'], stack_cogs_pred):.4f}")
    print(f"Stacking+ResEN MAPE Rev: {mape(y_train['Revenue'], final_rev_oof):.4f}  COGS: {mape(y_train['COGS'], final_cogs_oof):.4f}")

    sub = pd.DataFrame({
        "Date": pd.to_datetime(test_dates).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(final_rev, 2),
        "COGS": np.round(final_cogs, 2),
    })
    return sub, final_rev_oof, final_cogs_oof


def compare_stackers(oof, y_train):
    """Compare all meta-learners on OOF (3 models only) and return best."""
    results = []
    best_mape = np.inf
    best_name = None

    X_stack_rev, X_stack_cogs, _, _ = build_stack_features(oof, oof)

    for name, meta in META_LEARNERS.items():
        meta_rev = meta.__class__(**meta.get_params())
        meta_cogs = meta.__class__(**meta.get_params())
        meta_rev.fit(X_stack_rev, y_train["Revenue"].to_numpy())
        meta_cogs.fit(X_stack_cogs, y_train["COGS"].to_numpy())

        pred_rev = meta_rev.predict(X_stack_rev)
        pred_cogs = meta_cogs.predict(X_stack_cogs)
        pred_rev = np.maximum(pred_rev, 0.0)
        pred_cogs = np.maximum(pred_cogs, 0.0)
        pred_cogs = np.minimum(pred_cogs, pred_rev * 0.995)

        mape_rev = mape(y_train["Revenue"], pred_rev)
        mape_cogs = mape(y_train["COGS"], pred_cogs)
        avg_mape = (mape_rev + mape_cogs) / 2

        results.append({
            "meta_learner": name,
            "mape_revenue": mape_rev,
            "mape_cogs": mape_cogs,
            "avg_mape": avg_mape,
            "r2_revenue": r2_score_fn(y_train["Revenue"], pred_rev),
            "r2_cogs": r2_score_fn(y_train["COGS"], pred_cogs),
        })

        if avg_mape < best_mape:
            best_mape = avg_mape
            best_name = name

    results_df = pd.DataFrame(results).sort_values("avg_mape")
    print("\n=== Meta-Learner Comparison (OOF) ===")
    print(results_df.to_string(index=False))
    print(f"\nBest meta-learner: {best_name} (avg MAPE: {best_mape:.4f})")
    return best_name, results_df


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_results(dates, y, oof, stack_rev_train, stack_cogs_train, fold_df, fold_boundaries, out_dir):
    """
    Generate and save diagnostic plots.
    """
    dates = pd.to_datetime(dates)
    n_splits = len(fold_boundaries)

    # ---------- Plot 1: OOF Time Series with fold boundaries ----------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    models_rev = [("rf_rev", "RF"), ("lgb_rev", "LGB"), ("xgb_rev", "XGB")]
    models_cogs = [("rf_cogs", "RF"), ("lgb_cogs", "LGB"), ("xgb_cogs", "XGB")]

    colors_fold = plt.cm.Set1(np.linspace(0, 1, n_splits))

    axes[0].plot(dates, y["Revenue"], label="Actual", color="black", lw=1.5, zorder=5)
    for key, label in models_rev:
        axes[0].plot(dates, oof[key], label=label, lw=0.8, linestyle="--", alpha=0.6)
    for i, fb in enumerate(fold_boundaries):
        axes[0].axvline(x=fb["val_start"], color=colors_fold[i], linestyle=":", alpha=0.7, lw=1.5)
        axes[0].axvspan(fb["val_start"], fb["val_end"], alpha=0.1, color=colors_fold[i])
    axes[0].set_title("Revenue — OOF Predictions with CV Folds")
    axes[0].set_ylabel("Revenue")
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].plot(dates, y["COGS"], label="Actual", color="black", lw=1.5, zorder=5)
    for key, label in models_cogs:
        axes[1].plot(dates, oof[key], label=label, lw=0.8, linestyle="--", alpha=0.6)
    for i, fb in enumerate(fold_boundaries):
        axes[1].axvline(x=fb["val_start"], color=colors_fold[i], linestyle=":", alpha=0.7, lw=1.5)
        axes[1].axvspan(fb["val_start"], fb["val_end"], alpha=0.1, color=colors_fold[i])
    axes[1].set_title("COGS — OOF Predictions with CV Folds")
    axes[1].set_ylabel("COGS")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "plot_oof_timeseries.png", dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_dir / 'plot_oof_timeseries.png'}")

    # ---------- Plot 2: Scatter Actual vs Stacking OOF ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y["Revenue"], stack_rev_train, alpha=0.4, s=15)
    axes[0].plot([0, y["Revenue"].max()], [0, y["Revenue"].max()], "r--", lw=1)
    axes[0].set_xlabel("Actual Revenue")
    axes[0].set_ylabel("Predicted Revenue")
    axes[0].set_title(f"Stacking OOF Revenue | R2={r2_score_fn(y['Revenue'], stack_rev_train):.3f}")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y["COGS"], stack_cogs_train, alpha=0.4, s=15, color="orange")
    axes[1].plot([0, y["COGS"].max()], [0, y["COGS"].max()], "r--", lw=1)
    axes[1].set_xlabel("Actual COGS")
    axes[1].set_ylabel("Predicted COGS")
    axes[1].set_title(f"Stacking OOF COGS | R2={r2_score_fn(y['COGS'], stack_cogs_train):.3f}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "plot_scatter_stacking.png", dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_dir / 'plot_scatter_stacking.png'}")

    # ---------- Plot 3: Bar chart CV MAPE & R2 ----------
    summary = fold_df.groupby(["model", "target"]).agg({"mape": ["mean", "std"], "r2": ["mean", "std"]}).reset_index()
    summary.columns = ["model", "target", "mape_mean", "mape_std", "r2_mean", "r2_std"]
    summary["model_target"] = summary["model"] + "_" + summary["target"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(len(summary))
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))

    axes[0].bar(x_pos, summary["mape_mean"], yerr=summary["mape_std"], capsize=4, color=colors, alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(summary["model_target"], rotation=45, ha="right")
    axes[0].set_ylabel("MAPE")
    axes[0].set_title("CV MAPE (mean ± std)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x_pos, summary["r2_mean"], yerr=summary["r2_std"], capsize=4, color=colors, alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(summary["model_target"], rotation=45, ha="right")
    axes[1].set_ylabel("R2")
    axes[1].set_title("CV R2 (mean ± std)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "plot_cv_metrics.png", dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_dir / 'plot_cv_metrics.png'}")

    # ---------- Plot 4: Per-fold scatter Revenue & COGS ----------
    for target, tname, y_actual in [("rev", "Revenue", y["Revenue"]), ("cogs", "COGS", y["COGS"])]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        models = [("rf", "RF"), ("lgb", "LGB"), ("xgb", "XGB")]
        
        for i, (model_key, model_label) in enumerate(models):
            ax = axes[i]
            oof_key = f"{model_key}_{target}"
            
            for fold_num, fb in enumerate(fold_boundaries, 1):
                mask = (dates >= fb["val_start"]) & (dates <= fb["val_end"])
                if mask.sum() > 0:
                    fold_actual = y_actual[mask]
                    fold_pred = pd.Series(oof[oof_key])[mask]
                    fold_mape = mape(fold_actual, fold_pred)
                    fold_r2 = r2_score_fn(fold_actual, fold_pred)
                    ax.scatter(fold_actual, fold_pred, alpha=0.5, s=20, 
                              label=f"Fold {fold_num} (MAPE:{fold_mape:.3f}, R2:{fold_r2:.2f})")
            
            max_val = max(y_actual.max(), pd.Series(oof[oof_key]).max())
            ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5)
            ax.set_xlabel(f"Actual {tname}")
            ax.set_ylabel(f"Predicted {tname}")
            ax.set_title(f"{model_label} — {tname}")
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(alpha=0.3)
        
        ax = axes[5]
        stack_pred = stack_rev_train if target == "rev" else stack_cogs_train
        for fold_num, fb in enumerate(fold_boundaries, 1):
            mask = (dates >= fb["val_start"]) & (dates <= fb["val_end"])
            if mask.sum() > 0:
                fold_actual = y_actual[mask]
                fold_pred = pd.Series(stack_pred)[mask]
                fold_mape = mape(fold_actual, fold_pred)
                fold_r2 = r2_score_fn(fold_actual, fold_pred)
                ax.scatter(fold_actual, fold_pred, alpha=0.5, s=20,
                          label=f"Fold {fold_num} (MAPE:{fold_mape:.3f}, R2:{fold_r2:.2f})")
        max_val = max(y_actual.max(), pd.Series(stack_pred).max())
        ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5)
        ax.set_xlabel(f"Actual {tname}")
        ax.set_ylabel(f"Predicted {tname}")
        ax.set_title(f"Stacking — {tname}")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
        
        plt.suptitle(f"Per-fold Actual vs Predicted — {tname}", fontsize=14, y=1.00)
        plt.tight_layout()
        fig.savefig(out_dir / f"plot_fold_scatter_{target}.png", dpi=200)
        plt.close(fig)
        print(f"Saved plot: {out_dir / f'plot_fold_scatter_{target}.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading processed features...")
    train, test_feat = load_processed_features()

    print(f"Train rows: {len(train)} | Test rows: {len(test_feat)}")

    print("\nRunning TimeSeriesCV...")
    oof, test_avg, y, fold_df, dates, fold_boundaries = run_tscv(
        train, test_feat, n_splits=5
    )

    print("\n=== OOF CV Scores ===")
    for target in ["rev", "cogs"]:
        tname = "Revenue" if target == "rev" else "COGS"
        for model in ["rf", "lgb", "xgb"]:
            key = f"{model}_{target}"
            print(f"  {key.upper():12s} MAPE: {mape(y[tname], oof[key]):.4f} | R2: {r2_score_fn(y[tname], oof[key]):.4f}")

    # Prepare raw features for residual model
    feature_cols = [c for c in train.columns if c not in ["Date", "Revenue", "COGS"]]
    X_train_raw = train[feature_cols].copy()
    X_test_raw = test_feat[feature_cols].copy()
    median_vals = X_train_raw.median(numeric_only=True)
    X_train_raw = X_train_raw.fillna(median_vals).fillna(0.0).to_numpy()
    X_test_raw = X_test_raw.fillna(median_vals).fillna(0.0).to_numpy()

    # === Chosen meta-learner: GBM ===
    meta_name = "gbm"
    print(f"\n=== Stacking 3 models with meta-learner: {meta_name.upper()} ===")

    # Stacking 3 models only
    submission3, stack3_rev, stack3_cogs = stack_models(
        oof, test_avg, y, test_feat["Date"], meta_name=meta_name
    )
    out_path3 = OUT_DIR / f"stacking3_{meta_name}_tscv_submission.csv"
    submission3.to_csv(out_path3, index=False)
    print(f"Saved: {out_path3}")
    print(f"Stacking3 OOF — Revenue MAPE: {mape(y['Revenue'], stack3_rev):.4f} | R2: {r2_score_fn(y['Revenue'], stack3_rev):.4f}")
    print(f"Stacking3 OOF — COGS   MAPE: {mape(y['COGS'], stack3_cogs):.4f} | R2: {r2_score_fn(y['COGS'], stack3_cogs):.4f}")

    # Stacking 3 models + ResEN
    sub_res, stack_res_rev, stack_res_cogs = stack_with_residual(
        oof, test_avg, y, test_feat["Date"], X_train_raw, X_test_raw, meta_name=meta_name
    )
    out_path_res = OUT_DIR / f"stacking3_{meta_name}_resen_tscv_submission.csv"
    sub_res.to_csv(out_path_res, index=False)
    print(f"Saved: {out_path_res}")
    print(f"Stacking3+ResEN OOF — Revenue MAPE: {mape(y['Revenue'], stack_res_rev):.4f} | R2: {r2_score_fn(y['Revenue'], stack_res_rev):.4f}")
    print(f"Stacking3+ResEN OOF — COGS   MAPE: {mape(y['COGS'], stack_res_cogs):.4f} | R2: {r2_score_fn(y['COGS'], stack_res_cogs):.4f}")

    # Pick best variant for plotting
    if mape(y["Revenue"], stack_res_rev) + mape(y["COGS"], stack_res_cogs) < \
       mape(y["Revenue"], stack3_rev) + mape(y["COGS"], stack3_cogs):
        best_stack_rev = stack_res_rev
        best_stack_cogs = stack_res_cogs
    else:
        best_stack_rev = stack3_rev
        best_stack_cogs = stack3_cogs

    print("\nGenerating plots...")
    plot_results(dates, y, oof, best_stack_rev, best_stack_cogs, fold_df, fold_boundaries, OUT_DIR)


if __name__ == "__main__":
    main()
