"""
train_ensemble_tscv.py
Train tree-based models + Baseline+ Residual ElasticNet with TimeSeriesCV,
then stack by Huber Regression. Export submission CSV + diagnostic plots.

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
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

TRAIN_FEATURES = FEAT_DIR / "train_features_selected_v1_10.csv"
TEST_FEATURES = FEAT_DIR / "test_features_selected_v1_10.csv"
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
    # replace inf with nan, then drop
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
# 2. Build Baseline+ predictions
# ---------------------------------------------------------------------------
def build_baseline_plus(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()

    promo = pd.DataFrame()
    if PROMO_FILE.exists():
        promo = pd.read_csv(PROMO_FILE, parse_dates=["start_date", "end_date"])
        if "discount_value" not in promo.columns:
            promo["discount_value"] = 0.0
        promo["discount_value"] = promo["discount_value"].fillna(0.0).astype(float)

    train["month"] = train["Date"].dt.month
    train["day"] = train["Date"].dt.day
    train["weekday"] = train["Date"].dt.weekday
    train["year"] = train["Date"].dt.year

    base_year = int(train["year"].max()) - 1
    base_data = train[train["year"].isin([base_year, base_year + 1])].copy()

    base_rev = base_data["Revenue"].mean()
    base_cogs = base_data["COGS"].mean()

    seasonal = (
        base_data.groupby(["month", "day"])
        .agg(rev_norm_season=("Revenue", "mean"), cogs_norm_season=("COGS", "mean"))
        .reset_index()
    )
    seasonal["rev_norm_season"] /= base_rev
    seasonal["cogs_norm_season"] /= base_cogs

    weekday_prof = (
        base_data.groupby("weekday")
        .agg(rev_wday_factor=("Revenue", "mean"), cogs_wday_factor=("COGS", "mean"))
        .reset_index()
    )
    weekday_prof["rev_wday_factor"] /= weekday_prof["rev_wday_factor"].mean()
    weekday_prof["cogs_wday_factor"] /= weekday_prof["cogs_wday_factor"].mean()

    cal = train.merge(seasonal, on=["month", "day"], how="left").merge(weekday_prof, on="weekday", how="left")
    cal["rev_norm"] = cal["rev_norm_season"].fillna(1.0)
    cal["cogs_norm"] = cal["cogs_norm_season"].fillna(1.0)
    cal["rev_wday_factor"] = cal["rev_wday_factor"].fillna(1.0)
    cal["cogs_wday_factor"] = cal["cogs_wday_factor"].fillna(1.0)
    cal["years_ahead"] = cal["year"] - base_year

    val_years = sorted(cal["year"].unique())[-2:]
    cal_val = cal[cal["year"].isin(val_years)].copy()
    raw_growth_rev = float(np.clip(cal_val["Revenue"].mean() / (base_rev + 1e-9), 0.5, 2.0))
    raw_growth_cogs = float(np.clip(cal_val["COGS"].mean() / (base_cogs + 1e-9), 0.5, 2.0))

    split_date = cal["Date"].max() - pd.Timedelta(days=180)
    cal_val2 = cal[cal["Date"] > split_date].copy()

    best_lam_rev, best_lam_cogs = 1.0, 1.0
    best_mape_rev, best_mape_cogs = np.inf, np.inf
    for lam in np.linspace(0.0, 1.0, 11):
        g_rev = 1.0 + lam * (raw_growth_rev - 1.0)
        pred_rev = base_rev * (g_rev ** cal_val2["years_ahead"]) * cal_val2["rev_norm"] * cal_val2["rev_wday_factor"]
        s = mape(cal_val2["Revenue"], pred_rev)
        if s < best_mape_rev:
            best_mape_rev = s
            best_lam_rev = float(lam)

        g_cogs = 1.0 + lam * (raw_growth_cogs - 1.0)
        pred_cogs = base_cogs * (g_cogs ** cal_val2["years_ahead"]) * cal_val2["cogs_norm"] * cal_val2["cogs_wday_factor"]
        s = mape(cal_val2["COGS"], pred_cogs)
        if s < best_mape_cogs:
            best_mape_cogs = s
            best_lam_cogs = float(lam)

    growth_rev = 1.0 + best_lam_rev * (raw_growth_rev - 1.0)
    growth_cogs = 1.0 + best_lam_cogs * (raw_growth_cogs - 1.0)

    cal_base_rev = (base_rev * (growth_rev ** cal["years_ahead"]) * cal["rev_norm"] * cal["rev_wday_factor"]).clip(lower=0).fillna(0)
    cal_base_cogs = (base_cogs * (growth_cogs ** cal["years_ahead"]) * cal["cogs_norm"] * cal["cogs_wday_factor"]).clip(lower=0).fillna(0)

    def build_promo_daily(promo_df):
        if promo_df.empty:
            return pd.DataFrame(columns=["Date", "promo_count", "avg_discount"])
        chunks = []
        for _, r in promo_df.iterrows():
            start, end = r.get("start_date"), r.get("end_date")
            if pd.isna(start) or pd.isna(end) or end < start:
                continue
            days = pd.date_range(start, end, freq="D")
            chunks.append(pd.DataFrame({
                "Date": days,
                "promo_count": 1.0,
                "discount_value": float(r.get("discount_value", 0.0) or 0.0),
            }))
        if not chunks:
            return pd.DataFrame(columns=["Date", "promo_count", "avg_discount"])
        return (
            pd.concat(chunks, ignore_index=True)
            .groupby("Date", as_index=False)
            .agg(promo_count=("promo_count", "sum"), avg_discount=("discount_value", "mean"))
        )

    promo_daily = build_promo_daily(promo)

    TET_DATE = {
        2013: pd.Timestamp("2013-02-10"), 2014: pd.Timestamp("2014-01-31"),
        2015: pd.Timestamp("2015-02-19"), 2016: pd.Timestamp("2016-02-08"),
        2017: pd.Timestamp("2017-01-28"), 2018: pd.Timestamp("2018-02-16"),
        2019: pd.Timestamp("2019-02-05"), 2020: pd.Timestamp("2020-01-25"),
        2021: pd.Timestamp("2021-02-12"), 2022: pd.Timestamp("2022-02-01"),
        2023: pd.Timestamp("2023-01-22"), 2024: pd.Timestamp("2024-02-10"),
    }
    fixed_holidays = {(1, 1), (4, 30), (5, 1), (9, 2), (12, 24), (12, 25), (12, 31)}

    def is_tet_period(ts, pre_days=10, post_days=7):
        tet = TET_DATE.get(ts.year)
        if tet is None:
            return False
        return (tet - pd.Timedelta(days=pre_days)) <= ts <= (tet + pd.Timedelta(days=post_days))

    def add_event_features(df):
        out = df.merge(promo_daily, on="Date", how="left")
        out["promo_count"] = out["promo_count"].fillna(0.0)
        out["avg_discount"] = out["avg_discount"].fillna(0.0)
        out["is_holiday"] = out["Date"].apply(lambda x: (x.month, x.day) in fixed_holidays)
        out["is_tet"] = out["Date"].apply(is_tet_period)
        return out

    cal = add_event_features(cal)

    cal["rev_ratio"] = cal["Revenue"] / cal_base_rev.replace(0, np.nan)
    cal["cogs_ratio"] = cal["COGS"] / cal_base_cogs.replace(0, np.nan)
    global_rev_med = cal["rev_ratio"].dropna().median()
    global_cogs_med = cal["cogs_ratio"].dropna().median()
    if not np.isfinite(global_rev_med) or global_rev_med <= 0:
        global_rev_med = 1.0
    if not np.isfinite(global_cogs_med) or global_cogs_med <= 0:
        global_cogs_med = 1.0

    def median_ratio(series, mask, global_med):
        vals = series.loc[mask].dropna()
        return float(vals.median()) if len(vals) > 0 else float(global_med)

    promo_rev_mult = median_ratio(cal["rev_ratio"], cal["promo_count"] > 0, global_rev_med)
    promo_cogs_mult = median_ratio(cal["cogs_ratio"], cal["promo_count"] > 0, global_cogs_med)
    holiday_rev_mult = median_ratio(cal["rev_ratio"], cal["is_holiday"], global_rev_med)
    holiday_cogs_mult = median_ratio(cal["cogs_ratio"], cal["is_holiday"], global_cogs_med)
    tet_rev_mult = median_ratio(cal["rev_ratio"], cal["is_tet"], global_rev_med)
    tet_cogs_mult = median_ratio(cal["cogs_ratio"], cal["is_tet"], global_cogs_med)

    try:
        from sklearn.linear_model import LinearRegression
        rev_discount_mask = cal["avg_discount"] > 0
        if rev_discount_mask.sum() >= 5:
            lr_rev = LinearRegression()
            lr_rev.fit(cal.loc[rev_discount_mask, ["avg_discount"]], np.log(cal.loc[rev_discount_mask, "rev_ratio"].clip(lower=0.1)))
            rev_discount_coef = float(np.clip(lr_rev.coef_[0], -0.5, 0.5))
        else:
            rev_discount_coef = 0.0
        cogs_discount_mask = cal["avg_discount"] > 0
        if cogs_discount_mask.sum() >= 5:
            lr_cogs = LinearRegression()
            lr_cogs.fit(cal.loc[cogs_discount_mask, ["avg_discount"]], np.log(cal.loc[cogs_discount_mask, "cogs_ratio"].clip(lower=0.1)))
            cogs_discount_coef = float(np.clip(lr_cogs.coef_[0], -0.5, 0.5))
        else:
            cogs_discount_coef = 0.0
    except Exception:
        rev_discount_coef = 0.0
        cogs_discount_coef = 0.0

    def event_multiplier(df, promo_mult, holiday_mult, tet_mult, discount_coef):
        mult = np.ones(len(df), dtype=float)
        mult *= np.where(df["promo_count"] > 0, promo_mult, 1.0)
        mult *= np.where(df["is_holiday"], holiday_mult, 1.0)
        mult *= np.where(df["is_tet"], tet_mult, 1.0)
        mult *= (1.0 + discount_coef * df["avg_discount"])
        return np.clip(mult, 0.7, 1.6)

    cal_rev_event = event_multiplier(cal, promo_rev_mult, holiday_rev_mult, tet_rev_mult, rev_discount_coef)
    cal_cogs_event = event_multiplier(cal, promo_cogs_mult, holiday_cogs_mult, tet_cogs_mult, cogs_discount_coef)

    cal["Revenue_pred"] = (cal_base_rev * cal_rev_event).clip(lower=0)
    cal["COGS_pred"] = (cal_base_cogs * cal_cogs_event).clip(lower=0)
    cal["COGS_pred"] = np.minimum(cal["COGS_pred"], cal["Revenue_pred"] * 0.995)

    test["month"] = test["Date"].dt.month
    test["day"] = test["Date"].dt.day
    test["weekday"] = test["Date"].dt.weekday
    test["year"] = test["Date"].dt.year
    model_test = test.merge(seasonal, on=["month", "day"], how="left").merge(weekday_prof, on="weekday", how="left")
    model_test["rev_norm"] = model_test["rev_norm_season"].fillna(1.0)
    model_test["cogs_norm"] = model_test["cogs_norm_season"].fillna(1.0)
    model_test["rev_wday_factor"] = model_test["rev_wday_factor"].fillna(1.0)
    model_test["cogs_wday_factor"] = model_test["cogs_wday_factor"].fillna(1.0)
    model_test["years_ahead"] = model_test["year"] - base_year

    test_base_rev = (base_rev * (growth_rev ** model_test["years_ahead"]) * model_test["rev_norm"] * model_test["rev_wday_factor"]).fillna(0)
    test_base_cogs = (base_cogs * (growth_cogs ** model_test["years_ahead"]) * model_test["cogs_norm"] * model_test["cogs_wday_factor"]).fillna(0)
    model_test = add_event_features(model_test)
    test_rev_event = event_multiplier(model_test, promo_rev_mult, holiday_rev_mult, tet_rev_mult, rev_discount_coef)
    test_cogs_event = event_multiplier(model_test, promo_cogs_mult, holiday_cogs_mult, tet_cogs_mult, cogs_discount_coef)
    model_test["Revenue_pred"] = (test_base_rev * test_rev_event).clip(lower=0).round(2)
    model_test["COGS_pred"] = (test_base_cogs * test_cogs_event).clip(lower=0).round(2)
    model_test["COGS_pred"] = np.minimum(model_test["COGS_pred"], model_test["Revenue_pred"] * 0.995)

    train_pred = cal[["Date", "Revenue_pred", "COGS_pred"]].copy()
    test_pred = model_test[["Date", "Revenue_pred", "COGS_pred"]].copy()
    return train_pred, test_pred


# ---------------------------------------------------------------------------
# 3. Time-Series Cross Validation
# ---------------------------------------------------------------------------
def run_tscv(
    train,
    test_feat,
    baseline_train_pred,
    baseline_test_pred,
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
        "resid_rev": np.zeros(n_train), "resid_cogs": np.zeros(n_train),
    }
    test_fold = {
        "rf_rev": [], "rf_cogs": [],
        "lgb_rev": [], "lgb_cogs": [],
        "xgb_rev": [], "xgb_cogs": [],
        "resid_rev": [], "resid_cogs": [],
    }

    # collect per-fold metrics
    fold_records = []

    base_rev_train = baseline_train_pred["Revenue_pred"].to_numpy()
    base_cogs_train = baseline_train_pred["COGS_pred"].to_numpy()
    base_rev_test = baseline_test_pred["Revenue_pred"].to_numpy()
    base_cogs_test = baseline_test_pred["COGS_pred"].to_numpy()

    test_X = test_feat.drop(columns=["Date"], errors="ignore").copy()
    test_X = test_X.reindex(columns=feature_names)
    test_X = test_X.fillna(median_vals).fillna(0.0)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
        yl_tr, yl_va = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        base_rev_tr = base_rev_train[tr_idx]
        base_cogs_tr = base_cogs_train[tr_idx]
        base_rev_va = base_rev_train[val_idx]
        base_cogs_va = base_cogs_train[val_idx]

        # ---- Random Forest ----
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, n_jobs=-1, random_state=42,
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
             "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8,
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
             "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8,
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
             "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
             "colsample_bytree": 0.8, "nthread": -1},
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
             "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
             "colsample_bytree": 0.8, "nthread": -1},
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

        # ---- Residual ElasticNet ----
        resid_rev_train = yl_tr["Revenue"].to_numpy() - np.log1p(base_rev_tr)
        resid_cogs_train = yl_tr["COGS"].to_numpy() - np.log1p(base_cogs_tr)
        resid_rev_val_true = yl_va["Revenue"].to_numpy() - np.log1p(base_rev_va)
        resid_cogs_val_true = yl_va["COGS"].to_numpy() - np.log1p(base_cogs_va)

        en_rev = Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ])
        en_cogs = Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNet(alpha=0.001, l1_ratio=0.8, max_iter=5000)),
        ])
        en_rev.fit(X_tr, resid_rev_train)
        en_cogs.fit(X_tr, resid_cogs_train)

        resid_rev_pred = en_rev.predict(X_va)
        resid_cogs_pred = en_cogs.predict(X_va)
        pr_rev = np.maximum(safe_expm1(np.log1p(base_rev_va) + resid_rev_pred), 0.0)
        pr_cogs = np.maximum(safe_expm1(np.log1p(base_cogs_va) + resid_cogs_pred), 0.0)
        oof["resid_rev"][val_idx] = pr_rev
        oof["resid_cogs"][val_idx] = pr_cogs

        resid_rev_test = en_rev.predict(test_X)
        resid_cogs_test = en_cogs.predict(test_X)
        pt_rev = np.maximum(safe_expm1(np.log1p(base_rev_test) + resid_rev_test), 0.0)
        pt_cogs = np.maximum(safe_expm1(np.log1p(base_cogs_test) + resid_cogs_test), 0.0)
        test_fold["resid_rev"].append(pt_rev)
        test_fold["resid_cogs"].append(pt_cogs)
        fold_records.append({"fold": fold, "model": "resid_en", "target": "Revenue",
                             "mape": mape(y_va["Revenue"], pr_rev), "r2": r2_score_fn(y_va["Revenue"], pr_rev)})
        fold_records.append({"fold": fold, "model": "resid_en", "target": "COGS",
                             "mape": mape(y_va["COGS"], pr_cogs), "r2": r2_score_fn(y_va["COGS"], pr_cogs)})
        print(f"  ResEN MAPE Rev: {mape(y_va['Revenue'], pr_rev):.4f}  COGS: {mape(y_va['COGS'], pr_cogs):.4f} | R2 Rev: {r2_score_fn(y_va['Revenue'], pr_rev):.4f}  COGS: {r2_score_fn(y_va['COGS'], pr_cogs):.4f}")

    # Average test predictions across folds
    test_avg = {}
    for k, v in test_fold.items():
        test_avg[k] = np.mean(np.vstack(v), axis=0)

    fold_df = pd.DataFrame(fold_records)
    return oof, test_avg, y, fold_df, dates


# ---------------------------------------------------------------------------
# 4. Stacking with Huber Regression
# ---------------------------------------------------------------------------
def stack_with_huber(oof, test_avg, y_train, test_dates):
    X_stack_rev = np.column_stack([
        oof["rf_rev"], oof["lgb_rev"], oof["xgb_rev"],
        oof["resid_rev"],
    ])
    X_stack_cogs = np.column_stack([
        oof["rf_cogs"], oof["lgb_cogs"], oof["xgb_cogs"],
        oof["resid_cogs"],
    ])

    X_test_rev = np.column_stack([
        test_avg["rf_rev"], test_avg["lgb_rev"], test_avg["xgb_rev"],
        test_avg["resid_rev"],
    ])
    X_test_cogs = np.column_stack([
        test_avg["rf_cogs"], test_avg["lgb_cogs"], test_avg["xgb_cogs"],
        test_avg["resid_cogs"],
    ])

    huber_rev = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber_rev.fit(X_stack_rev, y_train["Revenue"].to_numpy())
    pred_rev = huber_rev.predict(X_test_rev)

    huber_cogs = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber_cogs.fit(X_stack_cogs, y_train["COGS"].to_numpy())
    pred_cogs = huber_cogs.predict(X_test_cogs)

    pred_rev = np.maximum(pred_rev, 0.0)
    pred_cogs = np.maximum(pred_cogs, 0.0)
    pred_cogs = np.minimum(pred_cogs, pred_rev * 0.995)

    # OOF stacking predictions
    stack_rev_train = huber_rev.predict(X_stack_rev)
    stack_cogs_train = huber_cogs.predict(X_stack_cogs)

    print("\nHuber stacking weights (Revenue):", dict(zip(["rf", "lgb", "xgb", "resid_en"], huber_rev.coef_.round(4))))
    print("Huber stacking intercept (Revenue):", round(huber_rev.intercept_, 2))
    print("Huber stacking weights (COGS):", dict(zip(["rf", "lgb", "xgb", "resid_en"], huber_cogs.coef_.round(4))))
    print("Huber stacking intercept (COGS):", round(huber_cogs.intercept_, 2))

    sub = pd.DataFrame({
        "Date": pd.to_datetime(test_dates).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(pred_rev, 2),
        "COGS": np.round(pred_cogs, 2),
    })
    return sub, stack_rev_train, stack_cogs_train


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_results(dates, y, oof, stack_rev_train, stack_cogs_train, fold_df, out_dir):
    """
    Generate and save diagnostic plots:
      1. OOF time-series: Actual vs each model (Revenue & COGS)
      2. Scatter actual vs predicted (Stacking)
      3. Bar chart: CV MAPE and R2 mean±std by model
    """
    dates = pd.to_datetime(dates)

    # ---------- Plot 1: OOF Time Series ----------
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    models_rev = [("rf_rev", "RF"), ("lgb_rev", "LGB"), ("xgb_rev", "XGB"), ("resid_rev", "ResEN")]
    models_cogs = [("rf_cogs", "RF"), ("lgb_cogs", "LGB"), ("xgb_cogs", "XGB"), ("resid_cogs", "ResEN")]

    axes[0].plot(dates, y["Revenue"], label="Actual", color="black", lw=1.2)
    for key, label in models_rev:
        axes[0].plot(dates, oof[key], label=label, lw=0.8, linestyle="--", alpha=0.8)
    axes[0].set_title("Revenue — OOF Predictions (5-fold TimeSeriesCV)")
    axes[0].set_ylabel("Revenue")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(dates, y["COGS"], label="Actual", color="black", lw=1.2)
    for key, label in models_cogs:
        axes[1].plot(dates, oof[key], label=label, lw=0.8, linestyle="--", alpha=0.8)
    axes[1].set_title("COGS — OOF Predictions (5-fold TimeSeriesCV)")
    axes[1].set_ylabel("COGS")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left", fontsize=8)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading processed features...")
    train, test_feat = load_processed_features()

    print("Building Baseline+ predictions...")
    baseline_train_pred, baseline_test_pred = build_baseline_plus(train, test_feat)

    print(f"Train rows: {len(train)} | Test rows: {len(test_feat)}")

    print("\nRunning 5-fold TimeSeriesCV...")
    oof, test_avg, y, fold_df, dates = run_tscv(
        train, test_feat, baseline_train_pred, baseline_test_pred, n_splits=5
    )

    print("\n=== OOF CV Scores ===")
    for target in ["rev", "cogs"]:
        tname = "Revenue" if target == "rev" else "COGS"
        for model in ["rf", "lgb", "xgb", "resid"]:
            key = f"{model}_{target}"
            print(f"  {key.upper():12s} MAPE: {mape(y[tname], oof[key]):.4f} | R2: {r2_score_fn(y[tname], oof[key]):.4f}")

    print("\nStacking with Huber Regression...")
    submission, stack_rev_train, stack_cogs_train = stack_with_huber(oof, test_avg, y, test_feat["Date"])

    print(f"\nStacking OOF — Revenue MAPE: {mape(y['Revenue'], stack_rev_train):.4f} | R2: {r2_score_fn(y['Revenue'], stack_rev_train):.4f}")
    print(f"Stacking OOF — COGS   MAPE: {mape(y['COGS'], stack_cogs_train):.4f} | R2: {r2_score_fn(y['COGS'], stack_cogs_train):.4f}")

    out_path = OUT_DIR / "stacking_huber_tscv_submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission.head(10))

    print("\nGenerating plots...")
    plot_results(dates, y, oof, stack_rev_train, stack_cogs_train, fold_df, OUT_DIR)


if __name__ == "__main__":
    main()
