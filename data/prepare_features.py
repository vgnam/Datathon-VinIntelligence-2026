"""
prepare_features.py
Feature engineering from raw data -> train/test features + target.
Run this before train_ensemble_tscv.py.

Design principle: ONLY use features that are either (a) calendar/date-derived
or (b) extrapolated / derived from historical sales (2012-2022). 
We deliberately DROP all concurrent operational data (traffic, orders, inventory,
promotions) because those do not exist for the 2023-2024 test horizon and cause
severe overfit.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "analytical"
OP_DIR = ROOT / "operational"
TRANS_DIR = ROOT / "transaction"
OUT_DIR = ROOT.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = RAW_DIR / "sales.csv"
TEST_FILE = RAW_DIR / "sample_submission.csv"
TRAFFIC_FILE = OP_DIR / "web_traffic.csv"
ORDERS_FILE = TRANS_DIR / "orders.csv"
ORDER_ITEMS_FILE = TRANS_DIR / "order_items.csv"
RETURNS_FILE = TRANS_DIR / "returns.csv"


def load_annual_metrics():
    """Derive annual structural metrics from transaction/operational data."""
    rows = []
    if ORDERS_FILE.exists():
        orders = pd.read_csv(ORDERS_FILE, parse_dates=["order_date"])
        orders["year"] = orders["order_date"].dt.year

        uniq = orders.groupby("year")["customer_id"].nunique().reset_index()
        uniq.columns = ["year", "annual_unique_customers"]

        total = orders.groupby("year").size().reset_index(name="annual_total_orders")

        first_year = orders.groupby("customer_id")["year"].min().reset_index()
        first_year.columns = ["customer_id", "first_year"]
        orders2 = orders.merge(first_year, on="customer_id", how="left")
        orders2["is_new"] = (orders2["year"] == orders2["first_year"]).astype(int)
        new_rate = orders2.groupby("year")["is_new"].mean().reset_index()
        new_rate.columns = ["year", "annual_new_customer_rate"]

        aov = None
        if ORDER_ITEMS_FILE.exists():
            items = pd.read_csv(ORDER_ITEMS_FILE)
            items["line_value"] = items["quantity"] * items["unit_price"]
            items = items.merge(orders[["order_id", "year"]].drop_duplicates(), on="order_id", how="left")
            aov = items.groupby("year").agg(total_value=("line_value", "sum"), total_discount=("discount_amount", "sum")).reset_index()
            aov = aov.merge(total, on="year", how="left")
            aov["annual_aov"] = aov["total_value"] / aov["annual_total_orders"]
            aov["annual_discount_intensity"] = aov["total_discount"] / aov["total_value"]

        ret = None
        if RETURNS_FILE.exists():
            returns = pd.read_csv(RETURNS_FILE)
            order_year = orders[["order_id", "year"]].drop_duplicates()
            returns = returns.merge(order_year, on="order_id", how="left")
            returned_orders = returns.groupby("year")["order_id"].nunique().reset_index()
            returned_orders.columns = ["year", "annual_returned_orders"]
            ret = total.merge(returned_orders, on="year", how="left")
            ret["annual_return_rate"] = ret["annual_returned_orders"] / ret["annual_total_orders"]

        df = uniq.merge(total, on="year").merge(new_rate, on="year")
        if aov is not None:
            df = df.merge(aov[["year", "annual_aov", "annual_discount_intensity"]], on="year", how="left")
        else:
            df["annual_aov"] = np.nan
            df["annual_discount_intensity"] = np.nan
        if ret is not None:
            df = df.merge(ret[["year", "annual_return_rate"]], on="year", how="left")
        else:
            df["annual_return_rate"] = np.nan
        rows.append(df)

    if TRAFFIC_FILE.exists():
        traffic = pd.read_csv(TRAFFIC_FILE, parse_dates=["date"])
        traffic["year"] = traffic["date"].dt.year
        sess = traffic.groupby("year")["sessions"].sum().reset_index()
        sess.columns = ["year", "annual_sessions"]
        sess["annual_sessions_growth"] = sess["annual_sessions"].pct_change()
        if rows:
            rows[0] = rows[0].merge(sess, on="year", how="left")
        else:
            rows.append(sess)

    if rows:
        return rows[0]
    return pd.DataFrame()


def main():
    print("Loading raw data...")
    sales = pd.read_csv(SALES_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test_dates = pd.read_csv(TEST_FILE, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # ========================================================================
    # 1. Calendar / time features
    # ========================================================================
    train_start = sales["Date"].min()
    for df in [sales, test_dates]:
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["weekday"] = df["Date"].dt.weekday
        df["dayofyear"] = df["Date"].dt.dayofyear
        df["year"] = df["Date"].dt.year
        df["quarter"] = df["Date"].dt.quarter
        df["days_since_start"] = (df["Date"] - train_start).dt.days
        # cyclic encoding
        df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        # polynomial trend terms (trees can split non-linear, but explicit helps linear meta-learner)
        df["days_since_start_sq"] = df["days_since_start"] ** 2
        df["year_sq"] = (df["year"] - 2012) ** 2

    # ========================================================================
    # 2. Historical (month,day) aggregates from train only
    # ========================================================================
    monthday_stats = (
        sales.groupby(["month", "day"])
        .agg(
            hist_monthday_revenue_mean=("Revenue", "mean"),
            hist_monthday_cogs_mean=("COGS", "mean"),
        )
        .reset_index()
    )
    sales = sales.merge(monthday_stats, on=["month", "day"], how="left")
    test_dates = test_dates.merge(monthday_stats, on=["month", "day"], how="left")

    # Recent (month,day) mean: last 2 years only (more sensitive to regime shift)
    recent_years = sales["year"].max() - 1  # e.g. 2021-2022 if max=2022
    monthday_recent = (
        sales[sales["year"] >= recent_years]
        .groupby(["month", "day"])
        .agg(
            hist_monthday_revenue_mean_2y=("Revenue", "mean"),
            hist_monthday_cogs_mean_2y=("COGS", "mean"),
        )
        .reset_index()
    )
    sales = sales.merge(monthday_recent, on=["month", "day"], how="left")
    test_dates = test_dates.merge(monthday_recent, on=["month", "day"], how="left")

    # Historical month-level mean (broader seasonal pattern)
    month_stats = (
        sales.groupby("month")
        .agg(
            hist_month_revenue_mean=("Revenue", "mean"),
            hist_month_cogs_mean=("COGS", "mean"),
        )
        .reset_index()
    )
    sales = sales.merge(month_stats, on="month", how="left")
    test_dates = test_dates.merge(month_stats, on="month", how="left")

    # Historical weekday-level mean (day-of-week pattern)
    weekday_stats = (
        sales.groupby("weekday")
        .agg(
            hist_weekday_revenue_mean=("Revenue", "mean"),
            hist_weekday_cogs_mean=("COGS", "mean"),
        )
        .reset_index()
    )
    sales = sales.merge(weekday_stats, on="weekday", how="left")
    test_dates = test_dates.merge(weekday_stats, on="weekday", how="left")

    # ========================================================================
    # 3. YoY growth (full-year aggregates)
    # ========================================================================
    annual = sales.groupby("year")[["Revenue", "COGS"]].sum().reset_index()
    annual["hist_yoy_revenue_growth"] = annual["Revenue"].pct_change()
    annual["hist_yoy_cogs_growth"] = annual["COGS"].pct_change()
    # 3-year smoothed growth (less noisy than single-year forward fill)
    annual["hist_yoy_rev_growth_3y"] = annual["hist_yoy_revenue_growth"].rolling(window=3, min_periods=1).mean()
    annual["hist_yoy_cogs_growth_3y"] = annual["hist_yoy_cogs_growth"].rolling(window=3, min_periods=1).mean()

    latest_rev_growth = annual["hist_yoy_revenue_growth"].iloc[-1]
    latest_cogs_growth = annual["hist_yoy_cogs_growth"].iloc[-1]
    latest_rev_growth_3y = annual["hist_yoy_rev_growth_3y"].iloc[-1]
    latest_cogs_growth_3y = annual["hist_yoy_cogs_growth_3y"].iloc[-1]

    sales = sales.merge(
        annual[["year", "hist_yoy_revenue_growth", "hist_yoy_cogs_growth",
               "hist_yoy_rev_growth_3y", "hist_yoy_cogs_growth_3y"]],
        on="year", how="left"
    )
    test_dates["hist_yoy_revenue_growth"] = latest_rev_growth
    test_dates["hist_yoy_cogs_growth"] = latest_cogs_growth
    test_dates["hist_yoy_rev_growth_3y"] = latest_rev_growth_3y
    test_dates["hist_yoy_cogs_growth_3y"] = latest_cogs_growth_3y

    # ========================================================================
    # 4. Year-ago lag (available for 2023 from 2022; 2024 from 2023 test which is missing -> use monthday mean fallback)
    # ========================================================================
    lag_df = sales[["Date", "Revenue", "COGS"]].copy()
    lag_df["lag_date"] = lag_df["Date"] + pd.Timedelta(days=365)
    lag_map = lag_df.set_index("lag_date")[["Revenue", "COGS"]]
    lag_map.columns = ["lag_1y_revenue", "lag_1y_cogs"]

    sales = sales.merge(lag_map, left_on="Date", right_index=True, how="left")
    test_dates = test_dates.merge(lag_map, left_on="Date", right_index=True, how="left")

    # For 2024 dates where lag_1y points to 2023 (test, not in train), fallback to hist_monthday mean
    missing_lag = test_dates["lag_1y_revenue"].isna()
    if missing_lag.any():
        test_dates.loc[missing_lag, "lag_1y_revenue"] = test_dates.loc[missing_lag, "hist_monthday_revenue_mean_2y"]
        test_dates.loc[missing_lag, "lag_1y_cogs"] = test_dates.loc[missing_lag, "hist_monthday_cogs_mean_2y"]

    # Also fill any remaining NaN in lags on train with full hist_monthday mean
    for col in ["lag_1y_revenue", "lag_1y_cogs"]:
        sales[col] = sales[col].fillna(sales["hist_monthday_revenue_mean"] if "revenue" in col else sales["hist_monthday_cogs_mean"])

    # ========================================================================
    # 5. Annual structural metrics from operational data (extrapolated to test)
    # ========================================================================
    ann_metrics = load_annual_metrics()
    if not ann_metrics.empty:
        test_years = sorted(test_dates["year"].unique())
        extra_rows = []
        for y in test_years:
            if y not in ann_metrics["year"].values:
                row = {"year": y}
                for col in ann_metrics.columns:
                    if col == "year":
                        continue
                    row[col] = ann_metrics[col].iloc[-3:].mean()
                extra_rows.append(row)
        if extra_rows:
            ann_metrics = pd.concat([ann_metrics, pd.DataFrame(extra_rows)], ignore_index=True)
        sales = sales.merge(ann_metrics, on="year", how="left")
        test_dates = test_dates.merge(ann_metrics, on="year", how="left")

    # ========================================================================
    # 6. Assemble outputs
    # ========================================================================
    feature_cols = [
        "Date",
        "hist_monthday_revenue_mean",
        "hist_monthday_cogs_mean",
        "hist_monthday_revenue_mean_2y",
        "hist_monthday_cogs_mean_2y",
        "hist_month_revenue_mean",
        "hist_month_cogs_mean",
        "hist_weekday_revenue_mean",
        "hist_weekday_cogs_mean",
        "lag_1y_revenue",
        "lag_1y_cogs",
        "day_cos",
        "hist_yoy_cogs_growth",
        "hist_yoy_cogs_growth_3y",
        "day_sin",
        "hist_yoy_revenue_growth",
        "hist_yoy_rev_growth_3y",
        "month",
        "year",
        "quarter",
        "days_since_start",
        "days_since_start_sq",
        "year_sq",
        "dayofyear",
        "weekday",
        "day",
    ]
    ann_cols = [c for c in ann_metrics.columns if c != "year"] if not ann_metrics.empty else []
    feature_cols.extend(ann_cols)

    train_features = sales[feature_cols].copy()
    train_features = train_features.fillna(train_features.median(numeric_only=True))

    test_features = test_dates[feature_cols].copy()
    test_features = test_features.fillna(test_features.median(numeric_only=True))

    train_target = sales[["Date", "Revenue", "COGS"]].copy()

    # ========================================================================
    # 6. Export
    # ========================================================================
    train_features.to_csv(OUT_DIR / "train_features.csv", index=False)
    test_features.to_csv(OUT_DIR / "test_features.csv", index=False)
    train_target.to_csv(OUT_DIR / "train_target.csv", index=False)

    print(f"Exported to {OUT_DIR}:")
    print(f"  train_features.csv  ({len(train_features)} rows, {len(train_features.columns)} cols)")
    print(f"  test_features.csv   ({len(test_features)} rows, {len(test_features.columns)} cols)")
    print(f"  train_target.csv    ({len(train_target)} rows)")
    print("\nFeature list:")
    for c in feature_cols[1:]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
