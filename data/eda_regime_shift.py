"""
eda_regime_shift.py
Quick EDA to understand why Revenue/COGS dropped after 2019.
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "analytical"
OP_DIR = ROOT / "operational"
MASTER_DIR = ROOT / "master"
OUT_DIR = ROOT.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES = RAW_DIR / "sales.csv"
TRAFFIC = OP_DIR / "web_traffic.csv"
INVENTORY = OP_DIR / "inventory.csv"
PROMO = MASTER_DIR / "promotions.csv"


def main():
    print("Loading data...")
    sales = pd.read_csv(SALES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sales["year"] = sales["Date"].dt.year
    sales["month"] = sales["Date"].dt.month
    sales["year_month"] = sales["Date"].dt.to_period("M")

    # -----------------------------------------------------------------------
    # 1. Annual aggregates
    # -----------------------------------------------------------------------
    annual = sales.groupby("year")[["Revenue", "COGS"]].agg(["sum", "mean", "std"]).reset_index()
    annual.columns = ["year", "rev_sum", "cogs_sum", "rev_mean", "cogs_mean", "rev_std", "cogs_std"]
    annual["rev_yoy"] = annual["rev_sum"].pct_change()
    annual["cogs_yoy"] = annual["cogs_sum"].pct_change()
    print("\n=== Annual Revenue / COGS ===")
    print(annual.to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. Monthly pattern before vs after 2019
    # -----------------------------------------------------------------------
    sales["era"] = np.where(sales["year"] <= 2019, "pre_2020", "post_2019")
    monthly_era = sales.groupby(["era", "month"])[["Revenue", "COGS"]].mean().reset_index()
    print("\n=== Monthly mean: pre_2020 vs post_2019 ===")
    print(monthly_era.to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. Plot time series with year markers
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(sales["Date"], sales["Revenue"], lw=0.5, label="Revenue")
    axes[0].axvline(pd.Timestamp("2020-01-01"), color="red", linestyle="--", label="2020")
    axes[0].axvline(pd.Timestamp("2019-01-01"), color="orange", linestyle="--", label="2019")
    axes[0].set_title("Revenue over time")
    axes[0].set_ylabel("Revenue")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(sales["Date"], sales["COGS"], lw=0.5, color="orange", label="COGS")
    axes[1].axvline(pd.Timestamp("2020-01-01"), color="red", linestyle="--")
    axes[1].axvline(pd.Timestamp("2019-01-01"), color="orange", linestyle="--")
    axes[1].set_title("COGS over time")
    axes[1].set_ylabel("COGS")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "eda_time_series.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'eda_time_series.png'}")

    # -----------------------------------------------------------------------
    # 4. Traffic trend
    # -----------------------------------------------------------------------
    if TRAFFIC.exists():
        traffic = pd.read_csv(TRAFFIC, parse_dates=["date"]).rename(columns={"date": "Date"})
        traffic_day = traffic.groupby("Date")["sessions"].sum().reset_index()
        traffic_day["year"] = traffic_day["Date"].dt.year
        traffic_annual = traffic_day.groupby("year")["sessions"].sum().reset_index()
        traffic_annual["sessions_yoy"] = traffic_annual["sessions"].pct_change()
        print("\n=== Annual Traffic ===")
        print(traffic_annual.to_string(index=False))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(traffic_day["Date"], traffic_day["sessions"], lw=0.5)
        ax.axvline(pd.Timestamp("2020-01-01"), color="red", linestyle="--")
        ax.set_title("Daily web traffic (sessions)")
        ax.set_ylabel("Sessions")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "eda_traffic.png", dpi=200)
        plt.close(fig)
        print(f"Saved: {OUT_DIR / 'eda_traffic.png'}")

    # -----------------------------------------------------------------------
    # 5. Promotions count per year
    # -----------------------------------------------------------------------
    if PROMO.exists():
        promo = pd.read_csv(PROMO, parse_dates=["start_date", "end_date"])
        promo["year"] = promo["start_date"].dt.year
        promo_count = promo.groupby("year").size().reset_index(name="promo_count")
        print("\n=== Promotions per year ===")
        print(promo_count.to_string(index=False))

    # -----------------------------------------------------------------------
    # 6. Rolling 365-day mean to see level shift
    # -----------------------------------------------------------------------
    sales = sales.sort_values("Date").reset_index(drop=True)
    sales["rev_ma365"] = sales["Revenue"].rolling(window=365, min_periods=30).mean()
    sales["cogs_ma365"] = sales["COGS"].rolling(window=365, min_periods=30).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sales["Date"], sales["rev_ma365"], label="Revenue MA365", lw=1.5)
    ax.plot(sales["Date"], sales["cogs_ma365"], label="COGS MA365", lw=1.5)
    ax.axvline(pd.Timestamp("2020-01-01"), color="red", linestyle="--", label="2020")
    ax.set_title("Rolling 365-day mean")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "eda_rolling_mean.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'eda_rolling_mean.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
