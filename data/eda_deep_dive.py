"""
eda_deep_dive.py
Deep-dive root-cause analysis: Why did Revenue drop after 2019?
Analyze traffic, orders, AOV, conversion, returns, customers, products.
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
TRANS_DIR = ROOT / "transaction"
OP_DIR = ROOT / "operational"
MASTER_DIR = ROOT / "master"
OUT_DIR = ROOT.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load core tables
sales = pd.read_csv(RAW_DIR / "sales.csv", parse_dates=["Date"])
traffic = pd.read_csv(OP_DIR / "web_traffic.csv", parse_dates=["date"])
orders = pd.read_csv(TRANS_DIR / "orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(TRANS_DIR / "order_items.csv")
returns = pd.read_csv(TRANS_DIR / "returns.csv", parse_dates=["return_date"])
customers = pd.read_csv(MASTER_DIR / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(MASTER_DIR / "products.csv")

sales["year"] = sales["Date"].dt.year
traffic["year"] = traffic["date"].dt.year
orders["year"] = orders["order_date"].dt.year

print("=" * 60)
print("1. CONVERSION FUNNEL (Traffic -> Orders -> Revenue)")
print("=" * 60)

# Annual traffic
traffic_annual = traffic.groupby("year").agg(
    total_sessions=("sessions", "sum"),
    total_visitors=("unique_visitors", "sum"),
).reset_index()

# Annual orders & revenue
orders_annual = orders.groupby("year").agg(
    total_orders=("order_id", "nunique"),
    total_customers=("customer_id", "nunique"),
).reset_index()
rev_annual = sales.groupby("year")[["Revenue", "COGS"]].sum().reset_index()

funnel = traffic_annual.merge(orders_annual, on="year", how="outer").merge(rev_annual, on="year", how="outer")
funnel["conversion_rate"] = funnel["total_orders"] / funnel["total_sessions"]
funnel["revenue_per_session"] = funnel["Revenue"] / funnel["total_sessions"]
funnel["aov"] = funnel["Revenue"] / funnel["total_orders"]
funnel["orders_per_customer"] = funnel["total_orders"] / funnel["total_customers"]

print(funnel.to_string(index=False))

print("\n" + "=" * 60)
print("2. ORDER SIZE & PRICING (order_items)")
print("=" * 60)

items_full = order_items.merge(orders[["order_id", "order_date", "year"]], on="order_id", how="left")
items_full["line_total"] = items_full["quantity"] * items_full["unit_price"] - items_full["discount_amount"]
items_full["discount_rate"] = items_full["discount_amount"] / (items_full["unit_price"] * items_full["quantity"] + items_full["discount_amount"]).replace(0, np.nan)

items_annual = items_full.groupby("year").agg(
    avg_quantity=("quantity", "mean"),
    avg_unit_price=("unit_price", "mean"),
    avg_discount_rate=("discount_rate", "mean"),
    avg_line_total=("line_total", "mean"),
    total_quantity=("quantity", "sum"),
).reset_index()

print(items_annual.to_string(index=False))

print("\n" + "=" * 60)
print("3. RETURN RATE")
print("=" * 60)

# returns vs orders
returns["year"] = returns["return_date"].dt.year
ret_annual = returns.groupby("year").size().reset_index(name="total_returns")
ret_rate = orders_annual.merge(ret_annual, on="year", how="left").fillna(0)
ret_rate["return_rate"] = ret_rate["total_returns"] / ret_rate["total_orders"]
print(ret_rate[["year", "total_orders", "total_returns", "return_rate"]].to_string(index=False))

print("\n" + "=" * 60)
print("4. CUSTOMER BASE (new vs existing)")
print("=" * 60)

customers["registration_year"] = customers["signup_date"].dt.year
cust_annual = customers.groupby("registration_year").size().reset_index(name="new_customers")
cust_annual.rename(columns={"registration_year": "year"}, inplace=True)

# active customers per year
active_cust = orders.groupby("year")["customer_id"].nunique().reset_index(name="active_customers")
cust_base = cust_annual.merge(active_cust, on="year", how="outer").fillna(0)
print(cust_base.to_string(index=False))

print("\n" + "=" * 60)
print("5. PRODUCT MIX (category share by year)")
print("=" * 60)

items_prod = items_full.merge(products[["product_id", "category", "segment"]], on="product_id", how="left")
prod_mix = items_prod.groupby(["year", "category"]).agg(
    total_quantity=("quantity", "sum"),
    total_revenue=("line_total", "sum"),
).reset_index()
prod_mix_share = prod_mix.merge(
    prod_mix.groupby("year")[["total_quantity", "total_revenue"]].sum().reset_index(),
    on="year", suffixes=("", "_all"),
)
prod_mix_share["quantity_share"] = prod_mix_share["total_quantity"] / prod_mix_share["total_quantity_all"]
prod_mix_share["revenue_share"] = prod_mix_share["total_revenue"] / prod_mix_share["total_revenue_all"]
print(prod_mix_share[["year", "category", "quantity_share", "revenue_share"]].to_string(index=False))

print("\n" + "=" * 60)
print("6. SEASONAL PATTERN CHANGE (pre/post 2019 by month)")
print("=" * 60)

sales["month"] = sales["Date"].dt.month
sales["era"] = np.where(sales["year"] <= 2018, "pre_2019", "2019_2022")
season = sales.groupby(["era", "month"])[["Revenue", "COGS"]].mean().reset_index()
season_pivot = season.pivot(index="month", columns="era", values="Revenue")
season_pivot["ratio_post_pre"] = season_pivot["2019_2022"] / season_pivot["pre_2019"]
print(season_pivot.round(4))

# ===========================================================================
# Plots
# ===========================================================================
print("\nGenerating plots...")

# --- Funnel metrics ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

axes[0].bar(funnel["year"], funnel["conversion_rate"])
axes[0].set_title("Conversion Rate (Orders / Sessions)")
axes[0].set_ylabel("Rate")

axes[1].bar(funnel["year"], funnel["aov"])
axes[1].set_title("Average Order Value (AOV)")
axes[1].set_ylabel("VND")

axes[2].bar(funnel["year"], funnel["revenue_per_session"])
axes[2].set_title("Revenue per Session")
axes[2].set_ylabel("VND")

axes[3].bar(items_annual["year"], items_annual["avg_unit_price"])
axes[3].set_title("Avg Unit Price")
axes[3].set_ylabel("VND")

axes[4].bar(items_annual["year"], items_annual["avg_discount_rate"].fillna(0))
axes[4].set_title("Avg Discount Rate")
axes[4].set_ylabel("Rate")

axes[5].bar(ret_rate["year"], ret_rate["return_rate"])
axes[5].set_title("Return Rate")
axes[5].set_ylabel("Rate")

plt.tight_layout()
fig.savefig(OUT_DIR / "eda_funnel_metrics.png", dpi=200)
plt.close(fig)
print("  Saved eda_funnel_metrics.png")

# --- Monthly ratio ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(season_pivot.index, season_pivot["ratio_post_pre"], marker="o")
ax.axhline(1.0, color="red", linestyle="--")
ax.set_title("Revenue Post-2019 / Pre-2019 Ratio by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Ratio")
ax.set_xticks(range(1, 13))
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "eda_monthly_ratio.png", dpi=200)
plt.close(fig)
print("  Saved eda_monthly_ratio.png")

print("\nDone.")
