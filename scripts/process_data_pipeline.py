from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean

# Date configuration
CUTOFF_DATE = date(2022, 12, 31)
TRAIN_START = date(2012, 7, 4)
FORECAST_START = date(2023, 1, 1)
FORECAST_END = date(2024, 7, 1)

# Feature configuration
CARRYOVER_WINDOWS = [3, 5, 7]
DECAY_FACTOR = 0.8
MIN_OBSERVATIONS_FOR_PROFILE = 30
MAX_FORWARD_FILL_DAYS = 7
MISSING_FLAG_THRESHOLD = 0.1

# Optional: switch to cleaned sales if available
USE_CLEANED_SALES = False

TET_DATES = {
    2013: date(2013, 2, 10),
    2014: date(2014, 1, 31),
    2015: date(2015, 2, 19),
    2016: date(2016, 2, 8),
    2017: date(2017, 1, 28),
    2018: date(2018, 2, 16),
    2019: date(2019, 2, 5),
    2020: date(2020, 1, 25),
    2021: date(2021, 2, 12),
    2022: date(2022, 2, 1),
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
}

PUBLIC_HOLIDAYS = {(1, 1), (4, 30), (5, 1), (9, 2)}

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_CLEANED_DIR = BASE_DIR / "data_cleaned"
OUTPUT_DIR = BASE_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

SALES_PATH = DATA_DIR / "analytical" / "sales.csv"
CLEANED_SALES_PATH = DATA_CLEANED_DIR / "analytical" / "sales_cleaned.csv"
PRODUCTS_PATH = DATA_DIR / "master" / "products.csv"
CUSTOMERS_PATH = DATA_DIR / "master" / "customers.csv"
GEOGRAPHY_PATH = DATA_DIR / "master" / "geography.csv"
PROMOTIONS_PATH = DATA_DIR / "master" / "promotions.csv"
RETURNS_PATH = DATA_DIR / "transaction" / "returns.csv"
ORDER_ITEMS_PATH = DATA_DIR / "transaction" / "order_items.csv"
ORDERS_PATH = DATA_DIR / "transaction" / "orders.csv"
WEB_TRAFFIC_PATH = DATA_DIR / "operational" / "web_traffic.csv"
INVENTORY_PATH = DATA_DIR / "operational" / "inventory.csv"

MASTER_TIMELINE_PATH = INTERMEDIATE_DIR / "master_timeline.csv"
PRODUCT_FEATURES_PATH = INTERMEDIATE_DIR / "product_features.csv"
GEO_CUSTOMER_FEATURES_PATH = INTERMEDIATE_DIR / "geo_customer_features.csv"
PROMO_HISTORICAL_PATH = INTERMEDIATE_DIR / "promo_historical_events.csv"
PROMO_PROPENSITY_PATH = INTERMEDIATE_DIR / "promo_seasonal_propensity.csv"
CATEGORY_RETURN_RATES_PATH = INTERMEDIATE_DIR / "category_return_rates.csv"
TRAFFIC_PROFILE_PATH = INTERMEDIATE_DIR / "traffic_seasonal_profile.csv"
CATEGORY_PROMO_EFFICIENCY_PATH = INTERMEDIATE_DIR / "category_promo_efficiency.csv"
CHANNEL_CONVERSION_PATH = INTERMEDIATE_DIR / "channel_conversion_rates.csv"
TRAFFIC_DAILY_PATH = INTERMEDIATE_DIR / "traffic_daily_features.csv"
INVENTORY_DAILY_PATH = INTERMEDIATE_DIR / "inventory_daily_proxy.csv"

TRAIN_FEATURES_PATH = OUTPUT_DIR / "train_features.csv"
TEST_FEATURES_PATH = OUTPUT_DIR / "test_features.csv"
TRAIN_TARGET_PATH = OUTPUT_DIR / "train_target.csv"
FEATURE_CATALOG_PATH = OUTPUT_DIR / "feature_catalog.json"
PROCESS_LOG_PATH = OUTPUT_DIR / "processing_log.txt"


@dataclass
class PromoDailyStats:
    active: int = 0
    discount_sum: float = 0.0
    promo_count: int = 0
    stackable_max: int = 0


@dataclass
class TrafficAgg:
    sessions: float = 0.0
    unique_visitors: float = 0.0
    page_views: float = 0.0
    bounce_sum: float = 0.0
    duration_sum: float = 0.0
    count: int = 0


@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)


def parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def label_encode(values: list[str]) -> dict[str, int]:
    uniq = sorted({v for v in values if v})
    return {v: idx for idx, v in enumerate(uniq)}


def norm_channel(value: str) -> str:
    value = value.strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return "_".join([part for part in cleaned.split("_") if part])


def build_master_timeline(timeline: list[date]) -> dict[date, dict[str, float]]:
    features: dict[date, dict[str, float]] = {}
    tet_dates = sorted(TET_DATES.values())

    for d in timeline:
        day_of_year = d.timetuple().tm_yday
        week_of_year = d.isocalendar().week
        day_of_week = d.weekday()
        day_sin = math.sin(2 * math.pi * day_of_year / 365.25)
        day_cos = math.cos(2 * math.pi * day_of_year / 365.25)
        week_sin = math.sin(2 * math.pi * week_of_year / 52.0)
        week_cos = math.cos(2 * math.pi * week_of_year / 52.0)

        is_weekend = 1 if day_of_week >= 5 else 0
        is_month_start = 1 if d.day <= 3 else 0
        is_month_end = 1 if d.day >= 28 else 0
        is_qtr_end = 1 if d.month in (3, 6, 9, 12) else 0

        is_public_holiday = 1 if (d.month, d.day) in PUBLIC_HOLIDAYS else 0

        tet_date = min(tet_dates, key=lambda t: abs((t - d).days)) if tet_dates else None
        days_to_tet = 0
        days_since_tet = 0
        tet_recency_weight = 0.0
        is_tet_period = 0
        if tet_date:
            delta = (tet_date - d).days
            if abs(delta) <= 30:
                days_to_tet = delta
                days_since_tet = -delta if delta < 0 else 0
                tet_recency_weight = 1.0 / (1.0 + abs(delta))
            if abs(delta) <= 7:
                is_tet_period = 1

        holiday_recency_weight = 0.0
        if PUBLIC_HOLIDAYS:
            distances = [abs((date(d.year, m, day) - d).days) for m, day in PUBLIC_HOLIDAYS]
            if distances:
                min_dist = min(distances)
                if min_dist <= 7:
                    holiday_recency_weight = 1.0 / (1.0 + min_dist)

        features[d] = {
            "day_of_year": day_of_year,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "week_of_year": week_of_year,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "is_qtr_end": is_qtr_end,
            "month": d.month,
            "year": d.year,
            "is_tet_period": is_tet_period,
            "days_to_tet": days_to_tet,
            "days_since_tet": days_since_tet,
            "tet_recency_weight": tet_recency_weight,
            "is_public_holiday": is_public_holiday,
            "holiday_recency_weight": holiday_recency_weight,
        }

    return features


def process_products(log: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing products file: {PRODUCTS_PATH}")

    rows: list[dict[str, str]] = []
    category_prices: dict[str, list[float]] = {}
    categories: list[str] = []
    segments: list[str] = []
    sizes: list[str] = []
    colors: list[str] = []
    product_category: dict[int, str] = {}

    with PRODUCTS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            category = row.get("category", "")
            price = safe_float(row.get("price", ""))
            category_prices.setdefault(category, []).append(price)
            if category:
                categories.append(category)
            if row.get("segment"):
                segments.append(row["segment"])
            if row.get("size"):
                sizes.append(row["size"])
            if row.get("color"):
                colors.append(row["color"])
            try:
                product_category[int(row["product_id"])] = category
            except (TypeError, ValueError):
                continue

    category_bins: dict[str, tuple[float, float, float]] = {}
    for category, prices in category_prices.items():
        sorted_prices = sorted(prices)
        q1 = quantile(sorted_prices, 0.25)
        q2 = quantile(sorted_prices, 0.5)
        q3 = quantile(sorted_prices, 0.75)
        category_bins[category] = (q1, q2, q3)

    category_enc = label_encode(categories)
    segment_enc = label_encode(segments)
    size_enc = label_encode(sizes)
    color_enc = label_encode(colors)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    with PRODUCT_FEATURES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product_id",
            "margin_rate",
            "price_tier",
            "category_enc",
            "segment_enc",
            "size_enc",
            "color_enc",
            "price_error",
        ])
        price_error_count = 0
        for row in rows:
            price = safe_float(row.get("price", ""))
            cogs = safe_float(row.get("cogs", ""))
            price_error = 0
            if price > 0 and cogs >= price:
                price_error = 1
                price_error_count += 1
                cogs = price * 0.6

            margin_rate = (price - cogs) / price if price > 0 else 0.0
            category = row.get("category", "")
            bins = category_bins.get(category, (0.0, 0.0, 0.0))
            if price <= bins[0]:
                tier = 1
            elif price <= bins[1]:
                tier = 2
            elif price <= bins[2]:
                tier = 3
            else:
                tier = 4

            writer.writerow([
                row.get("product_id", ""),
                f"{margin_rate:.6f}",
                tier,
                category_enc.get(category, -1),
                segment_enc.get(row.get("segment", ""), -1),
                size_enc.get(row.get("size", ""), -1),
                color_enc.get(row.get("color", ""), -1),
                price_error,
            ])

    log.append(f"products.csv: price errors fixed = {price_error_count}")
    return product_category, category_enc


def process_geo_customers(log: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if not CUSTOMERS_PATH.exists() or not GEOGRAPHY_PATH.exists():
        raise FileNotFoundError("Missing customers or geography file")

    geo_by_zip: dict[str, dict[str, str]] = {}
    with GEOGRAPHY_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geo_by_zip[row["zip"]] = row

    region_stats: dict[str, dict[str, float]] = {}
    region_counts: dict[str, int] = {}
    gender_counts: dict[str, dict[str, int]] = {}
    age_counts: dict[str, dict[str, int]] = {}
    channel_counts: dict[str, dict[str, int]] = {}

    overall_counts = {"total": 0, "female": 0, "male": 0, "unknown": 0}
    overall_age_counts: dict[str, int] = {}
    overall_channel_counts: dict[str, int] = {}
    overall_tenure_sum = 0.0

    with CUSTOMERS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zip_code = row.get("zip", "")
            geo = geo_by_zip.get(zip_code)
            if not geo:
                continue
            region = geo.get("region", "Unknown")
            region_counts[region] = region_counts.get(region, 0) + 1

            signup_date = parse_date(row.get("signup_date", "2012-01-01"))
            tenure_days = (CUTOFF_DATE - signup_date).days
            region_stats.setdefault(region, {"tenure_sum": 0.0})
            region_stats[region]["tenure_sum"] += tenure_days

            gender = (row.get("gender") or "").strip().lower()
            if gender == "female":
                gender_key = "female"
            elif gender == "male":
                gender_key = "male"
            else:
                gender_key = "unknown"
            gender_counts.setdefault(region, {})
            gender_counts[region][gender_key] = gender_counts[region].get(gender_key, 0) + 1

            age_group = row.get("age_group", "unknown") or "unknown"
            age_counts.setdefault(region, {})
            age_counts[region][age_group] = age_counts[region].get(age_group, 0) + 1

            channel = row.get("acquisition_channel", "unknown") or "unknown"
            channel_counts.setdefault(region, {})
            channel_counts[region][channel] = channel_counts[region].get(channel, 0) + 1

            overall_counts["total"] += 1
            overall_counts[gender_key] += 1
            overall_age_counts[age_group] = overall_age_counts.get(age_group, 0) + 1
            overall_channel_counts[channel] = overall_channel_counts.get(channel, 0) + 1
            overall_tenure_sum += tenure_days

    regions = sorted(region_counts.keys())
    region_enc = label_encode(regions)

    with GEO_CUSTOMER_FEATURES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["region", "region_enc", "customer_count", "avg_signup_tenure"]

        gender_keys = sorted({g for counts in gender_counts.values() for g in counts.keys()})
        age_keys = sorted({a for counts in age_counts.values() for a in counts.keys()})
        channel_keys = sorted({c for counts in channel_counts.values() for c in counts.keys()})

        header += [f"gender_pct_{g}" for g in gender_keys]
        header += [f"age_group_pct_{a}" for a in age_keys]
        header += [f"acquisition_pct_{c}" for c in channel_keys]
        writer.writerow(header)

        for region in regions:
            count = region_counts.get(region, 0)
            tenure_sum = region_stats.get(region, {}).get("tenure_sum", 0.0)
            avg_tenure = tenure_sum / count if count else 0.0

            row = [
                region,
                region_enc.get(region, -1),
                count,
                f"{avg_tenure:.2f}",
            ]

            for g in gender_keys:
                row.append(f"{gender_counts.get(region, {}).get(g, 0) / count:.6f}" if count else "0")
            for a in age_keys:
                row.append(f"{age_counts.get(region, {}).get(a, 0) / count:.6f}" if count else "0")
            for c in channel_keys:
                row.append(f"{channel_counts.get(region, {}).get(c, 0) / count:.6f}" if count else "0")

            writer.writerow(row)

    overall_features: dict[str, float] = {}
    total = max(1, overall_counts["total"])
    overall_features["customer_count_total"] = float(overall_counts["total"])
    overall_features["avg_signup_tenure"] = overall_tenure_sum / total
    overall_features["gender_pct_female"] = overall_counts["female"] / total
    overall_features["gender_pct_male"] = overall_counts["male"] / total
    overall_features["gender_pct_unknown"] = overall_counts["unknown"] / total

    for age_key, count in overall_age_counts.items():
        overall_features[f"age_group_pct_{age_key}"] = count / total
    for channel_key, count in overall_channel_counts.items():
        overall_features[f"acquisition_pct_{channel_key}"] = count / total

    return {r: {"region_enc": region_enc.get(r, -1)} for r in regions}, overall_features


def process_promotions(log: list[str]) -> tuple[dict[date, PromoDailyStats], dict[tuple[int, str], float]]:
    if not PROMOTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing promotions file: {PROMOTIONS_PATH}")

    promo_daily: dict[date, PromoDailyStats] = {}
    propensity_counts: dict[tuple[int, str], int] = {}

    with PROMOTIONS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        with PROMO_HISTORICAL_PATH.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                "Date",
                "promo_id",
                "promo_type",
                "discount_value",
                "applicable_category",
                "stackable_flag",
            ])

            for row in reader:
                start = parse_date(row["start_date"])
                end = parse_date(row["end_date"])
                if start > CUTOFF_DATE:
                    continue

                promo_id = row.get("promo_id", "")
                promo_type = row.get("promo_type", "")
                discount_value = safe_float(row.get("discount_value", ""))
                applicable_category = row.get("applicable_category", "") or "ALL"
                stackable_flag = safe_int(row.get("stackable_flag", ""))

                current = start
                while current <= end and current <= CUTOFF_DATE:
                    writer.writerow([
                        current.isoformat(),
                        promo_id,
                        promo_type,
                        f"{discount_value:.4f}",
                        applicable_category,
                        stackable_flag,
                    ])

                    stats = promo_daily.setdefault(current, PromoDailyStats())
                    stats.active = 1
                    stats.discount_sum += discount_value
                    stats.promo_count += 1
                    stats.stackable_max = max(stats.stackable_max, stackable_flag)

                    propensity_counts[(current.month, applicable_category)] = (
                        propensity_counts.get((current.month, applicable_category), 0) + 1
                    )

                    current += timedelta(days=1)

    # Propensity denominator per month across the train window
    total_days_by_month: dict[int, int] = {}
    for d in date_range(TRAIN_START, CUTOFF_DATE):
        total_days_by_month[d.month] = total_days_by_month.get(d.month, 0) + 1

    with PROMO_PROPENSITY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "applicable_category", "promo_propensity"])
        for (month, category), active_days in sorted(propensity_counts.items()):
            denom = total_days_by_month.get(month, 1)
            propensity = active_days / denom if denom else 0.0
            writer.writerow([month, category, f"{propensity:.6f}"])

    log.append(f"promotions.csv: daily promo records = {len(promo_daily)}")
    return promo_daily, {k: v / max(1, total_days_by_month.get(k[0], 1)) for k, v in propensity_counts.items()}


def load_orders(log: list[str]) -> tuple[dict[str, date], dict[tuple[str, int, int], int], dict[tuple[int, int], int]]:
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(f"Missing orders file: {ORDERS_PATH}")

    order_date_by_id: dict[str, date] = {}
    orders_by_channel_month: dict[tuple[str, int, int], int] = {}
    orders_by_month: dict[tuple[int, int], int] = {}

    with ORDERS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order_id = row.get("order_id", "")
            order_date = parse_date(row.get("order_date", "2012-01-01"))
            order_date_by_id[order_id] = order_date

            if order_date <= CUTOFF_DATE:
                source = row.get("order_source", "unknown") or "unknown"
                key = (source, order_date.year, order_date.month)
                orders_by_channel_month[key] = orders_by_channel_month.get(key, 0) + 1
                orders_by_month[(order_date.year, order_date.month)] = (
                    orders_by_month.get((order_date.year, order_date.month), 0) + 1
                )

    log.append(f"orders.csv: total orders loaded = {len(order_date_by_id)}")
    return order_date_by_id, orders_by_channel_month, orders_by_month


def process_order_items(
    log: list[str],
    order_date_by_id: dict[str, date],
    product_category: dict[int, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if not ORDER_ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing order_items file: {ORDER_ITEMS_PATH}")

    total_sold_by_category: dict[str, float] = {}
    promo_revenue_by_category: dict[str, float] = {}
    promo_discount_by_category: dict[str, float] = {}

    with ORDER_ITEMS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order_id = row.get("order_id", "")
            order_date = order_date_by_id.get(order_id)
            if not order_date or order_date > CUTOFF_DATE:
                continue

            product_id = safe_int(row.get("product_id", ""), -1)
            category = product_category.get(product_id)
            if not category:
                continue

            quantity = safe_float(row.get("quantity", ""))
            total_sold_by_category[category] = total_sold_by_category.get(category, 0.0) + quantity

            promo_id = (row.get("promo_id") or "").strip()
            promo_id_2 = (row.get("promo_id_2") or "").strip()
            if promo_id or promo_id_2:
                unit_price = safe_float(row.get("unit_price", ""))
                discount_amount = safe_float(row.get("discount_amount", ""))
                promo_revenue_by_category[category] = (
                    promo_revenue_by_category.get(category, 0.0) + quantity * unit_price
                )
                promo_discount_by_category[category] = (
                    promo_discount_by_category.get(category, 0.0) + discount_amount
                )

    log.append("order_items.csv: aggregated sales and promo metrics")
    return total_sold_by_category, promo_revenue_by_category, promo_discount_by_category


def process_returns(
    log: list[str],
    product_category: dict[int, str],
) -> dict[str, float]:
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"Missing returns file: {RETURNS_PATH}")

    returned_by_category: dict[str, float] = {}
    with RETURNS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return_date = parse_date(row.get("return_date", "2012-01-01"))
            if return_date > CUTOFF_DATE:
                continue

            product_id = safe_int(row.get("product_id", ""), -1)
            category = product_category.get(product_id)
            if not category:
                continue

            return_qty = safe_float(row.get("return_quantity", ""))
            returned_by_category[category] = returned_by_category.get(category, 0.0) + return_qty

    log.append("returns.csv: aggregated returns")
    return returned_by_category


def build_category_return_rates(
    total_sold_by_category: dict[str, float],
    returned_by_category: dict[str, float],
) -> dict[str, float]:
    return_rates: dict[str, float] = {}
    with CATEGORY_RETURN_RATES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "return_rate"])
        for category in sorted(total_sold_by_category.keys() | returned_by_category.keys()):
            total_sold = total_sold_by_category.get(category, 0.0)
            total_returned = returned_by_category.get(category, 0.0)
            rate = total_returned / (total_sold + 1e-8)
            return_rates[category] = rate
            writer.writerow([category, f"{rate:.6f}"])
    return return_rates


def build_category_promo_efficiency(
    promo_revenue_by_category: dict[str, float],
    promo_discount_by_category: dict[str, float],
) -> dict[str, float]:
    efficiency: dict[str, float] = {}
    with CATEGORY_PROMO_EFFICIENCY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "promo_efficiency"])
        for category in sorted(promo_revenue_by_category.keys() | promo_discount_by_category.keys()):
            revenue = promo_revenue_by_category.get(category, 0.0)
            discount = promo_discount_by_category.get(category, 0.0)
            value = revenue / (discount + 1e-8)
            efficiency[category] = value
            writer.writerow([category, f"{value:.6f}"])
    return efficiency


def process_traffic_profile(log: list[str]) -> tuple[dict[tuple[int, int], tuple[float, float]], float, float]:
    if not WEB_TRAFFIC_PATH.exists():
        raise FileNotFoundError(f"Missing web traffic file: {WEB_TRAFFIC_PATH}")

    stats: dict[tuple[int, int], Welford] = {}
    global_stats = Welford()

    with WEB_TRAFFIC_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = parse_date(row.get("date", "2012-01-01"))
            if d > CUTOFF_DATE:
                continue
            sessions = safe_float(row.get("sessions", ""))
            key = (d.weekday(), d.month)
            stats.setdefault(key, Welford()).update(sessions)
            global_stats.update(sessions)

    profile: dict[tuple[int, int], tuple[float, float]] = {}
    with TRAFFIC_PROFILE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dow", "month", "sessions_mean", "sessions_std", "volatility_index", "n_obs"])
        for key, agg in sorted(stats.items()):
            if agg.count < MIN_OBSERVATIONS_FOR_PROFILE:
                continue
            sessions_mean = agg.mean
            sessions_std = agg.std()
            volatility = sessions_std / (sessions_mean + 1e-8)
            profile[key] = (sessions_mean, volatility)
            writer.writerow([
                key[0],
                key[1],
                f"{sessions_mean:.4f}",
                f"{sessions_std:.4f}",
                f"{volatility:.6f}",
                agg.count,
            ])

    log.append("web_traffic.csv: seasonal profile computed")
    return profile, global_stats.mean, global_stats.std()


def process_traffic_daily(
    log: list[str],
    profile: dict[tuple[int, int], tuple[float, float]],
    global_mean: float,
    global_std: float,
    timeline: list[date],
) -> tuple[dict[date, dict[str, float]], dict[date, float]]:
    daily: dict[date, TrafficAgg] = {}

    with WEB_TRAFFIC_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = parse_date(row.get("date", "2012-01-01"))
            agg = daily.setdefault(d, TrafficAgg())
            agg.sessions += safe_float(row.get("sessions", ""))
            agg.unique_visitors += safe_float(row.get("unique_visitors", ""))
            agg.page_views += safe_float(row.get("page_views", ""))
            agg.bounce_sum += safe_float(row.get("bounce_rate", ""))
            agg.duration_sum += safe_float(row.get("avg_session_duration_sec", ""))
            agg.count += 1

    traffic_features: dict[date, dict[str, float]] = {}
    expected_sessions: dict[date, float] = {}

    with TRAFFIC_DAILY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date",
            "sessions_total",
            "unique_visitors_total",
            "page_views_total",
            "bounce_rate_avg",
            "avg_duration_avg",
            "expected_sessions",
            "traffic_uncertainty",
            "is_profile_based",
        ])

        for d in timeline:
            agg = daily.get(d)
            sessions_total = agg.sessions if agg else None
            unique_visitors_total = agg.unique_visitors if agg else None
            page_views_total = agg.page_views if agg else None
            bounce_rate_avg = (agg.bounce_sum / agg.count) if agg and agg.count else None
            avg_duration_avg = (agg.duration_sum / agg.count) if agg and agg.count else None

            profile_key = (d.weekday(), d.month)
            profile_stats = profile.get(profile_key)
            if profile_stats:
                profile_mean, profile_vol = profile_stats
            else:
                profile_mean, profile_vol = global_mean, global_std / (global_mean + 1e-8)

            if d <= CUTOFF_DATE and sessions_total is not None:
                expected = sessions_total
                is_profile_based = 0
            else:
                expected = profile_mean
                is_profile_based = 1

            expected_sessions[d] = expected
            traffic_features[d] = {
                "expected_sessions": expected,
                "traffic_uncertainty": profile_vol,
                "is_profile_based": is_profile_based,
            }

            writer.writerow([
                d.isoformat(),
                "" if sessions_total is None else f"{sessions_total:.4f}",
                "" if unique_visitors_total is None else f"{unique_visitors_total:.4f}",
                "" if page_views_total is None else f"{page_views_total:.4f}",
                "" if bounce_rate_avg is None else f"{bounce_rate_avg:.6f}",
                "" if avg_duration_avg is None else f"{avg_duration_avg:.4f}",
                f"{expected:.4f}",
                f"{profile_vol:.6f}",
                is_profile_based,
            ])

    log.append("web_traffic.csv: daily features computed")
    return traffic_features, expected_sessions


def build_inventory_daily(log: list[str], timeline: list[date]) -> dict[date, dict[str, float]]:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Missing inventory file: {INVENTORY_PATH}")

    snapshot_metrics: dict[str, dict[date, dict[str, float]]] = {}
    with INVENTORY_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "") or "unknown"
            snap_date = parse_date(row.get("snapshot_date", "2012-01-01"))
            metrics = snapshot_metrics.setdefault(category, {}).setdefault(
                snap_date,
                {"fill_rate": 0.0, "stockout_freq": 0.0, "sell_through": 0.0, "reorder_freq": 0.0, "count": 0},
            )
            metrics["fill_rate"] += safe_float(row.get("fill_rate", ""))
            metrics["stockout_freq"] += safe_float(row.get("stockout_flag", ""))
            metrics["sell_through"] += safe_float(row.get("sell_through_rate", ""))
            metrics["reorder_freq"] += safe_float(row.get("reorder_flag", ""))
            metrics["count"] += 1

    category_means: dict[str, dict[str, float]] = {}
    for category, snapshots in snapshot_metrics.items():
        values = {"fill_rate": [], "stockout_freq": [], "sell_through": [], "reorder_freq": []}
        for stats in snapshots.values():
            count = max(1, stats["count"])
            values["fill_rate"].append(stats["fill_rate"] / count)
            values["stockout_freq"].append(stats["stockout_freq"] / count)
            values["sell_through"].append(stats["sell_through"] / count)
            values["reorder_freq"].append(stats["reorder_freq"] / count)
        category_means[category] = {k: mean(v) if v else 0.0 for k, v in values.items()}

    overall_daily: dict[date, dict[str, float]] = {d: {"sum_fill": 0.0, "sum_stockout": 0.0, "sum_sell": 0.0, "sum_reorder": 0.0, "sum_days": 0.0, "sum_stale": 0.0, "count": 0} for d in timeline}

    with INVENTORY_DAILY_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date",
            "category",
            "fill_rate_daily",
            "stockout_freq_daily",
            "sell_through_daily",
            "reorder_freq_daily",
            "days_since_snapshot",
            "is_stale",
        ])

        for category, snapshots in snapshot_metrics.items():
            snap_dates = sorted(snapshots.keys())
            if not snap_dates:
                continue

            for idx, snap_date in enumerate(snap_dates):
                next_date = snap_dates[idx + 1] if idx + 1 < len(snap_dates) else FORECAST_END
                stats = snapshots[snap_date]
                count = max(1, stats["count"])
                fill_rate = stats["fill_rate"] / count
                stockout_freq = stats["stockout_freq"] / count
                sell_through = stats["sell_through"] / count
                reorder_freq = stats["reorder_freq"] / count

                current = max(snap_date, TRAIN_START)
                while current <= min(next_date, FORECAST_END):
                    days_since = (current - snap_date).days
                    if days_since <= MAX_FORWARD_FILL_DAYS:
                        value_fill = fill_rate
                        value_stockout = stockout_freq
                        value_sell = sell_through
                        value_reorder = reorder_freq
                        is_stale = 0
                    else:
                        mean_vals = category_means.get(category, {})
                        value_fill = mean_vals.get("fill_rate", 0.0)
                        value_stockout = mean_vals.get("stockout_freq", 0.0)
                        value_sell = mean_vals.get("sell_through", 0.0)
                        value_reorder = mean_vals.get("reorder_freq", 0.0)
                        is_stale = 1

                    writer.writerow([
                        current.isoformat(),
                        category,
                        f"{value_fill:.6f}",
                        f"{value_stockout:.6f}",
                        f"{value_sell:.6f}",
                        f"{value_reorder:.6f}",
                        days_since,
                        is_stale,
                    ])

                    overall = overall_daily[current]
                    overall["sum_fill"] += value_fill
                    overall["sum_stockout"] += value_stockout
                    overall["sum_sell"] += value_sell
                    overall["sum_reorder"] += value_reorder
                    overall["sum_days"] += days_since
                    overall["sum_stale"] += is_stale
                    overall["count"] += 1

                    current += timedelta(days=1)

    overall_features: dict[date, dict[str, float]] = {}
    for d, sums in overall_daily.items():
        count = max(1, sums["count"])
        overall_features[d] = {
            "inventory_fill_rate": sums["sum_fill"] / count,
            "inventory_stockout_freq": sums["sum_stockout"] / count,
            "inventory_sell_through": sums["sum_sell"] / count,
            "inventory_reorder_freq": sums["sum_reorder"] / count,
            "inventory_days_since_snapshot": sums["sum_days"] / count,
            "inventory_is_stale_rate": sums["sum_stale"] / count,
        }

    log.append("inventory.csv: daily proxy computed")
    return overall_features


def build_promo_state(timeline: list[date], promo_daily: dict[date, PromoDailyStats]) -> dict[date, dict[str, float]]:
    promo_active_series: list[int] = []
    promo_intensity_series: list[float] = []
    promo_stackable_series: list[int] = []

    for d in timeline:
        if d <= CUTOFF_DATE:
            stats = promo_daily.get(d, PromoDailyStats())
            active = stats.active
            intensity = stats.discount_sum / stats.promo_count if stats.promo_count else 0.0
            stackable = stats.stackable_max
        else:
            active = 0
            intensity = 0.0
            stackable = 0
        promo_active_series.append(active)
        promo_intensity_series.append(intensity)
        promo_stackable_series.append(stackable)

    promo_features: dict[date, dict[str, float]] = {}
    for idx, d in enumerate(timeline):
        row = {
            "promo_active": promo_active_series[idx],
            "promo_intensity": promo_intensity_series[idx],
            "promo_stackable": promo_stackable_series[idx],
        }

        for window in CARRYOVER_WINDOWS:
            carryover = 0
            for k in range(1, window + 1):
                if idx - k >= 0:
                    carryover += promo_active_series[idx - k]
            row[f"promo_carryover_{window}d"] = carryover

        recency = 0.0
        for k in range(0, 8):
            if idx - k >= 0:
                recency += promo_active_series[idx - k] * (DECAY_FACTOR ** k)
        row["promo_recency_weighted"] = recency

        promo_features[d] = row

    return promo_features


def build_traffic_lags(timeline: list[date], expected_sessions: dict[date, float]) -> dict[date, dict[str, float]]:
    series = [expected_sessions.get(d, 0.0) for d in timeline]
    features: dict[date, dict[str, float]] = {}

    for idx, d in enumerate(timeline):
        carryover = 0.0
        if idx - 1 >= 0:
            carryover += series[idx - 1]
        if idx - 2 >= 0:
            carryover += series[idx - 2]

        recency = 0.0
        for k in range(0, 8):
            if idx - k >= 0:
                recency += series[idx - k] * (DECAY_FACTOR ** k)

        features[d] = {
            "traffic_carryover_2d": carryover,
            "traffic_recency_weighted": recency,
        }

    return features


def process_conversion_rates(
    log: list[str],
    orders_by_channel_month: dict[tuple[str, int, int], int],
    orders_by_month: dict[tuple[int, int], int],
) -> tuple[dict[tuple[str, int], float], dict[tuple[int, int], float], dict[int, float]]:
    sessions_by_channel_month: dict[tuple[str, int, int], float] = {}
    sessions_by_month: dict[tuple[int, int], float] = {}

    with WEB_TRAFFIC_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = parse_date(row.get("date", "2012-01-01"))
            if d > CUTOFF_DATE:
                continue
            sessions = safe_float(row.get("sessions", ""))
            source = row.get("traffic_source", "unknown") or "unknown"
            sessions_by_channel_month[(source, d.year, d.month)] = (
                sessions_by_channel_month.get((source, d.year, d.month), 0.0) + sessions
            )
            sessions_by_month[(d.year, d.month)] = sessions_by_month.get((d.year, d.month), 0.0) + sessions

    traffic_norm_map = {norm_channel(k[0]): k[0] for k in sessions_by_channel_month.keys()}

    channel_rates: dict[tuple[str, int], float] = {}
    with CHANNEL_CONVERSION_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "month", "conversion_rate"])

        for (order_source, year, month), order_count in sorted(orders_by_channel_month.items()):
            norm_source = norm_channel(order_source)
            traffic_source = traffic_norm_map.get(norm_source)
            if not traffic_source:
                continue
            session_count = sessions_by_channel_month.get((traffic_source, year, month), 0.0)
            rate = order_count / (session_count + 1e-8)
            channel_rates[(traffic_source, month)] = rate
            writer.writerow([traffic_source, month, f"{rate:.6f}"])

    overall_rates: dict[tuple[int, int], float] = {}
    for key, order_count in orders_by_month.items():
        session_count = sessions_by_month.get(key, 0.0)
        overall_rates[key] = order_count / (session_count + 1e-8)

    # FIXED: conversion rate by month only (for test period lookup)
    conversion_by_month_only: dict[int, float] = {}
    for month in range(1, 13):
        month_rates = [r for (y, m), r in overall_rates.items() if m == month]
        if month_rates:
            conversion_by_month_only[month] = mean(month_rates)
        else:
            conversion_by_month_only[month] = 0.0

    log.append("orders + web_traffic: conversion rates computed")
    return channel_rates, overall_rates, conversion_by_month_only


def build_feature_catalog(columns: list[str]) -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for col in columns:
        if col == "Date":
            continue
        if col.startswith("promo_"):
            source = "promotions.csv + causal lag logic"
        elif col.startswith("traffic_") or col == "expected_sessions" or col == "is_profile_based":
            source = "web_traffic.csv + seasonal profile"
        elif col.startswith("inventory_"):
            source = "inventory.csv"
        elif col.startswith("gender_pct") or col.startswith("age_group_pct") or col.startswith("acquisition_pct"):
            source = "customers.csv + geography.csv"
        elif col.startswith("conversion_rate"):
            source = "orders.csv + web_traffic.csv"
        elif col.startswith("return_rate"):
            source = "returns.csv + order_items.csv + products.csv"
        elif col.startswith("promo_efficiency"):
            source = "promotions.csv + order_items.csv"
        else:
            source = "calendar"

        catalog.append(
            {
                "feature_name": col,
                "source": source,
                "granularity": "daily",
                "forecast_compatible": True,
                "computation_method": "see processing script",
                "assumptions": [],
                "missing_handling": "imputed where needed",
            }
        )
    return catalog


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    timeline = date_range(TRAIN_START, FORECAST_END)
    temporal_features = build_master_timeline(timeline)

    with MASTER_TIMELINE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date"])
        for d in timeline:
            writer.writerow([d.isoformat()])

    product_category, _ = process_products(log)
    _, geo_global = process_geo_customers(log)
    promo_daily, _ = process_promotions(log)

    order_date_by_id, orders_by_channel_month, orders_by_month = load_orders(log)
    total_sold, promo_revenue, promo_discount = process_order_items(log, order_date_by_id, product_category)
    returned = process_returns(log, product_category)

    category_return_rates = build_category_return_rates(total_sold, returned)
    category_promo_efficiency = build_category_promo_efficiency(promo_revenue, promo_discount)

    traffic_profile, traffic_global_mean, traffic_global_std = process_traffic_profile(log)
    traffic_daily, expected_sessions = process_traffic_daily(log, traffic_profile, traffic_global_mean, traffic_global_std, timeline)

    _, conversion_rates_by_month, conversion_by_month_only = process_conversion_rates(log, orders_by_channel_month, orders_by_month)

    inventory_overall = build_inventory_daily(log, timeline)
    promo_state = build_promo_state(timeline, promo_daily)
    traffic_lags = build_traffic_lags(timeline, expected_sessions)

    # FIXED: compute promo seasonal probabilities from train data
    promo_seasonal_prob: dict[tuple[int, int], float] = {}
    promo_monthly_prob: dict[int, float] = {}
    promo_active_train: list[int] = [promo_state[d]["promo_active"] for d in timeline if d <= CUTOFF_DATE]
    promo_dates_train: list[date] = [d for d in timeline if d <= CUTOFF_DATE]

    # By month-day
    md_counts: dict[tuple[int, int], int] = {}
    md_active: dict[tuple[int, int], int] = {}
    for d in promo_dates_train:
        md = (d.month, d.day)
        md_counts[md] = md_counts.get(md, 0) + 1
        if promo_state[d]["promo_active"]:
            md_active[md] = md_active.get(md, 0) + 1
    for md, cnt in md_counts.items():
        promo_seasonal_prob[md] = md_active.get(md, 0) / cnt

    # By month
    month_counts: dict[int, int] = {}
    month_active: dict[int, int] = {}
    for d in promo_dates_train:
        month_counts[d.month] = month_counts.get(d.month, 0) + 1
        if promo_state[d]["promo_active"]:
            month_active[d.month] = month_active.get(d.month, 0) + 1
    for m, cnt in month_counts.items():
        promo_monthly_prob[m] = month_active.get(m, 0) / cnt

    overall_return_rate = 0.0
    total_returned = sum(returned.values())
    total_sold_qty = sum(total_sold.values())
    if total_sold_qty > 0:
        overall_return_rate = total_returned / total_sold_qty

    overall_promo_efficiency = 0.0
    promo_discount_total = sum(promo_discount.values())
    promo_revenue_total = sum(promo_revenue.values())
    if promo_discount_total > 0:
        overall_promo_efficiency = promo_revenue_total / promo_discount_total

    # Build final feature rows
    feature_rows: list[dict[str, object]] = []
    imputed_conversion_flags: list[int] = []

    # Dynamic global feature column order
    global_feature_cols = [
        "customer_count_total",
        "avg_signup_tenure",
        "gender_pct_female",
        "gender_pct_male",
        "gender_pct_unknown",
    ]
    extra_geo_cols = sorted([k for k in geo_global.keys() if k not in global_feature_cols])
    global_feature_cols += extra_geo_cols

    temporal_cols = list(next(iter(temporal_features.values())).keys())
    promo_cols = list(next(iter(promo_state.values())).keys())
    traffic_cols = [
        "expected_sessions",
        "traffic_carryover_2d",
        "traffic_recency_weighted",
        "traffic_uncertainty",
        "is_profile_based",
    ]
    inventory_cols = list(next(iter(inventory_overall.values())).keys())
    conversion_cols = ["conversion_rate_overall", "is_imputed_conversion_rate"]
    pattern_cols = ["return_rate_overall", "promo_efficiency_overall"]

    columns = (
        ["Date"]
        + temporal_cols
        + promo_cols
        + ["promo_seasonal_prob", "promo_monthly_prob"]
        + traffic_cols
        + inventory_cols
        + conversion_cols
        + pattern_cols
        + global_feature_cols
    )

    for d in timeline:
        row: dict[str, object] = {"Date": d.isoformat()}
        row.update(temporal_features[d])
        row.update(promo_state[d])
        row.update(traffic_lags[d])
        row.update(traffic_daily[d])
        row.update(inventory_overall[d])

        # FIXED: conversion rate lookup
        if d <= CUTOFF_DATE:
            conversion_key = (d.year, d.month)
            conversion_rate = conversion_rates_by_month.get(conversion_key)
            if conversion_rate is None:
                conversion_rate = 0.0
                imputed_flag = 1
            else:
                imputed_flag = 0
        else:
            # Use month-only average for test period
            conversion_rate = conversion_by_month_only.get(d.month, 0.0)
            imputed_flag = 1  # still imputed but from historical monthly average
        row["conversion_rate_overall"] = conversion_rate
        row["is_imputed_conversion_rate"] = imputed_flag
        imputed_conversion_flags.append(imputed_flag)

        # FIXED: promo seasonal probabilities
        row["promo_seasonal_prob"] = promo_seasonal_prob.get((d.month, d.day), 0.0)
        row["promo_monthly_prob"] = promo_monthly_prob.get(d.month, 0.0)

        # FIXED: inventory in test period — mark as stale/projected
        if d > CUTOFF_DATE:
            for inv_key in inventory_cols:
                if inv_key == "inventory_is_stale_rate":
                    row[inv_key] = 1.0
                else:
                    row[inv_key] = ""

        row["return_rate_overall"] = overall_return_rate
        row["promo_efficiency_overall"] = overall_promo_efficiency

        for key in global_feature_cols:
            row[key] = geo_global.get(key, 0.0)

        feature_rows.append(row)

    # Split train/test
    train_rows = [row for row in feature_rows if parse_date(row["Date"]) <= CUTOFF_DATE]
    test_rows = [row for row in feature_rows if parse_date(row["Date"]) >= FORECAST_START]

    write_csv(TRAIN_FEATURES_PATH, train_rows, columns)
    write_csv(TEST_FEATURES_PATH, test_rows, columns)

    # Train target
    sales_source = SALES_PATH
    if USE_CLEANED_SALES and CLEANED_SALES_PATH.exists():
        sales_source = CLEANED_SALES_PATH
        log.append("Using cleaned sales for target.")

    sales_by_date: dict[date, tuple[float, float]] = {}
    with sales_source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = parse_date(row.get("Date", "2012-01-01"))
            if d < TRAIN_START or d > CUTOFF_DATE:
                continue
            revenue = safe_float(row.get("Revenue", ""))
            cogs = safe_float(row.get("COGS", ""))
            sales_by_date[d] = (revenue, cogs)

    with TRAIN_TARGET_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Revenue", "COGS"])
        for d in timeline:
            if d < TRAIN_START or d > CUTOFF_DATE:
                continue
            revenue, cogs = sales_by_date.get(d, (0.0, 0.0))
            writer.writerow([d.isoformat(), f"{revenue:.2f}", f"{cogs:.2f}"])

    # QC: missing conversion rate
    missing_rate = sum(imputed_conversion_flags) / max(1, len(imputed_conversion_flags))
    if missing_rate > MISSING_FLAG_THRESHOLD:
        log.append(f"Warning: conversion_rate_overall missing rate {missing_rate:.2%}")

    # QC: distribution shift on key features
    def calc_mean_std(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
        values = [float(r.get(key, 0.0)) for r in rows]
        if not values:
            return 0.0, 0.0
        avg = mean(values)
        var = mean([(v - avg) ** 2 for v in values])
        return avg, math.sqrt(var)

    train_tail = train_rows[-30:] if len(train_rows) >= 30 else train_rows
    test_head = test_rows[:30] if len(test_rows) >= 30 else test_rows
    for key in ["expected_sessions", "promo_active", "inventory_fill_rate", "conversion_rate_overall"]:
        train_mean, train_std = calc_mean_std(train_tail, key)
        test_mean, _ = calc_mean_std(test_head, key)
        if train_std > 0 and abs(test_mean - train_mean) > 3 * train_std:
            log.append(
                f"Warning: distribution shift on {key} (train_mean={train_mean:.4f}, test_mean={test_mean:.4f})"
            )

    catalog = build_feature_catalog(columns)
    with FEATURE_CATALOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    with PROCESS_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(log))
        f.write("\n")

    print("Train features:", TRAIN_FEATURES_PATH)
    print("Test features:", TEST_FEATURES_PATH)
    print("Train target:", TRAIN_TARGET_PATH)
    print("Feature catalog:", FEATURE_CATALOG_PATH)
    print("Processing log:", PROCESS_LOG_PATH)


if __name__ == "__main__":
    main()
