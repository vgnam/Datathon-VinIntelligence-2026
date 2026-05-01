# Time-Series Sales Forecasting — Tree Ensemble + Stacking

Stacked ensemble of tree-based regressors (Random Forest, LightGBM, XGBoost) with Time-Series Cross-Validation, meta-learned by Gradient Boosting (GBM). Optional Residual ElasticNet layer on top of stacking.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build features from raw data

```bash
cd data
python prepare_features.py
```

This reads raw CSVs in `data/analytical/`, `data/operational/`, and `data/transaction/`, then writes:
- `output/train_features.csv`
- `output/test_features.csv`
- `output/train_target.csv`

### 3. (Optional) Compute feature importances

```bash
python feature_importance.py
```

Exports to `output/`:
- `feature_importance.csv` — native tree importance (RF / LGB / XGB) + SHAP
- `feature_importance.png` — bar chart of top features
- `shap_*.png` — SHAP beeswarm plots per model & target

### 4. Train the ensemble

```bash
python train_ensemble_tscv.py
```

Outputs are written to `data/forecast/`:
- `stacking3_gbm_tscv_submission.csv` — stacking of 3 tree models via GBM
- `stacking3_gbm_resen_tscv_submission.csv` — stacking + Residual ElasticNet
- Diagnostic plots (`plot_*.png`)

## Pipeline Overview

```
Raw Data -> prepare_features.py -> output/*.csv -> train_ensemble_tscv.py -> forecast/*.csv

Base models (OOF via TimeSeriesSplit):
+-------------+   +-------------+   +-------------+
|  Random     |   |  LightGBM   |   |  XGBoost    |
|  Forest     |   |  (log)      |   |  (log)      |
+------+------+   +------+------+   +------+------+
       |                 |                 |
       +-----------------+-----------------+
                         |
                         v
              +-------------------+
              |  GBM Meta-Learner |
              +---------+---------+
                        |
        +---------------+---------------+
        |                               |
        v                               v
 stacking3_gbm_          stacking3_gbm_resen_
 tscv_submission.csv     tscv_submission.csv
```

All base models are trained with **5-fold TimeSeriesSplit** (OOF predictions), then GBM fits on the out-of-fold matrix. The optional ResEN variant fits ElasticNet on `log(actual) - log(stacking_pred)` to capture residual patterns.

## Data Layout

```
.
├── data/
│   ├── prepare_features.py         # feature engineering from raw data
│   ├── train_ensemble_tscv.py      # main training script
│   ├── feature_importance.py       # compute feature importances
│   ├── analytical/
│   │   ├── sales.csv               # historical daily sales (2012-2022)
│   │   └── sample_submission.csv   # submission template (2023-2024 dates)
│   ├── operational/
│   │   ├── web_traffic.csv         # daily web sessions
│   │   └── inventory.csv           # monthly inventory snapshots
│   ├── master/
│   │   ├── promotions.csv          # promotion calendar
│   │   ├── customers.csv           # customer master
│   │   ├── products.csv            # product master
│   │   └── geography.csv           # geographic master
│   ├── transaction/
│   │   ├── orders.csv              # order headers
│   │   ├── order_items.csv         # line items
│   │   ├── returns.csv             # returns log
│   │   ├── payments.csv            # payment transactions
│   │   ├── reviews.csv             # product reviews
│   │   └── shipments.csv           # shipment records
│   └── forecast/                   # generated outputs
├── output/                         # pre-computed features
│   ├── train_features.csv
│   ├── test_features.csv
│   └── train_target.csv
└── requirements.txt
```

## Reproducing from scratch

```bash
cd data
python prepare_features.py     # step 1: build features
python feature_importance.py   # step 2: compute native + SHAP feature importance
python train_ensemble_tscv.py  # step 3: train & predict
```

## Feature Groups

| Group | Features | Rationale |
|---|---|---|
| **Calendar** | `month`, `day`, `weekday`, `dayofyear`, `year`, `quarter`, `day_sin`, `day_cos` | Always available for any date; capture seasonality |
| **Trend** | `days_since_start`, `days_since_start_sq`, `year_sq` | Non-linear temporal trend |
| **Seasonal baselines** | `hist_monthday_*_mean`, `hist_month_*_mean`, `hist_weekday_*_mean`, `hist_monthday_*_mean_2y` | Historical averages by (month,day), month, weekday; 2y recent variants weight recent regime more |
| **Autoregressive** | `lag_1y_revenue`, `lag_1y_cogs` | Same-day-last-year value (fallback to recent monthday mean for 2024) |
| **YoY growth** | `hist_yoy_revenue_growth`, `hist_yoy_cogs_growth`, `hist_yoy_rev_growth_3y`, `hist_yoy_cogs_growth_3y` | Annual growth momentum; 3y-smoothed for stability |
| **Structural (annual)** | `annual_unique_customers`, `annual_total_orders`, `annual_new_customer_rate`, `annual_aov`, `annual_discount_intensity`, `annual_return_rate`, `annual_sessions`, `annual_sessions_growth` | Derived yearly from orders/traffic/returns 2012-2022; extrapolated to 2023-2024 via 3-year mean. Captures market size, customer behavior, pricing pressure without daily concurrent leakage |

## Key Design Choices

| Decision | Rationale |
|---|---|
| Log-transform targets | Revenue/COGS are right-skewed; log stabilizes variance |
| TimeSeriesSplit(5) | Preserves temporal ordering; no lookahead leakage |
| GBM meta-learner | Non-linear interaction among base-model predictions |
| Residual ElasticNet | Catches linear residual patterns that tree stacking misses |
| `COGS <= 0.995 * Revenue` | Business constraint enforced post-prediction |
| No concurrent operational data | Daily traffic, orders, inventory, promotions are **excluded** from features because they do not exist for 2023-2024 and cause severe overfit (e.g. `daily_order_value` is nearly Revenue itself) |
| Annual structural metrics only | Yearly aggregates (customers, AOV, return rate, sessions) are derived from 2012-2022 and extrapolated conservatively. They capture regime-shift drivers without daily leakage |
| No hardcoded regime flag | Removed `is_post_2019`; regime shifts are learned from trend features and structural metrics |
| Target leakage prevention | `train_features.csv` never contains `Revenue`/`COGS`; targets live in `train_target.csv` and are merged only at training time |

## Metrics

Primary metric: **MAPE** (Mean Absolute Percentage Error)  
Secondary: **R2** for sanity-checking fold quality

## License

MIT
