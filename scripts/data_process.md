# 📋 DATA PROCESSING GUIDE — DATATHON 2026
## Blind Forecast: Train [04/07/2012 → 31/12/2022] | Forecast [01/01/2023 → 01/07/2024]

> 🎯 **Mục tiêu**: Hướng dẫn agent xử lý 15 file CSV thành feature matrix daily, tuân thủ nguyên tắc "no future leakage".

---

## ⚙️ CONFIG & CONSTANTS

```yaml
# Date Configuration
CUTOFF_DATE: "2022-12-31"
TRAIN_START: "2012-07-04"
FORECAST_START: "2023-01-01"
FORECAST_END: "2024-07-01"
FREQUENCY: "daily"

# Lag Windows (cho causal lag features)
CARRYOVER_WINDOWS: [3, 5, 7]  # ngày
DECAY_FACTOR: 0.8             # cho recency weighting

# Thresholds
MIN_OBSERVATIONS_FOR_PROFILE: 30  # số ngày tối thiểu để tính seasonal profile
MAX_FORWARD_FILL_DAYS: 7          # số ngày tối đa forward-fill cho continuous features
MISSING_FLAG_THRESHOLD: 0.1       # flag nếu missing rate > 10%

# Vietnam Holidays (public knowledge, hardcode)
TET_2023: {start: "2023-01-22", end: "2023-01-26"}
TET_2024: {start: "2024-02-10", end: "2024-02-14"}
PUBLIC_HOLIDAYS_VN: [list các ngày lễ cố định]
```

---

## 🗂️ BƯỚC 1: CHUẨN BỊ MASTER TIMELINE

### 1.1 Tạo date range
- Tạo danh sách ngày liên tục từ `TRAIN_START` đến `FORECAST_END`
- Frequency: daily
- Timezone: UTC+7 (hoặc giữ naive datetime, nhất quán toàn pipeline)
- Output: `master_timeline.csv` với cột duy nhất `Date`

### 1.2 Thêm temporal base features (deterministic)
Với mỗi ngày trong timeline, tính:

| Feature | Công thức | Ghi chú |
|---------|-----------|---------|
| `day_of_year` | Ngày thứ mấy trong năm (1-366) | |
| `day_sin`, `day_cos` | sin/cos(2π × day_of_year / 365.25) | Cyclical encoding |
| `week_of_year` | Tuần thứ mấy trong năm (1-52) | |
| `week_sin`, `week_cos` | sin/cos(2π × week_of_year / 52) | |
| `day_of_week` | 0=Monday, 6=Sunday | |
| `is_weekend` | 1 nếu day_of_week ∈ {5,6}, else 0 | |
| `is_month_start` | 1 nếu day ∈ {1,2,3}, else 0 | |
| `is_month_end` | 1 nếu day ≥ 28, else 0 | |
| `is_qtr_end` | 1 nếu month ∈ {3,6,9,12}, else 0 | |
| `month` | 1-12 | |
| `year` | 2012-2024 | |
| `is_tet_period` | 1 nếu Date nằm trong khoảng Tet đã hardcode | |
| `days_to_tet` | Số ngày đến Tet nearest (negative nếu đã qua) | Chỉ tính khi gần Tet |
| `days_since_tet` | Số ngày kể từ Tet nearest | Chỉ tính sau Tet |

✅ **Kiểm tra**: Tất cả features trên chỉ dùng Date, không cần data từ CSV → an toàn cho forecast.

---

## 🗂️ BƯỚC 2: XỬ LÝ MASTER DATA

### 2.1 products.csv
**Input**: products.csv  
**Output**: `product_features.csv` (one row per product_id)

**Xử lý**:
1. Validate: `cogs < price` cho mọi row. Nếu vi phạm → flag `price_error=1`, impute `cogs = price × 0.6`
2. Tính derived features:
   - `margin_rate = (price - cogs) / price`
   - `price_tier = quartile của price trong cùng category` (dùng pd.qcut hoặc manual binning)
3. Encode categorical:
   - `category_enc`: Label encoding của `category`
   - `segment_enc`: Label encoding của `segment`
   - `size_enc`, `color_enc`: Optional, chỉ dùng nếu forecast granular level
4. Export: `product_id`, `margin_rate`, `price_tier`, `category_enc`, `segment_enc`, [+ optional encodings]

### 2.2 customers.csv + geography.csv
**Input**: customers.csv, geography.csv  
**Output**: `geo_customer_features.csv` (one row per zip hoặc region)

**Xử lý**:
1. Merge customers với geography qua `zip`
2. Aggregate lên level `region` (hoặc `city` nếu đủ data):
   - `customer_count`: số khách unique
   - `avg_signup_tenure`: trung bình ngày kể từ signup_date đến cutoff
   - `gender_distribution`: % male/female/unknown
   - `age_group_distribution`: % theo từng nhóm tuổi
   - `acquisition_channel_distribution`: % theo kênh
3. Encode:
   - `region_enc`: Label encoding của region
4. Export: `region`, `region_enc`, [các distribution features dưới dạng separate columns hoặc JSON string]

### 2.3 promotions.csv
**Input**: promotions.csv  
**Output**: 
- `promo_historical_events.csv` (cho train period analysis)
- `promo_seasonal_propensity.csv` (cho forecast proxy)

**Xử lý cho historical**:
1. Filter: chỉ giữ promo có `start_date ≤ CUTOFF_DATE`
2. Expand mỗi promo thành daily rows từ start_date đến end_date
3. Thêm columns:
   - `promo_id`, `promo_type`, `discount_value`, `applicable_category`, `stackable_flag`, `min_order_value`
4. Export: `promo_historical_events.csv` với columns: `Date`, `promo_id`, `promo_type`, `discount_value`, `applicable_category`, `stackable_flag`

**Xử lý cho forecast proxy**:
1. Group historical events theo `(month, applicable_category)`
2. Tính `promo_propensity = count(promo_active_days) / count(total_days)` cho mỗi group
3. Export: `promo_seasonal_propensity.csv` với columns: `month`, `applicable_category`, `promo_propensity`

✅ **Lưu ý**: Không tạo future promo events nếu không có public plan. Forecast period sẽ dùng propensity làm prior, không dùng actual promo flags.

---

## 🗂️ BƯỚC 3: TRÍCH XUẤT HISTORICAL PATTERNS (Train-only)

### 3.1 Return Rate theo Category
**Input**: returns.csv, order_items.csv, products.csv  
**Output**: `category_return_rates.csv`

**Xử lý**:
1. Filter returns: chỉ giữ `return_date ≤ CUTOFF_DATE`
2. Merge returns → order_items (qua order_id, product_id) → products (qua product_id để có category)
3. Aggregate theo category:
   - `total_returned = sum(return_quantity)`
   - `total_sold = sum(quantity)` từ order_items trong cùng period
   - `return_rate = total_returned / (total_sold + 1e-8)`
4. Export: `category`, `return_rate`

### 3.2 Seasonal Traffic Profile
**Input**: web_traffic.csv  
**Output**: `traffic_seasonal_profile.csv`

**Xử lý**:
1. Filter: chỉ giữ `date ≤ CUTOFF_DATE`
2. Extract: `dow = day_of_week`, `month` từ date
3. Groupby `(dow, month)`:
   - `sessions_mean = mean(sessions)`
   - `sessions_std = std(sessions)`
   - `n_obs = count`
4. Filter: chỉ giữ rows có `n_obs ≥ MIN_OBSERVATIONS_FOR_PROFILE`
5. Tính `volatility_index = sessions_std / (sessions_mean + 1e-8)`
6. Export: `dow`, `month`, `sessions_mean`, `sessions_std`, `volatility_index`

### 3.3 Promo Efficiency theo Category
**Input**: promotions.csv, order_items.csv, orders.csv, products.csv  
**Output**: `category_promo_efficiency.csv`

**Xử lý**:
1. Filter orders: `order_date ≤ CUTOFF_DATE`
2. Merge order_items → orders → products → promotions (left join, giữ cả rows không có promo)
3. Filter: chỉ giữ rows có `promo_id not null`
4. Groupby `category`:
   - `total_discount = sum(discount_amount)`
   - `total_revenue = sum(quantity × unit_price)`
   - `efficiency = total_revenue / (total_discount + 1e-8)`
5. Export: `category`, `promo_efficiency`

### 3.4 Conversion Proxy theo Channel
**Input**: orders.csv, web_traffic.csv  
**Output**: `channel_conversion_rates.csv`

**Xử lý**:
1. Aggregate orders theo `order_source` + `month`:
   - `order_count = count(order_id)`
2. Aggregate web_traffic theo `traffic_source` + `month`:
   - `session_count = sum(sessions)`
3. Map `order_source` ↔ `traffic_source` (cần mapping table hoặc heuristic: cùng tên hoặc manual mapping)
4. Join và tính: `conversion_rate = order_count / (session_count + 1e-8)`
5. Export: `channel`, `month`, `conversion_rate`

✅ **Lưu ý**: Tất cả patterns trên được tính **một lần** trên train period, export làm lookup table. Khi forecast, merge vào feature matrix dưới dạng constant theo key (category, channel, v.v.).

---

## 🗂️ BƯỚC 4: XỬ LÝ OPERATIONAL DATA

### 4.1 web_traffic.csv → Daily Features
**Input**: web_traffic.csv  
**Output**: `traffic_daily_features.csv` (one row per date)

**Xử lý cho historical period**:
1. Groupby `date`:
   - `sessions_total = sum(sessions)`
   - `unique_visitors_total = sum(unique_visitors)`
   - `page_views_total = sum(page_views)`
   - `bounce_rate_avg = mean(bounce_rate)`
   - `avg_duration_avg = mean(avg_session_duration_sec)`
2. Export: `date`, [các aggregated columns]

**Xử lý cho forecast period** (khi không có actual traffic):
1. Với mỗi date trong forecast range:
   - Extract `dow`, `month`
   - Lookup `traffic_seasonal_profile.csv` bằng `(dow, month)`
   - Nếu found: dùng `sessions_mean` làm `expected_sessions`
   - Nếu not found: fallback to global mean của sessions từ historical
   - Dùng `volatility_index` làm `traffic_uncertainty`
2. Export: `date`, `expected_sessions`, `traffic_uncertainty`, `is_profile_based=1`

### 4.2 inventory.csv → Daily Proxy
**Input**: inventory.csv  
**Output**: `inventory_daily_proxy.csv` (one row per date × product_id hoặc category)

**Xử lý**:
1. Vì inventory là monthly snapshot, quyết định granularity:
   - Option A (product-level): giữ nguyên product_id, resample về daily
   - Option B (category-level): aggregate lên category trước, rồi resample → recommended để giảm noise
2. Nếu Option B:
   - Merge inventory → products để có category
   - Groupby `(snapshot_date, category)`:
     - `fill_rate_avg = mean(fill_rate)`
     - `stockout_freq = mean(stockout_flag)`
     - `sell_through_avg = mean(sell_through_rate)`
     - `reorder_freq = mean(reorder_flag)`
3. Resample về daily:
   - Với mỗi category, forward-fill các metrics từ snapshot_date đến trước snapshot tiếp theo
   - Thêm column `days_since_snapshot`: số ngày kể từ last snapshot date
4. Export: `date`, `category`, `fill_rate_daily`, `stockout_freq_daily`, `sell_through_daily`, `reorder_freq_daily`, `days_since_snapshot`

✅ **Lưu ý**: Forward-fill chỉ áp dụng tối đa `MAX_FORWARD_FILL_DAYS`. Sau đó, revert về historical mean của category và flag `is_stale=1`.

---

## 🗂️ BƯỚC 5: XÂY DỰNG CAUSAL LAG FEATURES

### 5.1 Promo State Features
**Input**: master_timeline.csv, promo_historical_events.csv (cho train), promo_seasonal_propensity.csv (cho forecast)

**Xử lý cho train period**:
1. Với mỗi date trong train range:
   - `promo_active = 1` nếu có bất kỳ promo nào active tại date đó (từ promo_historical_events), else 0
   - `promo_intensity = mean(discount_value)` của các promo active tại date đó (chỉ tính percentage type, hoặc convert fixed về % equivalent)
   - `promo_stackable = max(stackable_flag)` của các promo active
2. Tính carryover windows:
   - Với mỗi window in `CARRYOVER_WINDOWS`:
     - `promo_carryover_{window}d = sum(promo_active[t-k] for k in 1..window)`
3. Tính recency decay:
   - `promo_recency_weighted = sum(promo_active[t-k] × DECAY_FACTOR^k for k in 0..7)`

**Xử lý cho forecast period**:
- Option Conservative: set `promo_active = 0`, `promo_intensity = 0`, carryover = 0
- Option Scenario: dùng `promo_seasonal_propensity` để sample hoặc set `promo_active = 1` với probability = propensity
- Carryover và recency: tính tương tự nếu promo_active=1, else = 0

**Export columns**: `promo_active`, `promo_intensity`, `promo_stackable`, `promo_carryover_3d`, `promo_carryover_5d`, `promo_carryover_7d`, `promo_recency_weighted`

### 5.2 Event/Holiday Lag Features
**Input**: master_timeline.csv, hardcoded holiday dates

**Xử lý**:
1. Với mỗi date:
   - `pre_event_window = max(0, event_start_date - date)` nếu date < event_start
   - `post_event_window = max(0, date - event_end_date)` nếu date > event_end
   - `event_recency_weight = 1 / (1 + min(pre_event_window, post_event_window))`
2. Áp dụng cho Tet và các public holidays quan trọng

**Export columns**: `days_to_tet`, `days_since_tet`, `tet_recency_weight`, [+ tương tự cho holidays khác nếu cần]

### 5.3 Traffic Recency Proxy
**Input**: traffic_daily_features.csv (historical) hoặc traffic_seasonal_profile (forecast)

**Xử lý**:
1. Với mỗi date:
   - `expected_sessions` (từ Bước 4.1)
   - `traffic_carryover_2d = expected_sessions[t-1] + expected_sessions[t-2]` (nếu có historical) hoặc dùng profile mean × 2
   - `traffic_recency_weighted = expected_sessions[t] × 1.0 + expected_sessions[t-1] × DECAY_FACTOR + ...`

**Export columns**: `expected_sessions`, `traffic_carryover_2d`, `traffic_recency_weighted`, `traffic_uncertainty`

---

## 🗂️ BƯỚC 6: MERGE & ALIGN TO MASTER TIMELINE

### 6.1 Chuẩn bị keys cho merge
- Tất cả tables phải có column `Date` (datetime) để join với master_timeline
- Product/category-level features: thêm column `category` hoặc `product_id` để merge sau
- Geo/customer features: merge qua `region` hoặc giữ làm global constants

### 6.2 Merge sequence (left-join vào master_timeline)
```
master_timeline
   ↓ left-join
product_features (via product_id) → nếu forecast product-level, else skip
   ↓ left-join  
geo_customer_features (via region) → aggregate lên global nếu cần
   ↓ left-join
promo_state_features (via Date)
   ↓ left-join
traffic_daily_features (via Date)
   ↓ left-join
inventory_daily_proxy (via Date + category)
   ↓ left-join
category_return_rates (via category) → constant per category
   ↓ left-join
category_promo_efficiency (via category) → constant per category
   ↓ left-join
channel_conversion_rates (via channel + month) → optional
```

### 6.3 Xử lý missing values sau merge
- **Categorical**: forward-fill vô hạn (ví dụ: category_enc không đổi)
- **Continuous có pattern** (sessions, fill_rate): forward-fill tối đa `MAX_FORWARD_FILL_DAYS`, sau đó revert to historical mean của category/global
- **State features** (promo_active, days_since): tính bằng logic, không interpolate
- **Luôn thêm flag**: `is_imputed_{feature_name} = 1` nếu giá trị được fill/reverted

### 6.4 Final feature matrix structure
```
Columns:
- Date (key)
- Temporal features (day_sin, is_weekend, is_tet_period, ...)
- Product static features (nếu product-level forecast)
- Geo/customer aggregates (nếu region-level forecast)
- Promo state & lag features
- Traffic proxy & lag features
- Inventory proxy features
- Historical pattern lookups (return_rate, promo_efficiency, ...)
- Missing flags (is_imputed_*)

Rows:
- Train: Dates từ TRAIN_START đến CUTOFF_DATE
- Test: Dates từ FORECAST_START đến FORECAST_END
```

---

## 🗂️ BƯỚC 7: QUALITY CONTROL & EXPORT

### 7.1 Kiểm tra leakage
- [ ] Audit từng feature column: nguồn gốc có phải train-only, calendar, hoặc master data không?
- [ ] Chạy script kiểm tra: không có feature nào truy cập data > CUTOFF_DATE
- [ ] Verify: sales.csv chỉ xuất hiện ở target columns (Revenue, COGS), không trong features

### 7.2 Kiểm tra distribution shift
- Tính mean/std của critical features cho:
  - Last 30 days of train period
  - First 30 days of forecast period
- Alert nếu: `|mean_test - mean_train| > 3 × std_train` cho bất kỳ feature quan trọng nào

### 7.3 Kiểm tra missingness
- Tính % missing cho mỗi feature trong test period
- Alert nếu: missing_rate > `MISSING_FLAG_THRESHOLD` cho features quan trọng

### 7.4 Export files
```
output/
├── train_features.csv      # Dates: TRAIN_START → CUTOFF_DATE
├── train_target.csv        # Columns: Date, Revenue, COGS (từ sales.csv)
├── test_features.csv       # Dates: FORECAST_START → FORECAST_END (no target)
├── feature_catalog.json    # Metadata: name, source, forecast_compatible, assumptions
└── processing_log.txt      # Audit trail: steps executed, warnings, errors
```

### 7.5 Feature catalog template (JSON)
```json
{
  "feature_name": "promo_recency_weighted",
  "source": "promotions.csv + causal lag logic",
  "granularity": "daily",
  "forecast_compatible": true,
  "computation_method": "sum(promo_active[t-k] * 0.8^k for k in 0..7)",
  "assumptions": [
    "Decay factor 0.8 phù hợp với promo impact trong fashion e-commerce",
    "Future promo unknown → default promo_active=0 cho forecast"
  ],
  "missing_handling": "forward-fill max 7 days, then revert to 0",
  "validation_note": "Feature importance rank: #5 trên pseudo-cutoff validation"
}
```

---

## 🚨 TROUBLESHOOTING GUIDE

| Vấn đề | Dấu hiệu | Hành động khắc phục |
|--------|---------|-------------------|
| Leakage phát hiện | Feature importance của biến "future" > 0 | Review pipeline, xóa feature vi phạm, re-run |
| Missing rate cao trong test | >10% missing cho critical features | Review projection logic, thêm fallback values, document assumption |
| Distribution drift lớn | Mean shift > 3σ giữa train end và test start | Xem xét recalibrate projection, hoặc thêm uncertainty features |
| Performance validation tệ | WAPE trên pseudo-cutoff > baseline đơn giản | Review feature set, giảm complexity, tăng regularization |
| Merge error | NaN xuất hiện sau join | Kiểm tra keys, data types, timezone consistency |

---

## ✅ FINAL CHECKLIST CHO AGENT

```
[ ] Master timeline created: [TRAIN_START, FORECAST_END] daily, no gaps
[ ] Temporal features added: all deterministic, calendar-based only
[ ] Master data processed: products, customers, promotions, geography → static features
[ ] Historical patterns extracted: return rates, traffic profile, promo efficiency (train-only)
[ ] Operational data projected: traffic seasonal proxy, inventory forward-fill with staleness flag
[ ] Causal lag features built: promo state, recency decay, event windows (no target lag)
[ ] All features merged to master timeline: daily granularity, consistent keys
[ ] Missing values handled: forward-fill limits, fallback to mean, imputation flags added
[ ] Leakage audit passed: no feature uses data > CUTOFF_DATE
[ ] Distribution check passed: no critical feature drift > 3σ
[ ] Feature catalog documented: source, method, assumptions for each feature
[ ] Export completed: train_features, train_target, test_features, catalog, log
```

---

> 📌 **Lưu ý cuối cùng cho agent**: 
> - Luôn ưu tiên **conservative assumptions** khi xử lý unknowns trong forecast period
> - Document mọi giả định trong `feature_catalog.json` để traceability
> - Nếu gặp ambiguity, chọn option an toàn hơn (ví dụ: giả định no promo thay vì guess promo plan)
> - Pipeline phải reproducible: chạy lại từ raw data → output giống hệt

*Tài liệu này đủ để agent thực hiện end-to-end data processing mà không cần code mẫu. Nếu cần clarification ở bước nào, agent nên hỏi trước khi proceed.*