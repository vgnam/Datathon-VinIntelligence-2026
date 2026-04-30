"""
pipeline_selected_v1_v2.py
============================================================
Pipeline hoàn chỉnh, chạy một lần để sinh:
  • train_features_selected.csv   (V1 – 10 features)
  • test_features_selected.csv    (V1 – 10 features)
  • train_features_selected_v2.csv (V2 – 15 features)
  • test_features_selected_v2.csv  (V2 – 15 features)

Thứ tự thực hiện:
  1. process_data_pipeline.py          → raw features + target
  2. fix_test_features.py              → fixed features (auto gọi reduce + historical)
  3. reduce_historical_features.py     → balanced features (35 cols)
  4. sync_promo_train.py               → bổ sung promo_seasonal_prob vào train
  5. fix_missing_cols.py               → bảo đảm month / day_of_week đầy đủ
  6. Tạo Selected V1 & V2 in-place     → output/ train/test selected
============================================================
"""

import subprocess
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN
# ------------------------------------------------------------------
BASE_DIR = Path(r'D:\Datathon-2026')
SCRIPT_DIR = BASE_DIR / 'scripts'
OUTPUT_DIR = BASE_DIR / 'output'

PYTHON = sys.executable  # dùng đúng interpreter hiện tại


def run_script(name: str) -> None:
    """Chạy một script trong thư mục scripts/ và in stdout."""
    path = SCRIPT_DIR / name
    print(f"\n{'='*70}")
    print(f"[RUN] {name}")
    print('='*70)
    result = subprocess.run(
        [PYTHON, str(path)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )
    # In toàn bộ stdout (hoặc 2000 ký tự cuối nếu quá dài)
    out = result.stdout
    if len(out) > 2000:
        print(out[-2000:])
    else:
        print(out)
    if result.returncode != 0:
        err = result.stderr
        print(f"[ERROR] in {name}:\n{err[-1000:] if len(err) > 1000 else err}")
        raise RuntimeError(f"Script {name} failed with code {result.returncode}")
    print(f"[OK] {name} completed.\n")


# ------------------------------------------------------------------
# BƯỚC 6 – TẠO SELECTED V1 & V2
# ------------------------------------------------------------------
def build_selected_v1_v2() -> None:
    print(f"\n{'='*70}")
    print("BUILDING SELECTED FEATURES V1 & V2")
    print('='*70)

    # --- Đọc dữ liệu nguồn -------------------------------------------------
    train_balanced = pd.read_csv(
        OUTPUT_DIR / 'train_features_balanced.csv', parse_dates=['Date']
    ).sort_values('Date').reset_index(drop=True)
    test_balanced = pd.read_csv(
        OUTPUT_DIR / 'test_features_balanced.csv', parse_dates=['Date']
    ).sort_values('Date').reset_index(drop=True)
    train_target = pd.read_csv(
        OUTPUT_DIR / 'train_target.csv', parse_dates=['Date']
    ).sort_values('Date').reset_index(drop=True)

    # Chuẩn hóa tên target
    if 'net_revenue' in train_target.columns and 'Revenue' not in train_target.columns:
        train_target = train_target.rename(columns={'net_revenue': 'Revenue'})

    print(f"Balanced train: {train_balanced.shape}")
    print(f"Balanced test : {test_balanced.shape}")

    # ======================================================================
    # V1 – 8 features (loại bỏ 4 cột yếu / không ổn định ở test)
    # Bỏ: inventory_sell_through, promo_seasonal_prob,
    #      inventory_days_since_snapshot, traffic_uncertainty
    # ======================================================================
    v1_features = [
        'hist_monthday_revenue_mean',
        'hist_monthday_cogs_mean',
        'day_cos',
        'hist_yoy_cogs_growth',
        'day_sin',
        'hist_yoy_revenue_growth',
        'expected_sessions',
        'month',
    ]

    # Kiểm tra tồn tại
    missing_train_v1 = [c for c in v1_features if c not in train_balanced.columns]
    missing_test_v1  = [c for c in v1_features if c not in test_balanced.columns]
    if missing_train_v1:
        print(f"⚠️  V1 missing in train: {missing_train_v1}")
    if missing_test_v1:
        print(f"⚠️  V1 missing in test : {missing_test_v1}")

    train_v1 = train_balanced[['Date'] + v1_features].copy()
    test_v1  = test_balanced[['Date'] + v1_features].copy()

    # Gắn target vào train
    train_v1 = train_v1.merge(train_target[['Date', 'Revenue', 'COGS']], on='Date', how='left')

    # Lưu V1
    train_v1.to_csv(OUTPUT_DIR / 'train_features_selected.csv', index=False)
    test_v1.to_csv(OUTPUT_DIR / 'test_features_selected.csv', index=False)
    print(f"\n[SAVED] V1  – train: {train_v1.shape}, test: {test_v1.shape}")
    for i, f in enumerate(v1_features, 1):
        print(f"   {i:2d}. {f}")

    # ======================================================================
    # V2 – 15 features (recency + calendar + promo fix)
    # ======================================================================
    # Tính thêm hist_monthday_*_mean_recent từ 2019-2022
    train_with_target = train_balanced.merge(
        train_target[['Date', 'Revenue', 'COGS']], on='Date', how='left'
    )
    train_with_target['year'] = train_with_target['Date'].dt.year
    train_with_target['month_day'] = train_with_target['Date'].dt.strftime('%m-%d')

    recent_train = train_with_target[train_with_target['year'] >= 2019].copy()
    hist_recent_rev = (
        recent_train.groupby('month_day')['Revenue']
        .mean().reset_index()
        .rename(columns={'Revenue': 'hist_monthday_revenue_mean_recent'})
    )
    hist_recent_cogs = (
        recent_train.groupby('month_day')['COGS']
        .mean().reset_index()
        .rename(columns={'COGS': 'hist_monthday_cogs_mean_recent'})
    )

    # Tạo month_day cho balanced trước khi merge
    train_balanced['month_day'] = train_balanced['Date'].dt.strftime('%m-%d')
    test_balanced['month_day'] = test_balanced['Date'].dt.strftime('%m-%d')

    # Merge vào balanced
    train_balanced = train_balanced.merge(hist_recent_rev, on='month_day', how='left')
    train_balanced = train_balanced.merge(hist_recent_cogs, on='month_day', how='left')
    test_balanced = test_balanced.merge(hist_recent_rev, on='month_day', how='left')
    test_balanced = test_balanced.merge(hist_recent_cogs, on='month_day', how='left')

    # Fill NaN recent bằng giá trị mean toàn kỳ
    for col in ['hist_monthday_revenue_mean_recent', 'hist_monthday_cogs_mean_recent']:
        train_balanced[col] = train_balanced[col].fillna(train_balanced[col.replace('_recent', '')])
        test_balanced[col]  = test_balanced[col].fillna(test_balanced[col.replace('_recent', '')])

    v2_features = [
        'hist_monthday_revenue_mean',
        'hist_monthday_revenue_mean_recent',   # NEW
        'hist_monthday_cogs_mean',
        'hist_monthday_cogs_mean_recent',      # NEW
        'day_cos',
        'hist_yoy_cogs_growth',
        'day_sin',
        'hist_yoy_revenue_growth',
        'expected_sessions',
        'month',
        'is_tet_period',                       # NEW
        'days_to_tet',                         # NEW
        'traffic_uncertainty',
        'promo_seasonal_prob',                 # FIXED
        'promo_monthly_prob',                  # NEW
    ]

    missing_train_v2 = [c for c in v2_features if c not in train_balanced.columns]
    missing_test_v2  = [c for c in v2_features if c not in test_balanced.columns]
    if missing_train_v2:
        print(f"\n⚠️  V2 missing in train: {missing_train_v2}")
    if missing_test_v2:
        print(f"⚠️  V2 missing in test : {missing_test_v2}")

    train_v2 = train_balanced[['Date'] + v2_features].copy()
    test_v2  = test_balanced[['Date'] + v2_features].copy()
    train_v2 = train_v2.merge(train_target[['Date', 'Revenue', 'COGS']], on='Date', how='left')

    # Lưu V2
    train_v2.to_csv(OUTPUT_DIR / 'train_features_selected_v2.csv', index=False)
    test_v2.to_csv(OUTPUT_DIR / 'test_features_selected_v2.csv', index=False)
    print(f"\n[SAVED] V2  – train: {train_v2.shape}, test: {test_v2.shape}")
    for i, f in enumerate(v2_features, 1):
        print(f"   {i:2d}. {f}")

    # ======================================================================
    # TÓM TẮT
    # ======================================================================
    print(f"\n{'='*70}")
    print("[DONE] PIPELINE HOAN TAT")
    print('='*70)
    print("Output files:")
    print(f"  • {OUTPUT_DIR / 'train_features_selected.csv'}")
    print(f"  • {OUTPUT_DIR / 'test_features_selected.csv'}")
    print(f"  • {OUTPUT_DIR / 'train_features_selected_v2.csv'}")
    print(f"  • {OUTPUT_DIR / 'test_features_selected_v2.csv'}")
    print('='*70)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main() -> None:
    print(f"\n{'#'*70}")
    print("# PIPELINE: SELECTED FEATURES V1 & V2")
    print(f"# Base dir: {BASE_DIR}")
    print(f"# Python  : {PYTHON}")
    print('#' * 70)

    # Bước 1 – Raw features + target
    run_script('process_data_pipeline.py')

    # Bước 2 – Fix test features (tự gọi reduce_features + add_historical_features bên trong)
    run_script('fix_test_features.py')

    # Bước 3 – Thu gọn historical features → balanced
    run_script('reduce_historical_features.py')

    # Bước 4 – Đồng bộ promo_seasonal_prob vào train balanced/enhanced
    run_script('sync_promo_train.py')

    # Bước 5 – Bảo đảm month / day_of_week không thiếu
    run_script('fix_missing_cols.py')

    # Bước 6 – Tạo Selected V1 & V2
    build_selected_v1_v2()


if __name__ == '__main__':
    main()
