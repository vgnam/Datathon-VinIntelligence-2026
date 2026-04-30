import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Doc du lieu
features = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv')
target = pd.read_csv(r'D:\Datathon-2026\output\train_target.csv')
test_features = pd.read_csv(r'D:\Datathon-2026\output\test_features_balanced.csv')

# Gop du lieu training
df = pd.merge(features, target, on='Date')

# Dua tren ket qua kiem dinh thong ke, chon top feature co Composite_Score > 0.1
# Danh sach 12 feature quan trong nhat
selected_features = [
    'hist_monthday_revenue_mean',
    'hist_monthday_cogs_mean',
    'day_cos',
    'hist_yoy_cogs_growth',
    'day_sin',
    'hist_yoy_revenue_growth',
    'expected_sessions',
    'month',
    'inventory_sell_through',
    'inventory_days_since_snapshot',
    'traffic_uncertainty',
    'promo_intensity'
]

# Them target
output_cols = ['Date'] + selected_features + ['Revenue', 'COGS']
train_selected = df[output_cols].copy()
test_selected = test_features[['Date'] + selected_features].copy()

# Luu ra CSV
train_path = r'D:\Datathon-2026\output\train_features_selected.csv'
test_path = r'D:\Datathon-2026\output\test_features_selected.csv'
train_selected.to_csv(train_path, index=False)
test_selected.to_csv(test_path, index=False)

print("=" * 70)
print("DA CHON FEATURE VA LUU FILE CSV")
print("=" * 70)
print(f"So feature duoc chon: {len(selected_features)}")
print(f"File train: {train_path}")
print(f"File test:  {test_path}")
print("\nCac feature duoc chon:")
for i, f in enumerate(selected_features, 1):
    print(f"  {i:2d}. {f}")

# ===========================
# TRAIN MODEL
# ===========================
print("\n" + "=" * 70)
print("TRAIN MODEL VOI FEATURE DA CHON")
print("=" * 70)

X = train_selected[selected_features]
y = train_selected['Revenue']

# Chia train/validation theo thoi gian (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# 1. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

# Danh gia
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{model_name}:")
    print(f"  MAE  = {mae:,.2f}")
    print(f"  RMSE = {rmse:,.2f}")
    print(f"  R2   = {r2:.4f}")
    print(f"  MAPE = {mape:.2f}%")
    return mae, rmse, r2, mape

print("\n" + "-" * 70)
print("KET QUA VALIDATION")
print("-" * 70)

mae_ridge, rmse_ridge, r2_ridge, mape_ridge = evaluate(y_val, y_pred_ridge, "Ridge Regression")
mae_rf, rmse_rf, r2_rf, mape_rf = evaluate(y_val, y_pred_rf, "Random Forest")

# Feature importance tu Random Forest
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "-" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)
print(importance_df.to_string(index=False))

# Chon model tot nhat va du doan test set
print("\n" + "=" * 70)
print("DU DOAN TEST SET")
print("=" * 70)

if mae_rf <= mae_ridge:
    best_model = rf
    best_name = "Random Forest"
else:
    best_model = ridge
    best_name = "Ridge Regression"

print(f"Model tot nhat: {best_name}")

# Du doan tren test set
X_test = test_selected[selected_features]
test_pred = best_model.predict(X_test)

# Luu submission
submission = pd.DataFrame({
    'Date': test_selected['Date'],
    'Revenue': test_pred
})
sub_path = r'D:\Datathon-2026\output\submission_selected_features.csv'
submission.to_csv(sub_path, index=False)
print(f"Da luu du doan test tai: {sub_path}")

# Luu model tot nhat
import joblib
model_path = r'D:\Datathon-2026\output\best_model_selected.pkl'
joblib.dump(best_model, model_path)
print(f"Da luu model tai: {model_path}")

print("\n" + "=" * 70)
print("TOM TAT")
print("=" * 70)
print(f"- So feature chon: {len(selected_features)}")
print(f"- Model tot nhat: {best_name}")
print(f"- Validation MAE:  {min(mae_ridge, mae_rf):,.2f}")
print(f"- Validation RMSE: {min(rmse_ridge, rmse_rf):,.2f}")
print(f"- Validation R2:   {max(r2_ridge, r2_rf):.4f}")
print("=" * 70)
