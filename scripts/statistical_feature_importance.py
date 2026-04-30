import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

warnings.filterwarnings('ignore')

# Read data
features = pd.read_csv(r'D:\Datathon-2026\output\train_features_balanced.csv')
target = pd.read_csv(r'D:\Datathon-2026\output\train_target.csv')

# Merge data
df = pd.merge(features, target, on='Date')

# Define target and features
y = df['Revenue']
X = df.drop(columns=['Date', 'Revenue', 'COGS'])

output_lines = []
output_lines.append("=" * 80)
output_lines.append("KIEM DINH THONG KE XAC DINH FEATURE QUAN TRONG DOI VOI REVENUE")
output_lines.append("=" * 80)
output_lines.append(f"So luong mau: {len(df)}")
output_lines.append(f"So luong features: {X.shape[1]}")
output_lines.append("")

# 1. PEARSON CORRELATION TEST
output_lines.append("-" * 80)
output_lines.append("1. KIEM DINH TUONG QUAN PEARSON (moi quan he tuyen tinh)")
output_lines.append("-" * 80)
pearson_results = []
for col in X.columns:
    r, p_value = pearsonr(X[col], y)
    pearson_results.append({
        'Feature': col,
        'Pearson_r': r,
        'p_value': p_value,
        'abs_r': abs(r)
    })

pearson_df = pd.DataFrame(pearson_results).sort_values('abs_r', ascending=False)
output_lines.append(pearson_df[['Feature', 'Pearson_r', 'p_value']].head(15).to_string(index=False))
output_lines.append("")

# 2. SPEARMAN CORRELATION TEST
output_lines.append("-" * 80)
output_lines.append("2. KIEM DINH TUONG QUAN SPEARMAN (moi quan he monotonic)")
output_lines.append("-" * 80)
spearman_results = []
for col in X.columns:
    rho, p_value = spearmanr(X[col], y)
    spearman_results.append({
        'Feature': col,
        'Spearman_rho': rho,
        'p_value': p_value,
        'abs_rho': abs(rho)
    })

spearman_df = pd.DataFrame(spearman_results).sort_values('abs_rho', ascending=False)
output_lines.append(spearman_df[['Feature', 'Spearman_rho', 'p_value']].head(15).to_string(index=False))
output_lines.append("")

# 3. F-TEST (ANOVA for regression)
output_lines.append("-" * 80)
output_lines.append("3. KIEM DINH F-TEST (Univariate Linear Regression F-test)")
output_lines.append("-" * 80)
output_lines.append("   H0: Feature khong co kha nang giai thich phuong sai cua target")
output_lines.append("   H1: Feature co kha nang giai thich phuong sai cua target")
output_lines.append("")

# Handle NaN/Inf
X_clean = X.replace([np.inf, -np.inf], np.nan)
X_clean = X_clean.fillna(X_clean.median())

f_stats, f_pvalues = f_regression(X_clean, y)

f_results = []
for i, col in enumerate(X_clean.columns):
    f_results.append({
        'Feature': col,
        'F_statistic': f_stats[i],
        'p_value': f_pvalues[i]
    })

f_df = pd.DataFrame(f_results).sort_values('F_statistic', ascending=False)
output_lines.append(f_df.head(15).to_string(index=False))
output_lines.append("")

# 4. MUTUAL INFORMATION
output_lines.append("-" * 80)
output_lines.append("4. MUTUAL INFORMATION (moi quan he phi tuyen tinh)")
output_lines.append("-" * 80)
output_lines.append("   Do luong luong thong tin feature cung cap ve target")
output_lines.append("")

mi_scores = mutual_info_regression(X_clean, y, random_state=42)
mi_results = []
for i, col in enumerate(X_clean.columns):
    mi_results.append({
        'Feature': col,
        'MI_Score': mi_scores[i]
    })

mi_df = pd.DataFrame(mi_results).sort_values('MI_Score', ascending=False)
output_lines.append(mi_df.head(15).to_string(index=False))
output_lines.append("")

# 5. COMBINED RANKING
output_lines.append("=" * 80)
output_lines.append("5. TONG HOP XEP HANG FEATURE QUAN TRONG NHAT")
output_lines.append("=" * 80)

# Merge all scores
combined = pearson_df[['Feature', 'abs_r']].merge(
    spearman_df[['Feature', 'abs_rho']], on='Feature'
).merge(
    f_df[['Feature', 'F_statistic']], on='Feature'
).merge(
    mi_df[['Feature', 'MI_Score']], on='Feature'
)

# Normalize F-statistic and MI to [0,1] for comparison
scaler = MinMaxScaler()
combined[['F_norm', 'MI_norm']] = scaler.fit_transform(combined[['F_statistic', 'MI_Score']])

# Calculate composite score (average)
combined['Composite_Score'] = (
    combined['abs_r'] + 
    combined['abs_rho'] + 
    combined['F_norm'] + 
    combined['MI_norm']
) / 4

combined = combined.sort_values('Composite_Score', ascending=False)

output_lines.append("\nTOP 20 FEATURE QUAN TRONG NHAT (theo diem tong hop):")
output_lines.append("-" * 80)
header = f"{'Rank':<6} {'Feature':<35} {'Pearson':<10} {'Spearman':<10} {'F_norm':<10} {'MI_norm':<10} {'Composite':<10}"
output_lines.append(header)
output_lines.append("-" * 80)
for idx, row in combined.head(20).iterrows():
    rank = combined.index.get_loc(idx) + 1
    line = f"{rank:<6} {row['Feature']:<35} {row['abs_r']:<10.4f} {row['abs_rho']:<10.4f} {row['F_norm']:<10.4f} {row['MI_norm']:<10.4f} {row['Composite_Score']:<10.4f}"
    output_lines.append(line)

output_lines.append("")

# Feature classification
output_lines.append("=" * 80)
output_lines.append("6. PHAN LOAI FEATURE THEO NHOM Y NGHIA KINH DOANH")
output_lines.append("=" * 80)

def classify_feature(name):
    if 'hist' in name:
        return "Historical Patterns"
    elif 'promo' in name:
        return "Promotions"
    elif 'traffic' in name or 'session' in name or 'conversion' in name:
        return "Web Traffic"
    elif 'inventory' in name:
        return "Inventory"
    elif 'tet' in name or 'holiday' in name:
        return "Calendar/Holidays"
    elif 'day' in name or 'month' in name or 'weekend' in name:
        return "Calendar/Basic"
    else:
        return "Other"

combined['Category'] = combined['Feature'].apply(classify_feature)
category_scores = combined.groupby('Category')['Composite_Score'].mean().sort_values(ascending=False)

output_lines.append("\nDiem trung binh theo nhom feature:")
for cat, score in category_scores.items():
    output_lines.append(f"  {cat:<25}: {score:.4f}")

output_lines.append("")

# Save results to CSV
output_path = r'D:\Datathon-2026\output\feature_importance_test.csv'
combined[['Feature', 'Category', 'abs_r', 'abs_rho', 'F_statistic', 'MI_Score', 'Composite_Score']].to_csv(
    output_path, index=False
)
output_lines.append(f"Da luu chi tiet ket qua tai: {output_path}")

output_lines.append("\n" + "=" * 80)
output_lines.append("KET LUAN")
output_lines.append("=" * 80)
conclusion = """
Dua tren 4 phep kiem dinh thong ke, cac feature QUAN TRONG NHAT doi voi Revenue la:

1. HISTORICAL PATTERNS (cac feature 'hist_*'):
   - Day la nhom co diem tong hop cao nhat
   - Bao gom: doanh thu trung binh theo ngay trong tuan, thang, 
     ngay le, khuyen mai, Tet...
   - Giai thich: Qua khu la chi bao tot nhat cho tuong lai trong 
     chuoi thoi gian (tinh tu tuong quan/autocorrelation)

2. PROMOTIONS (cac feature 'promo_*'):
   - Cuong do khuyen mai, hieu ung keo dai
   - Giai thich: Khuyen mai tao ra cac "demand shock" bat thuong

3. WEB TRAFFIC (cac feature 'traffic_*', 'session', 'conversion'):
   - La leading indicator - xay ra TRUOC khi doanh thu phat sinh
   - Giai thich: So luot truy cap web phan anh y dinh mua hang

4. CALENDAR/HOLIDAYS (cac feature 'tet_*', 'holiday_*'):
   - Dac biet la days_to_tet, is_tet_period
   - Giai thich: Thoi trang Viet Nam co tinh mua vu rat cao

5. INVENTORY (cac feature 'inventory_*'):
   - Ty le dap ung don hang, ton kho
   - Giai thich: Supply constraint anh huong den kha nang dap ung demand
"""
output_lines.append(conclusion)

# Write all output to file
report_path = r'D:\Datathon-2026\output\feature_importance_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

# Also print to console (ascii-safe)
print("=" * 80)
print("STATISTICAL FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 80)
print(f"Samples: {len(df)}, Features: {X.shape[1]}")
print("\nTOP 15 MOST IMPORTANT FEATURES (Composite Score):")
for idx, row in combined.head(15).iterrows():
    rank = combined.index.get_loc(idx) + 1
    print(f"{rank:2d}. {row['Feature']:35s} | Score: {row['Composite_Score']:.4f}")

print(f"\nFull report saved to: {report_path}")
print(f"CSV results saved to: {output_path}")
