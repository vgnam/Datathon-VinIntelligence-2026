"""
feature_importance.py
Compute normalized feature importances from RF, LGB, XGB trained on full data.
Includes native tree importance + SHAP (SHapley Additive exPlanations).
Exports CSVs + plots to output/.
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
FEAT_DIR = ROOT.parent / "output"
OUT_DIR = FEAT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FEATURES = FEAT_DIR / "train_features.csv"
TRAIN_TARGET = FEAT_DIR / "train_target.csv"


def plot_shap_summary(shap_values, X_display, out_path, title, max_display=15):
    """Save SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False, max_display=max_display, plot_size=None)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_FEATURES, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    target = pd.read_csv(TRAIN_TARGET, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    X = train.drop(columns=["Date"]).copy()
    y = target[["Revenue", "COGS"]].copy()
    y_log = y.copy()
    y_log["Revenue"] = np.log1p(y_log["Revenue"])
    y_log["COGS"] = np.log1p(y_log["COGS"])

    median_vals = X.median(numeric_only=True)
    X = X.fillna(median_vals).fillna(0.0)
    feature_names = list(X.columns)
    X_np = X.to_numpy()

    importances = pd.DataFrame({"feature": feature_names})

    # ========================================================================
    # 1. Random Forest
    # ========================================================================
    print("Training RF...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=42,
    )
    rf.fit(X, y_log)
    importances["rf"] = rf.feature_importances_

    print("  Computing SHAP (RF)...")
    explainer_rf = shap.TreeExplainer(rf)
    shap_rf = explainer_rf.shap_values(X_np)
    # shap_values returns list of 2 arrays for multi-output
    if isinstance(shap_rf, list):
        shap_rf_rev = np.abs(shap_rf[0]).mean(axis=0)
        shap_rf_cogs = np.abs(shap_rf[1]).mean(axis=0)
        shap_rf_mean = (shap_rf_rev + shap_rf_cogs) / 2.0
    else:
        shap_rf_mean = np.abs(shap_rf).mean(axis=0)
    importances["rf_shap"] = shap_rf_mean

    # ========================================================================
    # 2. LightGBM
    # ========================================================================
    print("Training LGB (Revenue)...")
    dtrain = lgb.Dataset(X, label=y_log["Revenue"], feature_name=feature_names)
    lgb_rev = lgb.train(
        {"objective": "regression", "metric": "mape", "verbosity": -1,
         "learning_rate": 0.03, "num_leaves": 20, "max_depth": 6,
         "min_child_samples": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5},
        dtrain, num_boost_round=500,
    )

    print("Training LGB (COGS)...")
    dtrain = lgb.Dataset(X, label=y_log["COGS"], feature_name=feature_names)
    lgb_cogs = lgb.train(
        {"objective": "regression", "metric": "mape", "verbosity": -1,
         "learning_rate": 0.03, "num_leaves": 20, "max_depth": 6,
         "min_child_samples": 20,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5},
        dtrain, num_boost_round=500,
    )

    lgb_rev_imp = pd.Series(lgb_rev.feature_importance(importance_type="gain"), index=feature_names)
    lgb_cogs_imp = pd.Series(lgb_cogs.feature_importance(importance_type="gain"), index=feature_names)
    importances["lgb"] = (lgb_rev_imp + lgb_cogs_imp).values / 2.0

    print("  Computing SHAP (LGB)...")
    explainer_lgb_rev = shap.TreeExplainer(lgb_rev)
    explainer_lgb_cogs = shap.TreeExplainer(lgb_cogs)
    shap_lgb_rev = np.abs(explainer_lgb_rev.shap_values(X_np)).mean(axis=0)
    shap_lgb_cogs = np.abs(explainer_lgb_cogs.shap_values(X_np)).mean(axis=0)
    importances["lgb_shap"] = (shap_lgb_rev + shap_lgb_cogs) / 2.0

    # ========================================================================
    # 3. XGBoost
    # ========================================================================
    print("Training XGB (Revenue)...")
    dtrain = xgb.DMatrix(X, label=y_log["Revenue"])
    xgb_rev = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": "mape", "seed": 42,
         "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0,
         "nthread": -1},
        dtrain, num_boost_round=500,
    )

    print("Training XGB (COGS)...")
    dtrain = xgb.DMatrix(X, label=y_log["COGS"])
    xgb_cogs = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": "mape", "seed": 42,
         "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0,
         "nthread": -1},
        dtrain, num_boost_round=500,
    )

    def xgb_importance_series(model):
        scores = model.get_score(importance_type="gain")
        mapped = {}
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                mapped[feature_names[idx]] = v
            else:
                mapped[k] = v
        s = pd.Series(mapped)
        s = s.reindex(feature_names, fill_value=0.0)
        return s

    xgb_rev_imp = xgb_importance_series(xgb_rev)
    xgb_cogs_imp = xgb_importance_series(xgb_cogs)
    importances["xgb"] = (xgb_rev_imp + xgb_cogs_imp).values / 2.0

    print("  Computing SHAP (XGB)...")
    explainer_xgb_rev = shap.TreeExplainer(xgb_rev)
    explainer_xgb_cogs = shap.TreeExplainer(xgb_cogs)
    shap_xgb_rev = np.abs(explainer_xgb_rev.shap_values(xgb.DMatrix(X_np))).mean(axis=0)
    shap_xgb_cogs = np.abs(explainer_xgb_cogs.shap_values(xgb.DMatrix(X_np))).mean(axis=0)
    importances["xgb_shap"] = (shap_xgb_rev + shap_xgb_cogs) / 2.0

    # ========================================================================
    # 4. Normalize & aggregate
    # ========================================================================
    for col in ["rf", "lgb", "xgb", "rf_shap", "lgb_shap", "xgb_shap"]:
        total = importances[col].sum()
        if total > 0:
            importances[col] = importances[col] / total

    importances["native_mean"] = importances[["rf", "lgb", "xgb"]].mean(axis=1)
    importances["shap_mean"] = importances[["rf_shap", "lgb_shap", "xgb_shap"]].mean(axis=1)
    importances["overall_mean"] = (importances["native_mean"] + importances["shap_mean"]) / 2.0
    importances = importances.sort_values("overall_mean", ascending=False).reset_index(drop=True)

    # Export CSV
    csv_path = OUT_DIR / "feature_importance.csv"
    importances.to_csv(csv_path, index=False)
    print(f"\nExported: {csv_path}")

    # ========================================================================
    # 5. Plots — native importance
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(importances))
    plot_df = importances.head(top_n).copy().sort_values("native_mean", ascending=True)
    plot_df.set_index("feature")[["rf", "lgb", "xgb"]].plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} Native Feature Importances (normalized)")
    ax.set_xlabel("Normalized Importance")
    plt.tight_layout()
    png_path = OUT_DIR / "feature_importance.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Exported: {png_path}")

    # ========================================================================
    # 6. Plots — SHAP summary (sample 500 rows for speed)
    # ========================================================================
    sample_idx = np.random.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].copy()
    X_sample_np = X_sample.to_numpy()

    print("\nGenerating SHAP summary plots...")
    # RF SHAP
    print("  RF SHAP summary...")
    if isinstance(shap_rf, list):
        plot_shap_summary(shap_rf[0][sample_idx], X_sample, OUT_DIR / "shap_rf_revenue.png", "RF SHAP — Revenue")
        plot_shap_summary(shap_rf[1][sample_idx], X_sample, OUT_DIR / "shap_rf_cogs.png", "RF SHAP — COGS")
    else:
        plot_shap_summary(shap_rf[sample_idx], X_sample, OUT_DIR / "shap_rf.png", "RF SHAP")

    # LGB SHAP
    print("  LGB SHAP summary...")
    shap_lgb_rev_vals = explainer_lgb_rev.shap_values(X_sample_np)
    shap_lgb_cogs_vals = explainer_lgb_cogs.shap_values(X_sample_np)
    plot_shap_summary(shap_lgb_rev_vals, X_sample, OUT_DIR / "shap_lgb_revenue.png", "LightGBM SHAP — Revenue")
    plot_shap_summary(shap_lgb_cogs_vals, X_sample, OUT_DIR / "shap_lgb_cogs.png", "LightGBM SHAP — COGS")

    # XGB SHAP
    print("  XGB SHAP summary...")
    shap_xgb_rev_vals = explainer_xgb_rev.shap_values(xgb.DMatrix(X_sample_np))
    shap_xgb_cogs_vals = explainer_xgb_cogs.shap_values(xgb.DMatrix(X_sample_np))
    plot_shap_summary(shap_xgb_rev_vals, X_sample, OUT_DIR / "shap_xgb_revenue.png", "XGBoost SHAP — Revenue")
    plot_shap_summary(shap_xgb_cogs_vals, X_sample, OUT_DIR / "shap_xgb_cogs.png", "XGBoost SHAP — COGS")

    # ========================================================================
    # 7. Print top features
    # ========================================================================
    print("\nTop 10 features (Native mean / SHAP mean / Overall):")
    top10 = importances.head(10)[["feature", "native_mean", "shap_mean", "overall_mean"]].copy()
    for _, r in top10.iterrows():
        print(f"  {r['feature']:35s}  Native:{r['native_mean']:.4f}  SHAP:{r['shap_mean']:.4f}  Overall:{r['overall_mean']:.4f}")


if __name__ == "__main__":
    main()
