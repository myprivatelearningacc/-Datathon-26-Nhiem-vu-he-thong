import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 0. CONFIG PATH
# =========================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "datathon-2026-round-1"
OUTPUT_DIR = PROJECT_ROOT / "datathon-output"

SALES_PATH = DATA_DIR / "sales.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission_final.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SALES_PATH:", SALES_PATH)
print("SAMPLE_PATH:", SAMPLE_PATH)
print("SUBMISSION_PATH:", SUBMISSION_PATH)

assert SALES_PATH.exists(), f"Không tìm thấy sales.csv tại: {SALES_PATH}"
assert SAMPLE_PATH.exists(), f"Không tìm thấy sample_submission.csv tại: {SAMPLE_PATH}"


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Đọc dữ liệu sales.csv...")
sales = pd.read_csv(SALES_PATH)
sales["Date"] = pd.to_datetime(sales["Date"])
sales.sort_values("Date", inplace=True)
sales.set_index("Date", inplace=True)

sub_template = pd.read_csv(SAMPLE_PATH)
future_dates = pd.to_datetime(sub_template["Date"])


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

data = sales.copy()
future_data = pd.DataFrame(index=future_dates)
all_data = pd.concat([data, future_data])

all_data["day_index"] = np.arange(len(all_data))
all_data["year"] = all_data.index.year
all_data["month"] = all_data.index.month
all_data["day"] = all_data.index.day
all_data["dayofweek"] = all_data.index.dayofweek
all_data["quarter"] = all_data.index.quarter
all_data["is_weekend"] = all_data["dayofweek"].isin([5, 6]).astype(int)

all_data["is_month_start"] = all_data["day"].isin([1, 2, 3]).astype(int)
all_data["is_month_end"] = all_data["day"].isin([29, 30, 31]).astype(int)
all_data["is_double_day"] = (all_data["month"] == all_data["day"]).astype(int)
all_data["is_mid_month"] = all_data["day"].isin([14, 15, 16]).astype(int)

tet_dates = pd.to_datetime([
    "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
    "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22",
    "2024-02-10", "2025-01-29"
])

all_data["days_to_tet"] = 999
all_data["days_after_tet"] = 999

for td in tet_dates:
    diff = (all_data.index - td).days

    mask_before = (diff >= -30) & (diff < 0)
    all_data.loc[mask_before, "days_to_tet"] = np.minimum(
        all_data.loc[mask_before, "days_to_tet"],
        np.abs(diff[mask_before])
    )

    mask_after = (diff >= 0) & (diff <= 15)
    all_data.loc[mask_after, "days_after_tet"] = np.minimum(
        all_data.loc[mask_after, "days_after_tet"],
        diff[mask_after]
    )

all_data["is_tet_week"] = (
    (all_data["days_after_tet"] <= 7) |
    (all_data["days_to_tet"] <= 3)
).astype(int)

all_data["is_holiday"] = 0
for h in ["01-01", "04-30", "05-01", "09-02"]:
    all_data.loc[all_data.index.strftime("%m-%d") == h, "is_holiday"] = 1


# =========================================================
# 3. TRAIN / FUTURE SPLIT
# =========================================================

train = all_data[all_data.index.year <= 2022].copy()
future = all_data[all_data.index.year > 2022].copy()

print("Loại bỏ giai đoạn giãn cách COVID-19 khỏi tập train...")
mask_covid_1 = (train.index >= "2020-03-15") & (train.index <= "2020-05-15")
mask_covid_2 = (train.index >= "2021-06-01") & (train.index <= "2021-10-31")
train_clean = train[~mask_covid_1 & ~mask_covid_2].copy()


# =========================================================
# 4. TARGET ENCODING
# =========================================================

print("Tạo Target Encoding features...")

m_rev = train_clean.groupby("month")["Revenue"].median()
d_rev = train_clean.groupby("dayofweek")["Revenue"].median()
m_cogs = train_clean.groupby("month")["COGS"].median()
d_cogs = train_clean.groupby("dayofweek")["COGS"].median()

train_clean["month_mean_rev"] = train_clean["month"].map(m_rev)
train_clean["dow_mean_rev"] = train_clean["dayofweek"].map(d_rev)
train_clean["month_mean_cogs"] = train_clean["month"].map(m_cogs)
train_clean["dow_mean_cogs"] = train_clean["dayofweek"].map(d_cogs)

future["month_mean_rev"] = future["month"].map(m_rev)
future["dow_mean_rev"] = future["dayofweek"].map(d_rev)
future["month_mean_cogs"] = future["month"].map(m_cogs)
future["dow_mean_cogs"] = future["dayofweek"].map(d_cogs)

features = [
    "month", "day", "dayofweek", "is_weekend", "quarter",
    "month_mean_rev", "dow_mean_rev", "month_mean_cogs", "dow_mean_cogs",
    "is_month_start", "is_month_end", "is_double_day", "is_mid_month",
    "days_to_tet", "days_after_tet", "is_tet_week", "is_holiday"
]

# =========================================================
# 4B. TIME-SERIES CROSS-VALIDATION
# =========================================================

from sklearn.metrics import mean_absolute_error

print("\nChạy Time-Series Cross-Validation...")

COVID_RANGES = [
    ("2020-03-15", "2020-05-15"),
    ("2021-06-01", "2021-10-31")
]

def remove_covid_period(df):
    out = df.copy()
    for start, end in COVID_RANGES:
        mask = (out.index >= start) & (out.index <= end)
        out = out[~mask]
    return out


def add_target_encoding_from_train(fold_train, fold_apply):
    """
    Tính target encoding chỉ từ fold_train để tránh leakage.
    Sau đó map sang cả fold_train và fold_apply.
    """
    fold_train = fold_train.copy()
    fold_apply = fold_apply.copy()

    m_rev = fold_train.groupby("month")["Revenue"].median()
    d_rev = fold_train.groupby("dayofweek")["Revenue"].median()
    m_cogs = fold_train.groupby("month")["COGS"].median()
    d_cogs = fold_train.groupby("dayofweek")["COGS"].median()

    global_rev = fold_train["Revenue"].median()
    global_cogs = fold_train["COGS"].median()

    for df in [fold_train, fold_apply]:
        df["month_mean_rev"] = df["month"].map(m_rev).fillna(global_rev)
        df["dow_mean_rev"] = df["dayofweek"].map(d_rev).fillna(global_rev)
        df["month_mean_cogs"] = df["month"].map(m_cogs).fillna(global_cogs)
        df["dow_mean_cogs"] = df["dayofweek"].map(d_cogs).fillna(global_cogs)

    return fold_train, fold_apply


def fit_predict_target(fold_train, fold_val, target_col, hybrid_seed=42, pure_seed=24):
    """
    Fit đúng logic model hiện tại:
    - Hybrid: LinearRegression trend trên log(target), LGBM học residual log
    - Pure: LGBM học trực tiếp log(target)
    - Ensemble: 70% hybrid + 30% pure
    """
    fold_train = fold_train.copy()
    fold_val = fold_val.copy()

    log_y = np.log1p(fold_train[target_col])

    # Hybrid trend
    lr = LinearRegression()
    lr.fit(fold_train[["day_index"]], log_y)

    residual_log = log_y - lr.predict(fold_train[["day_index"]])

    hybrid_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=6,
        colsample_bytree=0.7,
        random_state=hybrid_seed,
        verbose=-1
    )
    hybrid_model.fit(fold_train[features], residual_log)

    pred_hybrid = np.expm1(
        lr.predict(fold_val[["day_index"]]) +
        hybrid_model.predict(fold_val[features])
    )

    # Pure LGBM
    pure_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.005,
        max_depth=6,
        colsample_bytree=0.7,
        random_state=pure_seed,
        verbose=-1
    )
    pure_model.fit(fold_train[features], log_y)

    pred_pure = np.expm1(pure_model.predict(fold_val[features]))

    pred_final = pred_hybrid * 0.70 + pred_pure * 0.30
    pred_final = np.clip(pred_final, 0, None)

    return pred_final


cv_folds = [
    ("2018_H2", "2018-07-01", "2018-12-31"),
    ("2019_H1", "2019-01-01", "2019-06-30"),
    ("2019_H2", "2019-07-01", "2019-12-31"),
    ("2022_H1", "2022-01-01", "2022-06-30"),
    ("2022_H2", "2022-07-01", "2022-12-31"),
]

cv_rows = []
cv_predictions = []

for fold_name, val_start, val_end in cv_folds:
    val_start = pd.to_datetime(val_start)
    val_end = pd.to_datetime(val_end)

    fold_train = train[train.index < val_start].copy()
    fold_val = train[(train.index >= val_start) & (train.index <= val_end)].copy()

    fold_train = remove_covid_period(fold_train)

    # Bảo đảm validation có target thật
    fold_train = fold_train.dropna(subset=["Revenue", "COGS"])
    fold_val = fold_val.dropna(subset=["Revenue", "COGS"])

    fold_train, fold_val = add_target_encoding_from_train(fold_train, fold_val)

    pred_rev = fit_predict_target(fold_train, fold_val, "Revenue", hybrid_seed=42, pure_seed=24)
    pred_cogs = fit_predict_target(fold_train, fold_val, "COGS", hybrid_seed=42, pure_seed=24)

    rev_mae = mean_absolute_error(fold_val["Revenue"], pred_rev)
    cogs_mae = mean_absolute_error(fold_val["COGS"], pred_cogs)
    avg_mae = (rev_mae + cogs_mae) / 2

    cv_rows.append({
        "fold": fold_name,
        "train_start": fold_train.index.min(),
        "train_end": fold_train.index.max(),
        "val_start": fold_val.index.min(),
        "val_end": fold_val.index.max(),
        "n_train": len(fold_train),
        "n_val": len(fold_val),
        "revenue_mae": rev_mae,
        "cogs_mae": cogs_mae,
        "avg_mae": avg_mae
    })

    fold_pred_df = pd.DataFrame({
        "Date": fold_val.index,
        "fold": fold_name,
        "Revenue_actual": fold_val["Revenue"].values,
        "Revenue_pred": pred_rev,
        "COGS_actual": fold_val["COGS"].values,
        "COGS_pred": pred_cogs,
    })
    cv_predictions.append(fold_pred_df)

cv_metrics = pd.DataFrame(cv_rows)
cv_predictions = pd.concat(cv_predictions, ignore_index=True)

cv_metrics_path = OUTPUT_DIR / "time_series_cv_metrics.csv"
cv_pred_path = OUTPUT_DIR / "time_series_cv_predictions.csv"

cv_metrics.to_csv(cv_metrics_path, index=False)
cv_predictions.to_csv(cv_pred_path, index=False)

print("\nTime-Series CV metrics:")
print(cv_metrics)
print(f"Saved CV metrics to: {cv_metrics_path}")
print(f"Saved CV predictions to: {cv_pred_path}")

# =========================================================
# 5. MODEL TRAINING
# =========================================================

print("Train mô hình Hybrid + Pure LGBM...")

train_clean["log_rev"] = np.log1p(train_clean["Revenue"])
train_clean["log_cogs"] = np.log1p(train_clean["COGS"])

# Revenue - Hybrid
lr_rev = LinearRegression()
lr_rev.fit(train_clean[["day_index"]], train_clean["log_rev"])
train_clean["rev_res_log"] = train_clean["log_rev"] - lr_rev.predict(train_clean[["day_index"]])

hybrid_rev = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=6,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1
)
hybrid_rev.fit(train_clean[features], train_clean["rev_res_log"])

# COGS - Hybrid
lr_cogs = LinearRegression()
lr_cogs.fit(train_clean[["day_index"]], train_clean["log_cogs"])
train_clean["cogs_res_log"] = train_clean["log_cogs"] - lr_cogs.predict(train_clean[["day_index"]])

hybrid_cogs = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=6,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1
)
hybrid_cogs.fit(train_clean[features], train_clean["cogs_res_log"])

future_hybrid_rev = np.expm1(
    lr_rev.predict(future[["day_index"]]) + hybrid_rev.predict(future[features])
)

future_hybrid_cogs = np.expm1(
    lr_cogs.predict(future[["day_index"]]) + hybrid_cogs.predict(future[features])
)

# Revenue - Pure
pure_rev = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=6,
    colsample_bytree=0.7,
    random_state=24,
    verbose=-1
)
pure_rev.fit(train_clean[features], train_clean["log_rev"])

# COGS - Pure
pure_cogs = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=6,
    colsample_bytree=0.7,
    random_state=24,
    verbose=-1
)
pure_cogs.fit(train_clean[features], train_clean["log_cogs"])

future_pure_rev = np.expm1(pure_rev.predict(future[features]))
future_pure_cogs = np.expm1(pure_cogs.predict(future[features]))

future["Revenue"] = future_hybrid_rev * 0.70 + future_pure_rev * 0.30
future["COGS"] = future_hybrid_cogs * 0.70 + future_pure_cogs * 0.30

# =========================================================
# 5B. MODEL INTERPRETABILITY: SHAP + FEATURE IMPORTANCE
# =========================================================

print("\nTạo SHAP plots và feature importance...")

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    import shap
    import matplotlib.pyplot as plt

    X_explain = train_clean[features].copy()

    # Sample để SHAP chạy nhanh hơn
    if len(X_explain) > 1000:
        X_explain_sample = X_explain.sample(1000, random_state=42)
    else:
        X_explain_sample = X_explain

    def save_shap_summary(model, X, output_prefix, max_display=15):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Beeswarm summary
        shap.summary_plot(
            shap_values,
            X,
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"{output_prefix}_shap_beeswarm.png",
            dpi=220,
            bbox_inches="tight"
        )
        plt.close()

        # Bar summary
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"{output_prefix}_shap_bar.png",
            dpi=220,
            bbox_inches="tight"
        )
        plt.close()

        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns
        ).sort_values(ascending=False)

        return mean_abs_shap

    # Pure models: dễ giải thích nhất vì học trực tiếp log target
    pure_rev_shap = save_shap_summary(
        pure_rev,
        X_explain_sample,
        "pure_revenue"
    )

    pure_cogs_shap = save_shap_summary(
        pure_cogs,
        X_explain_sample,
        "pure_cogs"
    )

    # Hybrid models: giải thích residual adjustment sau linear trend
    hybrid_rev_shap = save_shap_summary(
        hybrid_rev,
        X_explain_sample,
        "hybrid_revenue_residual"
    )

    hybrid_cogs_shap = save_shap_summary(
        hybrid_cogs,
        X_explain_sample,
        "hybrid_cogs_residual"
    )

    # Weighted proxy cho final ensemble
    weighted_rev_shap = (
        0.70 * hybrid_rev_shap.reindex(features).fillna(0) +
        0.30 * pure_rev_shap.reindex(features).fillna(0)
    ).sort_values(ascending=False)

    weighted_cogs_shap = (
        0.70 * hybrid_cogs_shap.reindex(features).fillna(0) +
        0.30 * pure_cogs_shap.reindex(features).fillna(0)
    ).sort_values(ascending=False)

    shap_importance = pd.DataFrame({
        "feature": features,
        "pure_revenue_mean_abs_shap": pure_rev_shap.reindex(features).values,
        "hybrid_revenue_residual_mean_abs_shap": hybrid_rev_shap.reindex(features).values,
        "weighted_revenue_mean_abs_shap": weighted_rev_shap.reindex(features).values,
        "pure_cogs_mean_abs_shap": pure_cogs_shap.reindex(features).values,
        "hybrid_cogs_residual_mean_abs_shap": hybrid_cogs_shap.reindex(features).values,
        "weighted_cogs_mean_abs_shap": weighted_cogs_shap.reindex(features).values,
    })

    shap_importance_path = OUTPUT_DIR / "shap_feature_importance.csv"
    shap_importance.to_csv(shap_importance_path, index=False)

    print(f"Saved SHAP plots to: {FIG_DIR}")
    print(f"Saved SHAP importance to: {shap_importance_path}")

except ImportError:
    print("Không tìm thấy package shap. Cài bằng: pip install shap")
    print("Sẽ xuất LightGBM feature importance thay thế.")

# LightGBM feature importance fallback / supplement
lgb_importance = pd.DataFrame({
    "feature": features,
    "pure_rev_gain": pure_rev.booster_.feature_importance(importance_type="gain"),
    "hybrid_rev_gain": hybrid_rev.booster_.feature_importance(importance_type="gain"),
    "pure_cogs_gain": pure_cogs.booster_.feature_importance(importance_type="gain"),
    "hybrid_cogs_gain": hybrid_cogs.booster_.feature_importance(importance_type="gain"),
})

lgb_importance["weighted_rev_gain"] = (
    0.70 * lgb_importance["hybrid_rev_gain"] +
    0.30 * lgb_importance["pure_rev_gain"]
)

lgb_importance["weighted_cogs_gain"] = (
    0.70 * lgb_importance["hybrid_cogs_gain"] +
    0.30 * lgb_importance["pure_cogs_gain"]
)

lgb_importance_path = OUTPUT_DIR / "lgbm_feature_importance_gain.csv"
lgb_importance.to_csv(lgb_importance_path, index=False)

print(f"Saved LightGBM gain importance to: {lgb_importance_path}")

# Giải thích trend riêng
print("\nLinear trend coefficients:")
print("Revenue trend coef:", lr_rev.coef_[0])
print("COGS trend coef:", lr_cogs.coef_[0])


# =========================================================
# 6. SAFE LEVEL SHIFT
# =========================================================

print("Cân chỉnh baseline theo bản ổn định 690k...")

mean_rev_2022 = train[train.index.year == 2022]["Revenue"].mean()
mean_cogs_2022 = train[train.index.year == 2022]["COGS"].mean()

# Giữ lại mức ổn định cũ, không over-push nữa
expected_rev_2023 = mean_rev_2022 * 1.25
expected_cogs_2023 = mean_cogs_2022 * 1.25
expected_rev_2024 = mean_rev_2022 * 1.55
expected_cogs_2024 = mean_cogs_2022 * 1.55

mask_2023 = future.index.year == 2023
mask_2024 = future.index.year == 2024

if mask_2023.any():
    future.loc[mask_2023, "Revenue"] *= expected_rev_2023 / future.loc[mask_2023, "Revenue"].mean()
    future.loc[mask_2023, "COGS"] *= expected_cogs_2023 / future.loc[mask_2023, "COGS"].mean()

if mask_2024.any():
    future.loc[mask_2024, "Revenue"] *= expected_rev_2024 / future.loc[mask_2024, "Revenue"].mean()
    future.loc[mask_2024, "COGS"] *= expected_cogs_2024 / future.loc[mask_2024, "COGS"].mean()

future["Revenue"] = future["Revenue"].clip(lower=0)
future["COGS"] = future["COGS"].clip(lower=0)


# =========================================================
# 7. EXPORT ONLY ONE SUBMISSION FILE
# =========================================================

submission = future[["Revenue", "COGS"]].reset_index()
submission.rename(columns={"index": "Date"}, inplace=True)

# Đảm bảo đúng format như sample_submission
submission = submission[["Date", "Revenue", "COGS"]]

submission.to_csv(SUBMISSION_PATH, index=False)

print("Hoàn tất.")
print(f"Chỉ xuất 1 file submission tại: {SUBMISSION_PATH}")
print("Submission shape:", submission.shape)
print(submission.head())