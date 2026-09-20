# ============================================================================
# Individual Task 2 - Part 2 (Deliberation on Task 1):
#
# Upload train.csv, store.csv and online_retail_II.xlsx when prompted
# I used Google Collab for my analysis.
# ============================================================================

# ---- 0. Setup ----
!pip -q install fairlearn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fairlearn.metrics import MetricFrame

from google.colab import files
print("Upload train.csv, store.csv, online_retail_II.xlsx now:")
uploaded = files.upload()

# ---- 1. Rebuild the Task 1 daily-aggregate datasets ----
train = pd.read_csv("train.csv", dtype={"StateHoliday": str}, low_memory=False)
store = pd.read_csv("store.csv")
train["Date"] = pd.to_datetime(train["Date"])
train_open = train[train["Open"] == 1].copy()

daily_a = train_open.groupby("Date").agg(
    TotalSales=("Sales", "sum"),
    NumStoresOpen=("Store", "nunique"),
    PromoShare=("Promo", "mean"),
    SchoolHolidayShare=("SchoolHoliday", "mean"),
    AnyStateHoliday=("StateHoliday", lambda s: int((s != "0").any())),
).reset_index().sort_values("Date")
daily_a["DayOfWeek"] = daily_a["Date"].dt.dayofweek
daily_a["Month"] = daily_a["Date"].dt.month
daily_a["IsWeekend"] = (daily_a["DayOfWeek"] >= 5).astype(int)
daily_a["AvgSalesPerStore"] = daily_a["TotalSales"] / daily_a["NumStoresOpen"]
feat_a = ["DayOfWeek", "Month", "IsWeekend", "PromoShare", "SchoolHolidayShare", "AnyStateHoliday"]
target_a = "AvgSalesPerStore"
daily_a = daily_a.dropna(subset=feat_a + [target_a]).reset_index(drop=True)

sheets = pd.read_excel("online_retail_II.xlsx", sheet_name=None)
retail = pd.concat(sheets.values(), ignore_index=True)
retail.columns = [c.strip() for c in retail.columns]
retail["Invoice"] = retail["Invoice"].astype(str)
retail = retail[~retail["Invoice"].str.startswith("C")]
retail = retail[(retail["Quantity"] > 0) & (retail["Price"] > 0)]
retail["Revenue"] = retail["Quantity"] * retail["Price"]
retail["Date"] = pd.to_datetime(pd.to_datetime(retail["InvoiceDate"]).dt.date)

daily_b = retail.groupby("Date").agg(TotalRevenue=("Revenue", "sum")).reset_index().sort_values("Date")
uk_rev = retail[retail["Country"] == "United Kingdom"].groupby("Date")["Revenue"].sum()
tot_rev = retail.groupby("Date")["Revenue"].sum()
daily_b["UKShareRevenue"] = (uk_rev / tot_rev).reindex(daily_b["Date"]).fillna(0).values
daily_b["DayOfWeek"] = daily_b["Date"].dt.dayofweek
daily_b["Month"] = daily_b["Date"].dt.month
daily_b["IsWeekend"] = (daily_b["DayOfWeek"] >= 5).astype(int)
daily_b["IsDecember"] = (daily_b["Month"] == 12).astype(int)
feat_b = ["DayOfWeek", "Month", "IsWeekend", "IsDecember", "UKShareRevenue"]
target_b = "TotalRevenue"
daily_b = daily_b.dropna(subset=feat_b + [target_b]).reset_index(drop=True)

print(f"Rossmann daily rows: {len(daily_a)} | Retail II daily rows: {len(daily_b)}")

# ============================================================================
# (1) TIME-SERIES CROSS-VALIDATION  -- for Part 2, Q1
# ============================================================================
def run_cv(df, feats, target, n_splits=5, label=""):
    X, y = df[feats].values, df[target].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(Xtr)

        rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        knn = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(scaler.transform(Xtr), ytr)

        for name, pred in [("RandomForest", rf.predict(Xte)), ("kNN", knn.predict(scaler.transform(Xte)))]:
            rmse = np.sqrt(mean_squared_error(yte, pred))
            mae = mean_absolute_error(yte, pred)
            r2 = r2_score(yte, pred)
            rows.append({"dataset": label, "fold": fold, "model": name, "n_train": len(tr_idx),
                         "n_test": len(te_idx), "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(rows)

cv_a = run_cv(daily_a, feat_a, target_a, label="Rossmann")
cv_b = run_cv(daily_b, feat_b, target_b, label="Retail II")
cv_all = pd.concat([cv_a, cv_b], ignore_index=True)

print("\n=== 2.1  TimeSeriesSplit cross-validation (5 folds) ===")
print(cv_all.to_string(index=False))
summary = cv_all.groupby(["dataset", "model"])[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
print("\n--- Mean +/- SD across folds (COPY THIS INTO THE REPORT TABLE) ---")
print(summary)
cv_all.to_csv("task2_cv_results.csv", index=False)
files.download("task2_cv_results.csv")

# ============================================================================
# (2) LEARNING CURVES  -- for Part 2, Q2
# ============================================================================
def plot_learning_curve(df, feats, target, label, fname):
    X, y = df[feats].values, df[target].values
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=tscv, scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 6), shuffle=False,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(train_sizes, train_mean, "o-", color="#2f7ed8", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2f7ed8")
    ax.plot(train_sizes, test_mean, "o-", color="#d8572f", label="Validation score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#d8572f")
    ax.set_xlabel("Training examples (days)")
    ax.set_ylabel("R2 score")
    ax.set_title(f"Learning Curve -- {label} (Random Forest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved {fname}")
    files.download(fname)
    return train_sizes, train_mean, test_mean

print("\n=== 2.2  Learning curves ===")
plot_learning_curve(daily_a, feat_a, target_a, "Rossmann", "fig_learning_curve_rossmann.png")
plot_learning_curve(daily_b, feat_b, target_b, "Online Retail II", "fig_learning_curve_retail.png")

# ============================================================================
# (3) FAIRLEARN SUBGROUP FAIRNESS CHECK  -- for Part 2, Q3
# ============================================================================
# Rossmann: does prediction error differ systematically by StoreType?
# We deliberately EXCLUDE StoreType from the model's training features, so
# any gap in error across StoreType groups reflects a real blind spot in
# what the model was given to learn from -- not the model "seeing" the group.
store_daily = train_open.merge(store[["Store", "StoreType"]], on="Store", how="left")
store_daily["DayOfWeek"] = store_daily["Date"].dt.dayofweek
store_daily["Month"] = store_daily["Date"].dt.month
store_daily["IsWeekend"] = (store_daily["DayOfWeek"] >= 5).astype(int)
feat_fair_a = ["DayOfWeek", "Month", "IsWeekend", "Promo", "SchoolHoliday"]
store_daily = store_daily.dropna(subset=feat_fair_a + ["Sales", "StoreType"])
store_daily = store_daily.sort_values("Date").reset_index(drop=True)

split = int(len(store_daily) * 0.8)
tr, te = store_daily.iloc[:split], store_daily.iloc[split:]
rf_fair_a = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_fair_a.fit(tr[feat_fair_a], tr["Sales"])
pred_a = rf_fair_a.predict(te[feat_fair_a])

mf_a = MetricFrame(
    metrics={"MAE": mean_absolute_error, "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))},
    y_true=te["Sales"], y_pred=pred_a, sensitive_features=te["StoreType"],
)
print("\n=== 2.3  Fairlearn -- Rossmann error by StoreType ===")
print(mf_a.by_group)
print("Overall:", dict(mf_a.overall))
print("Max - min gap (MAE):", mf_a.difference()["MAE"])

# Online Retail II: does prediction error differ by top countries?
country_daily = retail.groupby(["Date", "Country"]).agg(Revenue=("Revenue", "sum")).reset_index()
top_countries = retail["Country"].value_counts().head(6).index.tolist()
country_daily = country_daily[country_daily["Country"].isin(top_countries)].copy()
country_daily["DayOfWeek"] = country_daily["Date"].dt.dayofweek
country_daily["Month"] = country_daily["Date"].dt.month
country_daily["IsWeekend"] = (country_daily["DayOfWeek"] >= 5).astype(int)
feat_fair_b = ["DayOfWeek", "Month", "IsWeekend"]
country_daily = country_daily.sort_values("Date").reset_index(drop=True)

split_b = int(len(country_daily) * 0.8)
tr_b, te_b = country_daily.iloc[:split_b], country_daily.iloc[split_b:]
rf_fair_b = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_fair_b.fit(tr_b[feat_fair_b], tr_b["Revenue"])
pred_b = rf_fair_b.predict(te_b[feat_fair_b])

mf_b = MetricFrame(
    metrics={"MAE": mean_absolute_error, "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))},
    y_true=te_b["Revenue"], y_pred=pred_b, sensitive_features=te_b["Country"],
)
print("\n=== 2.3  Fairlearn -- Online Retail II error by Country ===")
print(mf_b.by_group)
print("Overall:", dict(mf_b.overall))
print("Max - min gap (MAE):", mf_b.difference()["MAE"])

# Bar chart of both
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
mf_a.by_group["MAE"].plot(kind="bar", ax=axes[0], color="#2f7ed8")
axes[0].set_title("Rossmann: MAE by StoreType")
axes[0].set_ylabel("MAE")
mf_b.by_group["MAE"].plot(kind="bar", ax=axes[1], color="#d8572f")
axes[1].set_title("Online Retail II: MAE by Country")
axes[1].set_ylabel("MAE")
fig.tight_layout()
fig.savefig("fig_fairlearn_subgroup_mae.png", dpi=150)
plt.show()
files.download("fig_fairlearn_subgroup_mae.png")

print("\n\nDONE. You should have downloaded:")
print(" - task2_cv_results.csv")
print(" - fig_learning_curve_rossmann.png")
print(" - fig_learning_curve_retail.png")
print(" - fig_fairlearn_subgroup_mae.png")
print("Copy the printed numbers above into the report tables, and upload the")
print("4 files into your Overleaf project (project root, no subfolder needed).")
