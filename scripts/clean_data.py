"""
Cleans the real Big Mart Sales dataset — 8,523 real item-outlet sales
records across 10 stores of an Indian retail chain (Tier 1/2/3 cities).

Source: github.com/shrikant-temburwar/Big-Mart-Sales-Prediction
(originally an Analytics Vidhya hackathon dataset)
"""
import pandas as pd
import numpy as np

RAW = "/home/claude/projects-v3/smart-retail-analytics/data/bigmart_train.csv"
OUT = "/home/claude/projects-v3/smart-retail-analytics/data/bigmart_clean.csv"

df = pd.read_csv(RAW)
before = len(df)

# ---- Standardize inconsistent category labels (a real messy-data issue in this dataset) ----
df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
    "low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"
})

# ---- Impute missing Item_Weight with the mean weight for that item type ----
df["Item_Weight"] = df.groupby("Item_Type")["Item_Weight"].transform(
    lambda x: x.fillna(x.mean())
)

# ---- Impute missing Outlet_Size with the mode for that Outlet_Type ----
mode_by_type = df.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else "Medium"
)
df["Outlet_Size"] = df.apply(
    lambda r: mode_by_type[r["Outlet_Type"]] if pd.isna(r["Outlet_Size"]) else r["Outlet_Size"],
    axis=1
)

# ---- Derived fields ----
df["Outlet_Age_Years"] = 2013 - df["Outlet_Establishment_Year"]  # dataset reference year is 2013
df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)  # 0 visibility is a data artifact, not real
df["Item_Visibility"] = df.groupby("Item_Type")["Item_Visibility"].transform(
    lambda x: x.fillna(x.mean())
)

after = len(df)
print(f"Rows: {before:,} (all retained — missing values imputed, not dropped, since real dataset is small)")
print("Missing values remaining:\n", df.isnull().sum().sum())

df.to_csv(OUT, index=False)
print("Saved:", OUT)
print(df.head())
