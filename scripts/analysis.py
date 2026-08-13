"""
Smart Retail Analytics Platform — Analysis (real Indian data)
Sales-driver analysis and a sales-prediction model on the real Big Mart
dataset — 8,523 item-outlet records across 10 stores of an Indian
retail chain spanning Tier 1/2/3 cities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA = "/home/claude/projects-v3/smart-retail-analytics/data/bigmart_clean.csv"
OUT = "/home/claude/projects-v3/smart-retail-analytics/visuals"

df = pd.read_csv(DATA)

# ---------- 1. Total sales by outlet ----------
outlet_sales = df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(9, 6))
sns.barplot(x=outlet_sales.values, y=outlet_sales.index, color="#4C72B0")
plt.title("Total Sales by Outlet (10 Stores)")
plt.xlabel("Total Sales (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_outlet.png")
plt.close()

# ---------- 2. Sales by outlet type & location tier ----------
type_tier_sales = df.groupby(["Outlet_Type", "Outlet_Location_Type"])["Item_Outlet_Sales"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=type_tier_sales, x="Outlet_Type", y="Item_Outlet_Sales", hue="Outlet_Location_Type")
plt.title("Average Sales by Outlet Type and City Tier")
plt.ylabel("Average Sales per Item (₹)")
plt.xticks(rotation=20)
plt.legend(title="City Tier")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_outlet_type_tier.png")
plt.close()

# ---------- 3. Sales by item category ----------
category_sales = df.groupby("Item_Type")["Item_Outlet_Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(9, 8))
sns.barplot(x=category_sales.values, y=category_sales.index, color="#55A868")
plt.title("Total Sales by Product Category")
plt.xlabel("Total Sales (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_category.png")
plt.close()

# ---------- 4. MRP vs Sales relationship ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(2000, random_state=42), x="Item_MRP", y="Item_Outlet_Sales",
                 alpha=0.4, color="#DD8452")
plt.title("Item MRP vs Outlet Sales")
plt.xlabel("Item MRP (₹)")
plt.ylabel("Item Outlet Sales (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/mrp_vs_sales.png")
plt.close()

# ---------- 5. Outlet age vs average sales ----------
age_sales = df.groupby("Outlet_Age_Years")["Item_Outlet_Sales"].mean().sort_index()
plt.figure(figsize=(9, 5))
sns.lineplot(x=age_sales.index, y=age_sales.values, marker="o", color="#8172B2")
plt.title("Average Sales by Outlet Age (Years Since Establishment)")
plt.xlabel("Outlet Age (Years)")
plt.ylabel("Average Sales per Item (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_outlet_age.png")
plt.close()

# ---------- 6. Fat content vs sales (common retail question) ----------
fat_sales = df.groupby("Item_Fat_Content")["Item_Outlet_Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 5))
sns.barplot(x=fat_sales.index, y=fat_sales.values, color="#C44E52")
plt.title("Average Sales by Item Fat Content Label")
plt.ylabel("Average Sales (₹)")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_fat_content.png")
plt.close()

# ================= SALES PREDICTION MODEL =================
model_df = df[["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", "Item_MRP",
               "Outlet_Age_Years", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type",
               "Item_Outlet_Sales"]].copy()

model_df = pd.get_dummies(model_df, columns=["Item_Fat_Content", "Item_Type", "Outlet_Size",
                                              "Outlet_Location_Type", "Outlet_Type"], drop_first=True)

X = model_df.drop(columns=["Item_Outlet_Sales"])
y = model_df["Item_Outlet_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
cv_scores = cross_val_score(gbr, X_train, y_train, cv=5, scoring="r2")
gbr.fit(X_train, y_train)
preds = gbr.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# ---------- 7. Predicted vs actual ----------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.4, color="#4C72B0")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Sales (₹)")
plt.ylabel("Predicted Sales (₹)")
plt.title(f"Sales Prediction: Actual vs Predicted (R²={r2:.3f}, MAE=₹{mae:,.0f})")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_prediction_fit.png")
plt.close()

# ---------- 8. Feature importance ----------
importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(9, 7))
sns.barplot(x=importances.values, y=importances.index, color="#55A868")
plt.title("Top 12 Features Driving Sales Predictions")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_model_feature_importance.png")
plt.close()

summary = {
    "total_records": len(df),
    "unique_outlets": df["Outlet_Identifier"].nunique(),
    "unique_product_categories": df["Item_Type"].nunique(),
    "total_sales_inr": round(df["Item_Outlet_Sales"].sum(), 2),
    "top_outlet_by_sales": outlet_sales.index[0],
    "top_category_by_sales": category_sales.index[0],
    "model_test_r2": round(r2, 4),
    "model_cv_r2_mean": round(cv_scores.mean(), 4),
    "model_cv_r2_std": round(cv_scores.std(), 4),
    "model_mae_inr": round(mae, 2),
    "top_sales_driver": importances.index[0],
}
print(summary)

with open(f"{OUT}/summary_stats.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")
