# 🛍️ Smart Retail Analytics Platform — India (Big Mart)

Sales-driver analysis and a sales-prediction model on the real **Big Mart Sales dataset** — 8,523 real item-outlet sales records across 10 stores of an Indian retail chain, spanning Tier 1/2/3 cities.

## 📌 Project Overview

This is the real dataset behind a well-known Analytics Vidhya hackathon, built around one core question: **can you predict item-level sales at a given outlet from its attributes?** This project answers that, plus the sales-driver analysis around it:

- Which outlets, outlet types, and city tiers perform best?
- Which product categories drive the most revenue?
- How do item price (MRP) and outlet age relate to sales?
- **Can sales be predicted** from item/outlet attributes — and how well?

## 🗂️ Repository Structure

```
smart-retail-analytics/
├── data/
│   ├── bigmart_train.csv         # Real dataset (8,523 records)
│   ├── bigmart_test.csv          # Real holdout set (unlabeled, from the original hackathon)
│   └── bigmart_clean.csv         # Cleaned, imputed version
├── notebooks/
│   └── analysis.ipynb            # Full narrated analysis, outputs rendered
├── scripts/
│   ├── clean_data.py             # Cleaning + imputation pipeline
│   └── analysis.py               # Full analysis pipeline as a standalone script
├── visuals/
│   ├── sales_by_outlet.png
│   ├── sales_by_outlet_type_tier.png
│   ├── sales_by_category.png
│   ├── mrp_vs_sales.png
│   ├── sales_by_outlet_age.png
│   ├── sales_by_fat_content.png
│   ├── sales_prediction_fit.png
│   ├── sales_model_feature_importance.png
│   └── summary_stats.txt
└── README.md
```

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter`

## 📊 Key Findings

| Metric | Finding |
|---|---|
| Records analyzed | 8,523 (real) |
| Outlets | 10, across Tier 1/2/3 Indian cities |
| Product categories | 16 |
| Total sales | ₹1.86 crore (₹18.59M) |
| Top outlet by sales | OUT027 |
| Top category by sales | Fruits and Vegetables |
| Sales model test R² | 0.605 (5-fold CV mean: 0.589 ± 0.01) |
| Sales model MAE | ₹735.60 |
| Top sales driver | Item MRP (price) |

**Highlights:**
- **Item MRP (price) is overwhelmingly the strongest predictor of sales**, confirmed quantitatively by the model rather than assumed.
- **Outlet type and city tier meaningfully shift performance** — Supermarket Type3 outlets outperform smaller grocery-format stores on a per-item basis, a clear signal for format-investment decisions.
- **The R² of ~0.60 is intentionally reported as-is.** Real retail sales depend on local demand and customer behavior that item/outlet attributes alone can't fully capture — a model claiming a near-perfect fit here would be the actual red flag.

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb   # recommended: full narrated walkthrough
# or:
python scripts/clean_data.py                 # produces the cleaned CSV
python scripts/analysis.py                    # regenerates all charts + model
```

## 📁 Dataset Notes

Real data: the **Big Mart Sales** dataset, originally an Analytics Vidhya hackathon dataset (source: `github.com/shrikant-temburwar/Big-Mart-Sales-Prediction`). Unlike a per-transaction retail log, this data is aggregated to the item-outlet level with no customer IDs or transaction dates — so RFM/cohort-style customer segmentation isn't applicable here. Instead, the project focuses on what this real dataset actually supports: sales-driver analysis and predictive modeling, which is exactly the task the original hackathon was built around.

## 🚀 Possible Extensions

- Submit predictions against the included `bigmart_test.csv` holdout set (the original hackathon's actual task)
- Try XGBoost/LightGBM and compare against the Gradient Boosting baseline
- Add SHAP explainability to show per-item why a sales prediction came out the way it did
