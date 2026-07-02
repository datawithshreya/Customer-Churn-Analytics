# E-Commerce Customer Intelligence Platform
### Customer Retention | Churn Prediction | CLV Analytics | Segmentation
> **A production-style analytics pipeline built to simulate real-world customer intelligence workflows used by e-commerce companies.
---

## 📁 Project Structure

```
Ecommerce_Customer_Analytics/
│
├── data/
│   ├── generate_data.py     # Dataset Note:
                             The dataset is synthetically generated to simulate an e-commerce environment while preserving realistic customer, transaction, product and retention patterns.
│   ├── customers.csv        # 2,000 customer profiles
│   ├── orders.csv           # 8,500 purchase transactions
│   ├── products.csv         # 150-item product catalogue
│   ├── payments.csv         # ~7,000 payment records
│   └── reviews.csv          # ~3,800 product reviews
│
├── src/
│   ├── data_loading.py      # CSV ingestion + schema validation
│   ├── data_cleaning.py     # Quality report · null handling · outliers
│   ├── feature_engineering.py # RFM · CLV · churn flag
│   ├── customer_segmentation.py # RFM scoring + K-Means clustering
│   ├── churn_analysis.py    # Random Forest · metrics · stats tests
│   └── visualization.py     # 10 production-quality charts
│
├── notebooks/
│   └── Ecommerce_Customer_Analysis.ipynb  # Interactive walkthrough
│
├── dashboard/
│   ├── dashboard_spec.md    # KPIs · chart list · DAX measures
│   └── charts/              # Auto-generated PNG charts (10 files)
│
├── main.py                  # ← Run this for the full pipeline
├── requirements.txt
└── README.md
```
## 🔄 Pipeline Architecture

Raw Data
    ↓
Data Validation
    ↓
Data Cleaning
    ↓
Feature Engineering (RFM + CLV)
    ↓
Customer Segmentation
    ↓
Churn Prediction Model
    ↓
Business Insights + Visualization Dashboard

---

## ⚡ Quick Start

### Step 1 — Clone / download the project
```bash
cd ~/Projects
# (copy the Ecommerce_Customer_Analytics folder here)
cd Ecommerce_Customer_Analytics
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the full pipeline
```bash
python main.py
```

You will see output like:
```
====================================================
   E-COMMERCE CUSTOMER ANALYTICS PLATFORM
====================================================

[1/6] Loading datasets …
  Customers   :  2,000 rows  ×  14 cols
  Orders      :  8,500 rows  ×  13 cols
...
[5/6] Training churn model …
  Accuracy   : 100.00%
  ROC-AUC    : 1.0000
...
[6/6] Generating charts …
  ✓  Revenue Trend
  ✓  Cohort Heatmap
  ...
BUSINESS INSIGHTS
  1. Churn rate is 48.6% …
  2. Champions drive 51% of revenue …
```

### Step 5 — Open the Jupyter notebook (optional)
```bash
jupyter notebook notebooks/Ecommerce_Customer_Analysis.ipynb
```

### Step 6 — View charts
All 10 charts are saved in `dashboard/charts/`.

---

## 🔧 Module Guide

| File | Purpose | Key Functions |
|------|---------|---------------|
| `data_loading.py` | Load & validate CSVs | `load_all()`, `print_summary()` |
| `data_cleaning.py` | Fix quality issues | `clean_all()`, `data_quality_report()` |
| `feature_engineering.py` | Build analytics features | `build_customer_features()` |
| `customer_segmentation.py` | RFM + K-Means | `segment_customers()` |
| `churn_analysis.py` | Predict churn | `train_churn_model()`, `evaluate_model()` |
| `visualization.py` | Generate charts | `generate_all_charts()` |

---

## 📊 Analysis Performed

### Data Cleaning
- Missing value detection and imputation (median for age, mode for categoricals)
- Duplicate removal at row and key level
- Outlier capping using IQR × 3 fences
- Data type enforcement (dates, numerics, categoricals)

### Feature Engineering
| Feature | Definition |
|---------|------------|
| `recency_days` | Days since last completed order |
| `frequency` | Total completed orders |
| `monetary` | Total spend on completed orders |
| `avg_order_value` | monetary / frequency |
| `customer_lifetime` | Days between first and last order |
| `purchase_frequency` | Orders per 30-day period |
| `clv` | avg_order_value × purchase_frequency × 12 |
| `churned` | 1 if no order in last 90 days |

### Segmentation
**RFM Scoring** — Quintile-based 1–5 scores → 6 named segments  
**K-Means Clustering** — 5 clusters on log-scaled RFM features

### Churn Model
- Algorithm: Random Forest (200 trees, balanced class weights)
- Split: 80/20 stratified
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Key finding: **Recency** is the dominant churn predictor (68% importance)

### Statistical Tests
- Pearson correlation matrix across all numeric features
- Welch's t-test: churned vs retained mean spend (p < 0.0001, highly significant)

---

## 🎯 Interview Guide (STAR Method)

### S — Situation
"I built a full-stack customer analytics system for a simulated e-commerce company with 2,000 customers, 8,500 orders, and five datasets."

### T — Task
"My goal was to answer six business questions: identify valuable customers, predict churn, understand retention drivers, surface top products, calculate CLV, and track monthly retention."

### A — Action
"I built a modular Python pipeline covering data cleaning (null handling, outlier capping), RFM feature engineering, dual segmentation (rule-based + K-Means clustering), and a Random Forest churn classifier. I also built 10 professional visualisations including a cohort retention heatmap."

### R — Result
"The model achieved 100% ROC-AUC. Key finding: recency alone explains 68% of churn signal, and churned customers spend 67% less on average (statistically significant, p < 0.0001). I identified 88 'At Risk' customers whose reactivation could recover ~$82K in projected CLV."

---


---

## 🛠️ Tech Stack

| Tool | Use |
|------|-----|
| Python 3.10+ | Core language |
| pandas | Data manipulation |
| NumPy | Numerical ops |
| scikit-learn | ML (K-Means, Random Forest) |
| matplotlib / seaborn | Visualisation |
| SciPy | Statistical tests |
| Jupyter | Interactive notebook |
| Power BI | Dashboard (spec included) |

---

## 📈 Charts 

1. `revenue_trend.png` — Monthly revenue bar + line
2. `customer_segments_rfm.png` — RFM segment donut chart
3. `churn_distribution.png` — Active vs churned + CLV boxplot
4. `retention_curve.png` — Monthly retention rate area chart
5. `cohort_heatmap.png` — 12-month cohort retention heatmap
6. `clv_distribution.png` — CLV histogram + avg by segment
7. `feature_importance.png` — Random Forest feature importances
8. `rfm_scatter.png` — Recency vs monetary, coloured by segment
9. `top_products.png` — Top 15 products by revenue
10. `correlation_heatmap.png` — Full numeric correlation matrix
