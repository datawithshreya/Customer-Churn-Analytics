"""
visualization.py
----------------
Generates 10 publication-quality charts and saves them to
  Ecommerce_Customer_Analytics/dashboard/charts/

Charts produced
---------------
  1.  revenue_trend.png         – Monthly revenue line + bar
  2.  customer_segments_rfm.png – RFM segment distribution (donut)
  3.  churn_distribution.png    – Churned vs Active breakdown
  4.  retention_curve.png       – Monthly retention rate over time
  5.  cohort_heatmap.png        – Cohort retention heatmap
  6.  clv_distribution.png      – CLV histogram by segment
  7.  feature_importance.png    – Random Forest feature importances
  8.  rfm_scatter.png           – Recency vs Monetary coloured by segment
  9.  top_products.png          – Top 15 products by revenue
  10. correlation_heatmap.png   – Numeric feature correlation matrix
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────
PALETTE = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4","#FF5722"]
SEGMENT_COLORS = {
    "Champions":         "#2196F3",
    "Loyal Customers":   "#4CAF50",
    "Potential Loyalists":"#00BCD4",
    "At Risk":           "#FF9800",
    "Lost":              "#E91E63",
    "New Customers":     "#9C27B0",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR  = os.path.join(_SRC_DIR, "..", "dashboard", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def _save(fig, name: str) -> str:
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
# 1. REVENUE TREND
# ─────────────────────────────────────────────────────────────

def plot_revenue_trend(orders: pd.DataFrame) -> str:
    df = orders[orders["order_status"] == "Completed"].copy()
    df["month"] = df["order_date"].dt.to_period("M")
    monthly = df.groupby("month")["total_amount"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(monthly["month_str"], monthly["total_amount"] / 1000,
            color="#2196F3", alpha=0.6, label="Monthly Revenue")
    ax1.plot(monthly["month_str"], monthly["total_amount"] / 1000,
             color="#0D47A1", marker="o", markersize=4, linewidth=2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue (USD)")
    ax1.set_title("Monthly Revenue Trend")
    tick_step = max(1, len(monthly) // 12)
    ax1.set_xticks(range(0, len(monthly), tick_step))
    ax1.set_xticklabels(monthly["month_str"].iloc[::tick_step], rotation=45, ha="right")
    ax1.legend()
    fig.tight_layout()
    return _save(fig, "revenue_trend.png")


# ─────────────────────────────────────────────────────────────
# 2. CUSTOMER SEGMENTS (DONUT)
# ─────────────────────────────────────────────────────────────

def plot_customer_segments(feat: pd.DataFrame) -> str:
    counts = feat["rfm_segment"].value_counts()
    colors = [SEGMENT_COLORS.get(s, "#607D8B") for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(width=0.55),
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Customer Segments (RFM)")
    fig.tight_layout()
    return _save(fig, "customer_segments_rfm.png")


# ─────────────────────────────────────────────────────────────
# 3. CHURN DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_churn_distribution(feat: pd.DataFrame) -> str:
    churn_counts = feat["churned"].value_counts().rename({0: "Active", 1: "Churned"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = axes[0].bar(churn_counts.index, churn_counts.values,
                       color=["#4CAF50", "#E91E63"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, churn_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Active vs Churned Customers")
    axes[0].set_ylabel("Count")

    # CLV by churn status
    feat_plot = feat.copy()
    feat_plot["Status"] = feat_plot["churned"].map({0: "Active", 1: "Churned"})
    feat_plot[feat_plot["clv"] > 0].boxplot(column="clv", by="Status", ax=axes[1],
                                             patch_artist=True)
    axes[1].set_title("CLV Distribution: Active vs Churned")
    axes[1].set_xlabel("Status")
    axes[1].set_ylabel("Customer Lifetime Value ($)")
    plt.suptitle("")
    fig.tight_layout()
    return _save(fig, "churn_distribution.png")


# ─────────────────────────────────────────────────────────────
# 4. RETENTION CURVE
# ─────────────────────────────────────────────────────────────

def plot_retention_curve(orders: pd.DataFrame) -> str:
    df = orders[orders["order_status"] == "Completed"].copy()
    df["month"] = df["order_date"].dt.to_period("M")

    monthly_customers = df.groupby("month")["customer_id"].nunique()
    # Retention rate = repeat customers / total customers each month
    df["prev_order"] = df.sort_values("order_date").groupby("customer_id")["order_date"].shift(1)
    repeat = df[df["prev_order"].notna()].groupby("month")["customer_id"].nunique()
    retention = (repeat / monthly_customers * 100).fillna(0).reset_index()
    retention.columns = ["month", "retention_rate"]
    retention["month_str"] = retention["month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(range(len(retention)), retention["retention_rate"],
                    alpha=0.15, color="#2196F3")
    ax.plot(range(len(retention)), retention["retention_rate"],
            color="#2196F3", linewidth=2.5, marker="o", markersize=4)
    ax.axhline(retention["retention_rate"].mean(), color="#E91E63",
               linestyle="--", linewidth=1.5, label=f"Avg {retention['retention_rate'].mean():.1f}%")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylim(0, 100)
    tick_step = max(1, len(retention) // 12)
    ax.set_xticks(range(0, len(retention), tick_step))
    ax.set_xticklabels(retention["month_str"].iloc[::tick_step], rotation=45, ha="right")
    ax.set_title("Monthly Customer Retention Rate")
    ax.set_ylabel("Retention Rate (%)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "retention_curve.png")


# ─────────────────────────────────────────────────────────────
# 5. COHORT HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_cohort_heatmap(orders: pd.DataFrame, customers: pd.DataFrame) -> str:
    df = orders[orders["order_status"] == "Completed"].copy()
    cust = customers[["customer_id","registration_date"]].copy()

    # Cohort = registration month
    cust["cohort_month"] = pd.to_datetime(cust["registration_date"]).dt.to_period("M")
    df = df.merge(cust, on="customer_id", how="left")
    df["order_month"]   = df["order_date"].dt.to_period("M")
    df["period_number"] = (
        df["order_month"].apply(lambda x: x.ordinal) -
        df["cohort_month"].apply(lambda x: x.ordinal)
    )

    cohort_data = (
        df[df["period_number"] >= 0]
        .groupby(["cohort_month","period_number"])["customer_id"]
        .nunique()
        .reset_index()
    )
    cohort_sizes = (
        df[df["period_number"] == 0]
        .groupby("cohort_month")["customer_id"]
        .nunique()
    )
    cohort_data["cohort_size"] = cohort_data["cohort_month"].map(cohort_sizes)
    cohort_data["retention"]   = (cohort_data["customer_id"] / cohort_data["cohort_size"] * 100).round(1)

    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period_number", values="retention"
    )
    # Keep only first 12 periods & last 12 cohorts for readability
    cohort_pivot = cohort_pivot.iloc[-12:, :12]
    cohort_pivot.index = cohort_pivot.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(cohort_pivot, annot=True, fmt=".0f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Retention %"})
    ax.set_title("Cohort Retention Heatmap (%)")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort (Registration Month)")
    fig.tight_layout()
    return _save(fig, "cohort_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 6. CLV DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_clv_distribution(feat: pd.DataFrame) -> str:
    df = feat[feat["clv"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df["clv"].clip(upper=df["clv"].quantile(0.95)),
                 bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("CLV Distribution (95th pct cap)")
    axes[0].set_xlabel("CLV ($)")
    axes[0].set_ylabel("Count")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Average CLV by segment
    seg_clv = df.groupby("rfm_segment")["clv"].mean().sort_values(ascending=True)
    colors  = [SEGMENT_COLORS.get(s, "#607D8B") for s in seg_clv.index]
    axes[1].barh(seg_clv.index, seg_clv.values, color=colors)
    for i, v in enumerate(seg_clv.values):
        axes[1].text(v + 5, i, f"${v:,.0f}", va="center", fontsize=10)
    axes[1].set_title("Average CLV by Segment")
    axes[1].set_xlabel("CLV ($)")
    fig.tight_layout()
    return _save(fig, "clv_distribution.png")


# ─────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(imp: pd.DataFrame) -> str:
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1],
                   color=PALETTE[0], edgecolor="white")
    for bar, val in zip(bars, top["importance"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_title("Random Forest – Feature Importances (Churn Prediction)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


# ─────────────────────────────────────────────────────────────
# 8. RFM SCATTER
# ─────────────────────────────────────────────────────────────

def plot_rfm_scatter(feat: pd.DataFrame) -> str:
    df = feat[feat["monetary"] > 0].sample(min(500, len(feat)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    for seg in df["rfm_segment"].unique():
        mask = df["rfm_segment"] == seg
        ax.scatter(df.loc[mask, "recency_days"], df.loc[mask, "monetary"],
                   label=seg, alpha=0.6, s=40,
                   color=SEGMENT_COLORS.get(seg, "#607D8B"))
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Total Spend ($)")
    ax.set_title("RFM Scatter – Recency vs Monetary by Segment")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return _save(fig, "rfm_scatter.png")


# ─────────────────────────────────────────────────────────────
# 9. TOP PRODUCTS
# ─────────────────────────────────────────────────────────────

def plot_top_products(orders: pd.DataFrame, products: pd.DataFrame) -> str:
    df = (
        orders[orders["order_status"] == "Completed"]
        .groupby("product_id")["total_amount"].sum()
        .reset_index()
        .merge(products[["product_id","product_name","category"]], on="product_id")
        .sort_values("total_amount", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    ax.barh(df["product_name"][::-1], df["total_amount"][::-1] / 1000, color=colors[::-1])
    ax.set_xlabel("Revenue (USD Thousands)")
    ax.set_title("Top 15 Products by Revenue")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    fig.tight_layout()
    return _save(fig, "top_products.png")


# ─────────────────────────────────────────────────────────────
# 10. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(feat: pd.DataFrame) -> str:
    cols = [
        "age","recency_days","frequency","monetary",
        "avg_order_value","customer_lifetime","clv",
        "loyalty_points","avg_review_rating","churned",
    ]
    cols = [c for c in cols if c in feat.columns]
    corr = feat[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
# MASTER
# ─────────────────────────────────────────────────────────────

def generate_all_charts(
    orders:    pd.DataFrame,
    customers: pd.DataFrame,
    products:  pd.DataFrame,
    feat:      pd.DataFrame,
    imp:       pd.DataFrame,
) -> list:
    """
    Runs all 10 charts. Returns list of saved file paths.
    """
    paths = []
    steps = [
        ("Revenue Trend",         lambda: plot_revenue_trend(orders)),
        ("Customer Segments",     lambda: plot_customer_segments(feat)),
        ("Churn Distribution",    lambda: plot_churn_distribution(feat)),
        ("Retention Curve",       lambda: plot_retention_curve(orders)),
        ("Cohort Heatmap",        lambda: plot_cohort_heatmap(orders, customers)),
        ("CLV Distribution",      lambda: plot_clv_distribution(feat)),
        ("Feature Importance",    lambda: plot_feature_importance(imp)),
        ("RFM Scatter",           lambda: plot_rfm_scatter(feat)),
        ("Top Products",          lambda: plot_top_products(orders, products)),
        ("Correlation Heatmap",   lambda: plot_correlation_heatmap(feat)),
    ]
    print("\n--- Generating Visualisations ----------------")
    for name, fn in steps:
        path = fn()
        paths.append(path)
        print(f"  [OK] {name:<25} -> {os.path.basename(path)}")
    return paths
