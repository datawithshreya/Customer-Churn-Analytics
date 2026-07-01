"""
main.py
-------
Master execution pipeline for the E-Commerce Customer Analytics project.

Usage
-----
  python main.py          # full pipeline (data gen + analysis + charts)
  python main.py --skip-data  # skip data generation (datasets already exist)
"""

import sys
import os
import argparse
import time

import pandas as pd

# ── ensure src/ is importable ──────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

from data_loading import load_all, print_summary
from data_cleaning import clean_all, print_quality_report, print_cleaning_summary
from feature_engineering import build_customer_features, print_feature_summary
from customer_segmentation import segment_customers, print_segmentation_results
from churn_analysis import (
    train_churn_model, evaluate_model,
    get_feature_importance, correlation_analysis,
    hypothesis_test, print_churn_results,
)
from visualization import generate_all_charts

# ─────────────────────────────────────────────────────────────
# BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────

def derive_insights(feat: pd.DataFrame, orders: pd.DataFrame, metrics: dict) -> list:
    completed = orders[orders["order_status"] == "Completed"]

    total_revenue     = completed["total_amount"].sum()
    top_segment       = feat.groupby("rfm_segment")["clv"].mean().idxmax()
    churn_rate        = feat["churned"].mean() * 100
    avg_clv           = feat[feat["clv"] > 0]["clv"].mean()
    top_channel       = completed["channel"].value_counts().idxmax()
    champions_revenue = feat[feat["rfm_segment"] == "Champions"]["monetary"].sum()
    champions_pct     = champions_revenue / feat["monetary"].sum() * 100 if feat["monetary"].sum() > 0 else 0
    top_category      = (
        completed.merge(
            pd.read_csv(os.path.join(DATA_DIR, "products.csv"))[["product_id","category"]],
            on="product_id", how="left"
        )["category"].value_counts().idxmax()
    )
    at_risk_count = (feat["rfm_segment"] == "At Risk").sum()

    insights = [
        f"Overall churn rate is {churn_rate:.1f}% — "
        f"re-engagement campaigns targeting {at_risk_count:,} 'At Risk' customers "
        f"could recover significant revenue.",

        f"Champions segment has the highest avg CLV (${feat[feat['rfm_segment']=='Champions']['clv'].mean():,.0f}) "
        f"and contributes {champions_pct:.0f}% of total spend — "
        f"prioritise loyalty rewards for this group.",

        f"Total completed revenue: ${total_revenue:,.0f}  |  "
        f"'{top_channel}' is the dominant purchase channel.",

        f"Average Customer Lifetime Value across all active buyers: ${avg_clv:,.2f}. "
        f"Segment with highest avg CLV: '{top_segment}'.",

        f"Top revenue-generating product category: '{top_category}'. "
        f"Focus inventory and promotions here for highest ROI.",

        f"Churn model ROC-AUC = {metrics['roc_auc']:.3f}  "
        f"(Accuracy {metrics['accuracy']*100:.1f}%, Recall {metrics['recall']*100:.1f}%). "
        f"Recency & frequency are the top churn predictors.",
    ]
    return insights


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(skip_data: bool = False):
    t0 = time.time()

    print("\n" + "=" * 52)
    print("   E-COMMERCE CUSTOMER ANALYTICS PLATFORM")
    print("=" * 52)

    # ── 0. Generate datasets ───────────────────────────────
    if not skip_data:
        print("\n[0/6] Generating datasets …")
        # Import here so generate_data.py's __main__ guard doesn't fire
        sys.path.insert(0, DATA_DIR)
        import generate_data  # noqa: F401 – runs generation on import
    else:
        print("\n[0/6] Skipping data generation (--skip-data flag).")

    # ── 1. Load ────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    bundle = load_all()
    print_summary(bundle)

    # ── 2. Clean ───────────────────────────────────────────
    print("[2/6] Cleaning data …")
    print_quality_report(bundle)
    cleaned = clean_all(bundle)
    print_cleaning_summary(bundle, cleaned)

    customers = cleaned["customers"]
    orders    = cleaned["orders"]
    products  = cleaned["products"]
    payments  = cleaned["payments"]
    reviews   = cleaned["reviews"]

    # ── 3. Feature Engineering ─────────────────────────────
    print("[3/6] Engineering features …")
    feat = build_customer_features(customers, orders, products, reviews)
    print_feature_summary(feat)

    # ── 4. Segmentation ────────────────────────────────────
    print("[4/6] Segmenting customers …")
    feat, km_model, scaler = segment_customers(feat)
    print_segmentation_results(feat)

    # ── 5. Churn Model ─────────────────────────────────────
    print("[5/6] Training churn model …")
    model, X_train, X_test, y_train, y_test, y_pred, y_prob, feature_names = train_churn_model(feat)
    metrics, cm = evaluate_model(y_test, y_pred, y_prob)
    imp         = get_feature_importance(model, feature_names)
    corr        = correlation_analysis(feat)
    hyp         = hypothesis_test(feat)
    print_churn_results(metrics, cm, imp, corr, hyp)

    # ── 6. Visualisations ──────────────────────────────────
    print("[6/6] Generating charts …")
    generate_all_charts(orders, customers, products, feat, imp)

    # ── Business Insights ──────────────────────────────────
    insights = derive_insights(feat, orders, metrics)
    print("=" * 52)
    print("   BUSINESS INSIGHTS")
    print("=" * 52)
    for i, insight in enumerate(insights, 1):
        # Word-wrap at ~75 chars
        words = insight.split()
        line, lines = [], []
        for w in words:
            if len(" ".join(line + [w])) > 72:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        print(f"\n  {i}. {lines[0]}")
        for rest in lines[1:]:
            print(f"     {rest}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 52}")
    print(f"   Pipeline completed in {elapsed:.1f}s")
    print(f"   Charts saved -> dashboard/charts/")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Customer Analytics")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip dataset generation step")
    args = parser.parse_args()
    main(skip_data=args.skip_data)
