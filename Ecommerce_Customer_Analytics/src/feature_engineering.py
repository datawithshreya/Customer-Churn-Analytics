"""
feature_engineering.py
-----------------------
Builds a rich customer-level feature table used by segmentation,
churn prediction and the dashboard.

Features created
----------------
  RFM:
    recency_days        – days since last order
    frequency           – total number of completed orders
    monetary            – total spend (completed orders)

  Derived:
    avg_order_value     – monetary / frequency
    purchase_frequency  – orders per month of customer lifetime
    customer_lifetime   – days between first and last order
    clv                 – Customer Lifetime Value  (simple 12-month projection)
    avg_review_rating   – mean rating left by the customer
    total_reviews       – number of reviews written
    favourite_category  – most purchased product category
    preferred_channel   – most-used purchase channel

  Churn flag:
    churned             – 1 if no order in the last 90 days, else 0
"""

import pandas as pd
import numpy as np

CHURN_DAYS     = 90        # definition: no purchase in last N days
SNAPSHOT_DATE  = pd.Timestamp("2024-03-31")   # analysis reference date


# ─────────────────────────────────────────────────────────────
def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    """
    RFM from completed orders only.
    Returns one row per customer.
    """
    completed = orders[orders["order_status"] == "Completed"].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_date = ("order_date", "max"),
            first_order_date= ("order_date", "min"),
            frequency       = ("order_id",   "count"),
            monetary        = ("total_amount","sum"),
        )
        .reset_index()
    )

    rfm["recency_days"]     = (SNAPSHOT_DATE - rfm["last_order_date"]).dt.days
    rfm["customer_lifetime"]= (rfm["last_order_date"] - rfm["first_order_date"]).dt.days.clip(lower=1)
    rfm["avg_order_value"]  = (rfm["monetary"] / rfm["frequency"]).round(2)

    # Purchase frequency = orders per 30-day period of lifetime
    pf = rfm["frequency"] / (rfm["customer_lifetime"] / 30)
    rfm["purchase_frequency"] = (
        pf.where(np.isfinite(pf), rfm["frequency"])
        .round(3)
    )

    return rfm


def build_clv(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 12-month CLV projection.
    CLV = avg_order_value × purchase_frequency × 12
    """
    rfm = rfm.copy()
    rfm["clv"] = (rfm["avg_order_value"] * rfm["purchase_frequency"] * 12).round(2)
    return rfm


def build_churn_flag(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["churned"] = (rfm["recency_days"] > CHURN_DAYS).astype(int)
    return rfm


def build_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    rev_feat = (
        reviews.groupby("customer_id")
        .agg(
            avg_review_rating = ("rating",   "mean"),
            total_reviews     = ("review_id","count"),
        )
        .reset_index()
    )
    rev_feat["avg_review_rating"] = rev_feat["avg_review_rating"].round(2)
    return rev_feat


def build_category_features(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    merged = orders[orders["order_status"] == "Completed"].merge(
        products[["product_id","category"]], on="product_id", how="left"
    )
    cat_feat = (
        merged.groupby("customer_id")["category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        .reset_index()
        .rename(columns={"category": "favourite_category"})
    )
    chan_feat = (
        orders[orders["order_status"] == "Completed"]
        .groupby("customer_id")["channel"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        .reset_index()
        .rename(columns={"channel": "preferred_channel"})
    )
    return cat_feat.merge(chan_feat, on="customer_id", how="outer")


def build_customer_features(
    customers: pd.DataFrame,
    orders:    pd.DataFrame,
    products:  pd.DataFrame,
    reviews:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Master function – returns one row per customer with all features.
    Customers with zero completed orders get NaN for RFM and are flagged churned.
    """
    rfm  = build_rfm(orders)
    rfm  = build_clv(rfm)
    rfm  = build_churn_flag(rfm)

    rev  = build_review_features(reviews)
    cats = build_category_features(orders, products)

    # Merge everything onto the customer base
    feat = customers.merge(rfm,  on="customer_id", how="left")
    feat = feat.merge(rev,       on="customer_id", how="left")
    feat = feat.merge(cats,      on="customer_id", how="left")

    # Customers who never placed a completed order → churned
    feat["churned"]      = feat["churned"].fillna(1).astype(int)
    feat["frequency"]    = feat["frequency"].fillna(0).astype(int)
    feat["monetary"]     = feat["monetary"].fillna(0)
    feat["recency_days"] = feat["recency_days"].fillna(
        (SNAPSHOT_DATE - feat["registration_date"]).dt.days
    )
    feat["avg_order_value"]    = feat["avg_order_value"].fillna(0)
    feat["purchase_frequency"] = feat["purchase_frequency"].fillna(0)
    feat["customer_lifetime"]  = feat["customer_lifetime"].fillna(0).astype(int)
    feat["clv"]                = feat["clv"].fillna(0)
    feat["avg_review_rating"]  = feat["avg_review_rating"].fillna(0)
    feat["total_reviews"]      = feat["total_reviews"].fillna(0).astype(int)
    feat["favourite_category"] = feat["favourite_category"].fillna("None")
    feat["preferred_channel"]  = feat["preferred_channel"].fillna("None")

    return feat


def print_feature_summary(feat: pd.DataFrame) -> None:
    print("\n--- Feature Engineering Completed ----------------")
    print(f"  Total customers  : {len(feat):,}")
    print(f"  Churned          : {feat['churned'].sum():,}  ({feat['churned'].mean()*100:.1f}%)")
    print(f"  Avg CLV          : ${feat['clv'].mean():,.2f}")
    print(f"  Avg Recency      : {feat['recency_days'].mean():.0f} days")
    print(f"  Avg Frequency    : {feat['frequency'].mean():.1f} orders")
    print(f"  Avg Spend        : ${feat['monetary'].mean():,.2f}")
    print()
