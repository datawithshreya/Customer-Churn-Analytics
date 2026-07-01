"""
data_cleaning.py
----------------
Full data-quality pipeline:
  1. Data-quality report (nulls, dupes, dtypes)
  2. Missing-value imputation / removal
  3. Duplicate removal
  4. Outlier detection & capping (IQR method)
  5. Type corrections
Returns cleaned versions of every DataFrame.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────

def data_quality_report(bundle: dict) -> pd.DataFrame:
    """
    Build a single tidy DataFrame summarising quality metrics
    for every table.
    """
    rows = []
    for name, df in bundle.items():
        for col in df.columns:
            null_cnt  = df[col].isna().sum()
            null_pct  = round(null_cnt / len(df) * 100, 2)
            dup_cnt   = df.duplicated(subset=[col]).sum()
            rows.append({
                "table":        name,
                "column":       col,
                "dtype":        str(df[col].dtype),
                "null_count":   null_cnt,
                "null_%":       null_pct,
                "unique_vals":  df[col].nunique(),
                "duplicate_rows": df.duplicated().sum(),
            })
    report = pd.DataFrame(rows)
    return report


def print_quality_report(bundle: dict) -> None:
    report = data_quality_report(bundle)
    print("\n--- Data Quality Report ----------------")
    for table, grp in report.groupby("table"):
        nulls = grp[grp["null_count"] > 0][["column","null_count","null_%"]]
        dupes = bundle[table].duplicated().sum()
        print(f"\n  [{table.upper()}]  rows={len(bundle[table]):,}  duplicate_rows={dupes}")
        if nulls.empty:
            print("    No missing values.")
        else:
            for _, r in nulls.iterrows():
                print(f"    [WARNING] {r['column']:<25} nulls={r['null_count']:>5} ({r['null_%']}%)")

# ─────────────────────────────────────────────────────────────
# 2. CLEAN CUSTOMERS
# ─────────────────────────────────────────────────────────────

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove full duplicates
    df.drop_duplicates(inplace=True)

    # Remove duplicate customer_ids (keep first)
    df.drop_duplicates(subset=["customer_id"], keep="first", inplace=True)

    # Age: impute median; cap to [18, 90]
    median_age = df["age"].median()
    df["age"]  = df["age"].fillna(median_age).clip(18, 90).astype(int)

    # Gender: fill unknown
    df["gender"] = df["gender"].fillna("Unknown")

    # Phone: fill placeholder
    df["phone"] = df["phone"].fillna("N/A")

    # registration_date must not be null
    df.dropna(subset=["registration_date"], inplace=True)

    # Ensure loyalty_points >= 0
    df["loyalty_points"] = df["loyalty_points"].clip(lower=0)

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 3. CLEAN ORDERS
# ─────────────────────────────────────────────────────────────

def _cap_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """Cap outliers beyond factor*IQR to the fence values."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lo, hi)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["order_id"], keep="first", inplace=True)

    # Drop rows without essential IDs or dates
    df.dropna(subset=["order_id","customer_id","product_id","order_date"], inplace=True)

    # Numeric corrections
    df["quantity"]     = df["quantity"].clip(lower=1)
    df["unit_price"]   = df["unit_price"].clip(lower=0.01)
    df["discount"]     = df["discount"].clip(0, 0.99)
    df["total_amount"] = df["total_amount"].clip(lower=0)

    # Outlier capping on total_amount
    df["total_amount"] = _cap_outliers_iqr(df["total_amount"])

    # Delivery days: cap to [1, 60]
    df["delivery_days"] = pd.to_numeric(df["delivery_days"], errors="coerce").clip(1, 60)

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 4. CLEAN PRODUCTS
# ─────────────────────────────────────────────────────────────

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["product_id"], keep="first", inplace=True)
    df["price"]  = df["price"].clip(lower=0.01)
    df["cost"]   = df["cost"].clip(lower=0.01)
    df["rating"] = df["rating"].clip(1.0, 5.0)
    df["stock_qty"] = df["stock_qty"].clip(lower=0).fillna(0).astype(int)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 5. CLEAN PAYMENTS
# ─────────────────────────────────────────────────────────────

def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["payment_id"], keep="first", inplace=True)
    df.dropna(subset=["payment_id","order_id","amount"], inplace=True)
    df["amount"] = df["amount"].clip(lower=0)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# 6. CLEAN REVIEWS
# ─────────────────────────────────────────────────────────────

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset=["review_id"], keep="first", inplace=True)
    df.dropna(subset=["review_id","order_id","customer_id"], inplace=True)
    df["rating"] = df["rating"].clip(1, 5)
    df["review_text"] = df["review_text"].fillna("No review text")
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# MASTER CLEAN PIPELINE
# ─────────────────────────────────────────────────────────────

def clean_all(bundle: dict) -> dict:
    cleaned = {
        "customers": clean_customers(bundle["customers"]),
        "orders":    clean_orders(bundle["orders"]),
        "products":  clean_products(bundle["products"]),
        "payments":  clean_payments(bundle["payments"]),
        "reviews":   clean_reviews(bundle["reviews"]),
    }
    return cleaned


def print_cleaning_summary(raw: dict, cleaned: dict) -> None:
    print("\n--- Data Quality Report ---")
    for name in raw:
        before = len(raw[name])
        after  = len(cleaned[name])
        removed = before - after
        print(f"  {name.capitalize():<12}: {before:>6,} -> {after:>6,} rows  (removed {removed:,})")
    print()
