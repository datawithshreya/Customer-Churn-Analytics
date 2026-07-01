"""
data_loading.py
---------------
Loads all five CSV datasets, validates schemas, and returns a
single DataBundle dict used by every downstream module.
"""

import os
import pandas as pd

# Resolve data directory relative to this file
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "..", "data")

EXPECTED_COLUMNS = {
    "customers": ["customer_id","registration_date","age","gender","city","state","customer_segment","loyalty_points"],
    "orders":    ["order_id","customer_id","product_id","order_date","quantity","unit_price","discount","total_amount","order_status","channel"],
    "products":  ["product_id","product_name","category","price","cost","rating"],
    "payments":  ["payment_id","order_id","customer_id","payment_date","amount","payment_method","payment_status"],
    "reviews":   ["review_id","order_id","customer_id","product_id","rating","review_date"],
}

DATE_COLUMNS = {
    "customers": ["registration_date"],
    "orders":    ["order_date"],
    "payments":  ["payment_date"],
    "reviews":   ["review_date"],
}


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run  python data/generate_data.py  first."
        )
    df = pd.read_csv(path, low_memory=False)
    # Parse date columns
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _validate(name: str, df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}")


def load_all() -> dict:
    """
    Returns
    -------
    dict with keys: customers, orders, products, payments, reviews
    """
    bundle = {}
    names  = ["customers", "orders", "products", "payments", "reviews"]
    for name in names:
        df = _load_csv(name)
        _validate(name, df)
        bundle[name] = df

    return bundle


def print_summary(bundle: dict) -> None:
    print("\n" + "=" * 48)
    print("     E-COMMERCE CUSTOMER ANALYTICS PLATFORM")
    print("=" * 48)
    print("Dataset loaded successfully\n")
    for name, df in bundle.items():
        print(f"  {name.capitalize():<12}: {len(df):>6,} rows  ×  {df.shape[1]} cols")
    print()
