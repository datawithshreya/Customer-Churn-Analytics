"""
Dataset Generation Script
Generates realistic e-commerce datasets for portfolio project.
Run this once to create all CSV files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
N_CUSTOMERS   = 2000
N_ORDERS      = 8500
N_PRODUCTS    = 150
START_DATE    = datetime(2022, 1, 1)
END_DATE      = datetime(2024, 3, 31)

CITIES = [
    "New York","Los Angeles","Chicago","Houston","Phoenix",
    "Philadelphia","San Antonio","San Diego","Dallas","San Jose",
    "Austin","Jacksonville","Fort Worth","Columbus","Charlotte",
    "Indianapolis","Seattle","Denver","Washington","Nashville",
]
STATES = [
    "NY","CA","IL","TX","AZ","PA","TX","CA","TX","CA",
    "TX","FL","TX","OH","NC","IN","WA","CO","DC","TN",
]
SEGMENTS = ["B2C","B2B","Wholesale"]
GENDERS   = ["Male","Female","Non-binary","Prefer not to say"]
CATEGORIES = [
    "Electronics","Clothing","Home & Garden","Sports","Books",
    "Beauty","Toys","Automotive","Food & Grocery","Office Supplies",
]
PAYMENT_METHODS = ["Credit Card","Debit Card","PayPal","Apple Pay","Google Pay","Bank Transfer"]
ORDER_STATUSES  = ["Completed","Cancelled","Returned","Processing"]


def random_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


# ─────────────────────────────────────────
# 1. CUSTOMERS
# ─────────────────────────────────────────
def generate_customers():
    print("  Generating customers.csv ...")
    city_state = list(zip(CITIES, STATES))
    records = []
    for i in range(1, N_CUSTOMERS + 1):
        reg_date = random_date(START_DATE, END_DATE - timedelta(days=30))
        city, state = random.choice(city_state)
        age = int(np.random.normal(38, 12))
        age = max(18, min(80, age))
        records.append({
            "customer_id":        f"CUST{i:05d}",
            "first_name":         random.choice(["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Barbara","David","Elizabeth"]),
            "last_name":          random.choice(["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson","Martinez","Anderson","Taylor"]),
            "email":              f"user{i}@{'gmail' if i%3==0 else 'yahoo' if i%3==1 else 'outlook'}.com",
            "phone":              f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            "city":               city,
            "state":              state,
            "country":            "USA",
            "registration_date":  reg_date.strftime("%Y-%m-%d"),
            "age":                age,
            "gender":             random.choice(GENDERS),
            "customer_segment":   np.random.choice(SEGMENTS, p=[0.75, 0.18, 0.07]),
            "loyalty_points":     random.randint(0, 5000),
            "is_email_verified":  random.choice([True, True, True, False]),
        })
    df = pd.DataFrame(records)
    # Inject ~3% missing values
    for col in ["phone", "age", "gender"]:
        mask = np.random.rand(len(df)) < 0.03
        df.loc[mask, col] = np.nan
    df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    print(f"    → {len(df)} customers written.")
    return df


# ─────────────────────────────────────────
# 2. PRODUCTS
# ─────────────────────────────────────────
def generate_products():
    print("  Generating products.csv ...")
    records = []
    for i in range(1, N_PRODUCTS + 1):
        cat = random.choice(CATEGORIES)
        base = np.random.lognormal(mean=3.5, sigma=0.9)
        price = round(max(5.0, min(2000.0, base)), 2)
        records.append({
            "product_id":   f"PROD{i:04d}",
            "product_name": f"{cat} Item {i}",
            "category":     cat,
            "sub_category": f"{cat} Sub-{random.randint(1,4)}",
            "price":        price,
            "cost":         round(price * random.uniform(0.35, 0.65), 2),
            "stock_qty":    random.randint(0, 500),
            "supplier":     f"Supplier-{random.randint(1,20)}",
            "rating":       round(random.uniform(2.5, 5.0), 1),
            "review_count": random.randint(0, 2000),
        })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    print(f"    → {len(df)} products written.")
    return df


# ─────────────────────────────────────────
# 3. ORDERS
# ─────────────────────────────────────────
def generate_orders(customers_df, products_df):
    print("  Generating orders.csv ...")
    customer_ids = customers_df["customer_id"].tolist()
    reg_dates    = dict(zip(customers_df["customer_id"], pd.to_datetime(customers_df["registration_date"])))
    product_ids  = products_df["product_id"].tolist()
    prices       = dict(zip(products_df["product_id"], products_df["price"]))

    # Make ~20% of customers heavy buyers (simulate power users)
    heavy   = random.sample(customer_ids, int(0.20 * len(customer_ids)))
    heavy_s = set(heavy)

    records = []
    order_id = 1
    assigned = {c: 0 for c in customer_ids}

    for _ in range(N_ORDERS):
        cid = random.choice(heavy) if random.random() < 0.45 else random.choice(customer_ids)
        assigned[cid] += 1
        reg = reg_dates[cid]
        order_date = random_date(reg + timedelta(days=1), END_DATE)

        pid      = random.choice(product_ids)
        qty      = random.randint(1, 5)
        unit_p   = prices[pid]
        discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
        subtotal = round(unit_p * qty * (1 - discount), 2)
        shipping = round(random.uniform(0, 15), 2) if subtotal < 50 else 0.0
        total    = round(subtotal + shipping, 2)
        status   = np.random.choice(ORDER_STATUSES, p=[0.82, 0.08, 0.06, 0.04])

        records.append({
            "order_id":        f"ORD{order_id:06d}",
            "customer_id":     cid,
            "product_id":      pid,
            "order_date":      order_date.strftime("%Y-%m-%d"),
            "quantity":        qty,
            "unit_price":      unit_p,
            "discount":        discount,
            "subtotal":        subtotal,
            "shipping_cost":   shipping,
            "total_amount":    total,
            "order_status":    status,
            "delivery_days":   random.randint(1, 14) if status == "Completed" else None,
            "channel":         np.random.choice(["Website","Mobile App","In-Store","Phone"], p=[0.55,0.30,0.10,0.05]),
        })
        order_id += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)
    print(f"    → {len(df)} orders written.")
    return df


# ─────────────────────────────────────────
# 4. PAYMENTS
# ─────────────────────────────────────────
def generate_payments(orders_df):
    print("  Generating payments.csv ...")
    completed = orders_df[orders_df["order_status"] == "Completed"].copy()
    records   = []
    for _, row in completed.iterrows():
        pay_date = pd.to_datetime(row["order_date"]) + timedelta(days=random.randint(0, 2))
        records.append({
            "payment_id":     f"PAY{_ :06d}",
            "order_id":       row["order_id"],
            "customer_id":    row["customer_id"],
            "payment_date":   pay_date.strftime("%Y-%m-%d"),
            "amount":         row["total_amount"],
            "payment_method": random.choice(PAYMENT_METHODS),
            "payment_status": np.random.choice(["Success","Failed","Refunded"], p=[0.95,0.03,0.02]),
            "transaction_id": f"TXN{random.randint(10000000,99999999)}",
        })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "payments.csv"), index=False)
    print(f"    → {len(df)} payments written.")
    return df


# ─────────────────────────────────────────
# 5. REVIEWS
# ─────────────────────────────────────────
def generate_reviews(orders_df):
    print("  Generating reviews.csv ...")
    completed = orders_df[orders_df["order_status"] == "Completed"].sample(frac=0.55, random_state=42).copy()
    sentiments = {5:"Excellent product!", 4:"Very good, happy with purchase.",
                  3:"Average, could be better.", 2:"Disappointed with quality.", 1:"Terrible, do not buy."}
    records = []
    for _, row in completed.iterrows():
        rating  = np.random.choice([1,2,3,4,5], p=[0.05,0.08,0.15,0.32,0.40])
        records.append({
            "review_id":    f"REV{_ :06d}",
            "order_id":     row["order_id"],
            "customer_id":  row["customer_id"],
            "product_id":   row["product_id"],
            "rating":       rating,
            "review_text":  sentiments[rating],
            "review_date":  (pd.to_datetime(row["order_date"]) + timedelta(days=random.randint(3,30))).strftime("%Y-%m-%d"),
            "helpful_votes":random.randint(0, 150),
        })
    df = pd.DataFrame(records)
    # Inject a few nulls in review_text
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, "review_text"] = np.nan
    df.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)
    print(f"    → {len(df)} reviews written.")
    return df


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating e-commerce datasets...")
    customers = generate_customers()
    products  = generate_products()
    orders    = generate_orders(customers, products)
    payments  = generate_payments(orders)
    reviews   = generate_reviews(orders)
    print("\nAll datasets generated successfully!\n")
