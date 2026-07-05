"""
etl.py — E-Commerce Analytics Platform
Loads Olist Brazilian E-Commerce CSVs into a SQLite star schema warehouse.

Usage:
    python src/etl.py

Prerequisites:
    Place all Olist CSVs inside data/raw/
    Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime

RAW_PATH = "data/raw"
DB_PATH  = "data/ecommerce.db"
SQL_PATH = "sql/schema.sql"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def run_schema(engine):
    with open(SQL_PATH, "r") as f:
        schema_sql = f.read()
    with engine.connect() as conn:
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    print("✓ Schema created")


def csv(filename):
    path = os.path.join(RAW_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nMissing file: {path}\n"
            "Download the Olist dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
            "Unzip all CSVs into data/raw/\n"
        )
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# Dimension Builders
# ─────────────────────────────────────────────

def build_dim_customer(engine):
    df = csv("olist_customers_dataset.csv")
    df = df.rename(columns={
        "customer_city":  "city",
        "customer_state": "state"
    })[["customer_id", "customer_unique_id", "city", "state"]]
    df.drop_duplicates("customer_id", inplace=True)
    df.to_sql("dim_customer", engine, if_exists="replace", index=False)
    print(f"✓ dim_customer: {len(df):,} rows")
    return df


def build_dim_product(engine):
    products    = csv("olist_products_dataset.csv")
    translation = csv("product_category_name_translation.csv")
    df = products.merge(translation, on="product_category_name", how="left")
    df = df.rename(columns={
        "product_category_name":         "category_raw",
        "product_category_name_english": "category_english",
        "product_weight_g":              "weight_g",
        "product_length_cm":             "length_cm",
        "product_height_cm":             "height_cm",
        "product_width_cm":              "width_cm"
    })[["product_id", "category_raw", "category_english",
        "weight_g", "length_cm", "height_cm", "width_cm"]]
    df.drop_duplicates("product_id", inplace=True)
    df.to_sql("dim_product", engine, if_exists="replace", index=False)
    print(f"✓ dim_product: {len(df):,} rows")
    return df


def build_dim_seller(engine):
    df = csv("olist_sellers_dataset.csv")
    df = df.rename(columns={
        "seller_city":  "city",
        "seller_state": "state"
    })[["seller_id", "city", "state"]]
    df.drop_duplicates("seller_id", inplace=True)
    df.to_sql("dim_seller", engine, if_exists="replace", index=False)
    print(f"✓ dim_seller: {len(df):,} rows")
    return df


def build_dim_date(engine, date_series):
    dates = pd.to_datetime(date_series.dropna().unique())
    rows = []
    for d in dates:
        rows.append({
            "date_id":     d.strftime("%Y-%m-%d"),
            "year":        d.year,
            "quarter":     d.quarter,
            "month":       d.month,
            "month_name":  d.strftime("%B"),
            "week":        d.isocalendar().week,
            "day_of_week": d.weekday(),
            "day_name":    d.strftime("%A"),
            "is_weekend":  int(d.weekday() >= 5)
        })
    df = pd.DataFrame(rows).drop_duplicates("date_id")
    df.to_sql("dim_date", engine, if_exists="replace", index=False)
    print(f"✓ dim_date: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# Fact Builder
# ─────────────────────────────────────────────

def build_fact_orders(engine):
    orders   = csv("olist_orders_dataset.csv")
    items    = csv("olist_order_items_dataset.csv")
    payments = csv("olist_order_payments_dataset.csv")
    reviews  = csv("olist_order_reviews_dataset.csv")

    # Aggregate payments per order (sum across installments)
    pay_agg = (payments
        .groupby("order_id")
        .agg(payment_value=("payment_value", "sum"),
             payment_type =("payment_type",  lambda x: x.mode()[0]),
             installments =("payment_installments", "max"))
        .reset_index()
    )

    # Best review per order (some have multiple)
    rev_agg = (reviews
        .groupby("order_id")["review_score"]
        .mean()
        .reset_index()
    )

    # Parse date columns
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Build dim_date from all order dates
    all_dates = pd.concat([orders[c].dropna() for c in date_cols]).dt.normalize()
    build_dim_date(engine, all_dates.astype(str))

    # Merge everything onto items (one row per order-item)
    fact = (items
        .merge(orders,   on="order_id", how="left")
        .merge(pay_agg,  on="order_id", how="left")
        .merge(rev_agg,  on="order_id", how="left")
    )

    # Derived columns
    fact["order_date"]     = pd.to_datetime(fact["order_purchase_timestamp"]).dt.strftime("%Y-%m-%d")
    fact["approved_date"]  = pd.to_datetime(fact["order_approved_at"]).dt.strftime("%Y-%m-%d")
    fact["delivered_date"] = pd.to_datetime(fact["order_delivered_customer_date"]).dt.strftime("%Y-%m-%d")
    fact["estimated_date"] = pd.to_datetime(fact["order_estimated_delivery_date"]).dt.strftime("%Y-%m-%d")

    fact["delivery_days"] = (
        pd.to_datetime(fact["order_delivered_customer_date"]) -
        pd.to_datetime(fact["order_purchase_timestamp"])
    ).dt.days

    fact["is_late_delivery"] = (
        pd.to_datetime(fact["order_delivered_customer_date"]) >
        pd.to_datetime(fact["order_estimated_delivery_date"])
    ).astype(int)

    fact["order_item_key"] = fact["order_id"] + "_" + fact["order_item_id"].astype(str)

    fact = fact.rename(columns={"order_status": "order_status"})[
        ["order_item_key", "order_id", "customer_id", "product_id", "seller_id",
         "order_date", "approved_date", "delivered_date", "estimated_date",
         "price", "freight_value", "payment_value", "payment_type", "installments",
         "order_status", "review_score", "delivery_days", "is_late_delivery"]
    ]

    fact.dropna(subset=["customer_id", "order_date"], inplace=True)
    fact.to_sql("fact_orders", engine, if_exists="replace", index=False)
    print(f"✓ fact_orders: {len(fact):,} rows")
    return fact


# ─────────────────────────────────────────────
# Quality Report
# ─────────────────────────────────────────────

def quality_report(engine):
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    tables = ["dim_customer", "dim_product", "dim_seller", "dim_date", "fact_orders"]
    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t} LIMIT 0", engine)
        count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", engine)["n"][0]
        nulls = pd.read_sql(f"SELECT * FROM {t} LIMIT 5000", engine).isnull().sum().sum()
        print(f"  {t:<20} {count:>8,} rows   {nulls:>6} nulls in sample")

    delivered = pd.read_sql(
        "SELECT COUNT(*) AS n FROM fact_orders WHERE order_status='delivered'", engine
    )["n"][0]
    total = pd.read_sql("SELECT COUNT(*) AS n FROM fact_orders", engine)["n"][0]
    print(f"\n  Delivered orders: {delivered:,} / {total:,} ({delivered/total*100:.1f}%)")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("E-COMMERCE ANALYTICS — ETL PIPELINE")
    print("="*50 + "\n")

    engine = get_engine()
    run_schema(engine)

    build_dim_customer(engine)
    build_dim_product(engine)
    build_dim_seller(engine)
    build_fact_orders(engine)

    quality_report(engine)
    print("ETL complete. Database ready at:", DB_PATH)
