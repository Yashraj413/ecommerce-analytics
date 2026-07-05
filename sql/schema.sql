-- ============================================================
-- E-Commerce Analytics Platform — Star Schema
-- Compatible with SQLite (used in this project) and PostgreSQL
-- ============================================================

-- Dimension: Customers
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id        TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,
    city               TEXT,
    state              TEXT
);

-- Dimension: Products
CREATE TABLE IF NOT EXISTS dim_product (
    product_id        TEXT PRIMARY KEY,
    category_raw      TEXT,
    category_english  TEXT,
    weight_g          REAL,
    length_cm         REAL,
    height_cm         REAL,
    width_cm          REAL
);

-- Dimension: Sellers
CREATE TABLE IF NOT EXISTS dim_seller (
    seller_id   TEXT PRIMARY KEY,
    city        TEXT,
    state       TEXT
);

-- Dimension: Date (populated by ETL)
CREATE TABLE IF NOT EXISTS dim_date (
    date_id     TEXT PRIMARY KEY,
    year        INTEGER,
    quarter     INTEGER,
    month       INTEGER,
    month_name  TEXT,
    week        INTEGER,
    day_of_week INTEGER,
    day_name    TEXT,
    is_weekend  INTEGER
);

-- Fact: Orders (central table — one row per order-item)
CREATE TABLE IF NOT EXISTS fact_orders (
    order_item_key    TEXT PRIMARY KEY,
    order_id          TEXT NOT NULL,
    customer_id       TEXT REFERENCES dim_customer(customer_id),
    product_id        TEXT REFERENCES dim_product(product_id),
    seller_id         TEXT REFERENCES dim_seller(seller_id),
    order_date        TEXT REFERENCES dim_date(date_id),
    approved_date     TEXT,
    delivered_date    TEXT,
    estimated_date    TEXT,
    price             REAL,
    freight_value     REAL,
    payment_value     REAL,
    payment_type      TEXT,
    installments      INTEGER,
    order_status      TEXT,
    review_score      REAL,
    delivery_days     INTEGER,
    is_late_delivery  INTEGER
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_fact_customer  ON fact_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_fact_product   ON fact_orders(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_date      ON fact_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_fact_status    ON fact_orders(order_status);
