"""
generate_data.py — Synthetic Olist Dataset Generator
Produces realistic Brazilian e-commerce data matching the Olist schema exactly.
Use this if you don't have Kaggle access. For the real dataset:
  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Usage:
    python src/generate_data.py
"""

import os
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
N_CUSTOMERS   = 20000
N_PRODUCTS    = 3000
N_SELLERS     = 500
N_ORDERS      = 30000          # realistic for a medium dataset
START_DATE    = datetime(2017, 1, 1)
END_DATE      = datetime(2018, 8, 31)

BR_STATES = ["SP","RJ","MG","RS","PR","SC","BA","GO","ES","PE",
             "CE","MA","MT","MS","PA","RN","PB","AL","SE","PI"]
BR_CITIES = {
    "SP": ["São Paulo","Campinas","Santos","Ribeirão Preto","Sorocaba"],
    "RJ": ["Rio de Janeiro","Niterói","Nova Iguaçu","Duque de Caxias"],
    "MG": ["Belo Horizonte","Uberlândia","Contagem","Juiz de Fora"],
    "RS": ["Porto Alegre","Caxias do Sul","Pelotas","Canoas"],
    "PR": ["Curitiba","Londrina","Maringá","Ponta Grossa"],
    "SC": ["Florianópolis","Joinville","Blumenau","São José"],
    "BA": ["Salvador","Feira de Santana","Vitória da Conquista"],
    "GO": ["Goiânia","Aparecida de Goiânia","Anápolis"],
    "ES": ["Vitória","Vila Velha","Serra","Cariacica"],
    "PE": ["Recife","Caruaru","Petrolina","Olinda"],
}

CATEGORIES_RAW = [
    "cama_mesa_banho","beleza_saude","esporte_lazer","informatica_acessorios",
    "moveis_decoracao","utilidades_domesticas","relogios_presentes","telefonia",
    "automotivo","brinquedos","cool_stuff","ferramentas_jardim","perfumaria",
    "pet_shop","malas_acessorios","eletrodomesticos","construcao_ferramentas_seguranca",
    "fashion_bolsas_e_acessorios","livros_tecnicos","papelaria","musica","artes"
]

CATEGORY_TRANSLATION = {
    "cama_mesa_banho":                         "bed_bath_table",
    "beleza_saude":                            "health_beauty",
    "esporte_lazer":                           "sports_leisure",
    "informatica_acessorios":                  "computers_accessories",
    "moveis_decoracao":                        "furniture_decor",
    "utilidades_domesticas":                   "housewares",
    "relogios_presentes":                      "watches_gifts",
    "telefonia":                               "telephony",
    "automotivo":                              "auto",
    "brinquedos":                              "toys",
    "cool_stuff":                              "cool_stuff",
    "ferramentas_jardim":                      "garden_tools",
    "perfumaria":                              "perfumery",
    "pet_shop":                                "pet_shop",
    "malas_acessorios":                        "luggage_accessories",
    "eletrodomesticos":                        "home_appliances",
    "construcao_ferramentas_seguranca":        "construction_tools_safety",
    "fashion_bolsas_e_acessorios":             "fashion_bags_accessories",
    "livros_tecnicos":                         "books_technical",
    "papelaria":                               "stationery",
    "musica":                                  "music",
    "artes":                                   "art",
}

PAYMENT_TYPES = ["credit_card","boleto","voucher","debit_card"]
ORDER_STATUSES = ["delivered","delivered","delivered","delivered",
                  "shipped","canceled","invoiced","processing"]


def uid(): return str(uuid.uuid4())


def rand_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def city_state():
    state = random.choices(BR_STATES, weights=[30,15,10,7,7,5,4,3,3,3,
                                                2,2,2,2,1,1,1,1,1,1])[0]
    cities = BR_CITIES.get(state, [state + " City"])
    return random.choice(cities), state


# ── 1. Customers ───────────────────────────────────────────────
print("Generating customers...")
customer_unique_ids = [uid() for _ in range(int(N_CUSTOMERS * 0.85))]  # some repeat buyers
customer_rows = []
for _ in range(N_CUSTOMERS):
    cid   = uid()
    cuid  = random.choice(customer_unique_ids)
    city, state = city_state()
    zip_prefix  = str(random.randint(10000, 99999))
    customer_rows.append({
        "customer_id":         cid,
        "customer_unique_id":  cuid,
        "customer_zip_code_prefix": zip_prefix,
        "customer_city":       city,
        "customer_state":      state
    })
customers_df = pd.DataFrame(customer_rows)
customers_df.to_csv(f"{RAW}/olist_customers_dataset.csv", index=False)
print(f"  {len(customers_df):,} customers")

# ── 2. Sellers ─────────────────────────────────────────────────
print("Generating sellers...")
sellers = []
for _ in range(N_SELLERS):
    city, state = city_state()
    sellers.append({
        "seller_id":               uid(),
        "seller_zip_code_prefix":  str(random.randint(10000,99999)),
        "seller_city":             city,
        "seller_state":            state
    })
sellers_df = pd.DataFrame(sellers)
sellers_df.to_csv(f"{RAW}/olist_sellers_dataset.csv", index=False)
print(f"  {len(sellers_df):,} sellers")

# ── 3. Products ────────────────────────────────────────────────
print("Generating products...")
products = []
for _ in range(N_PRODUCTS):
    cat = random.choice(CATEGORIES_RAW)
    products.append({
        "product_id":                   uid(),
        "product_category_name":        cat,
        "product_name_lenght":          random.randint(20, 60),
        "product_description_lenght":   random.randint(100, 800),
        "product_photos_qty":           random.randint(1, 6),
        "product_weight_g":             random.randint(100, 8000),
        "product_length_cm":            random.randint(10, 60),
        "product_height_cm":            random.randint(5, 40),
        "product_width_cm":             random.randint(10, 50),
    })
products_df = pd.DataFrame(products)
products_df.to_csv(f"{RAW}/olist_products_dataset.csv", index=False)
print(f"  {len(products_df):,} products")

# ── 4. Category translation ────────────────────────────────────
trans_df = pd.DataFrame([
    {"product_category_name": k, "product_category_name_english": v}
    for k, v in CATEGORY_TRANSLATION.items()
])
trans_df.to_csv(f"{RAW}/product_category_name_translation.csv", index=False)
print(f"  {len(trans_df)} category translations")

# ── 5. Orders ──────────────────────────────────────────────────
print("Generating orders (this takes a moment)...")

# Simulate seasonality: more orders in Q4 and mid-year
def weighted_date():
    """Skew toward late 2017 and early 2018 — matches real Olist growth curve."""
    month_weights = [1,1,1.2,1.2,1.3,1.5,1.5,1.8,1.8,2.2,2.5,2.8,
                     2.5,2.3,2.0,1.8,1.8,1.6,1.5,1.3]
    days_total = (END_DATE - START_DATE).days
    # Simple approach: uniform then accept/reject by month
    for _ in range(50):
        d = START_DATE + timedelta(days=random.randint(0, days_total))
        month_idx = (d.year - 2017) * 12 + d.month - 1
        w = month_weights[min(month_idx, len(month_weights)-1)]
        if random.random() < w / 3.0:
            return d
    return START_DATE + timedelta(days=random.randint(0, days_total))

seller_ids  = sellers_df["seller_id"].tolist()
product_ids = products_df["product_id"].tolist()
customer_ids= customers_df["customer_id"].tolist()

orders_rows   = []
items_rows    = []
payment_rows  = []
review_rows   = []

for i in range(N_ORDERS):
    oid    = uid()
    cid    = random.choice(customer_ids)
    status = random.choices(ORDER_STATUSES,
                            weights=[70,70,70,70,8,4,2,1])[0]
    purchase_dt = weighted_date()

    # Approval: 0–2 days after purchase
    approved_dt = purchase_dt + timedelta(hours=random.randint(1, 48))

    # Delivery: 5–30 days, longer for remote states
    delivery_days = int(np.random.lognormal(2.4, 0.5))    # median ~11 days
    delivery_days = max(3, min(delivery_days, 60))
    delivered_dt  = approved_dt + timedelta(days=delivery_days)

    # Estimated: sometimes earlier, sometimes later than actual
    estimated_days = delivery_days + random.randint(-3, 10)
    estimated_days = max(delivery_days - 2, estimated_days)
    estimated_dt   = approved_dt + timedelta(days=estimated_days)

    if status != "delivered":
        delivered_dt = None

    orders_rows.append({
        "order_id":                          oid,
        "customer_id":                       cid,
        "order_status":                      status,
        "order_purchase_timestamp":          purchase_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "order_approved_at":                 approved_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_carrier_date":      (approved_dt + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_customer_date":     delivered_dt.strftime("%Y-%m-%d %H:%M:%S") if delivered_dt else None,
        "order_estimated_delivery_date":     estimated_dt.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Items: 1–3 per order
    n_items = random.choices([1,2,3], weights=[75,20,5])[0]
    order_total = 0
    for item_num in range(1, n_items + 1):
        price     = round(np.random.lognormal(4.0, 0.9), 2)   # median ~55 R$
        price     = max(5.0, min(price, 2000.0))
        freight   = round(random.uniform(5, 50), 2)
        order_total += price + freight
        items_rows.append({
            "order_id":              oid,
            "order_item_id":         item_num,
            "product_id":            random.choice(product_ids),
            "seller_id":             random.choice(seller_ids),
            "shipping_limit_date":   (approved_dt + timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "price":                 price,
            "freight_value":         freight,
        })

    # Payment
    pay_type = random.choices(PAYMENT_TYPES, weights=[70, 20, 5, 5])[0]
    installments = random.choices([1,2,3,6,10,12],
                                  weights=[50,15,10,10,8,7])[0] if pay_type == "credit_card" else 1
    payment_rows.append({
        "order_id":             oid,
        "payment_sequential":   1,
        "payment_type":         pay_type,
        "payment_installments": installments,
        "payment_value":        round(order_total * random.uniform(0.97, 1.03), 2),
    })

    # Review (only delivered orders, ~85% leave a review)
    if status == "delivered" and random.random() < 0.85:
        # Slight positive skew — real Olist has lots of 5-stars
        score = random.choices([1,2,3,4,5], weights=[5,5,10,20,60])[0]
        # Worse score if late delivery
        if delivered_dt and delivered_dt > estimated_dt:
            score = max(1, score - random.randint(1,2))
        review_rows.append({
            "review_id":                uid(),
            "order_id":                 oid,
            "review_score":             score,
            "review_comment_title":     "",
            "review_comment_message":   "",
            "review_creation_date":     (delivered_dt + timedelta(days=random.randint(1,15))).strftime("%Y-%m-%d %H:%M:%S"),
            "review_answer_timestamp":  (delivered_dt + timedelta(days=random.randint(2,20))).strftime("%Y-%m-%d %H:%M:%S"),
        })

orders_df   = pd.DataFrame(orders_rows)
items_df    = pd.DataFrame(items_rows)
payments_df = pd.DataFrame(payment_rows)
reviews_df  = pd.DataFrame(review_rows)

orders_df.to_csv(f"{RAW}/olist_orders_dataset.csv",         index=False)
items_df.to_csv(f"{RAW}/olist_order_items_dataset.csv",     index=False)
payments_df.to_csv(f"{RAW}/olist_order_payments_dataset.csv", index=False)
reviews_df.to_csv(f"{RAW}/olist_order_reviews_dataset.csv", index=False)

print(f"  {len(orders_df):,} orders")
print(f"  {len(items_df):,} order items")
print(f"  {len(payments_df):,} payments")
print(f"  {len(reviews_df):,} reviews")

# ── 6. Geolocation (optional but complete) ─────────────────────
print("Generating geolocation data...")
geo_rows = []
for _ in range(5000):
    city, state = city_state()
    lat = random.uniform(-33, -1)
    lng = random.uniform(-73, -35)
    geo_rows.append({
        "geolocation_zip_code_prefix": str(random.randint(10000,99999)),
        "geolocation_lat":  round(lat, 6),
        "geolocation_lng":  round(lng, 6),
        "geolocation_city": city,
        "geolocation_state":state
    })
geo_df = pd.DataFrame(geo_rows)
geo_df.to_csv(f"{RAW}/olist_geolocation_dataset.csv", index=False)

print("\n" + "="*50)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("="*50)
print(f"  Customers:    {len(customers_df):,}")
print(f"  Products:     {len(products_df):,}")
print(f"  Sellers:      {len(sellers_df):,}")
print(f"  Orders:       {len(orders_df):,}")
print(f"  Order Items:  {len(items_df):,}")
print(f"  Payments:     {len(payments_df):,}")
print(f"  Reviews:      {len(reviews_df):,}")
print(f"\n  All CSVs saved to: {RAW}/")
print("\n  Run next: python src/etl.py")
