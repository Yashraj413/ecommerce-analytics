"""
product_performance.py — Product & Category Analytics
ABC classification, revenue analysis, cross-sell basket rules (Apriori).

Usage:
    python src/product_performance.py

Output:
    outputs/product_performance.csv
    outputs/basket_rules.csv
    outputs/product_analysis.png
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine

DB_PATH  = "data/ecommerce.db"
OUT_PATH = "outputs"


def load_product_data(engine):
    df = pd.read_sql("""
        SELECT
            p.product_id,
            p.category_english                           AS category,
            f.order_id,
            f.customer_id,
            f.order_date,
            f.price,
            f.payment_value,
            f.freight_value,
            f.review_score,
            f.is_late_delivery,
            f.delivery_days
        FROM fact_orders f
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.order_status      = 'delivered'
          AND p.category_english IS NOT NULL
          AND f.order_date       IS NOT NULL
    """, engine)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["margin"]     = df["payment_value"] - df["freight_value"]
    print(f"  {len(df):,} delivered items | {df['category'].nunique()} categories")
    return df


def category_performance(df):
    """Revenue, volume, satisfaction, and ABC class per category."""
    agg = df.groupby("category").agg(
        orders          = ("order_id",         "count"),
        unique_customers= ("customer_id",       "nunique"),
        revenue         = ("payment_value",     "sum"),
        margin          = ("margin",            "sum"),
        avg_price       = ("price",             "mean"),
        avg_review      = ("review_score",      "mean"),
        late_rate       = ("is_late_delivery",  "mean"),
        avg_delivery    = ("delivery_days",     "mean")
    ).reset_index()

    agg["revenue_pct"]   = agg["revenue"] / agg["revenue"].sum() * 100
    agg["margin_pct"]    = agg["margin"]  / agg["margin"].sum()  * 100
    agg["repeat_rate"]   = (agg["unique_customers"] / agg["orders"]).clip(upper=1)

    # ABC Classification: A = top 70% revenue cumulative, B = 70-90%, C = rest
    agg_sorted = agg.sort_values("revenue", ascending=False)
    agg_sorted["cumulative_pct"] = agg_sorted["revenue_pct"].cumsum()
    agg_sorted["abc_class"] = agg_sorted["cumulative_pct"].apply(
        lambda x: "A" if x <= 70 else ("B" if x <= 90 else "C")
    )

    return agg_sorted.round(3)


def monthly_category_trend(df, top_n=5):
    """Monthly revenue trend for top N categories."""
    top_cats = (df.groupby("category")["payment_value"].sum()
                .nlargest(top_n).index.tolist())
    sub = df[df["category"].isin(top_cats)].copy()
    sub["month"] = sub["order_date"].dt.to_period("M")
    trend = (sub.groupby(["month", "category"])["payment_value"]
             .sum().reset_index())
    trend["month"] = trend["month"].astype(str)
    return trend


def run_basket_analysis(engine, min_support=0.002, min_confidence=0.05):
    """Apriori association rules — which categories are bought together."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        print("  mlxtend not found — skipping basket analysis")
        return pd.DataFrame()

    # Get order-category matrix (which categories appeared in each order)
    basket = pd.read_sql("""
        SELECT
            f.order_id,
            p.category_english AS category
        FROM fact_orders f
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.order_status      = 'delivered'
          AND p.category_english IS NOT NULL
        GROUP BY f.order_id, p.category_english
    """, engine)

    # Build transaction list
    transactions = basket.groupby("order_id")["category"].apply(list).tolist()
    # Keep only multi-category orders for meaningful rules
    transactions = [t for t in transactions if len(t) > 1]

    if len(transactions) < 100:
        print("  Not enough multi-category orders for basket analysis")
        return pd.DataFrame()

    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    df_enc = pd.DataFrame(te_arr, columns=te.columns_)

    frequent = apriori(df_enc, min_support=min_support, use_colnames=True, low_memory=True)

    if frequent.empty:
        print("  No frequent itemsets found at this support level")
        return pd.DataFrame()

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules = rules.sort_values("lift", ascending=False)
    print(f"  {len(rules)} association rules found")
    return rules


def plot_products(cat_perf, trend):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Product & Category Performance Analysis", fontsize=16, fontweight="bold")

    # 1. Top 15 categories by revenue (ABC colored)
    ax = axes[0, 0]
    top15 = cat_perf.head(15)
    abc_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
    colors = [abc_colors[c] for c in top15["abc_class"]]
    bars = ax.barh(top15["category"][::-1], top15["revenue"][::-1] / 1e3,
                   color=colors[::-1], alpha=0.85)
    ax.bar_label(bars, fmt="R$%.0fK", padding=3, fontsize=7)
    ax.set_xlabel("Revenue (R$ thousands)")
    ax.set_title("Top 15 Categories by Revenue (A/B/C class)")
    patches = [mpatches.Patch(color=c, label=f"Class {l}")
               for l, c in abc_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    # 2. Review score vs Revenue scatter
    ax = axes[0, 1]
    top40 = cat_perf.head(40)
    sc = ax.scatter(top40["avg_review"], top40["revenue"] / 1e3,
                    s=top40["orders"] / top40["orders"].max() * 500,
                    c=[{"A":"#2ecc71","B":"#f39c12","C":"#e74c3c"}[c] for c in top40["abc_class"]],
                    alpha=0.7)
    for _, row in top40.head(10).iterrows():
        ax.annotate(row["category"][:15], (row["avg_review"], row["revenue"] / 1e3),
                    fontsize=6, ha="left", va="bottom")
    ax.set_xlabel("Average Review Score")
    ax.set_ylabel("Revenue (R$ thousands)")
    ax.set_title("Review Score vs Revenue\n(bubble size = order volume)")

    # 3. ABC class breakdown (pie)
    ax = axes[1, 0]
    abc_counts = cat_perf["abc_class"].value_counts()
    abc_revenue = cat_perf.groupby("abc_class")["revenue"].sum()
    labels = [f"Class {k}\n{abc_counts[k]} cats\nR${abc_revenue[k]/1e6:.1f}M"
              for k in ["A","B","C"] if k in abc_counts]
    ax.pie([abc_revenue.get(k, 0) for k in ["A","B","C"] if k in abc_revenue],
           labels=labels,
           colors=["#2ecc71","#f39c12","#e74c3c"],
           autopct="%1.1f%%", startangle=90, pctdistance=0.75)
    ax.set_title("Revenue Share by ABC Class")

    # 4. Monthly trend for top 5 categories
    ax = axes[1, 1]
    if not trend.empty:
        palette = plt.cm.tab10.colors
        for i, (cat, grp) in enumerate(trend.groupby("category")):
            grp_sorted = grp.sort_values("month")
            ax.plot(grp_sorted["month"], grp_sorted["payment_value"] / 1e3,
                    marker="o", markersize=3, lw=2,
                    color=palette[i % len(palette)], label=cat[:20])
        ax.set_title("Monthly Revenue — Top 5 Categories")
        ax.set_ylabel("Revenue (R$ thousands)")
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/product_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/product_analysis.png")


def print_insights(cat_perf):
    a_cats = cat_perf[cat_perf["abc_class"] == "A"]
    print("\n" + "="*55)
    print("PRODUCT PERFORMANCE INSIGHTS")
    print("="*55)
    print(f"  Class A: {len(a_cats)} categories → "
          f"{a_cats['revenue_pct'].sum():.1f}% of revenue")
    print(f"  Highest avg review: "
          f"{cat_perf.nlargest(1,'avg_review')['category'].values[0]}")
    print(f"  Worst late delivery rate: "
          f"{cat_perf.nlargest(1,'late_rate')['category'].values[0]} "
          f"({cat_perf['late_rate'].max()*100:.1f}%)")
    print("="*55 + "\n")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_PATH, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")

    print("Loading product data...")
    df = load_product_data(engine)

    print("Computing category performance & ABC classification...")
    cat_perf = category_performance(df)
    print_insights(cat_perf)

    cat_perf.to_csv(f"{OUT_PATH}/product_performance.csv", index=False)
    print(f"✓ product_performance.csv saved ({len(cat_perf)} categories)")

    print("Computing monthly category trends...")
    trend = monthly_category_trend(df, top_n=5)

    print("Running basket analysis (Apriori)...")
    rules = run_basket_analysis(engine)
    if not rules.empty:
        rules.head(50).to_csv(f"{OUT_PATH}/basket_rules.csv", index=False)
        print(f"✓ basket_rules.csv saved (top 50 rules)")

    print("Generating charts...")
    plot_products(cat_perf, trend)

    print("\nProduct analysis complete.")
