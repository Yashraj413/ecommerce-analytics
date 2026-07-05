"""
rfm.py — RFM Customer Segmentation
Recency · Frequency · Monetary analysis on delivered orders.

Usage:
    python src/rfm.py

Output:
    outputs/rfm_segments.csv
    outputs/rfm_segment_summary.csv
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
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

DB_PATH  = "data/ecommerce.db"
OUT_PATH = "outputs"

SNAPSHOT_DATE = "2018-09-01"   # Dataset ends ~Aug 2018; use this as "today"

SEGMENT_MAP = {
    # (R_min, R_max, F_min, F_max) → label
    # Champions: bought recently, buy often, spend the most
    "Champions":          {"r": (4, 5), "f": (4, 5)},
    "Loyal Customers":    {"r": (2, 5), "f": (3, 5)},
    "Potential Loyalists":{"r": (3, 5), "f": (1, 3)},
    "Promising":          {"r": (4, 5), "f": (0, 1)},
    "Need Attention":     {"r": (2, 3), "f": (2, 3)},
    "About to Sleep":     {"r": (2, 3), "f": (0, 2)},
    "At Risk":            {"r": (0, 2), "f": (2, 5)},
    "Cant Lose Them":     {"r": (0, 1), "f": (4, 5)},
    "Hibernating":        {"r": (1, 2), "f": (1, 2)},
    "Lost":               {"r": (0, 2), "f": (0, 2)},
}

SEGMENT_COLORS = {
    "Champions":           "#2ecc71",
    "Loyal Customers":     "#27ae60",
    "Potential Loyalists": "#82e0aa",
    "Promising":           "#a9cce3",
    "Need Attention":      "#f39c12",
    "About to Sleep":      "#f8c471",
    "At Risk":             "#e74c3c",
    "Cant Lose Them":      "#c0392b",
    "Hibernating":         "#aab7b8",
    "Lost":                "#717d7e",
}


def load_data(engine):
    df = pd.read_sql("""
        SELECT
            c.customer_unique_id              AS customer_id,
            f.order_id,
            f.order_date,
            f.payment_value
        FROM fact_orders f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.order_status = 'delivered'
          AND f.payment_value > 0
          AND f.order_date IS NOT NULL
    """, engine)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def compute_rfm(df):
    snapshot = pd.to_datetime(SNAPSHOT_DATE)

    rfm = df.groupby("customer_id").agg(
        last_purchase = ("order_date",    "max"),
        frequency     = ("order_id",      "nunique"),
        monetary      = ("payment_value", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days

    # Score 1–5 (5 = best for all three)
    rfm["R"] = pd.qcut(rfm["recency"],
                        q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"),
                        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"),
                        q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)

    rfm["rfm_score"]  = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["rfm_total"]  = rfm[["R", "F", "M"]].sum(axis=1)
    rfm["segment"]    = rfm.apply(assign_segment, axis=1)

    return rfm


def assign_segment(row):
    r, f = row["R"], row["F"]
    if   r >= 4 and f >= 4:              return "Champions"
    elif r >= 2 and f >= 3:              return "Loyal Customers"
    elif r >= 3 and f <= 3:              return "Potential Loyalists"
    elif r >= 4 and f <= 1:              return "Promising"
    elif r in [2, 3] and f in [2, 3]:   return "Need Attention"
    elif r in [2, 3] and f <= 2:        return "About to Sleep"
    elif r <= 2 and f >= 2:             return "At Risk"
    elif r <= 1 and f >= 4:             return "Cant Lose Them"
    elif r in [1, 2] and f in [1, 2]:   return "Hibernating"
    else:                               return "Lost"


def segment_summary(rfm):
    summary = rfm.groupby("segment").agg(
        customers    = ("customer_id", "count"),
        avg_recency  = ("recency",    "mean"),
        avg_frequency= ("frequency",  "mean"),
        avg_monetary = ("monetary",   "mean"),
        total_revenue= ("monetary",   "sum")
    ).reset_index()

    total_rev = summary["total_revenue"].sum()
    summary["revenue_pct"]    = (summary["total_revenue"] / total_rev * 100).round(2)
    summary["customer_pct"]   = (summary["customers"] / summary["customers"].sum() * 100).round(2)
    summary["avg_recency"]    = summary["avg_recency"].round(0).astype(int)
    summary["avg_frequency"]  = summary["avg_frequency"].round(2)
    summary["avg_monetary"]   = summary["avg_monetary"].round(2)
    summary["total_revenue"]  = summary["total_revenue"].round(2)

    return summary.sort_values("total_revenue", ascending=False)


def print_insights(rfm, summary):
    print("\n" + "="*60)
    print("RFM SEGMENTATION RESULTS")
    print("="*60)
    print(summary[["segment","customers","customer_pct","total_revenue","revenue_pct"]]
          .to_string(index=False))

    top_segs  = ["Champions", "Loyal Customers"]
    top_rev   = summary[summary["segment"].isin(top_segs)]["total_revenue"].sum()
    total_rev = summary["total_revenue"].sum()
    top_cust  = summary[summary["segment"].isin(top_segs)]["customers"].sum()
    total_cust= summary["customers"].sum()

    print(f"\n★ Champions + Loyal Customers:")
    print(f"  → {top_cust:,} customers ({top_cust/total_cust*100:.1f}% of base)")
    print(f"  → R$ {top_rev:,.0f} revenue ({top_rev/total_rev*100:.1f}% of total)")
    print(f"\n  RESUME BULLET: 'Identified {top_cust:,} high-value customers")
    print(f"  contributing {top_rev/total_rev*100:.0f}% of total revenue via RFM.'")
    print("="*60 + "\n")


def plot_rfm(rfm, summary):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("RFM Customer Segmentation Analysis", fontsize=16, fontweight="bold", y=1.01)

    colors = [SEGMENT_COLORS.get(s, "#888") for s in summary["segment"]]

    # 1. Customer count by segment
    ax = axes[0, 0]
    bars = ax.barh(summary["segment"], summary["customers"], color=colors)
    ax.set_xlabel("Number of Customers")
    ax.set_title("Customers per Segment")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.invert_yaxis()

    # 2. Revenue by segment
    ax = axes[0, 1]
    bars = ax.barh(summary["segment"], summary["total_revenue"] / 1e6, color=colors)
    ax.set_xlabel("Revenue (R$ millions)")
    ax.set_title("Revenue per Segment")
    ax.bar_label(bars, fmt="%.2f M", padding=3, fontsize=8)
    ax.invert_yaxis()

    # 3. RFM score distribution (heatmap R vs F)
    ax = axes[1, 0]
    heatmap_data = rfm.groupby(["R", "F"]).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd",
                ax=ax, cbar_kws={"label": "Customer count"})
    ax.set_title("Customer Density: Recency vs Frequency Score")
    ax.set_xlabel("Frequency Score (1=low, 5=high)")
    ax.set_ylabel("Recency Score (1=low, 5=high)")

    # 4. Avg monetary by segment
    ax = axes[1, 1]
    top10 = summary.nlargest(8, "avg_monetary")
    bars  = ax.bar(range(len(top10)), top10["avg_monetary"],
                   color=[SEGMENT_COLORS.get(s, "#888") for s in top10["segment"]])
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10["segment"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg Spend per Customer (R$)")
    ax.set_title("Average Monetary Value by Segment")
    ax.bar_label(bars, fmt="R$%.0f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/rfm_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/rfm_analysis.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_PATH, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")

    print("Loading data...")
    df  = load_data(engine)
    print(f"  {len(df):,} transactions | {df['customer_id'].nunique():,} unique customers")

    print("Computing RFM scores...")
    rfm = compute_rfm(df)

    summary = segment_summary(rfm)
    print_insights(rfm, summary)

    rfm.to_csv(f"{OUT_PATH}/rfm_segments.csv", index=False)
    summary.to_csv(f"{OUT_PATH}/rfm_segment_summary.csv", index=False)
    print(f"✓ rfm_segments.csv saved ({len(rfm):,} rows)")
    print(f"✓ rfm_segment_summary.csv saved")

    print("Generating charts...")
    plot_rfm(rfm, summary)

    print("\nRFM analysis complete.")
