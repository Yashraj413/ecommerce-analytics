"""
churn.py — Customer Churn Prediction
XGBoost classifier with SHAP explainability.
Churn definition: no purchase in last 180 days (relative to snapshot date).

Usage:
    python src/churn.py

Output:
    outputs/churn_predictions.csv
    outputs/churn_analysis.png
    outputs/shap_summary.png
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
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import xgboost as xgb

DB_PATH  = "data/ecommerce.db"
OUT_PATH = "outputs"
SNAPSHOT = pd.to_datetime("2018-09-01")
CHURN_DAYS = 180   # customers with no order in last 180 days = churned


def build_features(engine, cutoff_date=None, end_date=None):
    """
    Build feature matrix from SQL — one row per unique customer.
    
    If cutoff_date is provided, builds a historical training cohort up to that date,
    with labels defined by whether they purchased between cutoff_date and end_date (leakage-free).
    If cutoff_date is None, builds the scoring cohort as of SNAPSHOT (all historical data).
    """
    if cutoff_date is None:
        ref_date = SNAPSHOT
        date_filter_clause = f"AND f.order_date < '{ref_date.strftime('%Y-%m-%d')}'"
    else:
        ref_date = cutoff_date
        date_filter_clause = f"AND f.order_date < '{ref_date.strftime('%Y-%m-%d')}'"

    df = pd.read_sql(f"""
        SELECT
            c.customer_unique_id                                    AS customer_id,
            COUNT(DISTINCT f.order_id)                              AS order_count,
            SUM(f.payment_value)                                    AS total_spend,
            AVG(f.payment_value)                                    AS avg_order_value,
            MAX(f.payment_value)                                    AS max_order_value,
            MIN(f.payment_value)                                    AS min_order_value,
            MAX(f.order_date)                                       AS last_order_date,
            MIN(f.order_date)                                       AS first_order_date,
            AVG(COALESCE(f.review_score, 3.0))                      AS avg_review_score,
            SUM(CASE WHEN f.review_score >= 4 THEN 1 ELSE 0 END)   AS positive_reviews,
            SUM(CASE WHEN f.review_score <= 2 THEN 1 ELSE 0 END)   AS negative_reviews,
            AVG(f.freight_value)                                    AS avg_freight,
            AVG(f.delivery_days)                                    AS avg_delivery_days,
            SUM(f.is_late_delivery)                                 AS late_deliveries,
            COUNT(DISTINCT p.category_english)                      AS category_diversity,
            COUNT(DISTINCT f.seller_id)                             AS unique_sellers,
            AVG(f.installments)                                     AS avg_installments
        FROM fact_orders f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        JOIN dim_product  p ON f.product_id  = p.product_id
        WHERE f.order_status = 'delivered'
          AND f.order_date IS NOT NULL
          {date_filter_clause}
        GROUP BY c.customer_unique_id
    """, engine)

    df["last_order_date"]   = pd.to_datetime(df["last_order_date"])
    df["first_order_date"]  = pd.to_datetime(df["first_order_date"])
    df["days_since_last"]   = (ref_date - df["last_order_date"]).dt.days
    df["customer_tenure"]   = (df["last_order_date"] - df["first_order_date"]).dt.days + 1
    df["order_frequency"]   = df["order_count"] / (df["customer_tenure"] / 30)  # orders/month
    df["spend_per_day"]     = df["total_spend"] / df["customer_tenure"]
    df["late_delivery_rate"]= df["late_deliveries"] / df["order_count"]
    df["positive_review_rate"] = df["positive_reviews"] / df["order_count"]

    if cutoff_date is not None and end_date is not None:
        # Determine target: did they purchase in the observation window?
        active_df = pd.read_sql(f"""
            SELECT DISTINCT c.customer_unique_id AS customer_id
            FROM fact_orders f
            JOIN dim_customer c ON f.customer_id = c.customer_id
            WHERE f.order_status = 'delivered'
              AND f.order_date IS NOT NULL
              AND f.order_date >= '{cutoff_date.strftime('%Y-%m-%d')}'
              AND f.order_date <= '{end_date.strftime('%Y-%m-%d')}'
        """, engine)
        active_ids = set(active_df["customer_id"])
        df["churned"] = df["customer_id"].apply(lambda x: 0 if x in active_ids else 1)
    else:
        # For the final scored output, we populate 'churned' based on their status as of SNAPSHOT
        # to ensure compatibility with Power BI summary metrics.
        df["churned"] = (df["days_since_last"] > CHURN_DAYS).astype(int)

    return df.fillna(0)


FEATURES = [
    "order_count", "total_spend", "avg_order_value", "max_order_value",
    "days_since_last", "customer_tenure", "order_frequency", "spend_per_day",
    "avg_review_score", "positive_review_rate", "negative_reviews",
    "avg_freight", "avg_delivery_days", "late_delivery_rate",
    "category_diversity", "unique_sellers", "avg_installments"
]

FEATURE_LABELS = {
    "order_count":          "Order Count",
    "total_spend":          "Total Spend (R$)",
    "avg_order_value":      "Avg Order Value",
    "max_order_value":      "Max Order Value",
    "days_since_last":      "Days Since Last Order",
    "customer_tenure":      "Customer Tenure (days)",
    "order_frequency":      "Order Frequency (orders/mo)",
    "spend_per_day":        "Avg Spend per Day",
    "avg_review_score":     "Avg Review Score",
    "positive_review_rate": "Positive Review Rate",
    "negative_reviews":     "Negative Review Count",
    "avg_freight":          "Avg Freight Cost",
    "avg_delivery_days":    "Avg Delivery Days",
    "late_delivery_rate":   "Late Delivery Rate",
    "category_diversity":   "Category Diversity",
    "unique_sellers":       "Unique Sellers",
    "avg_installments":     "Avg Installments",
}


def train_model(df):
    X = df[FEATURES]
    y = df["churned"]

    print(f"  Churn rate: {y.mean()*100:.1f}%  ({y.sum():,} churned / {len(y):,} total)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost with scale_pos_weight to handle class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators        = 300,
        max_depth           = 5,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,
        scale_pos_weight    = neg / pos,
        eval_metric         = "logloss",
        early_stopping_rounds = 20,
        random_state        = 42,
        verbosity           = 0
    )

    model.fit(
        X_train, y_train,
        eval_set        = [(X_test, y_test)],
        verbose         = False
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)

    print(f"\n  Model Performance:")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

    return model, X_test, y_test, y_proba


def compute_shap(model, X_test):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_churn(model, X_test, y_test, y_proba, shap_values, df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Churn Analysis — XGBoost + SHAP", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # 2. Churn probability distribution
    ax = axes[0, 1]
    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.7,
            color="#2ecc71", label="Active Customers", density=True)
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.7,
            color="#e74c3c", label="Churned Customers", density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Density")
    ax.set_title("Churn Probability Distribution")
    ax.legend()

    # 3. Confusion matrix
    ax = axes[1, 0]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted Active", "Predicted Churned"],
                yticklabels=["Actually Active", "Actually Churned"])
    ax.set_title("Confusion Matrix")

    # 4. Risk tier breakdown
    ax = axes[1, 1]
    df["churn_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    df["risk_tier"] = pd.cut(
        df["churn_probability"],
        bins   = [0, 0.33, 0.66, 1.0],
        labels = ["Low Risk", "Medium Risk", "High Risk"]
    )
    tier_counts = df["risk_tier"].value_counts()
    tier_colors = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}
    bars = ax.bar(tier_counts.index, tier_counts.values,
                  color=[tier_colors[t] for t in tier_counts.index])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Customers by Churn Risk Tier")
    ax.set_ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/churn_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/churn_analysis.png")

    # SHAP summary plot (separate file)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES],
        show=False,
        plot_size=None
    )
    plt.title("SHAP Feature Importance — Churn Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/shap_summary.png")

    return df


if __name__ == "__main__":
    import os
    os.makedirs(OUT_PATH, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")

    # 1. Train on historical cohort to avoid target leakage
    cutoff_date = SNAPSHOT - pd.Timedelta(days=CHURN_DAYS)
    print(f"Building leakage-free training cohort (cutoff: {cutoff_date.date()} to {SNAPSHOT.date()})...")
    df_train = build_features(engine, cutoff_date=cutoff_date, end_date=SNAPSHOT)
    print(f"  {len(df_train):,} training customers, {len(FEATURES)} features")

    print("\nTraining XGBoost churn model on training cohort...")
    model, X_test, y_test, y_proba = train_model(df_train)

    print("Computing SHAP values (explainability) on test set...")
    explainer, shap_values = compute_shap(model, X_test)

    # 2. Build scoring cohort as of SNAPSHOT (current customer states)
    print("\nBuilding current scoring cohort as of snapshot...")
    df_score = build_features(engine, cutoff_date=None)
    print(f"  {len(df_score):,} scoring customers")

    print("Generating charts...")
    # Pass df_score so the risk tier visualization reflects current customers
    df_score = plot_churn(model, X_test, y_test, y_proba, shap_values, df_score)

    # Score all current customers
    df_score["churn_probability"] = model.predict_proba(df_score[FEATURES])[:, 1]
    df_score["risk_tier"] = pd.cut(
        df_score["churn_probability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    output_cols = (
        ["customer_id", "churned", "churn_probability", "risk_tier"]
        + FEATURES
    )
    df_score[output_cols].to_csv(f"{OUT_PATH}/churn_predictions.csv", index=False)
    print(f"✓ churn_predictions.csv saved ({len(df_score):,} rows)")

    # Summary
    print("\n" + "="*50)
    print("CHURN RISK SUMMARY (CURRENT COHORT)")
    print("="*50)
    risk_summary = df_score.groupby("risk_tier").agg(
        customers    = ("customer_id",       "count"),
        avg_churn_prob = ("churn_probability","mean"),
        avg_spend    = ("total_spend",       "mean")
    ).round(2)
    print(risk_summary.to_string())
    print("="*50)
    print("\nChurn analysis complete.")
