"""
clv.py — Customer Lifetime Value Prediction
Uses the BG/NBD model for purchase frequency + Gamma-Gamma for spend.

Usage:
    python src/clv.py

Output:
    outputs/clv_predictions.csv
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
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions
)

DB_PATH  = "data/ecommerce.db"
OUT_PATH = "outputs"
SNAPSHOT = "2018-09-01"


def load_transactions(engine):
    df = pd.read_sql("""
        SELECT
            c.customer_unique_id AS customer_id,
            f.order_date,
            f.payment_value
        FROM fact_orders f
        JOIN dim_customer c ON f.customer_id = c.customer_id
        WHERE f.order_status  = 'delivered'
          AND f.payment_value > 0
          AND f.order_date   IS NOT NULL
    """, engine)
    df["order_date"] = pd.to_datetime(df["order_date"])
    print(f"  {len(df):,} transactions | {df['customer_id'].nunique():,} customers")
    return df


def build_rfm_summary(df):
    """Create the lifetimes summary table (frequency, recency, T, monetary_value)."""
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col        = "customer_id",
        datetime_col           = "order_date",
        monetary_value_col     = "payment_value",
        observation_period_end = SNAPSHOT,
        freq                   = "D"
    )
    # Remove customers with 0 frequency (only 1 purchase — no repeat data)
    summary = summary[summary["frequency"] > 0].copy()
    summary = summary[summary["monetary_value"] > 0].copy()
    print(f"  {len(summary):,} repeat customers in summary")
    return summary


def fit_bgf(summary):
    """BG/NBD model: predicts how many future transactions a customer will make."""
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    print(f"  BG/NBD fitted successfully")
    return bgf


def fit_ggf(summary):
    """Gamma-Gamma model: predicts average transaction value for repeat buyers."""
    # Gamma-Gamma assumption: frequency and monetary value are independent
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(summary["frequency"], summary["monetary_value"])
    print(f"  Gamma-Gamma fitted")
    return ggf


def predict_clv(summary, bgf, ggf, months=12):
    weeks = months * 4.33   # convert months to weeks

    summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        weeks, summary["frequency"], summary["recency"], summary["T"]
    )

    summary["prob_alive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    summary["expected_avg_value"] = ggf.conditional_expected_average_profit(
        summary["frequency"], summary["monetary_value"]
    )

    # Discount rate: 10% annual → monthly rate
    monthly_rate = 0.10 / 12
    summary["clv_12m"] = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency"],
        summary["T"],
        summary["monetary_value"],
        time           = months,
        freq           = "D",
        discount_rate  = monthly_rate
    )

    # Tier classification
    summary["clv_tier"] = pd.qcut(
        summary["clv_12m"], q=3,
        labels=["Low Value", "Mid Value", "High Value"]
    )

    return summary.reset_index()


def print_insights(clv_df):
    print("\n" + "="*55)
    print("CLV PREDICTION RESULTS")
    print("="*55)

    tier_summary = clv_df.groupby("clv_tier").agg(
        customers     = ("customer_id", "count"),
        avg_clv       = ("clv_12m",     "mean"),
        total_clv     = ("clv_12m",     "sum"),
        avg_prob_alive= ("prob_alive",  "mean")
    ).round(2)

    print(tier_summary.to_string())

    top10 = clv_df.nlargest(10, "clv_12m")[
        ["customer_id", "frequency", "monetary_value", "prob_alive", "clv_12m"]
    ].round(2)
    print(f"\nTop 10 highest-value customers (12-month CLV):")
    print(top10.to_string(index=False))

    avg_clv   = clv_df["clv_12m"].mean()
    total_clv = clv_df["clv_12m"].sum()
    high_cnt  = (clv_df["clv_tier"] == "High Value").sum()
    print(f"\n  Avg CLV (12m):    R$ {avg_clv:,.2f}")
    print(f"  Total projected:  R$ {total_clv:,.0f}")
    print(f"  High Value count: {high_cnt:,} customers")
    print("="*55 + "\n")


def plot_clv(summary, bgf, clv_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Customer Lifetime Value Analysis", fontsize=16, fontweight="bold")

    # 1. Expected purchases heatmap (manual — avoids lifetimes ax compatibility issue)
    ax = axes[0, 0]
    recency_vals   = np.linspace(0, float(summary["T"].max()), 30)
    frequency_vals = np.arange(1, 16)
    T_val = float(summary["T"].mean())
    Z = np.array([
        [bgf.conditional_expected_number_of_purchases_up_to_time(
            26, int(f), float(r), T_val) for r in recency_vals]
        for f in frequency_vals
    ])
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[0, recency_vals[-1], 1, 15], cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Expected purchases (26 weeks)")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Expected Future Purchases\n(Frequency × Recency)")

    # 2. Probability alive heatmap (manual)
    ax = axes[0, 1]
    Z2 = np.array([
        [bgf.conditional_probability_alive(int(f), float(r), T_val) for r in recency_vals]
        for f in frequency_vals
    ])
    im2 = ax.imshow(Z2, aspect="auto", origin="lower",
                    extent=[0, recency_vals[-1], 1, 15], cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax, label="Probability Alive")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Probability Customer is Still Active")

    # 3. CLV distribution by tier
    ax = axes[1, 0]
    tier_colors = {"Low Value": "#e74c3c", "Mid Value": "#f39c12", "High Value": "#2ecc71"}
    for tier, grp in clv_df.groupby("clv_tier"):
        ax.hist(grp["clv_12m"].clip(upper=grp["clv_12m"].quantile(0.95)),
                bins=40, alpha=0.7, label=tier, color=tier_colors.get(str(tier), "#888"))
    ax.set_xlabel("12-Month CLV (R$)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("CLV Distribution by Tier")
    ax.legend()

    # 4. Top 20 CLV customers
    ax = axes[1, 1]
    top20 = clv_df.nlargest(20, "clv_12m")
    ax.barh(range(len(top20)), top20["clv_12m"].values,
            color="#2980b9", alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"Customer {i+1}" for i in range(len(top20))], fontsize=8)
    ax.set_xlabel("Predicted 12-Month CLV (R$)")
    ax.set_title("Top 20 Customers by Predicted CLV")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/clv_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/clv_analysis.png")


def validate_clv_model(df):
    """
    Perform Calibration-Holdout validation on the BG/NBD model.
    Evaluates how well the model predicts transactions in a holdout period.
    """
    print("\nPerforming Calibration-Holdout validation...")
    from lifetimes.utils import calibration_and_holdout_data
    from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
    
    cal_end = pd.to_datetime(SNAPSHOT) - pd.Timedelta(days=180)
    
    cal_holdout = calibration_and_holdout_data(
        df,
        customer_id_col        = "customer_id",
        datetime_col           = "order_date",
        monetary_value_col     = "payment_value",
        calibration_period_end = cal_end.strftime("%Y-%m-%d"),
        observation_period_end = SNAPSHOT,
        freq                   = "D"
    )
    
    cal_holdout = cal_holdout[cal_holdout["frequency_cal"] > 0]
    
    if cal_holdout.empty:
        print("  Warning: No repeat customers in calibration data for validation")
        return
        
    bgf_val = BetaGeoFitter(penalizer_coef=0.01)
    bgf_val.fit(cal_holdout["frequency_cal"], cal_holdout["recency_cal"], cal_holdout["T_cal"])
    
    plt.figure(figsize=(10, 6))
    plot_calibration_purchases_vs_holdout_purchases(bgf_val, cal_holdout)
    plt.title("BG/NBD Calibration vs Holdout Validation (180-day holdout)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/clv_validation.png", dpi=150)
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/clv_validation.png")
    
    # Calculate metrics
    cal_holdout["predicted_holdout"] = bgf_val.predict(
        180, cal_holdout["frequency_cal"], cal_holdout["recency_cal"], cal_holdout["T_cal"]
    )
    
    mae = (cal_holdout["frequency_holdout"] - cal_holdout["predicted_holdout"]).abs().mean()
    corr = cal_holdout["frequency_holdout"].corr(cal_holdout["predicted_holdout"])
    
    print("\n" + "="*50)
    print("CLV MODEL VALIDATION SUMMARY (180-DAY HOLDOUT)")
    print("="*50)
    print(f"  Validation Cohort: {len(cal_holdout):,} repeat customers")
    print(f"  Holdout Purchase Prediction MAE: {mae:.4f}")
    print(f"  Holdout Correlation (Actual vs Pred): {corr:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_PATH, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")

    print("Loading transactions...")
    df = load_transactions(engine)

    # Run model validation
    validate_clv_model(df)

    print("Building lifetimes summary table...")
    summary = build_rfm_summary(df)

    print("Fitting BG/NBD model (purchase frequency)...")
    bgf = fit_bgf(summary)

    print("Fitting Gamma-Gamma model (spend prediction)...")
    ggf = fit_ggf(summary)

    print("Predicting 12-month CLV...")
    clv_df = predict_clv(summary.copy(), bgf, ggf, months=12)

    print_insights(clv_df)

    clv_df.to_csv(f"{OUT_PATH}/clv_predictions.csv", index=False)
    print(f"✓ clv_predictions.csv saved ({len(clv_df):,} rows)")

    print("Generating charts...")
    plot_clv(summary, bgf, clv_df)

    print("\nCLV analysis complete.")
