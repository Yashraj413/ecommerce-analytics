"""
forecast.py — Sales Forecasting
Uses Facebook Prophet for time-series forecasting with Brazilian holiday effects.
Falls back to ARIMA if Prophet is unavailable.

Usage:
    python src/forecast.py

Output:
    outputs/sales_forecast.csv
    outputs/category_forecast.csv
    outputs/forecast_analysis.png
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
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine

DB_PATH  = "data/ecommerce.db"
OUT_PATH = "outputs"
FORECAST_DAYS = 90


def load_daily_sales(engine):
    df = pd.read_sql("""
        SELECT
            order_date                       AS ds,
            SUM(payment_value)               AS revenue,
            COUNT(DISTINCT order_id)         AS orders,
            COUNT(DISTINCT customer_id)      AS unique_customers,
            AVG(payment_value)               AS avg_order_value,
            SUM(freight_value)               AS total_freight
        FROM fact_orders
        WHERE order_status  = 'delivered'
          AND order_date   IS NOT NULL
          AND order_date   >= '2017-01-01'
        GROUP BY order_date
        ORDER BY order_date
    """, engine)
    df["ds"]  = pd.to_datetime(df["ds"])
    df["y"]   = df["revenue"]
    print(f"  {len(df):,} days of sales data")
    print(f"  Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df


def load_category_sales(engine):
    df = pd.read_sql("""
        SELECT
            p.category_english               AS category,
            f.order_date                     AS ds,
            SUM(f.payment_value)             AS revenue
        FROM fact_orders f
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.order_status      = 'delivered'
          AND f.order_date       IS NOT NULL
          AND f.order_date       >= '2017-01-01'
          AND p.category_english IS NOT NULL
        GROUP BY p.category_english, f.order_date
        ORDER BY f.order_date
    """, engine)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def run_prophet(df):
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10,
        interval_width          = 0.80
    )

    # Brazilian public holidays as extra regressors
    try:
        model.add_country_holidays(country_name="BR")
        print("  Brazilian holidays added")
    except Exception:
        pass

    model.fit(df[["ds", "y"]])

    future   = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    return model, forecast


def run_arima_fallback(df):
    """Simple moving-average fallback if Prophet install fails."""
    print("  Using moving-average fallback (install prophet for full forecast)")
    history = df.set_index("ds")["y"]
    window  = 7
    smoothed = history.rolling(window, min_periods=1).mean()

    # Extend with trend + noise
    last_val = smoothed.iloc[-1]
    trend    = (smoothed.iloc[-1] - smoothed.iloc[-30]) / 30
    future_dates = pd.date_range(
        start=history.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS, freq="D"
    )
    yhat = [last_val + trend * i + np.random.normal(0, last_val * 0.05)
            for i in range(1, FORECAST_DAYS + 1)]

    history_df = pd.DataFrame({
        "ds":         history.index,
        "yhat":       smoothed.values,
        "yhat_lower": smoothed.values * 0.85,
        "yhat_upper": smoothed.values * 1.15,
        "y":          history.values
    })
    future_df = pd.DataFrame({
        "ds":         future_dates,
        "yhat":       yhat,
        "yhat_lower": [v * 0.85 for v in yhat],
        "yhat_upper": [v * 1.15 for v in yhat]
    })
    forecast = pd.concat([history_df, future_df], ignore_index=True)
    return None, forecast


def compute_metrics(df, forecast):
    """MAPE and RMSE on historical period."""
    historical = forecast[forecast["ds"] <= df["ds"].max()].copy()
    merged = df[["ds", "y"]].merge(historical[["ds", "yhat"]], on="ds", how="inner")
    merged = merged[merged["y"] > 0]
    mape = (abs(merged["y"] - merged["yhat"]) / merged["y"]).mean() * 100
    rmse = np.sqrt(((merged["y"] - merged["yhat"]) ** 2).mean())
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: R$ {rmse:,.2f}")
    return mape, rmse


def forecast_top_categories(cat_df, top_n=5):
    """Run Prophet on top N categories by total revenue."""
    top_cats = (cat_df.groupby("category")["revenue"].sum()
                .nlargest(top_n).index.tolist())
    results = []
    for cat in top_cats:
        sub = cat_df[cat_df["category"] == cat][["ds", "revenue"]].rename(
            columns={"revenue": "y"})
        sub = sub.groupby("ds")["y"].sum().reset_index()
        try:
            from prophet import Prophet
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False, changepoint_prior_scale=0.05,
                        interval_width=0.80)
            m.fit(sub)
            fut = m.make_future_dataframe(periods=FORECAST_DAYS)
            fc  = m.predict(fut)
            fc["category"] = cat
            results.append(fc[["ds", "category", "yhat", "yhat_lower", "yhat_upper"]])
        except Exception:
            pass
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def plot_forecast(df, forecast, model=None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Sales Forecast Analysis — Prophet Model", fontsize=16, fontweight="bold")

    split_date = df["ds"].max()

    # 1. Main forecast with confidence interval
    ax = axes[0, 0]
    hist = forecast[forecast["ds"] <= split_date]
    fut  = forecast[forecast["ds"] >  split_date]

    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.15, color="#2980b9", label="80% confidence")
    ax.plot(hist["ds"], hist["yhat"], color="#2980b9", lw=1.5, label="Model fit")
    ax.plot(fut["ds"],  fut["yhat"],  color="#e74c3c", lw=2,   label=f"{FORECAST_DAYS}-day forecast",
            linestyle="--")
    ax.scatter(df["ds"], df["y"], s=3, color="#333", alpha=0.4, label="Actual")
    ax.axvline(split_date, color="gray", linestyle=":", lw=1.5)
    ax.set_title("Revenue Forecast (90 Days)")
    ax.set_ylabel("Daily Revenue (R$)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30)

    # 2. Weekly seasonality
    ax = axes[0, 1]
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    df2 = df.copy()
    df2["dow"] = df2["ds"].dt.dayofweek
    weekly = df2.groupby("dow")["y"].mean()
    bars = ax.bar([dow_map[i] for i in weekly.index], weekly.values,
                  color=["#e74c3c" if i >= 5 else "#2980b9" for i in weekly.index])
    ax.bar_label(bars, fmt="R$%.0f", padding=3, fontsize=8)
    ax.set_title("Average Revenue by Day of Week")
    ax.set_ylabel("Avg Daily Revenue (R$)")

    # 3. Monthly trend
    ax = axes[1, 0]
    df2["month"] = df2["ds"].dt.to_period("M")
    monthly = df2.groupby("month")["y"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    ax.fill_between(range(len(monthly)), monthly["y"].values / 1e3,
                    alpha=0.3, color="#27ae60")
    ax.plot(range(len(monthly)), monthly["y"].values / 1e3,
            color="#27ae60", lw=2, marker="o", markersize=4)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue (R$ thousands)")

    # 4. Forecast summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    future_fc = forecast[forecast["ds"] > split_date]
    total_30  = future_fc.head(30)["yhat"].sum()
    total_60  = future_fc.head(60)["yhat"].sum()
    total_90  = future_fc.head(90)["yhat"].sum()
    avg_daily = future_fc["yhat"].mean()

    stats_text = f"""
FORECAST SUMMARY (Next 90 Days)

Total Revenue Projected:
  → Next 30 days:   R$ {total_30:>10,.0f}
  → Next 60 days:   R$ {total_60:>10,.0f}
  → Next 90 days:   R$ {total_90:>10,.0f}

Avg Daily Revenue:  R$ {avg_daily:>10,.0f}

Historical (last 30 days):
  → Actual avg/day: R$ {df.tail(30)['y'].mean():>10,.0f}

Growth Projected:
  → {(avg_daily/df.tail(30)['y'].mean()-1)*100:+.1f}% vs last 30d
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontfamily="monospace", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))
    ax.set_title("90-Day Forecast Summary")

    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/forecast_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved: {OUT_PATH}/forecast_analysis.png")

    if model is not None:
        try:
            comp_fig = model.plot_components(forecast)
            comp_fig.savefig(f"{OUT_PATH}/forecast_components.png",
                             dpi=150, bbox_inches="tight")
            plt.close(comp_fig)
            print(f"✓ Chart saved: {OUT_PATH}/forecast_components.png")
        except Exception:
            pass


if __name__ == "__main__":
    import os
    os.makedirs(OUT_PATH, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")

    print("Loading daily sales data...")
    df = load_daily_sales(engine)

    print("\nFitting forecast model...")
    try:
        model, forecast = run_prophet(df)
        print("  Prophet model fitted successfully")
    except ImportError:
        print("  Prophet not found — using fallback")
        model, forecast = run_arima_fallback(df)

    print("\nEvaluating model accuracy...")
    mape, rmse = compute_metrics(df, forecast)

    # Save full forecast (history + future)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out = out.merge(df[["ds", "y", "orders", "unique_customers"]], on="ds", how="left")
    out.to_csv(f"{OUT_PATH}/sales_forecast.csv", index=False)
    print(f"✓ sales_forecast.csv saved ({len(out):,} rows)")

    print("\nForecasting top 5 product categories...")
    cat_df     = load_category_sales(engine)
    cat_fc     = forecast_top_categories(cat_df, top_n=5)
    if not cat_fc.empty:
        cat_fc.to_csv(f"{OUT_PATH}/category_forecast.csv", index=False)
        print(f"✓ category_forecast.csv saved")

    print("\nGenerating charts...")
    plot_forecast(df, forecast, model)

    print("\nForecast analysis complete.")
