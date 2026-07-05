"""
run_all.py — Master Pipeline Runner
Runs the complete E-Commerce Analytics Platform end-to-end.

Usage:
    python run_all.py

Produces all outputs in /outputs/ directory ready for Power BI.
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from datetime import datetime


def section(title):
    width = 58
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step(name, module_path, func_name=None):
    """Import and run a module's __main__ block."""
    import importlib.util, traceback

    print(f"\n→ Running: {name}")
    t0 = time.time()
    try:
        spec   = importlib.util.spec_from_file_location("_mod", module_path)
        mod    = importlib.util.module_from_spec(spec)
        # Patch __name__ so if-__main__ blocks don't fire
        mod.__name__ = "__not_main__"
        spec.loader.exec_module(mod)

        # Call the primary public function
        if func_name and hasattr(mod, func_name):
            getattr(mod, func_name)()

        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s")
        return True
    except FileNotFoundError as e:
        print(f"  ✗ Skipped — {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def check_data():
    """Verify Olist CSVs exist before running anything."""
    required = [
        "data/raw/olist_orders_dataset.csv",
        "data/raw/olist_customers_dataset.csv",
        "data/raw/olist_products_dataset.csv",
        "data/raw/olist_order_items_dataset.csv",
        "data/raw/olist_order_payments_dataset.csv",
        "data/raw/olist_order_reviews_dataset.csv",
        "data/raw/olist_sellers_dataset.csv",
        "data/raw/product_category_name_translation.csv",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n✗ Missing Olist CSV files:")
        for f in missing:
            print(f"    {f}")
        print("\nDownload from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print("Then unzip all CSVs into data/raw/\n")
        sys.exit(1)
    print(f"  ✓ All {len(required)} Olist CSV files found")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    section("E-COMMERCE ANALYTICS PLATFORM")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    section("0. DATA VALIDATION")
    check_data()

    results = {}

    section("1. ETL — LOAD WAREHOUSE")
    print("  Building star-schema SQLite warehouse from Olist CSVs...")
    from src.etl import get_engine, run_schema, build_dim_customer, \
        build_dim_product, build_dim_seller, build_fact_orders, quality_report
    engine = get_engine()
    run_schema(engine)
    build_dim_customer(engine)
    build_dim_product(engine)
    build_dim_seller(engine)
    build_fact_orders(engine)
    quality_report(engine)
    results["ETL"] = True

    section("2. RFM SEGMENTATION")
    from src.rfm import load_data, compute_rfm, segment_summary, \
        print_insights, plot_rfm
    df_txn = load_data(engine)
    rfm    = compute_rfm(df_txn)
    summ   = segment_summary(rfm)
    print_insights(rfm, summ)
    rfm.to_csv("outputs/rfm_segments.csv", index=False)
    summ.to_csv("outputs/rfm_segment_summary.csv", index=False)
    plot_rfm(rfm, summ)
    results["RFM"] = True

    section("3. CUSTOMER LIFETIME VALUE")
    try:
        from src.clv import load_transactions, build_rfm_summary, \
            fit_bgf, fit_ggf, predict_clv, print_insights as clv_insights, plot_clv, validate_clv_model
        df_txn2 = load_transactions(engine)
        validate_clv_model(df_txn2)
        summary = build_rfm_summary(df_txn2)
        bgf     = fit_bgf(summary)
        ggf     = fit_ggf(summary)
        clv_df  = predict_clv(summary.copy(), bgf, ggf, months=12)
        clv_insights(clv_df)
        clv_df.to_csv("outputs/clv_predictions.csv", index=False)
        plot_clv(summary, bgf, clv_df)
        results["CLV"] = True
    except Exception as e:
        print(f"  CLV skipped: {e}")
        results["CLV"] = False

    section("4. CHURN PREDICTION")
    try:
        from src.churn import build_features, train_model, compute_shap, \
            plot_churn, FEATURES, SNAPSHOT, CHURN_DAYS
        import pandas as pd
        
        # 1. Train on historical cohort (no leakage)
        cutoff_date = SNAPSHOT - pd.Timedelta(days=CHURN_DAYS)
        print(f"  Building leakage-free training cohort (cutoff: {cutoff_date.date()})...")
        df_train = build_features(engine, cutoff_date=cutoff_date, end_date=SNAPSHOT)
        model_ch, X_test, y_test, y_proba = train_model(df_train)
        _, shap_vals = compute_shap(model_ch, X_test)
        
        # 2. Build scoring cohort as of SNAPSHOT (current customer states)
        print("  Building current scoring cohort as of snapshot...")
        df_score = build_features(engine, cutoff_date=None)
        
        df_score = plot_churn(model_ch, X_test, y_test, y_proba, shap_vals, df_score)
        df_score["churn_probability"] = model_ch.predict_proba(df_score[FEATURES])[:, 1]
        df_score["risk_tier"] = pd.cut(df_score["churn_probability"],
                                    bins=[0,.33,.66,1.],
                                    labels=["Low Risk","Medium Risk","High Risk"])
        out_cols = ["customer_id","churned","churn_probability","risk_tier"] + FEATURES
        df_score[out_cols].to_csv("outputs/churn_predictions.csv", index=False)
        results["Churn"] = True
    except Exception as e:
        print(f"  Churn skipped: {e}")
        results["Churn"] = False

    section("5. SALES FORECASTING")
    try:
        from src.forecast import (load_daily_sales, run_prophet, run_arima_fallback,
                                  compute_metrics, load_category_sales,
                                  forecast_top_categories, plot_forecast)
        df_sales = load_daily_sales(engine)
        try:
            model_f, forecast = run_prophet(df_sales)
        except ImportError:
            model_f, forecast = run_arima_fallback(df_sales)
        compute_metrics(df_sales, forecast)
        import pandas as pd
        out_fc = forecast[["ds","yhat","yhat_lower","yhat_upper"]].merge(
            df_sales[["ds","y","orders","unique_customers"]], on="ds", how="left")
        out_fc.to_csv("outputs/sales_forecast.csv", index=False)
        cat_df = load_category_sales(engine)
        cat_fc = forecast_top_categories(cat_df, top_n=5)
        if not cat_fc.empty:
            cat_fc.to_csv("outputs/category_forecast.csv", index=False)
        plot_forecast(df_sales, forecast, model_f)
        results["Forecast"] = True
    except Exception as e:
        print(f"  Forecast skipped: {e}")
        results["Forecast"] = False

    section("6. PRODUCT PERFORMANCE")
    try:
        from src.product_performance import (load_product_data, category_performance,
                                              monthly_category_trend, run_basket_analysis,
                                              plot_products, print_insights as prod_insights)
        df_prod  = load_product_data(engine)
        cat_perf = category_performance(df_prod)
        prod_insights(cat_perf)
        cat_perf.to_csv("outputs/product_performance.csv", index=False)
        trend = monthly_category_trend(df_prod, top_n=5)
        rules = run_basket_analysis(engine)
        if not rules.empty:
            rules.head(50).to_csv("outputs/basket_rules.csv", index=False)
        plot_products(cat_perf, trend)
        results["Products"] = True
    except Exception as e:
        print(f"  Products skipped: {e}")
        results["Products"] = False

    # ── Final summary ──────────────────────────────────────────
    section("PIPELINE COMPLETE")
    all_ok = all(results.values())
    for step_name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {step_name}")

    print("\n  Output files ready for Power BI:")
    for f in sorted(os.listdir("outputs")):
        if f.endswith(".csv"):
            size = os.path.getsize(f"outputs/{f}") / 1024
            print(f"    outputs/{f}  ({size:.0f} KB)")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Next step: Open Power BI Desktop → Get Data → import all CSVs from outputs/")


if __name__ == "__main__":
    main()
