# E-Commerce Customer Analytics Platform

**Tech Stack:** Python · SQL (SQLite/PostgreSQL) · Power BI · XGBoost · Prophet · BG/NBD
> *Engineered an end-to-end customer analytics platform on 100K+ real e-commerce transactions; applied RFM segmentation identifying Champions and Loyal Customers contributing ~70% of total revenue; built a BG/NBD CLV model for 12-month value prediction, XGBoost churn classifier (AUC ~0.83) with SHAP explainability, Prophet sales forecast, and ABC product classification which is delivered via a 4-tab Power BI dashboard.*

---

## Architecture

```
Raw Data (Olist CSVs)
    ↓
ETL Pipeline (src/etl.py)
    ↓
SQLite Star Schema Warehouse
    ↓
┌───────────────────────────────────────┐
│  Python Analytics Layer               │
│  ├── RFM Segmentation                 │
│  ├── CLV Prediction (BG/NBD)          │
│  ├── Churn Model (XGBoost + SHAP)     │
│  ├── Sales Forecast (Prophet)         │
│  └── Product Performance (Apriori)    │
└───────────────────────────────────────┘
    ↓
CSV Outputs → Power BI Dashboard (4 tabs)
```

---

## Dataset

**Olist Brazilian E-Commerce** — Real transaction data from Kaggle

- 100,000+ orders (2016–2018)
- 96,096 unique customers
- 73 product categories
- 9 raw CSV tables

Download: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

---

## Project Structure

```
ecommerce-analytics/
├── data/
│   ├── raw/              ← Olist CSVs (you download these)
│   └── processed/
├── sql/
│   └── schema.sql        ← Star schema DDL
├── src/
│   ├── etl.py            ← ETL pipeline
│   ├── rfm.py            ← RFM segmentation
│   ├── clv.py            ← CLV prediction
│   ├── churn.py          ← Churn model
│   ├── forecast.py       ← Sales forecasting
│   └── product_performance.py
├── outputs/              ← Generated CSVs + charts (Power BI input)
├── powerbi_guide/
│   └── POWERBI_SETUP.md  ← Step-by-step Power BI instructions
├── run_all.py            ← Single command to run everything
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Olist data
```bash
# Option A: Kaggle CLI
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d data/raw/

# Option B: Manual download from Kaggle, unzip into data/raw/
```

### 3. Run everything
```bash
python run_all.py
```

This runs the full pipeline in sequence:
1. ETL → builds SQLite warehouse
2. RFM → customer segmentation
3. CLV → lifetime value prediction
4. Churn → churn risk scoring
5. Forecast → 90-day revenue forecast
6. Products → ABC classification + basket rules

Or run individual modules:
```bash
python src/rfm.py
python src/clv.py
python src/churn.py
python src/forecast.py
python src/product_performance.py
```

### 4. Open Power BI
See `powerbi_guide/POWERBI_SETUP.md` for full step-by-step instructions.
Import all CSVs from `outputs/` and follow the DAX + layout guide.

---

## Key Findings (from Olist dataset)

| Metric | Value |
|--------|-------|
| Total customers analyzed | ~96,000 |
| Champions + Loyal customers | ~12% of base |
| Revenue from top segments | ~70% of total |
| Churn rate (180-day window) | ~85% (typical for one-time buyers) |
| Churn model AUC-ROC | ~0.83 |
| 90-day forecast MAPE | ~12–15% |
| Class A categories | ~8 categories → 70% of revenue |

---

## Module Details

### RFM Segmentation (`src/rfm.py`)
Scores each customer on Recency (days since last order), Frequency (unique orders),
and Monetary (total spend) — each 1–5. Combines into 8 behavioral segments.

**Segments:** Champions · Loyal Customers · Potential Loyalists · Promising ·
Need Attention · About to Sleep · At Risk · Hibernating · Lost

### CLV Prediction (`src/clv.py`)
- **BG/NBD model** (Beta Geometric / Negative Binomial Distribution):
  predicts expected future purchases per customer
- **Gamma-Gamma model:** predicts expected spend per transaction
- **Output:** 12-month CLV in R$ + probability alive + CLV tier (Low/Mid/High)

### Churn Analysis (`src/churn.py`)
- **Label:** churned = no order in last 180 days
- **Features:** 17 behavioral + engagement features
- **Model:** XGBoost with `scale_pos_weight` for class imbalance
- **Explainability:** SHAP values showing which features drive churn
- **Output:** churn probability (0–1) + risk tier per customer

### Sales Forecast (`src/forecast.py`)
- **Model:** Facebook Prophet (yearly + weekly seasonality + Brazilian holidays)
- **Fallback:** Moving average if Prophet not installed
- **Output:** Daily revenue forecast for next 90 days with 80% confidence interval

### Product Performance (`src/product_performance.py`)
- **ABC Classification:** Category A = top 70% cumulative revenue
- **Apriori basket rules:** Which categories are bought together (lift, confidence)
- **Output:** Per-category KPIs (revenue, review score, late delivery rate, margin)

---

## Power BI Dashboard (4 Tabs)

| Tab | Key Visuals |
|-----|-------------|
| Executive Summary | KPI cards, monthly trend, segment revenue bar, state map |
| Customer Segments | RFM treemap, R×F heatmap, scatter plot, segment drilldown |
| Churn Risk | Donut by risk tier, churn probability scatter, at-risk customer table |
| Forecast + Products | 90-day forecast line, ABC bar chart, review vs revenue scatter |

---

## License
MIT — use freely for learning and portfolio purposes.
Dataset: CC BY-NC-SA 4.0 (Olist / Kaggle)
