# Power BI Dashboard Setup Guide
## E-Commerce Customer Analytics Platform

---

## Step 1 — Import CSV Files

Open Power BI Desktop → **Home → Get Data → Text/CSV**

Import ALL of these files from your `outputs/` folder:

| File | Table Name in PBI |
|------|------------------|
| `rfm_segments.csv` | `RFM` |
| `rfm_segment_summary.csv` | `RFM_Summary` |
| `clv_predictions.csv` | `CLV` |
| `churn_predictions.csv` | `Churn` |
| `sales_forecast.csv` | `Forecast` |
| `product_performance.csv` | `Products` |

For each file: File Origin = **65001: Unicode (UTF-8)**, Delimiter = **Comma**

---

## Step 2 — Data Types (set after import)

**RFM table:**
- `customer_id` → Text
- `last_purchase` → Date
- `recency`, `frequency` → Whole Number
- `monetary` → Decimal Number
- `R`, `F`, `M`, `rfm_total` → Whole Number
- `segment` → Text

**CLV table:**
- `customer_id` → Text
- `frequency`, `recency`, `T` → Decimal Number
- `monetary_value`, `clv_12m`, `expected_avg_value` → Decimal Number
- `prob_alive` → Decimal Number (percentage)
- `clv_tier` → Text

**Churn table:**
- `customer_id` → Text
- `churned` → Whole Number
- `churn_probability` → Decimal Number
- `risk_tier` → Text
- All feature columns → Decimal Number

**Forecast table:**
- `ds` → Date
- `yhat`, `yhat_lower`, `yhat_upper`, `y` → Decimal Number
- `orders`, `unique_customers` → Whole Number

---

## Step 3 — Relationships (Model View)

Go to **Model view** (left sidebar icon).

Create these relationships:

```
RFM[customer_id]   → CLV[customer_id]    (Many-to-One)
RFM[customer_id]   → Churn[customer_id]  (Many-to-One)
```

(Forecast and Products are standalone tables — no joins needed)

---

## Step 4 — DAX Measures

Go to **Home → New Measure** for each of these.
Create all measures in the `RFM` table unless noted.

### Revenue Measures
```dax
Total Revenue =
SUM(RFM[monetary])

Avg Order Value =
AVERAGEX(RFM, RFM[monetary] / RFM[frequency])

Champions Revenue =
CALCULATE(
    SUM(RFM[monetary]),
    RFM[segment] = "Champions"
)

Top Segments Revenue % =
DIVIDE(
    CALCULATE(SUM(RFM[monetary]),
        RFM[segment] IN {"Champions","Loyal Customers"}),
    SUM(RFM[monetary])
) * 100

Revenue per Customer =
DIVIDE([Total Revenue], DISTINCTCOUNT(RFM[customer_id]))
```

### Customer Measures
```dax
Total Customers =
DISTINCTCOUNT(RFM[customer_id])

Churn Rate % =
DIVIDE(
    CALCULATE(COUNTROWS(Churn), Churn[churned] = 1),
    COUNTROWS(Churn)
) * 100

High Risk Customers =
CALCULATE(COUNTROWS(Churn), Churn[risk_tier] = "High Risk")

Avg CLV 12M =
AVERAGE(CLV[clv_12m])

High Value Customers =
CALCULATE(COUNTROWS(CLV), CLV[clv_tier] = "High Value")
```

### Forecast Measures
```dax
Next 30 Day Forecast =
CALCULATE(
    SUM(Forecast[yhat]),
    Forecast[ds] > MAX(Forecast[ds]) - 90,
    Forecast[y] = BLANK()
)
-- (Add in Forecast table)

Forecast vs Actual =
DIVIDE(
    CALCULATE(SUM(Forecast[yhat]), NOT ISBLANK(Forecast[y])),
    CALCULATE(SUM(Forecast[y]), NOT ISBLANK(Forecast[y]))
) * 100
```

---

## Step 5 — Build Report Pages

### Page 1: Executive Summary

**Add these visuals:**

1. **4 KPI Cards** (top row):
   - Total Customers → `[Total Customers]`
   - Total Revenue → `[Total Revenue]` (format: Currency R$)
   - Churn Rate → `[Churn Rate %]` (format: % 1dp)
   - Avg 12M CLV → `[Avg CLV 12M]` (format: Currency R$)

2. **Line Chart** — Monthly Revenue Trend:
   - X-axis: `Forecast[ds]` (set to Month hierarchy)
   - Y-axis: `Forecast[y]`
   - Title: "Monthly Revenue Trend"

3. **Clustered Bar Chart** — Segment Revenue:
   - X-axis: `RFM_Summary[total_revenue]`
   - Y-axis: `RFM_Summary[segment]`
   - Sort: descending by revenue
   - Data colors: manually set Champions=green, At Risk=red, Lost=gray

4. **Map Visual** — Revenue by State:
   - Location: `dim_customer[state]` (if you join)
   - Values: `[Total Revenue]`

5. **Text Card** — Resume Impact:
   - "Top 2 segments contribute **[Top Segments Revenue %]** of total revenue"

---

### Page 2: Customer Segments (RFM)

1. **Treemap**:
   - Group: `RFM[segment]`
   - Values: Count of `customer_id`
   - Color saturation: `[Total Revenue]`

2. **Matrix** (R × F heatmap):
   - Rows: `RFM[R]`
   - Columns: `RFM[F]`
   - Values: Count of `customer_id`
   - Conditional formatting: color scale on values

3. **Scatter Chart** — Customer Landscape:
   - X: `RFM[recency]`
   - Y: `RFM[frequency]`
   - Size: `RFM[monetary]`
   - Legend: `RFM[segment]`

4. **Stacked Bar** — Revenue + Customer mix per segment:
   - Group: `RFM_Summary[segment]`
   - Values: `revenue_pct` and `customer_pct` as two series

5. **Slicer** — Segment filter (apply to all visuals on this page)

---

### Page 3: Churn Risk Dashboard

1. **Donut Chart** — Risk Tier Breakdown:
   - Legend: `Churn[risk_tier]`
   - Values: Count of `customer_id`
   - Colors: High Risk=red, Medium=orange, Low=green

2. **Gauge** — Overall Churn Rate:
   - Value: `[Churn Rate %]`
   - Min: 0, Max: 100, Target: 30

3. **Scatter Chart** — Churn Probability vs Spend:
   - X: `Churn[churn_probability]`
   - Y: `Churn[total_spend]`
   - Color: `Churn[risk_tier]`

4. **Table** — Top 50 At-Risk Customers:
   - Columns: `customer_id`, `churn_probability`, `total_spend`,
     `days_since_last`, `order_count`, `risk_tier`
   - Filter: `risk_tier = "High Risk"`
   - Sort: descending by `churn_probability`
   - Conditional formatting on `churn_probability` column

5. **Bar Chart** — Feature Importance (manual):
   - Create a static table with SHAP values from your shap_summary.png
   - Visualize as horizontal bar

---

### Page 4: Sales Forecast & Products

**Left half — Forecast:**

1. **Line + Shaded Area Chart**:
   - X: `Forecast[ds]`
   - Line Y: `Forecast[yhat]` (predicted)
   - Line Y2: `Forecast[y]` (actual — will be null for future dates)
   - Error bars using `yhat_lower` and `yhat_upper`

2. **3 KPI Cards**:
   - `[Next 30 Day Forecast]`
   - `[Forecast vs Actual]`
   - Next 90-day total (manual calculation from CSV)

**Right half — Products:**

3. **Clustered Bar** — Top 10 Categories:
   - X: `Products[revenue]`
   - Y: `Products[category]`
   - Color: `Products[abc_class]` (green=A, orange=B, red=C)

4. **Scatter** — Review vs Revenue:
   - X: `Products[avg_review]`
   - Y: `Products[revenue]`
   - Size: `Products[orders]`
   - Color: `Products[abc_class]`

5. **KPI Card** — A-class impact:
   - "Class A categories drive X% of revenue"

---

## Step 6 — Formatting Tips

**Theme:** Use the "Executive" built-in theme or import a custom JSON theme.

**Consistent colors:**
```
Champions / Class A / Low Risk:  #2ecc71 (green)
At Risk / Class C / High Risk:   #e74c3c (red)
Neutral / Mid:                   #f39c12 (amber)
Forecast line:                   #2980b9 (blue)
```

**All pages:** Add your name + "E-Commerce Analytics Platform" in footer text box.

**Page navigation:** Add buttons with page navigation actions for a polished look.

---

## Step 7 — Publish & Screenshot

1. File → Save As → `ecommerce_analytics.pbix`
2. Take screenshots of all 4 pages at 1920×1080
3. Save screenshots to `powerbi_screenshots/` for GitHub README

---
