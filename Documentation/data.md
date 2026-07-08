# Data Structure & Schema Specifications

This document outlines the column names, data dimensions, types, and descriptions for all CSV files required by the Demand Approximation Framework.

---

## 1. Directory Structure

Mock data is stored in the `MockData/` directory, divided by category and product:

```
MockData/
├── README.md                           # Directory trait description
├── electronics/
│   ├── mobile_phone/
│   │   ├── sales_demand.csv            # * REQUIRED: Historical sales & cost
│   │   ├── competitor_pricing.csv      # Competitor price log
│   │   ├── marketing_promotions.csv     # Promo flag and spend
│   │   ├── inventory_status.csv         # Daily/weekly stock levels
│   │   ├── product_lifecycle.csv        # Launch metadata
│   │   └── consumer_sentiment.csv       # CCI index data
│   └── laptop/
│       └── [Same files...]
└── fmcg/
    ├── rice/
    │   └── [Same files...]
    ├── shampoo/
    │   └── [Same files...]
    └── face_wash/
        └── [Same files...]
```

---

## 2. File Schema Details

### 2.1 Sales & Demand CSV (Mandatory *)
This file contains the historical sales records. It is the primary dataset used to estimate price elasticity.
- **File Name**: `sales_demand.csv`
- **Granularity**: Weekly
- **Required Columns**:
  - `date` (YYYY-MM-DD): The start date of the week.
  - `sku_id` (String): Unique identifier of the product.
  - `unit_price` (Float): Average unit selling price in that week.
  - `units_sold` (Integer/Float): Number of units sold in that week.
  - `cost_per_unit` (Float): Cost of goods sold (COGS) per unit.

### 2.2 Competitor Pricing CSV (Optional)
Contains price logs for competitors to determine cross-price relationships and gaps.
- **File Name**: `competitor_pricing.csv`
- **Granularity**: Weekly
- **Required Columns**:
  - `date` (YYYY-MM-DD): The matching week date.
  - `comp_price_avg` (Float): Average competitor price in the market.
  - `comp_price_min` (Float): Minimum competitor price.
  - `comp_price_max` (Float): Maximum competitor price.

### 2.3 Marketing & Promotions CSV (Optional)
Tracks promotion flags and marketing campaigns.
- **File Name**: `marketing_promotions.csv`
- **Granularity**: Weekly
- **Required Columns**:
  - `date` (YYYY-MM-DD): The matching week date.
  - `is_promo` (Integer: 0 or 1): Flag indicating if a promotion (discount > 5% or campaign) was active.
  - `marketing_spend` (Float): Marketing spend in dollars/currency.

### 2.4 Inventory Status CSV (Optional)
Logs inventory levels to calculate stock coverage and control for censored demand.
- **File Name**: `inventory_status.csv`
- **Granularity**: Weekly (or aggregated daily)
- **Required Columns**:
  - `date` (YYYY-MM-DD): The matching week date.
  - `units_in_stock` (Integer): Stock available at the beginning/end of the week.
  - `avg_daily_sales_14d` (Float): Historical rolling daily sales rate.

### 2.5 Product Lifecycle CSV (Optional)
Metadata about the product lifecycle phase and age.
- **File Name**: `product_lifecycle.csv`
- **Required Columns**:
  - `sku_id` (String): Unique SKU.
  - `launch_date` (YYYY-MM-DD): The date the product was first launched.
  - `category` (String): Product category (e.g. Electronics, FMCG).

### 2.6 Consumer Sentiment CSV (Optional)
Economic indexes representing macro consumer demand sentiment.
- **File Name**: `consumer_sentiment.csv`
- **Granularity**: Weekly/Monthly
- **Required Columns**:
  - `date` (YYYY-MM-DD): Matching date.
  - `cci_current` (Float): Consumer Confidence Index for the current period (baseline = 100).
  - `cci_baseline` (Float): Baseline CCI (typically 100.0).
  - `google_trends_score` (Float): Category search interest score (0 to 100).
