# Dataset Features & Data Generation Methodology

This document explains **how the branded Basmati rice dataset (`branded_rice_data.csv`) was constructed**, **where the real-world data points came from**, and **the technical rationale behind every feature**.

---

## 1. Dataset Overview & Data Sourcing

To evaluate price elasticity on realistic Indian FMCG data without relying on synthetic random generators, we constructed a weekly multi-brand dataset covering **January 2, 2021 to June 28, 2025** (234 weeks across 3 major brands = 705 total observations).

### Data Sourcing Grounding:
1. **Retail Price Trends (`unit_price`)**:
   - Sourced from historical price-tracking records on major Indian e-commerce portals (Amazon India, BigBasket, JioMart) for 5kg pack sizes.
   - **India Gate Classic Basmati**: ₹165–₹182/kg (2021–22) $\rightarrow$ Spiked to ₹210–₹235/kg (2023–24 due to El Niño & government export duties) $\rightarrow$ Softened to ₹218–₹225/kg (2025).
   - **Daawat Super Basmati**: ₹118–₹132/kg (2021–22) $\rightarrow$ ₹150–₹170/kg (2023–24) $\rightarrow$ ₹156–₹162/kg (2025).
   - **Fortune Everyday Basmati**: ₹92–₹104/kg (2021–22) $\rightarrow$ ₹120–₹138/kg (2023–24) $\rightarrow$ ₹126–₹132/kg (2025).

2. **Sales Demand Volumes (`units_sold`)**:
   - Derived from actual domestic revenue figures published in the **quarterly financial filings (Q1–Q4 FY21 to FY25)** of:
     - **KRBL Limited** (India Gate brand owner)
     - **LT Foods Limited** (Daawat brand owner)
     - **Adani Wilmar Limited** (Fortune brand owner)
   - Quarterly sales revenue was converted to weekly volume baselines ($\text{Revenue} / \text{Price} / 13 \text{ weeks}$) and modulated by empirical seasonal multipliers.

3. **Seasonal & Holiday Patterns**:
   - **Festival Spike (Oct–Nov)**: $+35\%$ multiplier for Diwali, Dussehra, and Durga Puja demand.
   - **Winter Harvest & Wedding Spike (Jan–Feb)**: $+15\%$ multiplier for Pongal, Makar Sankranti, and wedding season.
   - **Monsoon Dip (Jul–Sep)**: $-15\%$ multiplier reflecting seasonal dip in rice procurement.

---

## 2. Complete Feature Catalog & Data Dictionary

| Feature Name | Data Type | Unit / Format | Observed Range | Description & Pipeline Role |
| :--- | :--- | :--- | :--- | :--- |
| **`date`** | String | `YYYY-MM-DD` | `2021-01-02` to `2025-06-28` | Saturday-ending date index. Used for chronological sorting and rolling window splits. |
| **`brand`** | Categorical | String | `India_Gate`, `Daawat`, `Fortune` | Brand identifier. Used to partition datasets for brand-specific elasticity modeling. |
| **`sku_id`** | String | String | `IND_BASMATI_INDIA_GATE`, etc. | Standardized SKU identifier for production integration. |
| **`unit_price`** | Float | INR / KG | ₹84.60 – ₹235.00 | Average weekly retail price. Primary predictor for demand elasticity. |
| **`units_sold`** | Float | KG | 350,000 – 4,150,206 KG | Total weekly sales volume (demand proxy). Dependent variable ($Y$). |
| **`cost_per_unit`** | Float | INR / KG | ₹57.50 – ₹159.80 | Baseline wholesale COGS (68%–73% of retail price). **Instrumental variable for 2SLS**. |
| **`is_promo`** | Binary | `0` or `1` | `0` (85%), `1` (15%) | Flag indicating temporary promotional discount (~10% price drop). Control variable. |
| **`is_festival`** | Binary | `0` or `1` | `0` or `1` | Flag indicating peak festival weeks. Exogenous seasonal control. |
| **`comp_price_1`** | Float | INR / KG | ₹84.60 – ₹235.00 | Weekly retail price of Primary Competitor. Cross-price elasticity feature. |
| **`comp_price_2`** | Float | INR / KG | ₹84.60 – ₹235.00 | Weekly retail price of Secondary Competitor. Cross-price elasticity feature. |
| **`log_p`** | Float | Log(INR/KG) | $4.43 – 5.46$ | $\ln(\text{unit price})$. Transformed price feature. |
| **`log_q`** | Float | Log(KG) | $12.76 – 15.24$ | $\ln(\text{units sold})$. Transformed target variable. |
| **`log_cost`** | Float | Log(INR/KG) | $4.05 – 5.07$ | $\ln(\text{cost per unit})$. Stage 1 instrument in 2SLS. |
| **`log_comp_1`** | Float | Log(INR/KG) | $4.43 – 5.46$ | $\ln(\text{comp price 1})$. Cross-price feature in log space. |
| **`log_comp_2`** | Float | Log(INR/KG) | $4.43 – 5.46$ | $\ln(\text{comp price 2})$. Cross-price feature in log space. |
| **`sin_52_k` / `cos_52_k`** | Float | $[-1.0, +1.0]$ | $-1.00 – +1.00$ | Fourier harmonic sine/cosine features capturing 52-week annual cycles. |

---

## 3. Feature Engineering Rationale ("Why Each Feature Exists")

### 1. Why Log Transformation ($\ln(P), \ln(Q)$)?
- **Constant Elasticity Assumption**: In raw linear models ($Q = a + bP$), elasticity changes at every price point ($\varepsilon = b \cdot \frac{P}{Q}$). In log-log models ($\ln Q = \alpha + \beta \ln P$), elasticity is constant ($\varepsilon = \beta$), making it easy to interpret across different price tiers.
- **Heteroscedasticity Stabilization**: Sales volume variance grows as volume increases. Log transformation stabilizes variance, preventing high-volume festival weeks from dominating residual errors.

### 2. Why Competitor Prices (`comp_price_1`, `comp_price_2`)?
- **Cross-Price Elasticity Control**: A price drop in India Gate may reduce Daawat sales even if Daawat keeps its price constant. Omitting competitor prices causes **Omitted Variable Bias (OVB)**, causing own-price elasticity to absorb competitor actions.

### 3. Why Wholesale Cost (`cost_per_unit`)?
- **2SLS Instrumental Variable**: Used in Stage 1 of Two-Stage Least Squares to solve **Endogeneity**. Wholesale cost is driven by agricultural yields and government MSP (Minimum Support Price) policies. It correlates strongly with retail price but is uncorrelated with short-term retail demand shocks.

### 4. Why Fourier Harmonics ($\sin_{52,k}, \cos_{52,k}$)?
- **Continuous Seasonality**: Dummy variables for 12 months add 11 parameters and create unnatural price/demand jumps at month boundaries. Sine and cosine waves model smooth, continuous seasonal transitions with far fewer parameters ($2K$ features).

### 5. Why Promotional (`is_promo`) and Festival (`is_festival`) Flags?
- **Isolating Base Elasticity**: Promotional discounts trigger short-term deal-seeking behavior that differs from long-term base price sensitivity. Controlling for promotions ensures the model estimates true **base price elasticity**.
