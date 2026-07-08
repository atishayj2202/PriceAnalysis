# Demand Approximation Framework & Pricing Analysis Dashboard

A multi-agent decision support system designed to assess how changing a product's price by \( x \% \) affects weekly consumer demand, revenue, and profit. The system operates on a parallel sub-agent architecture mapped to the pricing rules outlined in `demand_framework.pdf`.

---

## 🚀 How to Run the App

The project is managed using Poetry and Python 3.9.

### Step 1: Install Dependencies
If you haven't installed dependencies yet, run the following command in the project root:
```bash
poetry install
```

### Step 2: Run the Entire System (Tests & Dashboard)
Launch the system with a single entry point:
```bash
poetry run python src/main.py
```
This script will **automatically run the automated test suite first**. If all regression and factor checks pass successfully, it will launch the interactive Streamlit dashboard.

---

## 📁 Directory Structure & Datasets

- **`src/main.py`**: The single entry point script. Runs testing, then launches Streamlit.
- **`src/app.py`**: The Streamlit dashboard interface. Supports both CSV and Excel uploads.
- **`src/agents/`**: Contains the Coordinator Agent and 7 Factor Agents running in parallel.
- **`src/utils/`**: Implements Z-score based spike detection (`spike_detector.py`).
- **`src/run_tests.py`**: Automated test suite.
- **`Documentation/`**: Comprehensive markdown manuals describing formulas (`Agent.md`) and data structures (`data.md`).
- **`MockData/`**: Contains 3 years (156 weeks) of simulated sales, competitor, promo, inventory, and sentiment data across different scenarios.


---

## 📈 Comprehensive Dashboard Guide

The dashboard contains several sections, metrics, and indicators. Here is what each item signifies:

### 1. Key Performance Indicator Cards (Top)
- **Expected Revenue Increase**:
  - *Significance*: Projects the change in weekly gross revenue (\( \Delta \text{Revenue} = P_{\text{new}} \cdot Q_{\text{new}} - P_{\text{base}} \cdot Q_{\text{base}} \)).
  - *Meaning*: Shows whether the price change generates more gross money. A negative percentage indicates that the volume loss overrides the higher price.
- **Expected Profit Increase**:
  - *Significance*: Projects the change in weekly gross profit margin (\( \Delta \text{Profit} = (P_{\text{new}} - \text{Cost}) \cdot Q_{\text{new}} - (P_{\text{base}} - \text{Cost}) \cdot Q_{\text{base}} \)).
  - *Meaning*: This is the main metric for optimization. A positive profit increase indicates that the price shift yields a better margin outcome, even if demand or revenue drops.
- **Expected Error Percentage (MAPE)**:
  - *Significance*: Mean Absolute Percentage Error (MAPE) of the historical price-elasticity OLS regression.
  - *Meaning*: Shows the historical prediction error of the model. For example, a MAPE of `8.5%` signifies that historical predictions deviated from actual sales by an average of 8.5%. A higher percentage indicates that sales are volatile or poorly explained by price alone.

### 2. Projection Curves (Middle)
- **Weekly Demand vs. Price Change**:
  - *Blue line (P50)*: The median projected demand (units sold per week) at each simulated price change percentage from \(-30\%\) to \(+30\%\).
  - *Green/Red dashed lines (P90 / P10)*: The scenario confidence bands. These are derived from the 95% Confidence Interval of elasticity \( e \). The space between them visualizes estimation uncertainty. A wider band means less data or high noise.
- **Projected Profit vs. Price Change**:
  - *Gold line*: Shows projected profit across the price range.
  - *Green Dot (Optimal)*: Highlights the mathematical profit-maximizing price point (where marginal revenue equals marginal cost).

### 3. Active Factor Agents & Reliability Table
- **Sym**: Formula symbol representing the factor (e.g. \( e \)=Elasticity, \( S \)=Seasonality, \( C \)=Competitor, \( M \)=Promotions, \( I \)=Inventory, \( L \)=Lifecycle, \( X \)=Sentiment).
- **Multiplier / Modifier**: The value computed by the agent. If it is an *elasticity modifier* (\( C, L, X \)), it multiplies base elasticity (making it steeper or flatter). If it is a *demand multiplier* (\( S, I, 1+\text{lift}_M \)), it directly shifts base quantity up or down.
- **R² Explanatory Power**: The individual factor's contribution to explaining the remaining variance of demand residuals.
- **Normalized Weight \( w_i \)**: The relative weight of the factor in the denominator of the model, normalized by the sum of all active \( R^2 \) values.
- **Reliability Score**: Color-coded badges indicating the estimated reliability of each factor (e.g. F1=88% High, F7=38% Low). If data is thin, it is flagged as `PROVISIONAL`.
- **Usage Status**: Shows `Active` if the factor was included. Shows `Left Out` if the data file was missing or its \( R^2 \) contribution was too low (e.g. seasonality \( R^2 < 0.05 \) or inventory \( R^2 < 0.03 \)), in which case it was replaced by its neutral default (1.0 or 0.0).

### 4. Hard Stop Indicators & Checklists
The system evaluates safety limits and human check inputs. If active, a **Hard Stop** banner is displayed:
- **CCI drop > 10 pts**: Macro economic shock. Floors sentiment modifier \( X \) at 0.97 and flags projections.
- **Price change > 25%**: Outside safe extrapolation boundaries, requiring manual review.
- **|e| > 5.0**: Price elasticity estimate is highly suspect or corrupted.
- **CI width > 1.5**: Insufficient historical price variation to estimate elasticity.
- **Uncategorized Spike**: A demand spike occurred in history that wasn't explained or cleaned, blocking training.
- **Human Checklist Override (H1-H4, U1-U2)**: If a human answers `YES` or `UNKNOWN` to PR crises, new competitor launches, placement drops, or regulatory ceilings, the model halts and blocks automated execution.
