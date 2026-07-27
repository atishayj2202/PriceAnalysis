import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_notebook():
    print("Creating laptop_exploration.ipynb...")
    nb = nbf.v4.new_notebook()
    
    # 1. Title cell
    title_text = """# Exploratory Data Analysis (EDA): Laptop Market Pricing & Volume (2015-2025)
This notebook performs a visual and statistical analysis of our 10-year laptop dataset (Dell, HP, Lenovo, Asus). 
It covers:
1. **Price Trends**: Average Selling Price (ASP) changes and inflation over the last decade.
2. **Sales Volumes**: Volume growth and promotional dips/spikes.
3. **Seasonality Analysis**: Back-to-School and Festive sales cycles.
4. **Price Elasticity (Visual)**: The relationship between price changes and demand shifts.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(title_text))
    
    # 2. Imports and Data Loading
    imports_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set style for charts
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# Load dataset
df = pd.read_csv("laptop_pricing_data.csv")
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded dataset with {len(df)} rows.")
print("Columns:", list(df.columns))
df.head()"""
    nb.cells.append(nbf.v4.new_code_cell(imports_code))
    
    # 3. Summary Stats
    summary_code = """# Summary Statistics by Brand
print("--- SUMMARY STATISTICS ---")
summary = df.groupby("brand").agg({
    "unit_price": ["min", "mean", "max", "std"],
    "units_sold": ["min", "mean", "max", "sum"],
    "cost_per_unit": ["min", "mean", "max"]
})
print(summary)"""
    nb.cells.append(nbf.v4.new_code_cell(summary_code))
    
    # 4. Price Trends Plot
    price_code = """# 1. Retail Price Trends (2015 - 2025)
plt.figure(figsize=(12, 5))
for brand, bdf in df.groupby('brand'):
    plt.plot(bdf['date'], bdf['unit_price'], label=brand, linewidth=2, alpha=0.85)
plt.title("10-Year Laptop Retail Price Trends (2015-2025)")
plt.xlabel("Date")
plt.ylabel("Unit Price (INR)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(price_code))
    
    # 5. Quantity Trends Plot
    qty_code = """# 2. Sales Demand Volumes (2015 - 2025)
plt.figure(figsize=(12, 5))
for brand, bdf in df.groupby('brand'):
    plt.plot(bdf['date'], bdf['units_sold'] / 1e3, label=brand, linewidth=1.5, alpha=0.8)
plt.title("Weekly Laptop Shipment Volumes (2015-2025)")
plt.xlabel("Date")
plt.ylabel("Weekly Volume Sold (1,000s Units)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(qty_code))
    
    # 6. Seasonality Analysis
    season_code = """# 3. Monthly Demand Seasonality
df['month'] = df['date'].dt.month
monthly_sales = df.groupby(['month', 'brand'])['units_sold'].mean().unstack()

plt.figure(figsize=(12, 5))
monthly_sales.plot(kind='bar', figsize=(12, 5), width=0.8)
plt.title("Average Monthly Volume Demand (Back-to-School & Festive Seasonality)")
plt.xlabel("Month of Year (1=Jan, 12=Dec)")
plt.ylabel("Average Weekly Volume (Units)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Brand")
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(season_code))
    
    # 7. Price vs Quantity Scatter Plot (Elasticity Visual)
    elast_code = """# 4. Log-Log Price vs Quantity Regression Lines (Elasticity Visual)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

brands = df['brand'].unique()
for idx, brand in enumerate(brands):
    bdf = df[df['brand'] == brand]
    log_p = np.log(bdf['unit_price'])
    log_q = np.log(bdf['units_sold'])
    
    axes[idx].scatter(log_p, log_q, alpha=0.3, color='tab:blue')
    
    # Fit line
    m, b = np.polyfit(log_p, log_q, 1)
    axes[idx].plot(log_p, m*log_p + b, color='red', linewidth=2, label=f"Slope (Elasticity): {m:.2f}")
    
    axes[idx].set_title(f"{brand} Price Sensitivity")
    axes[idx].set_xlabel("Log(Unit Price)")
    axes[idx].set_ylabel("Log(Units Sold)")
    axes[idx].grid(True, linestyle='--', alpha=0.5)
    axes[idx].legend()

plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(elast_code))
    
    # Save notebook
    out_path = os.path.join("/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/mod_test", "laptop_exploration.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Notebook saved to {out_path}")
    
    # Execute notebook
    print("Executing notebook to populate outputs...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with open(out_path) as f:
        nb_to_run = nbf.read(f, as_version=4)
        
    ep.preprocess(nb_to_run, {'metadata': {'path': '/Users/atishayjain/PycharmProjects/PwC/PriceAnalysis/mod_test'}})
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_to_run, f)
        
    print("Successfully executed and updated laptop_exploration.ipynb!")

if __name__ == "__main__":
    create_notebook()
