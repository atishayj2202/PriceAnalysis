import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def create_notebook():
    print("Creating branded_rice_exploration.ipynb...")
    nb = nbf.v4.new_notebook()
    
    # 1. Title cell
    title_text = """# Exploratory Data Analysis (EDA): Branded Basmati Rice (2021-2025)
This notebook performs a visual and statistical analysis of our new, realistic branded Basmati rice dataset. 
It covers:
1. **Price Trends**: How prices for India Gate, Daawat, and Fortune have changed over the last 4 years.
2. **Sales Volumes**: How demand fluctuates and where the spikes/dips occur.
3. **Seasonality Analysis**: Annual demand patterns (festivals, harvests, and monsoons).
4. **Price Elasticity (Visual)**: The relationship between price changes and demand shifts.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(title_text))
    
    # 2. Imports and Data Loading
    imports_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for charts
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# Load dataset
df = pd.read_csv("branded_rice_data.csv")
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
    "cost_per_unit": ["mean"]
})
# Calculate average gross margin: (price - cost) / price
margins = {}
for brand in df['brand'].unique():
    b_df = df[df['brand'] == brand]
    margin = ((b_df['unit_price'] - b_df['cost_per_unit']) / b_df['unit_price']).mean() * 100
    margins[brand] = f"{margin:.1f}%"

print(summary)
print("\\nAverage Gross Margins:", margins)"""
    nb.cells.append(nbf.v4.new_code_cell(summary_code))
    
    # 4. Price Trends Plot
    price_plot_code = """# Price Trends Over Time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='unit_price', hue='brand', linewidth=2, palette='Set1')
plt.title("Retail Price Trends for Branded Basmati Rice (2021 - 2025)", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Retail Price (INR/KG)")
plt.legend(title="Brand", frameon=True)
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(price_plot_code))
    
    # 5. Quantity Trends Plot
    qty_plot_code = """# Demand (Quantity Sold) Trends Over Time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='units_sold', hue='brand', linewidth=1.5, alpha=0.8, palette='Set1')
plt.title("Weekly Sales Volume (Demand) Trends (2021 - 2025)", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Quantity Sold (KG)")
plt.legend(title="Brand", frameon=True)
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(qty_plot_code))
    
    # 6. Seasonality Analysis
    seasonality_code = """# Monthly Seasonality Analysis
df['month'] = df['date'].dt.strftime('%m-%B')
monthly_sales = df.groupby(['brand', 'month'])['units_sold'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.barplot(data=monthly_sales, x='month', y='units_sold', hue='brand', palette='Set1')
plt.title("Average Sales Volume by Month (Seasonality)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Avg Quantity Sold (KG)")
plt.legend(title="Brand")
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(seasonality_code))
    
    # 7. Price vs Quantity Scatter Plot (Elasticity Visual)
    elasticity_plot_code = """# Price vs. Quantity Log-Log Scatter Plot with Regression Line
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
brands = df['brand'].unique()

for idx, brand in enumerate(brands):
    b_df = df[df['brand'] == brand]
    
    # Log transform
    log_p = np.log(b_df['unit_price'])
    log_q = np.log(b_df['units_sold'])
    
    sns.regplot(x=log_p, y=log_q, ax=axes[idx], color='teal', 
                scatter_kws={'alpha':0.4, 's':25}, line_kws={'color':'red', 'linewidth':2})
    
    # Calculate simple correlation
    corr = np.corrcoef(log_p, log_q)[0, 1]
    
    axes[idx].set_title(f"{brand} (Corr: {corr:.2f})", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Log(Price)")
    axes[idx].set_ylabel("Log(Quantity)")

plt.suptitle("Log-Log Price vs. Quantity Relationship (Elasticity Slope)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(elasticity_plot_code))
    
    # 8. Explanatory Markdown
    explanations_text = """### Key Insights from EDA:
1. **Price Inflation**: Retail prices across all brands rose significantly from 2021 to early 2024. India Gate spiked from ~₹165/kg to ₹235/kg, and Daawat from ~₹118/kg to ₹170/kg. This aligns with real-world Basmati crop price trends and the Indian government's export restrictions in late 2023.
2. **Festival & Seasonal Peaks**: We see clear demand spikes every **October and November** (Diwali/festival season) and smaller spikes in **January and February** (harvest and wedding seasons). A noticeable dip occurs during the monsoon season (**July to September**).
3. **Negative Elasticity Slope**: The Log-Log scatter plots show a clear, downward-sloping regression line. As prices go up, quantity sold goes down. The correlations are strongly negative, indicating that customers are price-sensitive as expected.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(explanations_text))
    
    # Write the notebook
    nb_path = "mod_test/branded_rice_exploration.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Jupyter notebook saved to {nb_path}!")
    
    # Execute the notebook
    print("Executing notebook to save outputs...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_loaded = nbf.read(f, as_version=4)
        ep.preprocess(nb_loaded, {'metadata': {'path': 'mod_test'}})
        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb_loaded, f)
        print("Notebook executed successfully with outputs saved in-place!")
    except Exception as e:
        print(f"Error executing notebook: {str(e)}")

if __name__ == "__main__":
    create_notebook()
