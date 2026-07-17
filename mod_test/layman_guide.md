# A Layman's Guide to Price Elasticity & Seasonality Modeling

This guide explains how our pricing analysis model and pipeline work. It is designed for business managers, commercial teams, and anyone who wants to understand the concepts and technology without needing a degree in statistics or machine learning.

---

## 1. What is Price Elasticity?

Imagine you own a supermarket and you sell **Daawat Basmati Rice**. 

* If you **raise the price** of a 5kg bag by **10%**, you expect to sell **fewer bags**.
* If you **lower the price** by **10%**, you expect to sell **more bags**.

**Price Elasticity of Demand** is a number that tells you exactly *how sensitive* your customers are to these price changes.

### How to Read the Elasticity Number
The elasticity number is almost always negative because price and demand move in opposite directions:
* **Elasticity of -1.5**: This means if you **increase the price by 1%**, your sales volume will **drop by 1.5%**. Customers are sensitive to price changes.
* **Elasticity of -0.5**: This means if you **increase the price by 1%**, your sales volume will **only drop by 0.5%**. Customers are relatively insensitive to price changes (often because of brand loyalty).

---

## 2. The Core Problem: Why Omitted Variable Bias distort Elasticity?

If estimating elasticity was as simple as dividing sales changes by price changes, we wouldn't need advanced software. In the real world, many factors affect demand simultaneously.

For example, look at this scenario:
1. **Monsoon season** hits, and overall rice consumption naturally dips.
2. At the same time, you run a **discount promotion** on India Gate rice.
3. Your sales volume stays flat.

If your model only looks at price and quantity, it might conclude: *"A discount had zero effect on sales, so elasticity is 0!"* 

This is **Omitted Variable Bias**. The model missed the monsoon dip, which masked the positive effect of the discount. To find the *true* price sensitivity, we must control for seasonality, competitor pricing, and promotions.

---

## 3. How We Model Seasonality (Fourier Equation)

To control for seasonality, we use **Fourier Series (Sine and Cosine waves)**. 

### Why not just use monthly flags?
Historically, analysts used simple "monthly flags" (e.g., 1 for December, 0 otherwise). But demand doesn't jump abruptly on December 1st and drop on January 1st. It flows smoothly.

Sine and cosine waves are perfect for this because they are periodic and smooth. They repeat every 52 weeks (yearly cycle) or 13 weeks (quarterly cycle).

Our pipeline implements seasonality in **3 different ways**:

1. **Fixed Fourier Seasonality (Approach A)**:
   - Uses two waves (harmonics) at a fixed 52-week period.
   - It captures a smooth annual cycle (representing the winter harvest and peak holiday demand).
   - This is simple, robust, and rarely overfits the data.

2. **Adaptive Seasonality (Approach B)**:
   - The pipeline programmatically tests different wave complexities (from 1 to 5 harmonics).
   - It uses an information criterion called **AICc** (which acts like a judge that penalizes complexity to prevent overfitting).
   - It automatically selects the optimal wave structure that fits the historical data best.

3. **Multi-Period Seasonality (Approach C)**:
   - It stacks multiple periods together: an annual period (52 weeks) AND a quarterly period (13 weeks).
   - This captures both agricultural seasons (yearly) and corporate promotional cycles (quarterly).

---

## 4. Addressing Endogeneity (The Biggest Pitfall in Pricing)

In business, prices are not set randomly. 
* Retailers **increase prices** during peak seasons (like Diwali) because they know demand is strong.
* They **lower prices** during off-seasons to clear stock.

This creates a loop: **Price affects Demand, but Demand also affects Price.** 
In statistics, this is called **Endogeneity**. If you fit a simple regression line to this data, it will suggest that higher prices lead to higher sales, which violates the laws of economics!

### How we solve it: Two-Stage Least Squares (2SLS)
To break the loop, we use an industry-standard technique called **Two-Stage Least Squares (2SLS)** using **Instrumental Variables**:
1. **Stage 1 (Isolate the Price Shift)**: We predict the retail price using the **wholesale cost of goods sold (COGS)** and seasonal variables. Wholesale cost is driven by farm-gate prices and supply chain factors, *not* by retail demand. We call this predicted price the "clean price."
2. **Stage 2 (Measure Sensitivity)**: We measure how the "clean price" affects sales quantity. 

By using the cost-driven wholesale price as a bridge, we isolate the true, exogenous customer sensitivity, free from demand-driven price increases.

---

## 5. Our Pipeline Architecture: Finding the Best Fit

Our pipeline script (`elasticity_pipeline.py`) acts as a massive laboratory. It tests **192 different combinations** across:
* **3 Brands**: India Gate, Daawat, and Fortune.
* **8 Regression & Machine Learning Models**:
  - *Linear (OLS/WLS)*: Simple, standard lines.
  - *IV_2SLS*: Cost-instrumented causal model.
  - *Ridge/Huber*: Robust lines that handle outliers.
  - *SVR / Random Forest / Gradient Boosting / MLP (Neural Net)*: Advanced models that capture non-linear behaviors.
* **4 Seasonality Settings**: Baseline (none), Fixed, Adaptive, and Multi-period.
* **2 Decay Options**: With and without time-decay (recent weeks weighted higher to reflect changing customer habits).

### The Ranking System
We don't just pick the model with the lowest training error. That leads to overfitting. Instead, we rank configurations using a balanced score:
1. **Accuracy (40%)**: Measured using **WMAPE** (Weighted Mean Absolute Percentage Error). It checks how close predictions are to actual sales.
2. **Plausibility (35%)**: Checks if the estimated elasticity makes economic sense (must be negative and between -0.5 and -4.0).
3. **Stability (25%)**: Checks if the elasticity estimate is stable over time or if it fluctuates wildly.

This ranking ensures the winning model is both highly accurate and highly reliable for real-world pricing decisions.
