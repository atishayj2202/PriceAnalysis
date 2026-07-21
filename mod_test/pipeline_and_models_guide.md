# Technical Pipeline & Model Architecture Guide

This document provides a comprehensive, step-by-step technical explanation of the **Elasticity & Seasonality Pipeline** (`elasticity_pipeline.py`) and how each underlying statistical and machine learning model operates.

---

## 1. Pipeline Architecture Overview

The pipeline is built as an automated, modular framework designed to evaluate price elasticity and forecast weekly demand out-of-sample.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION ENGINE                            │
│  - Loads branded_rice_data.csv                                          │
│  - Filters by brand (India Gate, Daawat, Fortune)                       │
│  - Computes log transformations: log_p, log_q, log_comp_1, log_comp_2  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FOURIER SEASONALITY ENGINE                         │
│  - Fixed (K=2 harmonics, 52-week period)                                │
│  - Adaptive (AICc-optimized K ∈ [1..5])                                 │
│  - Multi-Period (Stacked 52-week K=2 + 13-week K=1)                     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROLLING BACKTEST SPLITTER                            │
│  - 104-week rolling training window                                     │
│  - 130 test weeks (out-of-sample forward step = 1 week)                 │
│  - Applies sample decay weighting: λ = 0.95^t                           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     MODEL FACTORY & FITTING ENGINE                      │
│  - Linear (OLS/WLS)          - Support Vector Regression (SVR)           │
│  - Two-Stage Least Sq (2SLS) - Random Forest (RF)                       │
│  - Ridge Regression          - Gradient Boosting (GB)                   │
│  - Huber Regressor           - Multi-Layer Perceptron (MLP)             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ELASTICITY EXTRACTION ENGINE                        │
│  - Direct Coefficient: β_1 from log-log regression                      │
│  - Finite Difference: [f(log_p + h) - f(log_p - h)] / 2h (h = 10^-4)     │
│  - Clamps elasticity to plausible range [-10.0, +2.0]                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     EVALUATION & RANKING ENGINE                         │
│  - WMAPE, MAPE, RMSE out-of-sample metrics                              │
│  - Plausibility & Stability Scoring                                     │
│  - Generates pipeline_results.csv                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Pipeline Mechanics

### Step 1: Preprocessing & Log Transformations
The pipeline transforms price and quantity into natural logarithmic space:
$$\text{log_p} = \ln(\text{unit price}), \quad \text{log_q} = \ln(\text{units sold})$$
$$\text{log_comp_1} = \ln(\text{comp price 1}), \quad \text{log_comp_2} = \ln(\text{comp price 2})$$

**Why log transformation?** In log-log space, the regression slope $\beta_1$ is constant and directly equals the price elasticity:
$$\frac{d \ln(Q)}{d \ln(P)} = \frac{dQ/Q}{dP/P} = \varepsilon$$

### Step 2: Fourier Seasonality Generation
Instead of using 12 monthly dummy variables (which introduce 11 extra parameters and sharp step jumps), the pipeline generates continuous sine and cosine harmonic pairs:

$$\sin_{52, k}(t) = \sin\left(\frac{2\pi \cdot k \cdot t}{52}\right), \quad \cos_{52, k}(t) = \cos\left(\frac{2\pi \cdot k \cdot t}{52}\right)$$

The pipeline supports 4 options:
1. **Baseline**: No seasonality terms added.
2. **Fixed**: Adds $K=2$ harmonics (4 features: $\sin_{52,1}, \cos_{52,1}, \sin_{52,2}, \cos_{52,2}$).
3. **Adaptive**: Evaluates $K \in \{1, 2, 3, 4, 5\}$ on the training slice and selects $K^*$ that minimizes corrected AIC ($\text{AICc}$):
   $$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{N - k - 1}$$
4. **Multi-Period**: Combines annual ($P=52, K=2$) and quarterly ($P=13, K=1$) harmonics (6 seasonal features total).

### Step 3: Rolling Window & Time-Decay Weighting
- **Training Window**: 104 weeks (2 full years).
- **Test Step**: 1 week ahead.
- **Decay Weighting**: When `use_decay = True`, observation $t$ weeks prior to the test date receives weight:
  $$w_t = 0.95^t$$
  Weights are normalized so that $\sum w_t = N_{\text{train}}$. This acts as a smooth Weighted Least Squares (WLS) exponential memory filter.

### Step 4: Elasticity Extraction
- **Linear Models (OLS, WLS, IV_2SLS, Ridge, Huber)**:
  Elasticity is extracted directly from the coefficient corresponding to `log_p`.
- **Non-Linear / ML Models (SVR, RF, GB, MLP)**:
  Elasticity is calculated numerically via **Finite Differences** around the mean feature vector $\bar{\mathbf{x}}$:
  $$\varepsilon \approx \frac{\hat{f}(\log P + h, \bar{\mathbf{x}}_{\text{other}}) - \hat{f}(\log P - h, \bar{\mathbf{x}}_{\text{other}})}{2h}, \quad \text{where } h = 10^{-4}$$

---

## 3. Deep Dive: How Each Model Type Works

### 1. Ordinary & Weighted Least Squares (`Linear`)
* **Mathematical Equation**:
  $$\ln(Q) = \beta_0 + \beta_1 \ln(P) + \beta_2 \ln(P_{\text{comp1}}) + \beta_3 \ln(P_{\text{comp2}}) + \beta_4 \text{Promo} + \sum_{k} \left[\alpha_k \sin_{52,k} + \gamma_k \cos_{52,k}\right] + \varepsilon$$
* **Fitting Method**:
  - OLS: Minimizes $\sum e_i^2$.
  - WLS: Minimizes $\sum w_i e_i^2$ using statsmodels `sm.WLS`.
* **Strengths**: Fully interpretable, structurally stable, direct elasticity $\beta_1$.

### 2. Two-Stage Least Squares (`IV_2SLS`)
* **Problem Addressed**: **Endogeneity**. Prices and demand are determined simultaneously. High demand in peak season allows retailers to charge higher prices, creating a positive feedback loop that biases naive OLS elasticity toward zero.
* **Solution**: Use **Cost of Goods Sold ($\ln(\text{Cost})$)** as an instrumental variable. Wholesale cost is driven by farm-gate supply factors, not retail demand.
* **Two-Stage Workflow**:
  - **Stage 1 (First Stage Regression)**:
    $$\ln(P) = \pi_0 + \pi_1 \ln(\text{Cost}) + \mathbf{Z}\mathbf{\pi}_2 + v$$
    Obtain fitted values $\hat{\ln(P)}$.
  - **Stage 2 (Second Stage Regression)**:
    $$\ln(Q) = \beta_0 + \beta_1 \hat{\ln(P)} + \mathbf{Z}\mathbf{\beta}_2 + u$$
* **Result**: $\beta_1$ represents the **pure causal price elasticity**, purged of demand-driven endogeneity.

### 3. Ridge Regression (`Ridge`)
* **Mathematical Equation**: Minimizes penalised residual sum of squares:
  $$\min_{\mathbf{\beta}} \sum_{i=1}^N (y_i - \mathbf{x}_i^T \mathbf{\beta})^2 + \alpha \sum_{j=1}^p \beta_j^2$$
* **Hyperparameter**: $\alpha = 1.0$.
* **Why it fails for elasticity**: When `log_p` and `log_comp_1` are correlated, Ridge shrinks both coefficients equally toward zero. This suppresses own-price elasticity from -0.57 to -0.01.

### 4. Huber Regressor (`Huber`)
* **Mathematical Equation**: Uses Huber loss instead of squared loss:
  $$L_{\delta}(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \le \delta \\ \delta (|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
* **Hyperparameter**: $\delta = 1.35$.
* **Strengths**: Outliers (e.g., massive holiday demand spikes) do not distort the regression line.

### 5. Support Vector Regression (`SVR`)
* **Mathematical Equation**: Fits an $\epsilon$-insensitive tube around the data in a high-dimensional kernel space:
  $$\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum (\xi_i + \xi_i^*)$$
* **Kernel**: Radial Basis Function (RBF) $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.
* **Why it fails for elasticity**: RBF SVR maps inputs to an infinite-dimensional space. The local partial derivative $\frac{\partial \hat{y}}{\partial x_{\log p}}$ fluctuates wildly or flattens out, leading to 0 elasticity.

### 6. Random Forest (`RF`) & Gradient Boosting (`GB`)
* **Random Forest**: Ensemble of 50 decision trees trained on bootstrap samples with max depth 4.
* **Gradient Boosting**: Sequential ensemble of 50 decision trees optimizing squared error with max depth 3.
* **Why tree models fail for elasticity**: Decision trees split feature space into hyper-rectangles (constant step functions). The derivative $\frac{d}{dx}\text{StepFunction}(x)$ is zero almost everywhere. Finite difference calculations yield elasticities near **0.000**.

### 7. Multi-Layer Perceptron (`MLP`)
* **Architecture**: Input Layer $\rightarrow$ Dense Layer (8 neurons, `tanh`) $\rightarrow$ Dense Layer (4 neurons, `tanh`) $\rightarrow$ Output Layer (1 neuron).
* **Decay Implementation**: Scikit-learn's `MLPRegressor` does not accept `sample_weight`. We implemented **Weighted Bootstrap Resampling**:
  $$p_i = \frac{w_i}{\sum w_k}, \quad \text{Sample } N \text{ rows with replacement using probabilities } p_i$$
* **Strengths & Weaknesses**: Captures non-linear interactions, but requires high iteration counts and exhibits random initialization variance.
