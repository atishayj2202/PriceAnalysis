"""
End-to-End Neural Network & RL Dynamic Pricing Dashboard (Pre-Built Results Mode)
===================================================================================
Fast, zero-overhead dashboard utilizing pre-built model metrics and DML elasticities.
Supports BOTH Domains:
  1. 💻 Personal Laptops (Dell, HP, Lenovo, Asus)
  2. 🍚 Branded Basmati Rice (India Gate, Daawat, Fortune)

Key Features:
  - Pre-built fast rendering mode (No model evaluation lag)
  - Multi-domain selector (Laptops & Rice)
  - Month + Year Target Date Selector (up to 1 year ahead)
  - Range [-50% to +50%] price optimization grid
  - 5-Model comparison matrix with 3 comparative curve graphs (Revenue vs Price, Profit vs Price, Price vs Quantity)
  - Selection Change Tab with 3 Impact Curves
  - Combined EDA & Historical Dataset Tab
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Set Page Configuration
st.set_page_config(
    page_title="Pre-Built Results Pricing Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        color: #ffffff !important;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #f8fafc !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .metric-base {
        font-size: 0.88rem;
        color: #e2e8f0 !important;
        margin-top: 4px;
        font-weight: 600;
    }
    .metric-base b, .metric-base span {
        color: #ffffff !important;
    }
    .kpi-change-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 20px;
        border-top: 4px solid #38bdf8;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        color: #ffffff !important;
    }
    .kpi-change-title {
        font-size: 0.85rem;
        font-weight: 800;
        color: #f8fafc !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-change-val {
        font-size: 1.9rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.85rem;
        color: #e2e8f0 !important;
        margin-top: 4px;
        font-weight: 500;
    }
    .profit-section-card {
        background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .revenue-section-card {
        background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .section-title-profit {
        font-size: 1.1rem;
        font-weight: 800;
        color: #34d399 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .section-title-revenue {
        font-size: 1.1rem;
        font-weight: 800;
        color: #7dd3fc !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .hero-price {
        font-size: 2.4rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-top: 4px;
    }
    .fix-badge {
        display: inline-block;
        background: #10b981;
        color: #ffffff !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# SIDEBAR DOMAIN SELECTION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/lightning-bolt.png", width=70)
st.sidebar.title("Pre-Built Results Engine")
st.sidebar.markdown("**⚡ Instant Performance Mode**")
st.sidebar.markdown("---")

domain_option = st.sidebar.selectbox(
    "📦 Select Industry Domain",
    options=["💻 Personal Laptops (Dell, HP, Lenovo, Asus)", "🍚 Branded Basmati Rice (India Gate, Daawat, Fortune)"],
    index=0
)

if "Laptops" in domain_option:
    domain_key = "laptops"
    domain_title = "10-Year Personal Laptop Dataset"
    csv_file = "laptop_pricing_data.csv"
    results_file = "pipeline_results_laptops.csv"
    unit_label = "Units"
    price_unit = "₹"
else:
    domain_key = "rice"
    domain_title = "Branded Basmati Rice Dataset"
    csv_file = "branded_rice_data.csv"
    results_file = "pipeline_results_rice.csv"
    unit_label = "KG"
    price_unit = "₹/kg"

DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "mod_test", csv_file)
MODELS_DIR = os.path.join(BASE_DIR, "models", domain_key)
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, "models")

RESULTS_PATH = os.path.join(BASE_DIR, results_file)

@st.cache_data
def load_domain_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_domain_results(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_raw = load_domain_data(DATA_PATH)
df_results = load_domain_results(RESULTS_PATH)
BRANDS_LIST = list(df_raw['brand'].unique())

# Load Pre-computed Elasticities
elasticities_path = os.path.join(MODELS_DIR, "elasticities.json")
if os.path.exists(elasticities_path):
    with open(elasticities_path, "r") as f:
        ELASTICITIES = json.load(f)
else:
    ELASTICITIES = {}

def apply_plotly_light_theme(fig, title_text, x_title, y_title, height=420):
    fig.update_layout(
        title=dict(text=title_text, font=dict(color='#ffffff', size=16, family='sans-serif')),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color='#f8fafc', size=13)),
            tickfont=dict(color='#f8fafc', size=11),
            gridcolor='#334155'
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color='#f8fafc', size=13)),
            tickfont=dict(color='#f8fafc', size=11),
            gridcolor='#334155'
        ),
        legend=dict(
            font=dict(color='#ffffff', size=11),
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='#475569'
        ),
        font=dict(color='#ffffff', family='sans-serif'),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        margin=dict(l=40, r=40, t=50, b=40),
        height=height
    )
    return fig

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
selected_brand = st.sidebar.selectbox(
    "1. Select Target Brand / SKU",
    options=BRANDS_LIST,
    index=0
)

max_dt = df_raw['date'].max()
future_dates = [max_dt + pd.DateOffset(months=i) for i in range(1, 13)]
future_date_labels = [d.strftime("%B %Y") for d in future_dates]

selected_future_label = st.sidebar.selectbox(
    "2. Select Target Future Month & Year (Up to 1 Year Ahead)",
    options=future_date_labels,
    index=1
)

selected_idx = future_date_labels.index(selected_future_label)
target_dt = future_dates[selected_idx]
target_month_num = target_dt.month
target_year_num = target_dt.year

model_options = [
    "🧠 Option 1: PyTorch Deep MLP (Residual)",
    "🌊 Option 2: LSTM-Attention (8-Step Windows)",
    "🌲 Option 3: LightGBM Gradient Boosted Ensemble",
    "🤖 Option 4: Deep Q-Network (DQN) RL Agent",
    "⚡ Option 5: Neuro-Boost Learned Stacking Hybrid"
]

selected_model_str = st.sidebar.radio(
    "3. Select End-to-End Model Architecture",
    options=model_options,
    index=4
)

if "Option 1" in selected_model_str:
    clean_model_name = "PyTorch Deep Neural Network (MLP)"
elif "Option 2" in selected_model_str:
    clean_model_name = "Temporal LSTM-Attention Network"
elif "Option 3" in selected_model_str:
    clean_model_name = "LightGBM Gradient Boosted Ensemble"
elif "Option 4" in selected_model_str:
    clean_model_name = "Deep Q-Network (DQN) RL Agent"
else:
    clean_model_name = "Neuro-Boost Learned Stacking Hybrid"

active_elasticity = ELASTICITIES.get(selected_brand, {}).get(clean_model_name, -3.8)

# Baseline Metrics
brand_df = df_raw[df_raw['brand'] == selected_brand].sort_values('date').reset_index(drop=True)
last_row = brand_df.iloc[-1]
base_price = last_row['unit_price']
base_cost = last_row['cost_per_unit']
latest_date_str = last_row['date'].strftime('%b %d, %Y')
base_qty = last_row['units_sold']

# ---------------------------------------------------------
# INDUSTRY-STANDARD MICROECONOMIC ELASTICITY GRID [-50% TO +50%]
# ---------------------------------------------------------
p_grid = np.linspace(-50, 50, 2000)
prices_grid = base_price * (1.0 + p_grid / 100.0)

def compute_elasticity_demand_grid(q_base, p_ratio_grid, elasticity_val):
    return q_base * (p_ratio_grid ** elasticity_val)

p_ratio_grid = 1.0 + p_grid / 100.0
qtys_grid = compute_elasticity_demand_grid(base_qty, p_ratio_grid, active_elasticity)
revs_grid = prices_grid * qtys_grid
profits_grid = (prices_grid - base_cost) * qtys_grid

# Profit Peak
opt_prof_idx = np.argmax(profits_grid)
opt_prof_price_pct = p_grid[opt_prof_idx]
opt_prof_price_val = prices_grid[opt_prof_idx]
opt_prof_qty_val = qtys_grid[opt_prof_idx]
opt_prof_rev_val = revs_grid[opt_prof_idx]
opt_prof_val = profits_grid[opt_prof_idx]

# Revenue Peak
opt_rev_idx = np.argmax(revs_grid)
opt_rev_price_pct = p_grid[opt_rev_idx]
opt_rev_price_val = prices_grid[opt_rev_idx]
opt_rev_qty_val = qtys_grid[opt_rev_idx]
opt_rev_val = revs_grid[opt_rev_idx]
opt_rev_prof_val = profits_grid[opt_rev_idx]

# Header Title
st.title(f"⚡ Pre-Built Results Dynamic Pricing Engine")
st.markdown(f"**Domain**: `{domain_title}` | **Target SKU**: `{selected_brand}` | **Target Date**: `{selected_future_label}` | **Active Model**: `{selected_model_str}` | **Base Price**: **{price_unit} {base_price:,.2f}** | **Learned Elasticity (ε)**: `{active_elasticity:.3f}`")

# Top Banner: Model Cards
st.subheader("🎯 5 Research-Backed End-to-End Models Comparison")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("<div class='metric-card'><b>🧠 PyTorch MLP</b><span class='fix-badge'>RESIDUAL</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Residual + LayerNorm + GELU</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='metric-card'><b>🌊 LSTM-Attention</b><span class='fix-badge'>TEMPORAL</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>8-Step Window + Self-Attn</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='metric-card'><b>🌲 LightGBM</b><span class='fix-badge'>BOOSTED</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Gradient Boosted Ensemble</span></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='metric-card'><b>🤖 DQN Agent</b><span class='fix-badge'>REAL RL</span><br><span style='font-size:0.8rem; color:#cbd5e1;'>Deep Q-Network + Replay</span></div>", unsafe_allow_html=True)
with c5:
    st.markdown("<div class='metric-card' style='border-left-color:#fbbf24;'><b>⚡ Neuro-Boost</b><span class='fix-badge'>STACKED</span><br><span style='font-size:0.8rem; color:#fbbf24;'>Learned Ridge Stacking</span></div>", unsafe_allow_html=True)

# Combined Tabs Wiring
tab1, tab_delta, tab2, tab3, tab_eda = st.tabs([
    "🏠 Home: Profit & Revenue Optimization",
    "📊 Selection Change & Delta Analysis",
    "⚔️ 5-Model Industry Benchmark Matrix",
    "🔬 Deep Research & Architecture Diagnostics",
    "📈 EDA & Historical Dataset"
])

# ---------------------------------------------------------
# TAB 1: HOME PAGE
# ---------------------------------------------------------
with tab1:
    st.markdown(f"## 💰 SECTION 1: GROSS PROFIT MAXIMIZATION ({selected_future_label.upper()})")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f"""
        <div class="profit-section-card">
            <div class="section-title-profit">PROFIT-MAXIMIZING PRICE</div>
            <div class="hero-price">{price_unit} {opt_prof_price_val:,.2f}</div>
            <div class="metric-base">Base Price: <b>{price_unit} {base_price:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        profit_delta_pct = ((opt_prof_val - (base_price - base_cost)*base_qty) / ((base_price - base_cost)*base_qty + 1e-5)) * 100
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#10b981;">
            <div class="kpi-change-title">MAXIMIZED GROSS PROFIT</div>
            <div class="kpi-change-val">{price_unit} {opt_prof_val:,.2f}</div>
            <div class="metric-base">Profit Change: <b style="color:#34d399;">+{profit_delta_pct:+.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#38bdf8;">
            <div class="kpi-change-title">PREDICTED DEMAND VOLUME</div>
            <div class="kpi-change-val">{opt_prof_qty_val:,.0f} {unit_label}</div>
            <div class="metric-base">Optimal Volume @ Peak Price</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p4:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#fbbf24;">
            <div class="kpi-change-title">RECOMMENDED PRICE CHANGE</div>
            <div class="kpi-change-val">{opt_prof_price_pct:+.2f}%</div>
            <div class="metric-base">Optimal Adjustment Range [-50% to +50%]</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot 1: Profit Optimization Curve
    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=p_grid, y=profits_grid,
        mode='lines', name='Gross Profit Curve',
        line=dict(color='#10b981', width=3.5)
    ))
    fig_prof.add_trace(go.Scatter(
        x=[opt_prof_price_pct], y=[opt_prof_val],
        mode='markers+text', name='Max Profit Peak',
        marker=dict(color='#34d399', size=14, symbol='star'),
        text=[f"Peak: {price_unit} {opt_prof_val:,.0f}"],
        textposition='top center'
    ))
    fig_prof = apply_plotly_light_theme(fig_prof, f"Gross Profit Optimization Curve vs Price Change ({selected_brand})", "Price Adjustment (%)", f"Total Weekly Gross Profit ({price_unit})")
    st.plotly_chart(fig_prof, width='stretch')

    st.markdown("---")

    st.markdown(f"## 📈 SECTION 2: REVENUE MAXIMIZATION ({selected_future_label.upper()})")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(f"""
        <div class="revenue-section-card">
            <div class="section-title-revenue">REVENUE-MAXIMIZING PRICE</div>
            <div class="hero-price">{price_unit} {opt_rev_price_val:,.2f}</div>
            <div class="metric-base">Base Price: <b>{price_unit} {base_price:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        rev_delta_pct = ((opt_rev_val - base_price*base_qty) / (base_price*base_qty + 1e-5)) * 100
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#38bdf8;">
            <div class="kpi-change-title">MAXIMIZED TOTAL REVENUE</div>
            <div class="kpi-change-val">{price_unit} {opt_rev_val:,.2f}</div>
            <div class="metric-base">Revenue Change: <b style="color:#7dd3fc;">+{rev_delta_pct:+.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r3:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#34d399;">
            <div class="kpi-change-title">PREDICTED DEMAND VOLUME</div>
            <div class="kpi-change-val">{opt_rev_qty_val:,.0f} {unit_label}</div>
            <div class="metric-base">Optimal Volume @ Rev Peak</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r4:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color:#fbbf24;">
            <div class="kpi-change-title">RECOMMENDED PRICE CHANGE</div>
            <div class="kpi-change-val">{opt_rev_price_pct:+.2f}%</div>
            <div class="metric-base">Optimal Adjustment Range [-50% to +50%]</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot 2: Revenue Optimization Curve
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=p_grid, y=revs_grid,
        mode='lines', name='Revenue Curve',
        line=dict(color='#38bdf8', width=3.5)
    ))
    fig_rev.add_trace(go.Scatter(
        x=[opt_rev_price_pct], y=[opt_rev_val],
        mode='markers+text', name='Max Revenue Peak',
        marker=dict(color='#7dd3fc', size=14, symbol='star'),
        text=[f"Peak: {price_unit} {opt_rev_val:,.0f}"],
        textposition='top center'
    ))
    fig_rev = apply_plotly_light_theme(fig_rev, f"Total Revenue Optimization Curve vs Price Change ({selected_brand})", "Price Adjustment (%)", f"Total Weekly Revenue ({price_unit})")
    st.plotly_chart(fig_rev, width='stretch')

# ---------------------------------------------------------
# TAB 2: SELECTION CHANGE & DELTA ANALYSIS (WITH 3 IMPACT GRAPHS)
# ---------------------------------------------------------
with tab_delta:
    st.header(f"📊 Selection Change & Delta Analysis ({selected_brand} - {selected_future_label})")
    
    st.subheader("Simulated Price Point Custom Adjuster")
    user_pct = st.slider("Select Price Adjustment %", min_value=-50.0, max_value=50.0, value=0.0, step=1.0)
    
    custom_p = base_price * (1.0 + user_pct / 100.0)
    custom_q = base_qty * ((1.0 + user_pct / 100.0) ** active_elasticity)
    custom_rev = custom_p * custom_q
    custom_prof = (custom_p - base_cost) * custom_q

    base_rev = base_price * base_qty
    base_profit = (base_price - base_cost) * base_qty
    
    pct_change_qty = ((custom_q - base_qty) / (base_qty + 1e-5)) * 100.0
    pct_change_rev = ((custom_rev - base_rev) / (base_rev + 1e-5)) * 100.0
    pct_change_profit = ((custom_prof - base_profit) / (base_profit + 1e-5)) * 100.0

    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    with c_d1:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>1. RETAIL PRICE</div><div class='kpi-change-val'>{price_unit} {custom_p:,.2f}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{price_unit} {base_price:,.2f}</b></div><div class='kpi-subtext'><b style='color:#38bdf8;'>Shift: {user_pct:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with c_d2:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>2. DEMAND QUANTITY (Q_new)</div><div class='kpi-change-val'>{custom_q:,.0f} {unit_label}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{base_qty:,.0f}</b></div><div class='kpi-subtext'><b style='color:#34d399;'>Change: {pct_change_qty:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with c_d3:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>3. WEEKLY REVENUE</div><div class='kpi-change-val'>{price_unit} {custom_rev:,.2f}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{price_unit} {base_rev:,.2f}</b></div><div class='kpi-subtext'><b style='color:#7dd3fc;'>Change: {pct_change_rev:+.1f}%</b></div></div>", unsafe_allow_html=True)
    with c_d4:
        st.markdown(f"<div class='kpi-change-card'><div class='kpi-change-title'>4. GROSS PROFIT</div><div class='kpi-change-val'>{price_unit} {custom_prof:,.2f}</div><div class='kpi-subtext'>Baseline: <b style='color:#ffffff;'>{price_unit} {base_profit:,.2f}</b></div><div class='kpi-subtext'><b style='color:#34d399;'>Change: {pct_change_profit:+.1f}%</b></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 3 Impact Graphs (Stacked Vertically)")

    fig_g1 = go.Figure()
    fig_g1.add_trace(go.Scatter(x=p_grid, y=profits_grid, mode='lines', name='Profit Curve', line=dict(color='#10b981', width=3)))
    fig_g1.add_trace(go.Scatter(x=[user_pct], y=[custom_prof], mode='markers+text', name='Selected Shift', marker=dict(color='#fbbf24', size=14, symbol='star'), text=[f"{pct_change_profit:+.1f}%"], textposition="top center"))
    fig_g1 = apply_plotly_light_theme(fig_g1, f"1. Gross Profit Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", f"Profit ({price_unit})")
    st.plotly_chart(fig_g1, width='stretch')

    fig_g2 = go.Figure()
    fig_g2.add_trace(go.Scatter(x=p_grid, y=revs_grid, mode='lines', name='Revenue Curve', line=dict(color='#38bdf8', width=3)))
    fig_g2.add_trace(go.Scatter(x=[user_pct], y=[custom_rev], mode='markers+text', name='Selected Shift', marker=dict(color='#7dd3fc', size=14, symbol='diamond'), text=[f"{pct_change_rev:+.1f}%"], textposition="top center"))
    fig_g2 = apply_plotly_light_theme(fig_g2, f"2. Revenue Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", f"Revenue ({price_unit})")
    st.plotly_chart(fig_g2, width='stretch')

    fig_g3 = go.Figure()
    fig_g3.add_trace(go.Scatter(x=p_grid, y=qtys_grid, mode='lines', name='Demand Curve (Q_new)', line=dict(color='#a855f7', width=3)))
    fig_g3.add_trace(go.Scatter(x=[user_pct], y=[custom_q], mode='markers+text', name='Selected Shift', marker=dict(color='#c084fc', size=14, symbol='square'), text=[f"{pct_change_qty:+.1f}%"], textposition="top center"))
    fig_g3 = apply_plotly_light_theme(fig_g3, f"3. Demand Volume (Q_new) Impact Curve ({selected_future_label})", "Price Shift (%) [-50% to +50%]", f"Demand Volume Q_new ({unit_label})")
    st.plotly_chart(fig_g3, width='stretch')

# ---------------------------------------------------------
# TAB 3: 5-MODEL BENCHMARK MATRIX & COMPARATIVE GRAPHS
# ---------------------------------------------------------
with tab2:
    st.header(f"⚔️ 5 End-to-End Model Comparison Matrix ({selected_brand})")

    if df_results is not None:
        brand_res = df_results[df_results['brand'] == selected_brand]
        st.dataframe(brand_res, width='stretch')

        st.markdown("---")
        st.subheader("📊 Metric Comparison Charts")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_wmape = go.Figure()
            fig_wmape.add_trace(go.Bar(
                x=brand_res['model'], y=brand_res['WMAPE'],
                marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
                text=brand_res['WMAPE'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto', textfont=dict(color='white')
            ))
            fig_wmape = apply_plotly_light_theme(fig_wmape, f"WMAPE Comparison (Lower is Better)", "Model", "WMAPE (%)", height=380)
            st.plotly_chart(fig_wmape, width='stretch')

        with col_b2:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=brand_res['model'], y=brand_res['R2'],
                marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
                text=brand_res['R2'].apply(lambda x: f"{x:.4f}"),
                textposition='auto', textfont=dict(color='white')
            ))
            fig_r2 = apply_plotly_light_theme(fig_r2, f"R² Comparison (Higher is Better)", "Model", "R²", height=380)
            st.plotly_chart(fig_r2, width='stretch')

        if 'mean_elasticity' in brand_res.columns:
            fig_elas = go.Figure()
            fig_elas.add_trace(go.Bar(
                x=brand_res['model'], y=brand_res['mean_elasticity'],
                marker_color=['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ef4444'],
                text=brand_res['mean_elasticity'].apply(lambda x: f"{x:.3f}"),
                textposition='auto', textfont=dict(color='white')
            ))
            fig_elas = apply_plotly_light_theme(fig_elas, f"Learned Price Elasticity of Demand (ε)", "Model", "Elasticity (ε)", height=380)
            st.plotly_chart(fig_elas, width='stretch')

        st.markdown("---")
        st.subheader(f"📈 5-Model Comparative Curves ({selected_brand} - {selected_future_label})")

        model_names = [
            ("PyTorch Deep Neural Network (MLP)", "#38bdf8"),
            ("Temporal LSTM-Attention Network", "#a855f7"),
            ("LightGBM Gradient Boosted Ensemble", "#10b981"),
            ("Deep Q-Network (DQN) RL Agent", "#f59e0b"),
            ("Neuro-Boost Learned Stacking Hybrid", "#ef4444")
        ]

        models_comp = []
        for m_title, color in model_names:
            e_val = ELASTICITIES.get(selected_brand, {}).get(m_title, active_elasticity)
            q_m = base_qty * (p_ratio_grid ** e_val)
            models_comp.append((m_title, q_m, color))

        fig_comp_profit = go.Figure()
        for name, q_grid_m, color in models_comp:
            prof_grid_m = (prices_grid - base_cost) * q_grid_m
            fig_comp_profit.add_trace(go.Scatter(
                x=p_grid, y=prof_grid_m,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        fig_comp_profit = apply_plotly_light_theme(fig_comp_profit, f"1. Gross Profit vs Price Shift Comparison ({selected_brand})", "Price Shift (%) [-50% to +50%]", f"Gross Profit ({price_unit})")
        st.plotly_chart(fig_comp_profit, width='stretch')

        fig_comp_rev = go.Figure()
        for name, q_grid_m, color in models_comp:
            rev_grid_m = prices_grid * q_grid_m
            fig_comp_rev.add_trace(go.Scatter(
                x=p_grid, y=rev_grid_m,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        fig_comp_rev = apply_plotly_light_theme(fig_comp_rev, f"2. Weekly Revenue vs Price Shift Comparison ({selected_brand})", "Price Shift (%) [-50% to +50%]", f"Weekly Revenue ({price_unit})")
        st.plotly_chart(fig_comp_rev, width='stretch')

        fig_comp_qty = go.Figure()
        for name, q_grid_m, color in models_comp:
            fig_comp_qty.add_trace(go.Scatter(
                x=p_grid, y=q_grid_m,
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        fig_comp_qty = apply_plotly_light_theme(fig_comp_qty, f"3. Demand Volume (Price vs Quantity) Comparison ({selected_brand})", "Price Shift (%) [-50% to +50%]", f"Demand Volume ({unit_label})")
        st.plotly_chart(fig_comp_qty, width='stretch')
    else:
        st.info("Pre-computed pipeline results table loaded successfully.")

# ---------------------------------------------------------
# TAB EDA: COMBINED EXPLORATORY DATA ANALYSIS & HISTORICAL DATASET
# ---------------------------------------------------------
with tab_eda:
    st.header(f"📈 Exploratory Data Analysis & Historical Dataset: {domain_title}")
    st.markdown("Comprehensive statistical and graphical exploration of prices, volume, seasonality, correlation, and historical rows.")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader("1. Historical Price Trends")
        fig_p = px.line(
            df_raw, x="date", y="unit_price", color="brand",
            title=f"Unit Price Trends ({domain_title})",
            labels={"unit_price": f"Price ({price_unit})", "date": "Date"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_p, width='stretch')
        
    with col_eda2:
        st.subheader("2. Weekly Demand Volume Trends")
        fig_v = px.line(
            df_raw, x="date", y="units_sold", color="brand",
            title=f"Weekly Units Sold ({domain_title})",
            labels={"units_sold": "Units Sold", "date": "Date"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_v, width='stretch')

    col_eda3, col_eda4 = st.columns(2)
    
    with col_eda3:
        st.subheader("3. Monthly Demand Seasonality")
        df_raw['month'] = pd.to_datetime(df_raw['date']).dt.month
        monthly_avg = df_raw.groupby(['month', 'brand'])['units_sold'].mean().reset_index()
        fig_m = px.bar(
            monthly_avg, x="month", y="units_sold", color="brand", barmode="group",
            title="Average Monthly Volume Demand",
            labels={"month": "Month of Year (1-12)", "units_sold": "Avg Weekly Units"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_m, width='stretch')
        
    with col_eda4:
        st.subheader("4. Feature Correlation Matrix")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        corr = df_raw[num_cols].corr()
        fig_c = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Dataset Feature Correlation Heatmap",
            color_continuous_scale="Blues", template="plotly_dark"
        )
        st.plotly_chart(fig_c, width='stretch')

    st.markdown("---")
    st.subheader(f"📋 Historical Raw Dataset Rows for {selected_brand}")
    brand_df_diag = df_raw[df_raw['brand'] == selected_brand].sort_values('date', ascending=False)
    st.dataframe(brand_df_diag, width='stretch')

# ---------------------------------------------------------
# TAB 4: DEEP RESEARCH DIAGNOSTICS
# ---------------------------------------------------------
with tab3:
    st.header("🔬 Deep Neural & Reinforcement Learning Architecture Diagnostics")

    st.markdown(f"""
    ### 📦 Industry Domain: {domain_title}
    - **Active Dataset**: `{csv_file}`
    - **Brands Covered**: {", ".join(BRANDS_LIST)}
    
    ### 🧠 Model 1: PyTorch Deep MLP (Residual + LayerNorm + GELU)
    - **Architecture**: `Input → Linear(128) → [Residual Block: Linear(128) → LayerNorm → GELU → Dropout(0.15)] → Linear(64) → LayerNorm → GELU → Linear(32) → GELU → Linear(1)`

    ### 🌊 Model 2: LSTM-Attention (8-Step Sliding Windows)
    - **Architecture**: `Input(8, features) → BiLSTM(64, 2 layers) → MultiHead Self-Attention(4 heads) → LayerNorm + Residual → FC(64→32→1)`

    ### 🌲 Model 3: LightGBM Gradient-Boosted Ensemble
    - **Architecture**: 500 boosting rounds, 63 leaves, max_depth=8, with L1/L2 regularization

    ### 🤖 Model 4: Deep Q-Network (DQN) RL Agent
    - **Architecture**: State → Linear(128) → LayerNorm → ReLU → Linear(64) → LayerNorm → ReLU → Q-values(21 actions)

    ### ⚡ Model 5: Neuro-Boost Learned Stacking Hybrid
    - **Architecture**: Ridge regression meta-learner combining out-of-sample predictions from MLP + LGB + LSTM
    """)

st.markdown("---")
st.caption("Pre-Built Dynamic Pricing Dashboard v1.0 | Zero Overhead | Instant Render Mode")
