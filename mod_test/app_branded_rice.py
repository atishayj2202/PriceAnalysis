import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Set Page Configuration
st.set_page_config(
    page_title="Branded Rice Dynamic Pricing Dashboard",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-base {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .kpi-change-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 20px;
        border-top: 4px solid #38bdf8;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
    }
    .kpi-change-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-change-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }
    .kpi-change-sub {
        font-size: 0.88rem;
        color: #f1f5f9;
        margin-top: 4px;
    }
    .profit-section-card {
        background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);
        margin-bottom: 20px;
    }
    .revenue-section-card {
        background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25);
        margin-bottom: 20px;
    }
    .section-title-profit {
        font-size: 1.1rem;
        font-weight: 800;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .section-title-revenue {
        font-size: 1.1rem;
        font-weight: 800;
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .hero-price {
        font-size: 2.6rem;
        font-weight: 900;
        color: #ffffff;
        margin-top: 4px;
    }
    .badge-gold {
        background-color: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #f59e0b;
    }
    .badge-silver {
        background-color: rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #94a3b8;
    }
    .badge-bronze {
        background-color: rgba(168, 85, 247, 0.25);
        color: #c084fc;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #a855f7;
    }
</style>
""", unsafe_allow_html=True)

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "branded_rice_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "pipeline_results.csv")

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Run `construct_branded_rice_data.py` first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        st.error(f"Pipeline results not found at {RESULTS_PATH}. Run `elasticity_pipeline.py` first.")
        st.stop()
    df_res = pd.read_csv(RESULTS_PATH)
    return df_res

df_raw = load_dataset()
df_results = load_results()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/rice-bowl.png", width=70)
st.sidebar.title("Branded Rice Pricing Engine")
st.sidebar.markdown("---")

# Brand Selection
selected_brand = st.sidebar.selectbox(
    "1. Select Target Brand",
    options=["Daawat", "India_Gate", "Fortune"],
    index=0
)

# Dynamically filter and sort Top 3 Models for the selected brand
brand_results_sorted = df_results[df_results['brand'] == selected_brand].sort_values(by="Rank_Score", ascending=False)
top3_df = brand_results_sorted.head(3).reset_index(drop=True)

# Dynamic Top 3 Options list
top3_options = [
    f"🥇 Option 1: {top3_df.iloc[0]['model']} ({top3_df.iloc[0]['seasonality']} + {top3_df.iloc[0]['decay']})",
    f"🥈 Option 2: {top3_df.iloc[1]['model']} ({top3_df.iloc[1]['seasonality']} + {top3_df.iloc[1]['decay']})",
    f"🥉 Option 3: {top3_df.iloc[2]['model']} ({top3_df.iloc[2]['seasonality']} + {top3_df.iloc[2]['decay']})"
]

# Dynamic Model Selector with brand-specific key to reset selection on brand change
selected_model_idx = st.sidebar.radio(
    "2. Select Model Option (Top 3 for Brand)",
    options=[0, 1, 2],
    format_func=lambda i: top3_options[i],
    index=0,
    key=f"dynamic_top3_radio_{selected_brand}"
)

selected_model_row = top3_df.iloc[selected_model_idx]

# Calculate Base and Optimal Metrics based on LATEST Active Week in Dataset
brand_df = df_raw[df_raw['brand'] == selected_brand].sort_values('date').reset_index(drop=True)
last_row = brand_df.iloc[-1]
base_price = last_row['unit_price']
base_cost = last_row['cost_per_unit']
base_qty = last_row['units_sold']
base_rev = base_price * base_qty
base_profit = (base_price - base_cost) * base_qty
latest_date_str = last_row['date'].strftime('%b %d, %Y')

# Calculate Elasticity for the selected model
elas = selected_model_row['mean_elasticity']
if elas == 0.0 or np.isnan(elas):
    elas = -0.574  # Fallback for models with 0 derivative

# Generate fine grid for optimization (-25% to +25%)
p_grid = np.linspace(-25, 25, 501)
prices_grid = base_price * (1.0 + p_grid / 100.0)
qtys_grid = base_qty * ((prices_grid / base_price) ** elas)
revs_grid = prices_grid * qtys_grid
profits_grid = (prices_grid - base_cost) * qtys_grid

# 1. Optimal Profit Point Calculation
opt_prof_idx = np.argmax(profits_grid)
opt_prof_price_pct = p_grid[opt_prof_idx]
opt_prof_price_val = prices_grid[opt_prof_idx]
opt_prof_qty_val = qtys_grid[opt_prof_idx]
opt_prof_rev_val = revs_grid[opt_prof_idx]
opt_prof_val = profits_grid[opt_prof_idx]

# 2. Optimal Revenue Point Calculation
opt_rev_idx = np.argmax(revs_grid)
opt_rev_price_pct = p_grid[opt_rev_idx]
opt_rev_price_val = prices_grid[opt_rev_idx]
opt_rev_qty_val = qtys_grid[opt_rev_idx]
opt_rev_val = revs_grid[opt_rev_idx]
opt_rev_prof_val = profits_grid[opt_rev_idx]

# Helper function to apply high-visibility light font styling to all Plotly figures
def apply_plotly_light_theme(fig, title_text, x_title, y_title, height=400):
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

# Main Title & Header
st.title("🍚 Branded Rice Dynamic Pricing Strategy Dashboard")
st.markdown(f"**Target SKU / Brand**: `{selected_brand}` | **Current Active Baseline Date**: `{latest_date_str}` (Price: **₹{base_price:.2f}/kg**) | **Active Model**: `{selected_model_row['model']}` ({selected_model_row['seasonality']} + {selected_model_row['decay']})")

# Top Banner: Dynamic Model Switcher Cards
st.subheader("🎯 Dynamic Top 3 Model Selection")
c_m1, c_m2, c_m3 = st.columns(3)

with c_m1:
    r1 = top3_df.iloc[0]
    is_sel1 = (selected_model_idx == 0)
    card_border = "border: 2px solid #fbbf24;" if is_sel1 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-gold">🥇 OPTION 1 (RANK 1)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r1['model']}</div>
        <div style="font-size:0.8rem; color:#fde68a;">{r1['seasonality'].upper()} + {r1['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r1['WMAPE']:.2f}%</b> | Elasticity: <b>{r1['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r1['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_m2:
    r2 = top3_df.iloc[1]
    is_sel2 = (selected_model_idx == 1)
    card_border = "border: 2px solid #38bdf8;" if is_sel2 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-silver">🥈 OPTION 2 (RANK 2)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r2['model']}</div>
        <div style="font-size:0.8rem; color:#cbd5e1;">{r2['seasonality'].upper()} + {r2['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r2['WMAPE']:.2f}%</b> | Elasticity: <b>{r2['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r2['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_m3:
    r3 = top3_df.iloc[2]
    is_sel3 = (selected_model_idx == 2)
    card_border = "border: 2px solid #a855f7;" if is_sel3 else ""
    st.markdown(f"""
    <div class="metric-card" style="{card_border}">
        <span class="badge-bronze">🥉 OPTION 3 (RANK 3)</span>
        <div style="font-size:1.4rem; font-weight:800; color:#ffffff; margin-top:4px;">{r3['model']}</div>
        <div style="font-size:0.8rem; color:#e9d5ff;">{r3['seasonality'].upper()} + {r3['decay']}</div>
        <div style="margin-top:6px; font-size:0.85rem; color:#cbd5e1;">
            WMAPE: <b>{r3['WMAPE']:.2f}%</b> | Elasticity: <b>{r3['mean_elasticity']:.3f}</b><br>
            Rank Score: <b>{r3['Rank_Score']:.3f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tabs Navigation Wiring (Properly Mapping Each Tab)
tab1, tab_delta, tab2, tab3, tab4 = st.tabs([
    "🏠 Home: Profit & Revenue Optimization",
    "📊 Selection Change & Delta Analysis",
    "⚔️ Top 3 Models Comparison Matrix",
    "🌊 Seasonality & Fourier Rhythms",
    "🔬 Diagnostics & Exploration"
])

# ---------------------------------------------------------
# TAB 1: HOME PAGE - 2 SECTIONS (PROFIT & REVENUE)
# ---------------------------------------------------------
with tab1:
    st.markdown("## 💰 SECTION 1: GROSS PROFIT MAXIMIZATION STRATEGY")
    st.markdown(f"Calculates the exact retail price that maximizes **Weekly Gross Profit Margin** taking unit cost (₹{base_cost:.2f}/kg) into account.")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    with col_p1:
        st.markdown(f"""
        <div class="profit-section-card">
            <div class="section-title-profit">PROFIT-MAXIMIZING PRICE</div>
            <div class="hero-price">₹{opt_prof_price_val:.2f}<span style="font-size:1.1rem; font-weight:normal;">/kg</span></div>
            <div class="metric-base">Current Baseline: <b>₹{base_price:.2f}/kg</b> ({latest_date_str})</div>
            <div style="color:#34d399; font-size:0.9rem; font-weight:bold; margin-top:4px;">Shift: {opt_prof_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_prof_val/1e7:.2f} Cr</div>
            <div class="metric-base">Current Baseline: <b>₹{base_profit/1e7:.2f} Cr</b></div>
            <div style="color:#34d399; font-size:0.85rem; font-weight:bold; margin-top:4px;">Profit Lift: {((opt_prof_val - base_profit)/base_profit)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_prof_rev_val/1e7:.2f} Cr</div>
            <div class="metric-base">Current Baseline: <b>₹{base_rev/1e7:.2f} Cr</b></div>
            <div style="color:#38bdf8; font-size:0.85rem; font-weight:bold; margin-top:4px;">Revenue Shift: {((opt_prof_rev_val - base_rev)/base_rev)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume</div>
            <div class="metric-value">{opt_prof_qty_val/1e3:.1f}k KG</div>
            <div class="metric-base">Current Baseline: <b>{base_qty/1e3:.1f}k KG</b></div>
            <div style="color:#f1f5f9; font-size:0.85rem; font-weight:bold; margin-top:4px;">Volume Shift: {((opt_prof_qty_val - base_qty)/base_qty)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Plotly Gross Profit Optimization Chart (Stacked Full Width)
    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=p_grid, y=profits_grid / 1e7,
        mode='lines',
        name='Gross Profit (₹ Cr)',
        line=dict(color='#10b981', width=3)
    ))
    fig_prof.add_trace(go.Scatter(
        x=[opt_prof_price_pct], y=[opt_prof_val / 1e7],
        mode='markers+text',
        name=f'Optimal Profit Peak (₹{opt_prof_price_val:.2f}/kg)',
        marker=dict(color='#fbbf24', size=14, symbol='star'),
        text=[f"Max Profit: ₹{opt_prof_val/1e7:.2f} Cr"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=12)
    ))
    fig_prof.add_vline(
        x=0.0, line_dash="dash", line_color="#ffffff",
        annotation_text=f"Current Price (₹{base_price:.2f}/kg on {latest_date_str})",
        annotation_font=dict(color="#ffffff", size=12)
    )
    apply_plotly_light_theme(fig_prof, f"Gross Profit Optimization Curve for {selected_brand}", "Price Change (%)", "Weekly Gross Profit (₹ Crore)", height=420)
    st.plotly_chart(fig_prof, use_container_width=True)

    st.markdown("---")

    # SECTION 2: REVENUE MAXIMIZATION STRATEGY
    st.markdown("## 📈 SECTION 2: REVENUE MAXIMIZATION STRATEGY")
    st.markdown("Calculates the exact retail price that maximizes **Total Weekly Top-Line Revenue** to capture market share.")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.markdown(f"""
        <div class="revenue-section-card">
            <div class="section-title-revenue">REVENUE-MAXIMIZING PRICE</div>
            <div class="hero-price">₹{opt_rev_price_val:.2f}<span style="font-size:1.1rem; font-weight:normal;">/kg</span></div>
            <div class="metric-base">Current Baseline: <b>₹{base_price:.2f}/kg</b> ({latest_date_str})</div>
            <div style="color:#7dd3fc; font-size:0.9rem; font-weight:bold; margin-top:4px;">Shift: {opt_rev_price_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Revenue</div>
            <div class="metric-value">₹{opt_rev_val/1e7:.2f} Cr</div>
            <div class="metric-base">Current Baseline: <b>₹{base_rev/1e7:.2f} Cr</b></div>
            <div style="color:#38bdf8; font-size:0.85rem; font-weight:bold; margin-top:4px;">Revenue Lift: {((opt_rev_val - base_rev)/base_rev)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Weekly Profit</div>
            <div class="metric-value">₹{opt_rev_prof_val/1e7:.2f} Cr</div>
            <div class="metric-base">Current Baseline: <b>₹{base_profit/1e7:.2f} Cr</b></div>
            <div style="color:#34d399; font-size:0.85rem; font-weight:bold; margin-top:4px;">Profit Shift: {((opt_rev_prof_val - base_profit)/base_profit)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Demand Volume</div>
            <div class="metric-value">{opt_rev_qty_val/1e3:.1f}k KG</div>
            <div class="metric-base">Current Baseline: <b>{base_qty/1e3:.1f}k KG</b></div>
            <div style="color:#f1f5f9; font-size:0.85rem; font-weight:bold; margin-top:4px;">Volume Shift: {((opt_rev_qty_val - base_qty)/base_qty)*100:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Plotly Revenue Optimization Chart (Stacked Full Width)
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=p_grid, y=revs_grid / 1e7,
        mode='lines',
        name='Weekly Revenue (₹ Cr)',
        line=dict(color='#38bdf8', width=3)
    ))
    fig_rev.add_trace(go.Scatter(
        x=[opt_rev_price_pct], y=[opt_rev_val / 1e7],
        mode='markers+text',
        name=f'Optimal Revenue Peak (₹{opt_rev_price_val:.2f}/kg)',
        marker=dict(color='#7dd3fc', size=14, symbol='diamond'),
        text=[f"Max Revenue: ₹{opt_rev_val/1e7:.2f} Cr"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=12)
    ))
    fig_rev.add_vline(
        x=0.0, line_dash="dash", line_color="#ffffff",
        annotation_text=f"Current Price (₹{base_price:.2f}/kg on {latest_date_str})",
        annotation_font=dict(color="#ffffff", size=12)
    )
    apply_plotly_light_theme(fig_rev, f"Weekly Revenue Optimization Curve for {selected_brand}", "Price Change (%)", "Weekly Revenue (₹ Crore)", height=420)
    st.plotly_chart(fig_rev, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: SELECTION CHANGE & DELTA ANALYSIS (SLIDER MOVED HERE & STACKED GRAPHS)
# ---------------------------------------------------------
with tab_delta:
    st.header(f"📊 Impact Analysis of Selected Price Shift ({selected_brand})")
    st.markdown(f"Adjust the manual slider below to examine the exact delta shifts in **Price, Demand Quantity, Revenue, and Gross Profit** against the active baseline date (**{latest_date_str}**).")
    
    # Manual Slider moved directly into Tab 2
    custom_price_change = st.slider(
        "💡 Adjust Manual Price Shift vs. Baseline (%)",
        min_value=-25.0, max_value=+25.0, value=0.0, step=0.5,
        key=f"tab2_custom_price_slider_{selected_brand}"
    )
    
    # Selected Point Metrics based on slider
    cust_price_val = base_price * (1.0 + custom_price_change / 100.0)
    cust_qty_val = base_qty * ((cust_price_val / base_price) ** elas)
    cust_rev_val = cust_price_val * cust_qty_val
    cust_profit_val = (cust_price_val - base_cost) * cust_qty_val

    pct_change_price = custom_price_change
    pct_change_qty = ((cust_qty_val - base_qty) / base_qty) * 100.0
    pct_change_rev = ((cust_rev_val - base_rev) / base_rev) * 100.0
    pct_change_profit = ((cust_profit_val - base_profit) / base_profit) * 100.0

    # KPI Row with Current Baseline, Projected Data, and Percent Change
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    color_price_delta = "#34d399" if pct_change_price >= 0 else "#f43f5e"
    color_qty_delta = "#34d399" if pct_change_qty >= 0 else "#f43f5e"
    color_rev_delta = "#34d399" if pct_change_rev >= 0 else "#f43f5e"
    color_prof_delta = "#34d399" if pct_change_profit >= 0 else "#f43f5e"
    
    with col_d1:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color: #38bdf8;">
            <div class="kpi-change-title">1. RETAIL PRICE</div>
            <div class="kpi-change-val">₹{cust_price_val:.2f} <span style="font-size:1.0rem; font-weight:normal;">/kg</span></div>
            <div class="kpi-change-sub">Current Base: <b>₹{base_price:.2f}/kg</b> ({latest_date_str})</div>
            <div style="font-size:0.95rem; font-weight:bold; color:{color_price_delta}; margin-top:6px;">
                Shift: {pct_change_price:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color: #a855f7;">
            <div class="kpi-change-title">2. DEMAND QUANTITY</div>
            <div class="kpi-change-val">{cust_qty_val/1e3:.1f}k <span style="font-size:1.0rem; font-weight:normal;">KG</span></div>
            <div class="kpi-change-sub">Current Base: <b>{base_qty/1e3:.1f}k KG</b></div>
            <div style="font-size:0.95rem; font-weight:bold; color:{color_qty_delta}; margin-top:6px;">
                Change: {pct_change_qty:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d3:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color: #7dd3fc;">
            <div class="kpi-change-title">3. WEEKLY REVENUE</div>
            <div class="kpi-change-val">₹{cust_rev_val/1e7:.2f} <span style="font-size:1.0rem; font-weight:normal;">Cr</span></div>
            <div class="kpi-change-sub">Current Base: <b>₹{base_rev/1e7:.2f} Cr</b></div>
            <div style="font-size:0.95rem; font-weight:bold; color:{color_rev_delta}; margin-top:6px;">
                Change: {pct_change_rev:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d4:
        st.markdown(f"""
        <div class="kpi-change-card" style="border-top-color: #10b981;">
            <div class="kpi-change-title">4. GROSS PROFIT</div>
            <div class="kpi-change-val">₹{cust_profit_val/1e7:.2f} <span style="font-size:1.0rem; font-weight:normal;">Cr</span></div>
            <div class="kpi-change-sub">Current Base: <b>₹{base_profit/1e7:.2f} Cr</b></div>
            <div style="font-size:0.95rem; font-weight:bold; color:{color_prof_delta}; margin-top:6px;">
                Change: {pct_change_profit:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 3 Dedicated Impact Graphs (Stacked Full Width, One Below Another)")

    # Graph 1: Profit Change Curve (Full Width)
    fig_g1 = go.Figure()
    fig_g1.add_trace(go.Scatter(x=p_grid, y=profits_grid/1e7, mode='lines', name='Profit Curve', line=dict(color='#10b981', width=3)))
    fig_g1.add_trace(go.Scatter(x=[0.0], y=[base_profit/1e7], mode='markers', name='Baseline (0%)', marker=dict(color='#e2e8f0', size=10, symbol='circle')))
    fig_g1.add_trace(go.Scatter(
        x=[custom_price_change], y=[cust_profit_val/1e7], mode='markers+text',
        name=f'Selected ({custom_price_change:+.1f}%)',
        marker=dict(color='#fbbf24', size=14, symbol='star'),
        text=[f"{pct_change_profit:+.1f}%"], textposition="top center",
        textfont=dict(color="#ffffff", size=13)
    ))
    fig_g1.add_vline(x=custom_price_change, line_dash="dot", line_color="#fbbf24")
    apply_plotly_light_theme(fig_g1, "1. Gross Profit Impact Curve vs. Price Adjustment", "Price Change (%)", "Profit (₹ Cr)", height=420)
    st.plotly_chart(fig_g1, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graph 2: Revenue Change Curve (Full Width)
    fig_g2 = go.Figure()
    fig_g2.add_trace(go.Scatter(x=p_grid, y=revs_grid/1e7, mode='lines', name='Revenue Curve', line=dict(color='#38bdf8', width=3)))
    fig_g2.add_trace(go.Scatter(x=[0.0], y=[base_rev/1e7], mode='markers', name='Baseline (0%)', marker=dict(color='#e2e8f0', size=10, symbol='circle')))
    fig_g2.add_trace(go.Scatter(
        x=[custom_price_change], y=[cust_rev_val/1e7], mode='markers+text',
        name=f'Selected ({custom_price_change:+.1f}%)',
        marker=dict(color='#7dd3fc', size=14, symbol='diamond'),
        text=[f"{pct_change_rev:+.1f}%"], textposition="top center",
        textfont=dict(color="#ffffff", size=13)
    ))
    fig_g2.add_vline(x=custom_price_change, line_dash="dot", line_color="#7dd3fc")
    apply_plotly_light_theme(fig_g2, "2. Revenue Impact Curve vs. Price Adjustment", "Price Change (%)", "Revenue (₹ Cr)", height=420)
    st.plotly_chart(fig_g2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graph 3: Quantity (Demand) Change Curve (Full Width)
    fig_g3 = go.Figure()
    fig_g3.add_trace(go.Scatter(x=p_grid, y=qtys_grid/1e3, mode='lines', name='Demand Curve', line=dict(color='#a855f7', width=3)))
    fig_g3.add_trace(go.Scatter(x=[0.0], y=[base_qty/1e3], mode='markers', name='Baseline (0%)', marker=dict(color='#e2e8f0', size=10, symbol='circle')))
    fig_g3.add_trace(go.Scatter(
        x=[custom_price_change], y=[cust_qty_val/1e3], mode='markers+text',
        name=f'Selected ({custom_price_change:+.1f}%)',
        marker=dict(color='#c084fc', size=14, symbol='square'),
        text=[f"{pct_change_qty:+.1f}%"], textposition="top center",
        textfont=dict(color="#ffffff", size=13)
    ))
    fig_g3.add_vline(x=custom_price_change, line_dash="dot", line_color="#c084fc")
    apply_plotly_light_theme(fig_g3, "3. Demand Volume Impact Curve vs. Price Adjustment", "Price Change (%)", "Demand Volume (k KG)", height=420)
    st.plotly_chart(fig_g3, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: TOP 3 MODELS COMPARISON MATRIX (STACKED FULL WIDTH GRAPHS)
# ---------------------------------------------------------
with tab2:
    st.header(f"⚔️ Top 3 Model Options Comparison for {selected_brand}")
    st.markdown("Detailed technical comparison of the **Top 3 Recommended Models** for this brand across Profit, Revenue, Quantity, and Price dynamics.")
    
    m1 = top3_df.iloc[0]
    m2 = top3_df.iloc[1] if len(top3_df) > 1 else m1
    m3 = top3_df.iloc[2] if len(top3_df) > 2 else m1
    
    e1 = m1['mean_elasticity'] if m1['mean_elasticity'] != 0.0 else -0.574
    q1 = base_qty * ((prices_grid / base_price) ** e1)
    prof1 = (prices_grid - base_cost) * q1
    rev1 = prices_grid * q1
    
    e2 = m2['mean_elasticity'] if m2['mean_elasticity'] != 0.0 else -0.574
    q2 = base_qty * ((prices_grid / base_price) ** e2)
    prof2 = (prices_grid - base_cost) * q2
    rev2 = prices_grid * q2
    
    e3 = m3['mean_elasticity'] if m3['mean_elasticity'] != 0.0 else -0.574
    q3 = base_qty * ((prices_grid / base_price) ** e3)
    prof3 = (prices_grid - base_cost) * q3
    rev3 = prices_grid * q3
    
    st.subheader("📈 Overlaid Model Comparison Graphs (Stacked Full Width)")
    
    # Graph 1: Profit Comparison (Full Width)
    fig_c1 = go.Figure()
    fig_c1.add_trace(go.Scatter(x=p_grid, y=prof1/1e7, mode='lines', name=f"Rank 1: {m1['model']}", line=dict(color='#fbbf24', width=3)))
    fig_c1.add_trace(go.Scatter(x=p_grid, y=prof2/1e7, mode='lines', name=f"Rank 2: {m2['model']}", line=dict(color='#38bdf8', width=3, dash='dash')))
    fig_c1.add_trace(go.Scatter(x=p_grid, y=prof3/1e7, mode='lines', name=f"Rank 3: {m3['model']}", line=dict(color='#c084fc', width=3, dash='dot')))
    apply_plotly_light_theme(fig_c1, "1. Gross Profit Comparison Across Top 3 Models", "Price Change (%)", "Gross Profit (₹ Cr)", height=420)
    st.plotly_chart(fig_c1, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graph 2: Revenue Comparison (Full Width)
    fig_c2 = go.Figure()
    fig_c2.add_trace(go.Scatter(x=p_grid, y=rev1/1e7, mode='lines', name=f"Rank 1: {m1['model']}", line=dict(color='#fbbf24', width=3)))
    fig_c2.add_trace(go.Scatter(x=p_grid, y=rev2/1e7, mode='lines', name=f"Rank 2: {m2['model']}", line=dict(color='#38bdf8', width=3, dash='dash')))
    fig_c2.add_trace(go.Scatter(x=p_grid, y=rev3/1e7, mode='lines', name=f"Rank 3: {m3['model']}", line=dict(color='#c084fc', width=3, dash='dot')))
    apply_plotly_light_theme(fig_c2, "2. Revenue Comparison Across Top 3 Models", "Price Change (%)", "Revenue (₹ Cr)", height=420)
    st.plotly_chart(fig_c2, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graph 3: Quantity Comparison (Full Width)
    fig_c3 = go.Figure()
    fig_c3.add_trace(go.Scatter(x=p_grid, y=q1/1e3, mode='lines', name=f"Rank 1: {m1['model']}", line=dict(color='#fbbf24', width=3)))
    fig_c3.add_trace(go.Scatter(x=p_grid, y=q2/1e3, mode='lines', name=f"Rank 2: {m2['model']}", line=dict(color='#38bdf8', width=3, dash='dash')))
    fig_c3.add_trace(go.Scatter(x=p_grid, y=q3/1e3, mode='lines', name=f"Rank 3: {m3['model']}", line=dict(color='#c084fc', width=3, dash='dot')))
    apply_plotly_light_theme(fig_c3, "3. Demand Quantity Comparison Across Top 3 Models", "Price Change (%)", "Demand Volume (k KG)", height=420)
    st.plotly_chart(fig_c3, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graph 4: Price vs Quantity Elasticity Demand Curve (Full Width)
    fig_c4 = go.Figure()
    fig_c4.add_trace(go.Scatter(x=prices_grid, y=q1/1e3, mode='lines', name=f"Rank 1 (ε={e1:.2f})", line=dict(color='#fbbf24', width=3)))
    fig_c4.add_trace(go.Scatter(x=prices_grid, y=q2/1e3, mode='lines', name=f"Rank 2 (ε={e2:.2f})", line=dict(color='#38bdf8', width=3, dash='dash')))
    fig_c4.add_trace(go.Scatter(x=prices_grid, y=q3/1e3, mode='lines', name=f"Rank 3 (ε={e3:.2f})", line=dict(color='#c084fc', width=3, dash='dot')))
    fig_c4.add_vline(x=base_price, line_dash="dash", line_color="#ffffff", annotation_text=f"Base Price: ₹{base_price:.1f}/kg", annotation_font=dict(color="#ffffff", size=12))
    apply_plotly_light_theme(fig_c4, "4. Price vs. Quantity Elasticity Slopes Across Top 3 Models", "Retail Price (₹/kg)", "Demand Volume (k KG)", height=420)
    st.plotly_chart(fig_c4, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Top 3 Models Comparison Matrix Table")
    
    comp_matrix = pd.DataFrame({
        "Metric": [
            "Model Name",
            "Seasonality Mode",
            "Decay Setting",
            "WMAPE Forecast Error",
            "Seasonal R² Variance",
            "Seasonal WMAPE Gain",
            "Own-Price Elasticity (ε)",
            "Optimal Profit Price (₹/kg)",
            "Optimal Revenue Price (₹/kg)",
            "Composite Rank Score",
            "Operational Verdict"
        ],
        f"Option 1 ({m1['model']})": [
            m1['model'],
            m1['seasonality'].upper(),
            m1['decay'],
            f"{m1['WMAPE']:.2f}%",
            f"{m1['seasonal_r2']*100:.1f}%",
            f"+{m1['seasonal_wmape_gain']:.2f}%",
            f"{m1['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof1)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev1)]/100.0):.2f}",
            f"{m1['Rank_Score']:.3f}",
            "RECOMMENDED WINNER"
        ],
        f"Option 2 ({m2['model']})": [
            m2['model'],
            m2['seasonality'].upper(),
            m2['decay'],
            f"{m2['WMAPE']:.2f}%",
            f"{m2['seasonal_r2']*100:.1f}%",
            f"+{m2['seasonal_wmape_gain']:.2f}%",
            f"{m2['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof2)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev2)]/100.0):.2f}",
            f"{m2['Rank_Score']:.3f}",
            "Strong Alternative"
        ],
        f"Option 3 ({m3['model']})": [
            m3['model'],
            m3['seasonality'].upper(),
            m3['decay'],
            f"{m3['WMAPE']:.2f}%",
            f"{m3['seasonal_r2']*100:.1f}%",
            f"+{m3['seasonal_wmape_gain']:.2f}%",
            f"{m3['mean_elasticity']:.3f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(prof3)]/100.0):.2f}",
            f"₹{base_price*(1.0 + p_grid[np.argmax(rev3)]/100.0):.2f}",
            f"{m3['Rank_Score']:.3f}",
            "Robust Runner-up"
        ]
    })
    
    st.dataframe(comp_matrix, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: SEASONALITY & FOURIER RHYTHMS
# ---------------------------------------------------------
with tab3:
    st.header("🌊 Fourier Seasonality & Demand Rhythms")
    st.markdown("Our pipeline uses continuous Fourier Sine and Cosine terms ($\sin_{52,k}, \cos_{52,k}$) to model annual and quarterly demand rhythms.")
    
    st.subheader("Empirical Seasonal Multiplier Curve (52-Week Cycle)")
    brand_df_s = df_raw[df_raw['brand'] == selected_brand].copy()
    brand_df_s['week_of_year'] = brand_df_s['date'].dt.isocalendar().week
    weekly_pattern = brand_df_s.groupby('week_of_year')['units_sold'].mean()
    weekly_pattern_norm = weekly_pattern / weekly_pattern.mean()
    
    # Plotly Seasonal Chart (Full Width)
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=list(weekly_pattern_norm.index),
        y=list(weekly_pattern_norm.values),
        mode='lines',
        name='Seasonal Multiplier',
        line=dict(color='#c084fc', width=3)
    ))
    fig_s.add_hline(y=1.0, line_dash="dot", line_color="#ffffff", annotation_text="Baseline Demand (1.0x)", annotation_font=dict(color="#ffffff", size=12))
    apply_plotly_light_theme(fig_s, f"{selected_brand} Seasonal Demand Multiplier", "Week of Year (1 to 52)", "Seasonal Multiplier (x)", height=420)
    st.plotly_chart(fig_s, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Seasonal Accuracy & Value Summary")
    st.markdown(f"""
    - **Seasonal Variance Explained ($R^2$)**: **{selected_model_row['seasonal_r2']*100:.1f}%** of weekly sales fluctuations are driven by predictable calendar rhythms.
    - **WMAPE Accuracy Gain**: Adding Fixed Fourier seasonality reduced out-of-sample forecasting error by **+{selected_model_row['seasonal_wmape_gain']:.2f}% percentage points**.
    - **Peak Demand Windows**:
      - *Festival Surge (Weeks 40-46 / Oct-Nov)*: **+35% demand lift** (Diwali/Durga Puja).
      - *Winter Harvest & Wedding Surge (Weeks 2-8 / Jan-Feb)*: **+15% demand lift**.
      - *Monsoon Dip (Weeks 27-36 / Jul-Sep)*: **-15% demand dip**.
    """)

# ---------------------------------------------------------
# TAB 5: DIAGNOSTICS & DATA EXPLORATION (INCLUDES TABULAR DATA)
# ---------------------------------------------------------
with tab4:
    st.header("🔬 Model Diagnostics & Historical Data Exploration")
    st.markdown(f"Examine historical price vs. cost trends, sales volume patterns, and inspect raw weekly dataset rows for **{selected_brand}**.")
    
    brand_df_diag = df_raw[df_raw['brand'] == selected_brand].sort_values('date', ascending=False).reset_index(drop=True)
    
    # Plotly Time Series Chart (Full Width)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=brand_df_diag['date'], y=brand_df_diag['unit_price'], mode='lines', name='Retail Price (₹/kg)', line=dict(color='#38bdf8', width=2.5)))
    fig_ts.add_trace(go.Scatter(x=brand_df_diag['date'], y=brand_df_diag['cost_per_unit'], mode='lines', name='Wholesale COGS (₹/kg)', line=dict(color='#cbd5e1', width=2, dash='dash')))
    apply_plotly_light_theme(fig_ts, f"{selected_brand} Historical Retail Price vs. COGS (2021-2025)", "Date", "Price (₹/kg)", height=420)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # Interactive Tabular View of Historical Data
    st.subheader(f"📋 Historical Dataset Rows for {selected_brand} (Tabular View)")
    st.markdown(f"Below is the complete weekly historical record (234 weeks) for **{selected_brand}**, sorted from latest week (**{latest_date_str}**) to oldest.")

    tabular_df = brand_df_diag.copy()
    tabular_df['date_str'] = tabular_df['date'].dt.strftime('%Y-%m-%d')
    tabular_df['gross_margin_%'] = ((tabular_df['unit_price'] - tabular_df['cost_per_unit']) / tabular_df['unit_price'] * 100).map('{:.1f}%'.format)
    tabular_df['unit_price_str'] = tabular_df['unit_price'].map('₹{:.2f}'.format)
    tabular_df['cost_per_unit_str'] = tabular_df['cost_per_unit'].map('₹{:.2f}'.format)
    tabular_df['units_sold_str'] = tabular_df['units_sold'].map('{:,.0f} KG'.format)
    tabular_df['weekly_revenue_cr'] = (tabular_df['unit_price'] * tabular_df['units_sold'] / 1e7).map('₹{:.2f} Cr'.format)
    
    disp_table = tabular_df[[
        'date_str', 'brand', 'unit_price_str', 'cost_per_unit_str', 'gross_margin_%', 
        'units_sold_str', 'weekly_revenue_cr', 'is_promo', 'is_festival'
    ]].rename(columns={
        'date_str': 'Date',
        'brand': 'Brand',
        'unit_price_str': 'Retail Price',
        'cost_per_unit_str': 'Wholesale COGS',
        'gross_margin_%': 'Gross Margin',
        'units_sold_str': 'Weekly Volume',
        'weekly_revenue_cr': 'Weekly Revenue',
        'is_promo': 'Promo Flag',
        'is_festival': 'Festival Flag'
    })

    st.dataframe(disp_table, use_container_width=True, height=380)

st.markdown("---")
st.caption("Branded Rice Pricing Engine | Interactive Plotly Dashboard (No Seaborn)")
